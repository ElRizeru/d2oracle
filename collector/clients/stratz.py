"""Async STRATZ GraphQL API client with header-aware rate limiting and retries."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from collector.rate_limit import MultiWindowRateLimiter
from shared.constants import LEADERBOARD_DIVISIONS, STRATZ_GRAPHQL_URL

logger = logging.getLogger(__name__)


class StratzTransientError(RuntimeError):
    """Retryable STRATZ API error (429, 5xx, or transient transport error)."""


class StratzGraphQLError(RuntimeError):
    """Non-retryable STRATZ GraphQL query or schema error."""


class StratzBatchFallbackError(StratzGraphQLError):
    """Raised when a batched GetMatches query should fall back to single-match calls."""


@dataclass(slots=True)
class BatchMatchesResult:
    """Parsed result of a batched GetMatches GraphQL query."""

    matches: dict[int, dict[str, Any] | None]
    failed_ids: set[int] = field(default_factory=set)


GAME_VERSIONS_QUERY = """
query GetGameVersions {
  constants {
    gameVersions {
      id
      name
      asOfDateTime
    }
  }
}
"""

CONSTANTS_QUERY = """
query GetConstants($gv: Short!) {
  constants {
    gameVersions {
      id
      name
      asOfDateTime
    }
    heroes(gameVersionId: $gv) {
      id
      shortName
      displayName
      talents { abilityId slot }
      abilities { slot abilityId }
    }
    items(gameVersionId: $gv) {
      id
      shortName
      displayName
      isSupportFullItem
      stat { cost isRecipe isPurchasable isStackable neutralItemTier }
    }
    abilities(gameVersionId: $gv) {
      id
      name
      isTalent
      language { displayName }
    }
  }
}
"""

MATCH_SELECTION_FIELDS = """
    id
    didRadiantWin
    durationSeconds
    startDateTime
    sequenceNum
    gameVersionId
    gameMode
    lobbyType
    bracket
    rank
    averageRank
    actualRank
    parsedDateTime
    leagueId
    topLaneOutcome
    midLaneOutcome
    bottomLaneOutcome
    radiantNetworthLeads
    radiantExperienceLeads
    players {
      steamAccountId
      playerSlot
      isRadiant
      heroId
      variant
      position
      leaverStatus
      kills
      deaths
      assists
      numLastHits
      numDenies
      goldPerMinute
      experiencePerMinute
      networth
      level
      heroDamage
      towerDamage
      heroHealing
      item0Id
      item1Id
      item2Id
      item3Id
      item4Id
      item5Id
      neutral0Id
      abilities {
        abilityId
        time
        level
        isTalent
      }
      stats {
        itemPurchases { time itemId }
        heroDamageReport {
          receivedTotal { physicalDamage magicalDamage pureDamage }
        }
      }
    }
"""

MATCH_QUERY = f"""
query GetMatch($id: Long!) {{
  match(id: $id) {{{MATCH_SELECTION_FIELDS}  }}
}}
"""

LEADERBOARD_QUERY_TEMPLATE = """
query GetLeaderboard($skip: Long, $take: Long) {{
  leaderboard {{
    season(request: {{ leaderBoardDivision: {division} }}) {{
      players(skip: $skip, take: $take) {{
        steamAccountId
      }}
    }}
  }}
}}
"""


class StratzClient:
    """Async client for STRATZ GraphQL API."""

    def __init__(
        self,
        token: str,
        rate_limiter: MultiWindowRateLimiter,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = STRATZ_GRAPHQL_URL,
    ) -> None:
        self._token = token
        self.rate_limiter = rate_limiter
        self.base_url = base_url
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=30.0)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "User-Agent": "STRATZ_API",
            "Content-Type": "application/json",
        }

    async def _execute_with_errors(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        *,
        allow_partial_data: bool = False,
    ) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
        payload: dict[str, Any] = {"query": query}
        if variables is not None:
            payload["variables"] = variables

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((StratzTransientError, httpx.TransportError, httpx.TimeoutException)),
            stop=stop_after_attempt(4),
            wait=wait_exponential_jitter(initial=1.0, max=20.0),
            reraise=True,
        ):
            with attempt:
                await self.rate_limiter.acquire()
                resp = await self._client.post(
                    self.base_url,
                    json=payload,
                    headers=self._headers(),
                    timeout=30.0,
                )
                self.rate_limiter.update_from_headers(resp.headers)

                if resp.status_code == 429:
                    wait = self.rate_limiter.notify_rate_limited(resp.headers)
                    logger.warning("STRATZ returned 429; cooling down %.1fs", wait)
                    raise StratzTransientError("STRATZ HTTP 429")
                if resp.status_code >= 500:
                    raise StratzTransientError(f"STRATZ HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    raise StratzGraphQLError(
                        f"STRATZ HTTP {resp.status_code}: {resp.text[:300]}"
                    )

                body = resp.json()
                errors = body.get("errors")
                data = body.get("data")
                if errors and not (allow_partial_data and isinstance(data, dict)):
                    msg = "; ".join(str(e.get("message", e)) for e in errors)
                    raise StratzGraphQLError(f"STRATZ GraphQL error: {msg}")
                if errors and allow_partial_data:
                    logger.warning("STRATZ partial GraphQL errors: %s", errors[0])
                if not isinstance(data, dict):
                    raise StratzGraphQLError("STRATZ response missing data object")
                return data, list(errors) if isinstance(errors, list) else None

        raise AssertionError("Unreachable retry termination")

    async def execute(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        *,
        allow_partial_data: bool = False,
    ) -> dict[str, Any]:
        data, _errors = await self._execute_with_errors(
            query, variables=variables, allow_partial_data=allow_partial_data
        )
        return data

    async def get_game_versions(self) -> list[dict[str, Any]]:
        data = await self.execute(GAME_VERSIONS_QUERY)
        return list((data.get("constants") or {}).get("gameVersions") or [])

    async def get_constants(self, game_version_id: int) -> dict[str, Any]:
        data = await self.execute(CONSTANTS_QUERY, variables={"gv": int(game_version_id)})
        return dict(data.get("constants") or {})

    async def get_match(self, match_id: int) -> dict[str, Any] | None:
        data = await self.execute(MATCH_QUERY, variables={"id": int(match_id)})
        return data.get("match")

    async def get_matches_batch(self, match_ids: list[int]) -> BatchMatchesResult:
        """Fetch multiple full matches in one aliased GraphQL query."""
        if not match_ids:
            return BatchMatchesResult(matches={})

        aliases = "\n".join(
            f"  m{idx}: match(id: {int(mid)}) {{{MATCH_SELECTION_FIELDS}  }}"
            for idx, mid in enumerate(match_ids)
        )
        query = f"query GetMatchesBatch {{\n{aliases}\n}}"
        data, errors = await self._execute_with_errors(query, allow_partial_data=True)

        failed_ids: set[int] = set()
        if errors:
            for err in errors:
                msg = str(err.get("message", err)) if isinstance(err, dict) else str(err)
                lower_msg = msg.lower()
                if "rebuilt at a time" in lower_msg or "not cached" in lower_msg:
                    raise StratzBatchFallbackError(msg)
                path = err.get("path") if isinstance(err, dict) else None
                if (
                    isinstance(path, list)
                    and path
                    and isinstance(path[0], str)
                    and path[0].startswith("m")
                    and path[0][1:].isdigit()
                ):
                    idx = int(path[0][1:])
                    if 0 <= idx < len(match_ids):
                        failed_ids.add(int(match_ids[idx]))
                        continue
                raise StratzBatchFallbackError(msg)

            if len(failed_ids) == len(match_ids):
                raise StratzBatchFallbackError("All matches in batch had GraphQL errors")

        matches: dict[int, dict[str, Any] | None] = {}
        for idx, mid in enumerate(match_ids):
            mid_int = int(mid)
            if mid_int not in failed_ids:
                matches[mid_int] = data.get(f"m{idx}")
        return BatchMatchesResult(matches=matches, failed_ids=failed_ids)

    async def get_players_matches(
        self,
        steam_account_ids: list[int],
        game_version_id: int | Sequence[int],
        *,
        take: int = 100,
        start_datetime: int | None = None,
    ) -> dict[int, list[dict[str, Any]]]:
        """Batch BFS query for up to 10 players without playbackData."""
        if not steam_account_ids:
            return {}
        if len(steam_account_ids) > 10:
            raise ValueError("STRATZ player batch query supports at most 10 players at once")

        if isinstance(game_version_id, int):
            gv_ids = [int(game_version_id)]
        else:
            gv_ids = [int(x) for x in game_version_id]
        if not gv_ids:
            raise ValueError("game_version_id must contain at least one game version ID")

        multi_gv = len(gv_ids) > 1
        gv_filter = "gameVersionIds: $gvs" if multi_gv else "gameVersionIds: [$gv]"
        gv_decl = "$gvs: [Int]" if multi_gv else "$gv: Int!"
        use_since = start_datetime is not None and int(start_datetime) > 0
        since_clause = ", startDateTime: $since" if use_since else ""
        aliases: list[str] = []
        for idx, sid in enumerate(steam_account_ids):
            aliases.append(
                f"p{idx}: player(steamAccountId: {int(sid)}) {{ "
                f"matches(request: {{ take: {int(take)}, isParsed: true, "
                f"{gv_filter}, lobbyTypeIds: [7], bracketIds: [8]{since_clause} }}) {{ "
                f"id startDateTime parsedDateTime durationSeconds gameMode }} }}"
            )
        if use_since:
            query = f"query GetPlayersMatches({gv_decl}, $since: Long) {{ {' '.join(aliases)} }}"
            variables: dict[str, Any] = (
                {"gvs": gv_ids, "since": int(start_datetime)}
                if multi_gv
                else {"gv": gv_ids[0], "since": int(start_datetime)}
            )
        else:
            query = f"query GetPlayersMatches({gv_decl}) {{ {' '.join(aliases)} }}"
            variables = {"gvs": gv_ids} if multi_gv else {"gv": gv_ids[0]}

        data = await self.execute(
            query, variables=variables, allow_partial_data=True
        )

        results: dict[int, list[dict[str, Any]]] = {}
        for idx, sid in enumerate(steam_account_ids):
            player_node = data.get(f"p{idx}") or {}
            matches = player_node.get("matches") or []
            results[int(sid)] = list(matches)
        return results

    async def get_leaderboard_players(
        self, division: str, *, skip: int = 4500, take: int = 100
    ) -> list[int]:
        if division not in LEADERBOARD_DIVISIONS:
            raise ValueError(f"Unsupported leaderboard division: {division}")
        query = LEADERBOARD_QUERY_TEMPLATE.format(division=division)
        data = await self.execute(
            query,
            variables={"skip": int(skip), "take": int(take)},
            allow_partial_data=True,
        )
        season = (data.get("leaderboard") or {}).get("season") or {}
        players = season.get("players") or []
        return [int(p["steamAccountId"]) for p in players if p.get("steamAccountId")]
