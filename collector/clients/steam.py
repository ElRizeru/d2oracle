"""Async Steam Web API client for Dota 2 match data and live discovery."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from collector.rate_limit import MultiWindowRateLimiter
from shared.constants import STEAM_API_BASE_URL

logger = logging.getLogger(__name__)
_KEY_QUERY_RE = re.compile(r"([?&]key=)[^&\s\"']+", flags=re.IGNORECASE)


class SteamTransientError(RuntimeError):
    """Retryable Steam Web API error (429, 5xx, transport, or timeout)."""


class SteamAPIError(RuntimeError):
    """Non-retryable Steam Web API error (401/403/4xx or invalid payload)."""


class SteamClient:
    """Async client for Steam Web API (IDOTA2Match_570 and IGCVersion_570)."""

    def __init__(
        self,
        api_key: str,
        rate_limiter: MultiWindowRateLimiter,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = STEAM_API_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self.rate_limiter = rate_limiter
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=25.0)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _redact_secrets(self, text: str) -> str:
        scrubbed = _KEY_QUERY_RE.sub(r"\1[REDACTED]", text)
        if self._api_key:
            scrubbed = scrubbed.replace(self._api_key, "[REDACTED]")
        return scrubbed

    def _headers(self) -> dict[str, str]:
        return {"x-webapi-key": self._api_key}

    async def _get_json(
        self, interface_method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{interface_method.lstrip('/')}"
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(SteamTransientError),
            stop=stop_after_attempt(4),
            wait=wait_exponential_jitter(multiplier=1.0, max=15.0),
            reraise=True,
        ):
            with attempt:
                await self.rate_limiter.acquire()
                try:
                    resp = await self._client.get(
                        url,
                        params=params,
                        headers=self._headers(),
                        timeout=25.0,
                    )
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    raise SteamTransientError(
                        self._redact_secrets(
                            f"Steam API transport error on {interface_method}: {exc}"
                        )
                    ) from None

                self.rate_limiter.update_from_headers(resp.headers)

                if resp.status_code == 429:
                    wait = self.rate_limiter.notify_rate_limited(resp.headers)
                    logger.warning("Steam Web API returned 429; cooling down %.1fs", wait)
                    raise SteamTransientError("Steam Web API HTTP 429")
                if resp.status_code >= 500:
                    raise SteamTransientError(
                        self._redact_secrets(
                            f"Steam Web API HTTP {resp.status_code} on {interface_method}"
                        )
                    )
                if resp.status_code >= 400:
                    raise SteamAPIError(
                        self._redact_secrets(
                            f"Steam Web API HTTP {resp.status_code} on {interface_method}: {resp.text[:300]}"
                        )
                    )

                try:
                    body = resp.json()
                except ValueError as exc:
                    raise SteamAPIError(
                        f"Steam Web API invalid JSON on {interface_method}"
                    ) from exc

                if not isinstance(body, dict):
                    raise SteamAPIError(
                        f"Steam Web API returned {type(body).__name__} on {interface_method}"
                    )
                return body

        raise AssertionError("Unreachable retry termination")

    async def get_matches_by_seq_num(
        self, start_at_match_seq_num: int, *, matches_requested: int = 1
    ) -> list[dict[str, Any]]:
        """Fetch matches ordered by match_seq_num via GetMatchHistoryBySequenceNum."""
        req_count = max(1, min(100, int(matches_requested)))
        body = await self._get_json(
            "IDOTA2Match_570/GetMatchHistoryBySequenceNum/v1/",
            params={
                "start_at_match_seq_num": int(start_at_match_seq_num),
                "matches_requested": req_count,
            },
        )
        result = body.get("result") or {}
        if not isinstance(result, dict):
            return []
        matches = result.get("matches")
        return [m for m in matches if isinstance(m, dict)] if isinstance(matches, list) else []

    async def get_matches_for_seq_nums(
        self,
        seq_to_match_id: dict[int, int],
        *,
        max_concurrency: int = 4,
        coalesce_span: int = 90,
    ) -> dict[int, dict[str, Any]]:
        """Fetch Steam match payloads for known sequence numbers with window coalescing."""
        if not seq_to_match_id:
            return {}

        span = max(0, min(99, int(coalesce_span)))
        sorted_pairs = sorted((int(seq), int(mid)) for seq, mid in seq_to_match_id.items())

        clusters: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        cluster_start = 0
        for seq, mid in sorted_pairs:
            if not current:
                current = [(seq, mid)]
                cluster_start = seq
            elif (seq - cluster_start) <= span:
                current.append((seq, mid))
            else:
                clusters.append(current)
                current = [(seq, mid)]
                cluster_start = seq
        if current:
            clusters.append(current)

        sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        matched_by_mid: dict[int, dict[str, Any]] = {}
        missed_pairs: list[tuple[int, int]] = []

        async def _fetch_cluster(cluster: list[tuple[int, int]]) -> None:
            start_seq = cluster[0][0]
            end_seq = cluster[-1][0]
            count = min(100, max(1, (end_seq - start_seq) + 1))
            async with sem:
                matches = await self.get_matches_by_seq_num(
                    start_seq, matches_requested=count
                )
            by_mid: dict[int, dict[str, Any]] = {}
            for m in matches:
                raw_mid = m.get("match_id")
                if raw_mid is not None:
                    by_mid[int(raw_mid)] = m
            for seq, expected_mid in cluster:
                hit = by_mid.get(expected_mid)
                if hit is not None:
                    matched_by_mid[expected_mid] = hit
                elif len(cluster) > 1:
                    missed_pairs.append((seq, expected_mid))

        await asyncio.gather(*(_fetch_cluster(c) for c in clusters))

        if missed_pairs:
            async def _fetch_single(seq: int, expected_mid: int) -> None:
                async with sem:
                    matches = await self.get_matches_by_seq_num(seq, matches_requested=1)
                for m in matches:
                    if m.get("match_id") is not None and int(m["match_id"]) == expected_mid:
                        matched_by_mid[expected_mid] = m
                        break

            await asyncio.gather(*(_fetch_single(s, mid) for s, mid in missed_pairs))

        return matched_by_mid

    async def get_match_history(
        self,
        *,
        account_id: int | None = None,
        league_id: int | None = None,
        start_at_match_id: int | None = None,
        matches_requested: int = 100,
    ) -> list[dict[str, Any]]:
        """Query Steam GetMatchHistory for a player or league; returns [] on private profiles."""
        params: dict[str, Any] = {
            "matches_requested": max(1, min(100, int(matches_requested)))
        }
        if account_id is not None:
            params["account_id"] = int(account_id)
        if league_id is not None:
            params["league_id"] = int(league_id)
        if start_at_match_id is not None:
            params["start_at_match_id"] = int(start_at_match_id)

        body = await self._get_json("IDOTA2Match_570/GetMatchHistory/v1/", params=params)
        result = body.get("result") or {}
        if not isinstance(result, dict):
            return []
        status = result.get("status")
        if status == 15:
            # Private match history for this player
            return []
        matches = result.get("matches")
        return [m for m in matches if isinstance(m, dict)] if isinstance(matches, list) else []

    async def get_top_live_games(self, *, partner: int = 0) -> list[dict[str, Any]]:
        """Fetch live high-MMR matches and their active players via GetTopLiveGame."""
        body = await self._get_json(
            "IDOTA2Match_570/GetTopLiveGame/v1/", params={"partner": int(partner)}
        )
        games = body.get("game_list")
        return [g for g in games if isinstance(g, dict)] if isinstance(games, list) else []

    async def get_live_league_games(self) -> list[dict[str, Any]]:
        """Fetch active tournament/league matches via GetLiveLeagueGames."""
        body = await self._get_json("IDOTA2Match_570/GetLiveLeagueGames/v1/")
        games = (body.get("result") or {}).get("games")
        return [g for g in games if isinstance(g, dict)] if isinstance(games, list) else []

    async def get_server_version(self) -> dict[str, Any]:
        """Fetch current Dota 2 GC active deploy version via IGCVersion_570/GetServerVersion."""
        body = await self._get_json("IGCVersion_570/GetServerVersion/v1/")
        result = body.get("result")
        return dict(result) if isinstance(result, dict) else {}
