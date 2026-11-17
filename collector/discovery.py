"""Match and player discovery from non-ladder OpenDota seeds, STRATZ BFS, proMatches, and leaderboard tail."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from collector.clients.opendota import OpenDotaClient
from collector.clients.stratz import StratzClient
from collector.db import CollectorDB
from collector.rate_limit import QuotaExhaustedError
from shared.constants import (
    LEADERBOARD_DIVISIONS,
    MIN_DURATION_SECONDS,
    RANKED_LOBBY_TYPE_ID,
    VALID_PUB_GAME_MODES,
)

logger = logging.getLogger(__name__)
DEFAULT_BFS_MAX_AGE_SECONDS: int = 14 * 86400
MIN_PUBLIC_AVG_RANK_TIER: int = 80


class DiscoveryEngine:
    """Coordinates all 3 Immortal & Pro match discovery sources."""

    def __init__(
        self,
        *,
        db: CollectorDB,
        stratz: StratzClient,
        opendota: OpenDotaClient,
        public_min_delay_seconds: int = 3600,
        public_max_delay_seconds: int = 10800,
        pro_min_delay_seconds: int = 1800,
        bfs_batch_size: int = 10,
        bfs_cooldown_seconds: int = 14400,
        bfs_max_age_seconds: int = DEFAULT_BFS_MAX_AGE_SECONDS,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.stratz = stratz
        self.opendota = opendota
        self.public_min_delay_seconds = public_min_delay_seconds
        self.public_max_delay_seconds = public_max_delay_seconds
        self.pro_min_delay_seconds = pro_min_delay_seconds
        self.bfs_batch_size = bfs_batch_size
        self.bfs_cooldown_seconds = bfs_cooldown_seconds
        self.bfs_max_age_seconds = max(0, int(bfs_max_age_seconds))
        self._wall_clock = wall_clock

    async def discover_public_seeds(self, active_gv_as_of: int) -> int:
        """Seed Immortal ranked matches from OpenDota publicMatches?min_rank=80 offset 1-3h back."""
        try:
            head_page = await self.opendota.get_public_matches(min_rank=80)
        except QuotaExhaustedError as exc:
            logger.warning("Skipping OpenDota publicMatches: %s", exc)
            return 0

        if not head_page:
            return 0

        latest_ts = max(int(m.get("start_time") or 0) for m in head_page)
        latest_id = max(int(m.get("match_id") or 0) for m in head_page)
        if latest_id <= 0:
            return 0

        raw_cursor = self.db.get_meta("opendota_public_cursor")
        cursor = int(raw_cursor) if raw_cursor else max(1, latest_id - 220_000)

        page: list[dict[str, Any]] = []
        for _ in range(3):
            try:
                page = await self.opendota.get_public_matches(
                    min_rank=80, less_than_match_id=cursor
                )
            except QuotaExhaustedError as exc:
                logger.warning("OpenDota quota paused during publicMatches offset: %s", exc)
                return 0
            if not page:
                cursor = max(1, latest_id - 220_000)
                break

            page_newest_ts = max(int(m.get("start_time") or 0) for m in page)
            age_sec = latest_ts - page_newest_ts
            if age_sec < self.public_min_delay_seconds:
                # Too fresh for STRATZ parsing; step back further
                step = max(100_000, int((self.public_min_delay_seconds - age_sec + 1800) * 60))
                cursor = max(1, min(int(m["match_id"]) for m in page) - step)
                continue
            break

        if not page:
            return 0

        candidates: list[tuple[int, str]] = []
        oldest_ts = latest_ts
        oldest_id = cursor
        for m in page:
            mid = int(m.get("match_id") or 0)
            st = int(m.get("start_time") or 0)
            dur = int(m.get("duration") or 0)
            lobby = m.get("lobby_type")
            raw_rank_tier = m.get("avg_rank_tier")
            try:
                rank_tier_ok = (
                    raw_rank_tier is not None and int(raw_rank_tier) >= MIN_PUBLIC_AVG_RANK_TIER
                )
            except (TypeError, ValueError):
                rank_tier_ok = False
            oldest_ts = min(oldest_ts, st)
            if mid > 0:
                oldest_id = min(oldest_id, mid)
            if (
                mid > 0
                and lobby == RANKED_LOBBY_TYPE_ID
                and rank_tier_ok
                and dur >= MIN_DURATION_SECONDS
                and st >= active_gv_as_of
                and (latest_ts - st) >= self.public_min_delay_seconds
            ):
                candidates.append((mid, "opendota_public"))

        # Reset cursor back to the 1-3h window once we drift too far back or before patch start
        if oldest_ts < active_gv_as_of or (latest_ts - oldest_ts) > max(
            86400, self.public_max_delay_seconds * 4
        ):
            next_cursor = max(1, latest_id - 220_000)
        else:
            next_cursor = oldest_id
        self.db.set_meta("opendota_public_cursor", str(next_cursor))

        inserted = self.db.enqueue_matches(candidates)
        logger.info(
            "Public seed discovery: scanned=%d filtered=%d enqueued=%d",
            len(page),
            len(candidates),
            inserted,
        )
        return inserted

    async def discover_pro_matches(
        self, active_gv_as_of: int, *, max_pages: int = 2
    ) -> int:
        """Discover tournament pro matches >= active_gv_as_of from OpenDota proMatches."""
        raw_newest = self.db.get_meta("opendota_pro_newest_id")
        known_newest = int(raw_newest) if raw_newest else 0
        now_ts = int(time.time())

        less_than_id: int | None = None
        newest_seen = known_newest
        candidates: list[tuple[int, str]] = []

        for _ in range(max_pages):
            try:
                page = await self.opendota.get_pro_matches(less_than_match_id=less_than_id)
            except QuotaExhaustedError as exc:
                logger.warning("Skipping OpenDota proMatches: %s", exc)
                break
            if not page:
                break

            stop_paging = False
            for m in page:
                mid = int(m.get("match_id") or 0)
                st = int(m.get("start_time") or 0)
                dur = int(m.get("duration") or 0)
                league_id = m.get("leagueid")
                if known_newest and mid <= known_newest:
                    stop_paging = True
                    continue
                if st < active_gv_as_of:
                    stop_paging = True
                    continue
                # Only advance watermark for matches that have already cleared the parse delay window
                if (now_ts - st) < self.pro_min_delay_seconds:
                    continue
                if mid > newest_seen:
                    newest_seen = mid
                if mid > 0 and league_id is not None and dur >= MIN_DURATION_SECONDS:
                    candidates.append((mid, "opendota_pro"))

            min_page_id = min(int(m["match_id"]) for m in page if m.get("match_id"))
            if stop_paging or min_page_id <= 1:
                break
            less_than_id = min_page_id

        if newest_seen > known_newest:
            self.db.set_meta("opendota_pro_newest_id", str(newest_seen))

        inserted = self.db.enqueue_matches(candidates)
        if inserted:
            logger.info("Pro match discovery: enqueued=%d", inserted)
        return inserted

    async def discover_bfs_from_players(
        self, active_gv_id: int, active_gv_as_of: int = 0
    ) -> int:
        """Run one 10-player aliased STRATZ BFS batch (without playbackData)."""
        batch = self.db.pop_players_for_bfs(
            limit=self.bfs_batch_size,
            cooldown_seconds=self.bfs_cooldown_seconds,
        )
        if not batch:
            return 0

        player_ids = [sid for sid, _ in batch]
        watermarks = dict(batch)
        now_ts = int(self._wall_clock())
        min_start_ts = (
            max(int(active_gv_as_of or 0), now_ts - self.bfs_max_age_seconds)
            if self.bfs_max_age_seconds > 0
            else int(active_gv_as_of or 0)
        )

        bfs_take = 25 if all(wm is not None for _sid, wm in batch) else 100
        matches_by_player = await self.stratz.get_players_matches(
            player_ids,
            active_gv_id,
            take=bfs_take,
            start_datetime=min_start_ts if min_start_ts > 0 else None,
        )

        to_enqueue: list[tuple[int, str]] = []
        fetched_updates: list[tuple[int, int | None]] = []
        for sid in player_ids:
            matches = matches_by_player.get(sid) or []
            prev_watermark = watermarks.get(sid)
            max_id = prev_watermark
            for m in matches:
                mid = m.get("id")
                if mid is None:
                    continue
                mid_int = int(mid)
                if max_id is None or mid_int > max_id:
                    max_id = mid_int
                if prev_watermark is not None and mid_int <= prev_watermark:
                    continue
                if "parsedDateTime" in m and m.get("parsedDateTime") is None:
                    continue
                if "startDateTime" in m:
                    st = int(m.get("startDateTime") or 0)
                    if st > 0 and min_start_ts > 0 and st < min_start_ts:
                        continue
                dur = int(m.get("durationSeconds") or 0)
                mode = str(m.get("gameMode") or "")
                if dur >= MIN_DURATION_SECONDS and mode in VALID_PUB_GAME_MODES:
                    to_enqueue.append((mid_int, "bfs"))
            fetched_updates.append((sid, max_id))

        self.db.mark_players_fetched_batch(fetched_updates)

        inserted = self.db.enqueue_matches(to_enqueue)
        logger.info(
            "BFS discovery: polled=%d players, matched=%d, enqueued=%d",
            len(player_ids),
            len(to_enqueue),
            inserted,
        )
        return inserted

    async def discover_leaderboard_tail(self) -> int:
        """Seed fallback Immortal players from the lower tail of regional leaderboards."""
        raw = self.db.get_meta("leaderboard_cursors")
        state: dict[str, Any] = json.loads(raw) if raw else {
            "div_idx": 0,
            "skips": dict.fromkeys(LEADERBOARD_DIVISIONS, 4500),
        }
        div_idx = int(state.get("div_idx") or 0) % len(LEADERBOARD_DIVISIONS)
        division = LEADERBOARD_DIVISIONS[div_idx]
        skips: dict[str, int] = dict(state.get("skips") or {})
        skip = int(skips.get(division, 4500))

        players = await self.stratz.get_leaderboard_players(
            division, skip=skip, take=100
        )
        if not players and skip > 500:
            skips[division] = max(0, skip - 1000)
        elif players:
            # Move slightly up/down tail on next visit to avoid scanning identical slice
            next_skip = skip - 100 if skip >= 200 else 4500
            skips[division] = next_skip

        state["div_idx"] = (div_idx + 1) % len(LEADERBOARD_DIVISIONS)
        state["skips"] = skips
        self.db.set_meta("leaderboard_cursors", json.dumps(state))

        inserted = self.db.enqueue_players(players, source="leaderboard_tail", priority=0)
        logger.info(
            "Leaderboard tail seed (%s skip=%d): fetched=%d enqueued=%d",
            division,
            skip,
            len(players),
            inserted,
        )
        return inserted
