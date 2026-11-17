"""CLI entrypoint and main orchestration loop for the d2oracle match collector."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from collector.clients.hf_storage import HFDatasetStorage
from collector.clients.opendota import OpenDotaClient
from collector.clients.stratz import (
    StratzBatchFallbackError,
    StratzClient,
    StratzGraphQLError,
    StratzTransientError,
)
from collector.config import CollectorSettings
from collector.db import CollectorDB
from collector.discovery import DiscoveryEngine
from collector.patch_manager import ActivePatchContext, PatchManager
from collector.rate_limit import (
    QuotaExhaustedError,
    build_opendota_rate_limiter,
    build_stratz_rate_limiter,
)
from collector.validator import validate_and_transform_match

logger = logging.getLogger("d2oracle.collector")
STRATZ_MATCH_BATCH_SIZE: int = 20
STRATZ_FETCH_CAP: int = 40
PUBLIC_DISCOVERY_PENDING_THRESHOLD: int = 15


@dataclass
class RunSummary:
    """Summary returned at the end of a collector run."""

    gv_id: int
    gv_name: str
    collected: int
    rejected: int
    retried: int
    flushed_shards: int
    stats: dict[str, int]


class CollectorRunner:
    """Coordinates patch management, discovery, validation, buffering, and HF flushing."""

    def __init__(
        self,
        *,
        settings: CollectorSettings,
        db: CollectorDB,
        stratz: StratzClient,
        opendota: OpenDotaClient,
        hf_storage: HFDatasetStorage,
        patch_manager: PatchManager,
        discovery: DiscoveryEngine,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.db = db
        self.stratz = stratz
        self.opendota = opendota
        self.hf_storage = hf_storage
        self.patch_manager = patch_manager
        self.discovery = discovery
        self._clock = clock
        self._stop_requested = False
        self._stop_event = asyncio.Event()
        self._last_pro_discovery_at: float | None = None
        self._last_public_discovery_at: float | None = None

    def request_stop(self) -> None:
        self._stop_requested = True
        self._stop_event.set()

    async def _sleep_interruptible(self, seconds: float) -> None:
        if self._stop_requested or seconds <= 0.0:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), timeout=float(seconds))

    async def _run_periodic_discovery(
        self, ctx: ActivePatchContext, *, pending_count: int
    ) -> None:
        """Run periodic pro discovery always, and public OpenDota discovery only when queue is low."""
        now = self._clock()
        if (
            self._last_pro_discovery_at is None
            or (now - self._last_pro_discovery_at)
            >= self.settings.pro_discovery_interval_seconds
        ):
            await self.discovery.discover_pro_matches(ctx.as_of_datetime)
            self._last_pro_discovery_at = self._clock()

        if pending_count < PUBLIC_DISCOVERY_PENDING_THRESHOLD and (
            self._last_public_discovery_at is None
            or (now - self._last_public_discovery_at)
            >= self.settings.public_discovery_interval_seconds
        ):
            await self.discovery.discover_public_seeds(ctx.as_of_datetime)
            self._last_public_discovery_at = self._clock()

    async def _ensure_queue_has_matches(self, ctx: ActivePatchContext) -> int:
        cooldown = self.settings.match_retry_cooldown_seconds
        pending_count = self.db.count_pending_matches(retry_cooldown_seconds=cooldown)
        await self._run_periodic_discovery(ctx, pending_count=pending_count)

        pending_count = self.db.count_pending_matches(retry_cooldown_seconds=cooldown)
        if pending_count >= PUBLIC_DISCOVERY_PENDING_THRESHOLD:
            return pending_count

        # 1. Snowball BFS from already known Immortal players (fresh <=14d matches)
        if self.db.has_players():
            await self.discovery.discover_bfs_from_players(
                ctx.gv_id, active_gv_as_of=ctx.as_of_datetime
            )

        pending_count = self.db.count_pending_matches(retry_cooldown_seconds=cooldown)
        if pending_count < PUBLIC_DISCOVERY_PENDING_THRESHOLD:
            # 2. Immediate fallback seed if queue is still low
            await self.discovery.discover_public_seeds(ctx.as_of_datetime)
            self._last_public_discovery_at = self._clock()
            await self.discovery.discover_pro_matches(ctx.as_of_datetime)
            self._last_pro_discovery_at = self._clock()

        pending_count = self.db.count_pending_matches(retry_cooldown_seconds=cooldown)
        if pending_count == 0:
            # 3. Backup seed from leaderboard tail
            await self.discovery.discover_leaderboard_tail()
            await self.discovery.discover_bfs_from_players(
                ctx.gv_id, active_gv_as_of=ctx.as_of_datetime
            )
            pending_count = self.db.count_pending_matches(retry_cooldown_seconds=cooldown)

        return pending_count

    async def _handle_match_payload(
        self,
        match_id: int,
        source: str,
        raw_match: dict[str, object] | None,
        *,
        ctx: ActivePatchContext,
        limit: int | None,
        collected_so_far: int,
    ) -> tuple[int, int, int]:
        """Validate and persist one match outcome; return (collected_d, rejected_d, retried_d)."""
        result = validate_and_transform_match(
            raw_match,
            active_gv_id=ctx.gv_id,
            active_gv_name=ctx.gv_name,
            source=source,
            vocab=ctx.vocab,
        )

        if result.valid and result.record is not None:
            self.db.mark_match_collected(match_id, ctx.gv_id, result.record)
            self.db.enqueue_players(
                result.player_steam_ids, source="match_bfs", priority=10
            )
            new_total = collected_so_far + 1
            logger.info(
                "Collected match %d (%s, is_pro=%s) [%d%s]",
                match_id,
                source,
                result.record["is_pro"],
                new_total,
                f"/{limit}" if limit else "",
            )
            return 1, 0, 0

        if result.retryable:
            max_attempts = (
                1 if source == "opendota_public" and result.reason == "unparsed" else 2
            )
            new_status = self.db.mark_match_retry(
                match_id, result.reason or "retry", max_attempts=max_attempts
            )
            logger.debug(
                "Retryable match %d (%s) -> %s",
                match_id,
                result.reason,
                new_status,
            )
            return (0, 1, 0) if new_status == "rejected" else (0, 0, 1)

        self.db.mark_match_rejected(match_id, result.reason or "invalid")
        logger.debug("Rejected match %d: %s", match_id, result.reason)
        return 0, 1, 0

    async def _fetch_and_process_single(
        self,
        match_id: int,
        source: str,
        *,
        ctx: ActivePatchContext,
        limit: int | None,
        collected_so_far: int,
        batch_gql_errors: int,
    ) -> tuple[int, int, int, int]:
        """Fetch one match via single GetMatch and return metric deltas + updated gql error streak."""
        try:
            raw_match = await self.stratz.get_match(match_id)
            next_gql_errors = 0
        except StratzGraphQLError as exc:
            err_text = str(exc)
            next_gql_errors = batch_gql_errors + 1
            if "HTTP 401" in err_text or "HTTP 403" in err_text or next_gql_errors >= 3:
                raise
            new_status = self.db.mark_match_retry(
                match_id, "stratz_graphql_error", max_attempts=2
            )
            logger.warning(
                "STRATZ GraphQL error on match %d -> %s: %s",
                match_id,
                new_status,
                exc,
            )
            return (
                (0, 1, 0, next_gql_errors)
                if new_status == "rejected"
                else (0, 0, 1, next_gql_errors)
            )

        c_d, r_d, rt_d = await self._handle_match_payload(
            match_id,
            source,
            raw_match,
            ctx=ctx,
            limit=limit,
            collected_so_far=collected_so_far,
        )
        return c_d, r_d, rt_d, next_gql_errors

    async def _process_chunk_sequential(
        self,
        chunk: list[tuple[int, str, int]],
        *,
        ctx: ActivePatchContext,
        limit: int | None,
        collected: int,
        rejected: int,
        retried: int,
        batch_gql_errors: int,
    ) -> tuple[int, int, int, int]:
        """Process a chunk via single-match GetMatch calls (1-item chunk or batch fallback)."""
        for match_id, source, _attempts in chunk:
            if self._stop_requested or (limit is not None and collected >= limit):
                break
            c_d, r_d, rt_d, batch_gql_errors = await self._fetch_and_process_single(
                match_id,
                source,
                ctx=ctx,
                limit=limit,
                collected_so_far=collected,
                batch_gql_errors=batch_gql_errors,
            )
            collected += c_d
            rejected += r_d
            retried += rt_d
        return collected, rejected, retried, batch_gql_errors

    async def _process_batch_chunk_results(
        self,
        chunk: list[tuple[int, str, int]],
        batch_matches: dict[int, dict[str, object] | None],
        failed_ids: set[int],
        *,
        ctx: ActivePatchContext,
        limit: int | None,
        collected: int,
        rejected: int,
        retried: int,
        batch_gql_errors: int,
    ) -> tuple[int, int, int, int]:
        """Validate a batch chunk and persist all outcomes in one SQLite transaction."""
        to_collect: list[tuple[int, dict[str, object]]] = []
        collect_logs: list[tuple[int, str, bool]] = []
        to_reject: list[tuple[int, str]] = []
        to_retry: list[tuple[int, str, int]] = []
        bfs_players: list[int] = []
        fallback_items: list[tuple[int, str, int]] = []

        for match_id, source, attempts in chunk:
            if match_id in failed_ids:
                fallback_items.append((match_id, source, attempts))
                continue
            result = validate_and_transform_match(
                batch_matches.get(match_id),
                active_gv_id=ctx.gv_id,
                active_gv_name=ctx.gv_name,
                source=source,
                vocab=ctx.vocab,
            )
            if result.valid and result.record is not None:
                to_collect.append((match_id, result.record))
                collect_logs.append((match_id, source, bool(result.record["is_pro"])))
                bfs_players.extend(result.player_steam_ids)
            elif result.retryable:
                max_att = (
                    1 if source == "opendota_public" and result.reason == "unparsed" else 2
                )
                to_retry.append((match_id, result.reason or "retry", max_att))
            else:
                to_reject.append((match_id, result.reason or "invalid"))

        retry_statuses = self.db.record_batch_results(
            game_version_id=ctx.gv_id,
            collected=to_collect,
            rejected=to_reject,
            retries=to_retry,
            bfs_player_ids=bfs_players,
        )

        for match_id, source, is_pro in collect_logs:
            collected += 1
            logger.info(
                "Collected match %d (%s, is_pro=%s) [%d%s]",
                match_id,
                source,
                is_pro,
                collected,
                f"/{limit}" if limit else "",
            )
        rejected += len(to_reject) + sum(
            1 for st in retry_statuses.values() if st == "rejected"
        )
        retried += sum(1 for st in retry_statuses.values() if st == "pending")

        if fallback_items:
            collected, rejected, retried, batch_gql_errors = (
                await self._process_chunk_sequential(
                    fallback_items,
                    ctx=ctx,
                    limit=limit,
                    collected=collected,
                    rejected=rejected,
                    retried=retried,
                    batch_gql_errors=batch_gql_errors,
                )
            )
        return collected, rejected, retried, batch_gql_errors

    async def _flush_buffer_if_ready(
        self,
        ctx: ActivePatchContext,
        *,
        dry_run: bool,
        once: bool,
        no_upload: bool,
        hf_error_streak: int,
        error_backoff_seconds: float,
    ) -> tuple[int, int]:
        """Flush buffered matches to HF once ready, isolating transient HF errors with backoff."""
        if no_upload or self.db.count_buffered_matches(ctx.gv_id) < self.settings.batch_size:
            return 0, hf_error_streak
        try:
            await self.patch_manager.flush_buffer_for_patch(
                gv_id=ctx.gv_id,
                gv_name=ctx.gv_name,
                as_of_datetime=ctx.as_of_datetime,
                is_active_patch=True,
                limit=self.settings.batch_size,
            )
            return 1, 0
        except Exception as exc:
            if dry_run or once:
                raise
            next_streak = hf_error_streak + 1
            base = max(1.0, float(error_backoff_seconds))
            sleep_for = min(300.0, base * (2 ** min(5, next_streak - 1)))
            logger.warning(
                "HF shard flush failed (%d consecutive, preserving %d buffered matches): %s; sleeping %.1fs",
                next_streak,
                self.db.count_buffered_matches(ctx.gv_id),
                exc,
                sleep_for,
            )
            await self._sleep_interruptible(sleep_for)
            return 0, next_streak

    async def run(
        self,
        *,
        limit: int | None = None,
        dry_run: bool = False,
        once: bool = False,
        no_upload: bool = False,
        idle_sleep_seconds: float = 30.0,
        error_backoff_seconds: float = 10.0,
    ) -> RunSummary:
        ctx: ActivePatchContext | None = None
        collected = 0
        rejected = 0
        retried = 0
        flushed_shards = 0
        empty_cycles = 0
        consecutive_errors = 0
        hf_error_streak = 0

        while not self._stop_requested:
            if limit is not None and collected >= limit:
                break

            try:
                if ctx is None or self.patch_manager.should_check_patch(
                    self.settings.patch_check_interval_seconds
                ):
                    initial_sync = ctx is None
                    ctx = await self.patch_manager.sync_patch()
                    if initial_sync:
                        logger.info(
                            "Collector active on patch %s (gv=%d, asOf=%d), dry_run=%s, limit=%s",
                            ctx.gv_name,
                            ctx.gv_id,
                            ctx.as_of_datetime,
                            dry_run,
                            limit,
                        )
                    elif ctx.switched:
                        self._last_pro_discovery_at = None
                        self._last_public_discovery_at = None
                        logger.info(
                            "Switched cleanly to new patch %s (gv=%d)",
                            ctx.gv_name,
                            ctx.gv_id,
                        )

                await self._ensure_queue_has_matches(ctx)
                fetch_cap = (
                    STRATZ_FETCH_CAP
                    if limit is None
                    else min(STRATZ_FETCH_CAP, max(1, limit - collected))
                )
                batch = self.db.pop_pending_matches(
                    limit=fetch_cap,
                    retry_cooldown_seconds=self.settings.match_retry_cooldown_seconds,
                )

                if not batch:
                    consecutive_errors = 0
                    empty_cycles += 1
                    logger.info("No pending matches available (cycle %d)", empty_cycles)
                    if once or (dry_run and empty_cycles >= 3):
                        break
                    await self._sleep_interruptible(idle_sleep_seconds)
                    continue

                empty_cycles = 0
                batch_gql_errors = 0
                for start_i in range(0, len(batch), STRATZ_MATCH_BATCH_SIZE):
                    if self._stop_requested or (limit is not None and collected >= limit):
                        break
                    chunk = batch[start_i : start_i + STRATZ_MATCH_BATCH_SIZE]
                    if len(chunk) == 1:
                        collected, rejected, retried, batch_gql_errors = (
                            await self._process_chunk_sequential(
                                chunk,
                                ctx=ctx,
                                limit=limit,
                                collected=collected,
                                rejected=rejected,
                                retried=retried,
                                batch_gql_errors=batch_gql_errors,
                            )
                        )
                        continue

                    try:
                        batch_res = await self.stratz.get_matches_batch(
                            [mid for mid, _src, _att in chunk]
                        )
                    except QuotaExhaustedError:
                        raise
                    except (
                        StratzBatchFallbackError,
                        StratzGraphQLError,
                        StratzTransientError,
                        httpx.HTTPError,
                    ) as exc:
                        err_text = str(exc)
                        if "HTTP 401" in err_text or "HTTP 403" in err_text:
                            raise
                        logger.info(
                            "Batch GetMatches fallback to single GetMatch (%d matches): %s",
                            len(chunk),
                            exc,
                        )
                        collected, rejected, retried, batch_gql_errors = (
                            await self._process_chunk_sequential(
                                chunk,
                                ctx=ctx,
                                limit=limit,
                                collected=collected,
                                rejected=rejected,
                                retried=retried,
                                batch_gql_errors=batch_gql_errors,
                            )
                        )
                        continue

                    batch_gql_errors = 0
                    collected, rejected, retried, batch_gql_errors = (
                        await self._process_batch_chunk_results(
                            chunk,
                            batch_res.matches,
                            batch_res.failed_ids,
                            ctx=ctx,
                            limit=limit,
                            collected=collected,
                            rejected=rejected,
                            retried=retried,
                            batch_gql_errors=batch_gql_errors,
                        )
                    )

                fl_delta, hf_error_streak = await self._flush_buffer_if_ready(
                    ctx,
                    dry_run=dry_run,
                    once=once,
                    no_upload=no_upload,
                    hf_error_streak=hf_error_streak,
                    error_backoff_seconds=error_backoff_seconds,
                )
                flushed_shards += fl_delta
                consecutive_errors = 0
                if once:
                    break

            except QuotaExhaustedError as exc:
                logger.warning("Quota exhausted: %s", exc)
                if dry_run or once:
                    break
                sleep_for = min(3600.0, max(60.0, exc.retry_after_seconds))
                logger.info("Sleeping %.0fs until quota recovers", sleep_for)
                await self._sleep_interruptible(sleep_for)

            except (StratzGraphQLError, StratzTransientError, httpx.HTTPError) as exc:
                consecutive_errors += 1
                if once:
                    logger.error("Collector failed in --once mode: %s", exc)
                    raise
                base = max(0.01, float(error_backoff_seconds))
                sleep_for = min(300.0, base * (2 ** min(5, consecutive_errors - 1)))
                logger.warning(
                    "Transient collector error (%d consecutive): %s; sleeping %.1fs",
                    consecutive_errors,
                    exc,
                    sleep_for,
                )
                await self._sleep_interruptible(sleep_for)

        if (
            dry_run
            and not no_upload
            and ctx is not None
            and self.db.count_buffered_matches(ctx.gv_id) > 0
        ):
            await self.patch_manager.flush_buffer_for_patch(
                gv_id=ctx.gv_id,
                gv_name=ctx.gv_name,
                as_of_datetime=ctx.as_of_datetime,
                is_active_patch=True,
            )
            flushed_shards += 1

        stats = self.db.stats()
        return RunSummary(
            gv_id=ctx.gv_id if ctx is not None else -1,
            gv_name=ctx.gv_name if ctx is not None else "",
            collected=collected,
            rejected=rejected,
            retried=retried,
            flushed_shards=flushed_shards,
            stats=stats,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="d2oracle-collector",
        description="Collect Dota 2 Immortal & Pro matches for d2oracle.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect a small real batch (default --limit 20), pack Parquet, and upload to HF.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after collecting N valid matches (defaults to 20 under --dry-run).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Execute a single discovery + collection pass and exit.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip uploading buffered shards to Hugging Face (useful for local debugging).",
    )
    return parser


async def _run_from_args(args: argparse.Namespace) -> int:
    settings = CollectorSettings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    limit = args.limit
    if args.dry_run and limit is None:
        limit = 20

    db = CollectorDB(settings.sqlite_path)
    stratz_rl = build_stratz_rate_limiter(
        rps=settings.stratz_rps_limit,
        rpm=settings.stratz_rpm_limit,
        rph=settings.stratz_rph_limit,
        rpd=settings.stratz_rpd_limit,
    )
    od_rl = build_opendota_rate_limiter(
        rpm=settings.opendota_rpm_limit,
        rpd=settings.opendota_rpd_limit,
    )

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        stratz = StratzClient(
            token=settings.stratz_token.get_secret_value(),
            rate_limiter=stratz_rl,
            http_client=http_client,
        )
        opendota = OpenDotaClient(rate_limiter=od_rl, http_client=http_client)
        hf_storage = HFDatasetStorage(
            repo_id=settings.hf_repo_id,
            token=settings.hf_token.get_secret_value(),
        )
        patch_manager = PatchManager(
            target_game_version=settings.target_game_version,
            db=db,
            stratz=stratz,
            hf_storage=hf_storage,
            http_client=http_client,
            constants_cache_dir=settings.constants_cache_dir,
            parquet_dir=settings.parquet_dir,
        )
        discovery = DiscoveryEngine(
            db=db,
            stratz=stratz,
            opendota=opendota,
            public_min_delay_seconds=settings.public_match_min_delay_seconds,
            public_max_delay_seconds=settings.public_match_max_delay_seconds,
            pro_min_delay_seconds=settings.pro_match_min_delay_seconds,
            bfs_batch_size=settings.player_bfs_batch_size,
            bfs_cooldown_seconds=settings.player_bfs_cooldown_seconds,
            bfs_max_age_seconds=settings.player_bfs_max_age_seconds,
        )
        runner = CollectorRunner(
            settings=settings,
            db=db,
            stratz=stratz,
            opendota=opendota,
            hf_storage=hf_storage,
            patch_manager=patch_manager,
            discovery=discovery,
        )

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, runner.request_stop)

        try:
            summary = await runner.run(
                limit=limit,
                dry_run=args.dry_run,
                once=args.once,
                no_upload=args.no_upload,
            )
            logger.info(
                "Finished run: patch=%s(gv=%d) collected=%d rejected=%d retried=%d flushed_shards=%d stats=%s",
                summary.gv_name,
                summary.gv_id,
                summary.collected,
                summary.rejected,
                summary.retried,
                summary.flushed_shards,
                summary.stats,
            )
        finally:
            db.close()

    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_from_args(args))


if __name__ == "__main__":
    sys.exit(cli())
