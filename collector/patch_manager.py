"""Patch detection, auto-switching, buffer flushing, and vocab synchronization."""

from __future__ import annotations

import gc
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from collector.clients.hf_storage import HFDatasetStorage, patch_data_prefix
from collector.clients.stratz import StratzClient
from collector.db import CollectorDB
from collector.parquet_writer import write_parquet_shard
from shared.constants import PARQUET_SCHEMA_VERSION, VOCAB_SCHEMA_VERSION
from shared.dotaconstants import fetch_dotaconstants
from shared.valve_datafeed import fetch_valve_constants, fetch_valve_patchnotes_list
from shared.vocab import build_vocab

logger = logging.getLogger(__name__)
_VALVE_AHEAD_RECENT_LOOKBACK_SEC = 7200
_VALVE_AHEAD_DEPLOY_OFFSET_SEC = 43200


@dataclass
class ActivePatchContext:
    """Currently active Dota 2 patch metadata and built vocabulary."""

    gv_id: int
    gv_name: str
    as_of_datetime: int
    vocab: dict[str, Any]
    switched: bool = False
    stratz_gv_ids: tuple[int, ...] = ()
    min_start_datetime: int = 0
    max_start_datetime: int | None = None
    valve_ahead: bool = False


class PatchManager:
    """Manages active gameVersionId resolution, patch transitions, and HF shard flushes."""

    def __init__(
        self,
        *,
        target_game_version: str,
        db: CollectorDB,
        stratz: StratzClient,
        hf_storage: HFDatasetStorage,
        http_client: httpx.AsyncClient,
        constants_cache_dir: Path,
        parquet_dir: Path,
        wall_clock: Callable[[], float] = time.time,
        use_valve_datafeed: bool = True,
    ) -> None:
        self.target_game_version = target_game_version
        self.db = db
        self.stratz = stratz
        self.hf_storage = hf_storage
        self.http_client = http_client
        self.constants_cache_dir = constants_cache_dir
        self.parquet_dir = parquet_dir
        self._wall_clock = wall_clock
        self.use_valve_datafeed = use_valve_datafeed
        self.constants_cache_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self._last_checked_at: float = 0.0
        self.current: ActivePatchContext | None = None

    def _select_target_version(self, versions: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [v for v in versions if v.get("id") is not None and v.get("asOfDateTime") is not None]
        if not valid:
            raise RuntimeError("STRATZ returned no valid gameVersions")

        if self.target_game_version == "auto":
            return max(valid, key=lambda v: (int(v["asOfDateTime"]), int(v["id"])))

        target_id = int(self.target_game_version)
        for v in valid:
            if int(v["id"]) == target_id:
                return v
        raise RuntimeError(f"TARGET_GAME_VERSION={target_id} not found in STRATZ gameVersions")

    def _resolve_target_patch(
        self,
        stratz_versions: list[dict[str, Any]],
        valve_patches: list[dict[str, Any]],
        prev_active: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], int, tuple[int, ...], int, int | None, bool]:
        """Resolve active patch combining STRATZ gameVersions and Valve patchnoteslist.

        Returns (target_dict, stratz_constants_gv_id, stratz_gv_ids,
                 min_start_datetime, max_start_datetime, valve_ahead).
        """
        stratz_target = self._select_target_version(stratz_versions)
        stratz_id = int(stratz_target["id"])
        stratz_name = str(stratz_target["name"])
        stratz_as_of = int(stratz_target["asOfDateTime"])

        clean_valve = sorted(
            (
                p
                for p in valve_patches
                if isinstance(p, dict)
                and p.get("patch_number")
                and p.get("patch_timestamp") is not None
            ),
            key=lambda p: int(p["patch_timestamp"]),
        )

        if self.target_game_version != "auto":
            max_start: int | None = None
            target_norm = stratz_name.strip().lower()
            matched_idx = next(
                (
                    i
                    for i, p in enumerate(clean_valve)
                    if str(p["patch_number"]).strip().lower() == target_norm
                ),
                None,
            )
            if matched_idx is not None and matched_idx + 1 < len(clean_valve):
                max_start = int(clean_valve[matched_idx + 1]["patch_timestamp"])
            elif clean_valve:
                later = [
                    p
                    for p in clean_valve
                    if int(p["patch_timestamp"]) > stratz_as_of + 86400
                ]
                if later:
                    max_start = int(later[0]["patch_timestamp"])
            return stratz_target, stratz_id, (stratz_id,), 0, max_start, False

        if not clean_valve:
            return stratz_target, stratz_id, (stratz_id,), 0, None, False

        valve_latest = clean_valve[-1]
        valve_latest_name = str(valve_latest["patch_number"]).strip()
        valve_latest_ts = int(valve_latest["patch_timestamp"])
        stratz_names_lower = {
            str(v.get("name") or "").strip().lower() for v in stratz_versions
        }

        if (
            valve_latest_name.lower() not in stratz_names_lower
            and valve_latest_ts > stratz_as_of
        ):
            stratz_norm = stratz_name.strip().lower()
            matched_idx = next(
                (
                    i
                    for i, p in enumerate(clean_valve)
                    if str(p["patch_number"]).strip().lower() == stratz_norm
                ),
                None,
            )
            if matched_idx is not None:
                offset = max(1, len(clean_valve) - 1 - matched_idx)
            else:
                offset = max(
                    1,
                    sum(1 for p in clean_valve if int(p["patch_timestamp"]) > stratz_as_of + 86400),
                )

            synthetic_id = stratz_id + offset
            now_ts = int(self._wall_clock())
            prev_same_name = (
                prev_active is not None
                and str(prev_active.get("name") or "").strip().lower()
                == valve_latest_name.lower()
            )
            deploy_floor = valve_latest_ts + _VALVE_AHEAD_DEPLOY_OFFSET_SEC
            if prev_same_name and prev_active is not None and prev_active.get("minStartDateTime"):
                min_start = int(prev_active["minStartDateTime"])
            elif now_ts >= deploy_floor:
                min_start = deploy_floor
            else:
                min_start = max(valve_latest_ts, now_ts - _VALVE_AHEAD_RECENT_LOOKBACK_SEC)

            target = {
                "id": synthetic_id,
                "name": valve_latest_name,
                "asOfDateTime": valve_latest_ts,
            }
            return target, stratz_id, (stratz_id, synthetic_id), min_start, None, True

        return stratz_target, stratz_id, (stratz_id,), 0, None, False

    def _vocab_cache_path(self, gv_id: int) -> Path:
        return self.constants_cache_dir / f"vocab_{int(gv_id)}.json"

    @staticmethod
    def _load_valid_local_vocab(
        vocab_path: Path, gv_id: int, expected_name: str | None = None
    ) -> dict[str, Any] | None:
        if not vocab_path.exists():
            return None
        try:
            data = json.loads(vocab_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("schema_version") != VOCAB_SCHEMA_VERSION:
            return None
        if int(data.get("game_version_id") or -1) != int(gv_id):
            return None
        if expected_name is not None and str(data.get("game_version_name") or "") != expected_name:
            return None
        return data

    @staticmethod
    def _hf_manifest_needs_update(
        hf_latest: dict[str, Any] | None,
        *,
        gv_id: int,
        expected_vocab_path: str,
    ) -> bool:
        if not isinstance(hf_latest, dict):
            return True
        if int(hf_latest.get("game_version_id") or -1) != int(gv_id):
            return True
        if hf_latest.get("schema_version") != PARQUET_SCHEMA_VERSION:
            return True
        if hf_latest.get("parquet_schema_version") != PARQUET_SCHEMA_VERSION:
            return True
        if hf_latest.get("vocab_schema_version") != VOCAB_SCHEMA_VERSION:
            return True
        return hf_latest.get("vocab_path") != expected_vocab_path

    async def flush_buffer_for_patch(
        self,
        *,
        gv_id: int,
        gv_name: str,
        as_of_datetime: int,
        is_active_patch: bool,
        limit: int | None = None,
    ) -> dict[str, Any] | None:
        """Pack buffered matches for gv_id into a zstd Parquet shard, upload to HF, and clean up."""
        records = self.db.fetch_buffered_matches(gv_id, limit=limit)
        if not records:
            return None

        match_ids = [int(r["match_id"]) for r in records]
        shard_path, shard_meta = write_parquet_shard(records, self.parquet_dir)
        records.clear()
        del records
        gc.collect()
        try:
            manifest = await self.hf_storage.upload_shard_and_update_manifest(
                gv_id=gv_id,
                gv_name=gv_name,
                as_of_datetime=as_of_datetime,
                shard_path=shard_path,
                shard_meta=shard_meta,
                is_active_patch=is_active_patch,
            )
            self.db.delete_buffered_matches(match_ids)
            return manifest
        finally:
            if shard_path.exists():
                shard_path.unlink()

    async def _build_and_save_vocab(
        self,
        target: dict[str, Any],
        *,
        stratz_constants_gv_id: int | None = None,
    ) -> dict[str, Any]:
        gv_id = int(target["id"])
        gv_name = str(target["name"])
        query_gv_id = int(stratz_constants_gv_id if stratz_constants_gv_id is not None else gv_id)

        stratz_consts = await self.stratz.get_constants(query_gv_id)

        valve_consts: dict[str, Any] | None = None
        if self.use_valve_datafeed:
            try:
                valve_consts = await fetch_valve_constants(
                    self.http_client,
                    patch_name=gv_name,
                    cache_dir=self.constants_cache_dir,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch Valve datafeed constants (%s); continuing with STRATZ + dotaconstants",
                    exc,
                )

        try:
            odota_consts = await fetch_dotaconstants(
                self.http_client, cache_dir=self.constants_cache_dir
            )
        except Exception as exc:
            if valve_consts is None:
                raise
            logger.warning(
                "Failed to fetch odota/dotaconstants (%s); continuing with Valve datafeed + STRATZ",
                exc,
            )
            odota_consts = {}

        vocab = build_vocab(
            target,
            stratz_consts,
            odota_consts,
            valve_datafeed=valve_consts,
        )
        self._vocab_cache_path(gv_id).write_text(
            json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return vocab

    async def sync_patch(
        self, *, force: bool = False, skip_hf_upload: bool = False
    ) -> ActivePatchContext:
        """Check STRATZ gameVersions + Valve patchnoteslist and handle setup or live switch."""
        self._last_checked_at = time.monotonic()
        versions = await self.stratz.get_game_versions()
        valve_patches: list[dict[str, Any]] = []
        if self.use_valve_datafeed:
            try:
                valve_patches = await fetch_valve_patchnotes_list(
                    self.http_client, cache_dir=self.constants_cache_dir
                )
            except Exception as exc:
                logger.warning(
                    "Failed to fetch Valve patchnoteslist (%s); using STRATZ gameVersions",
                    exc,
                )

        prev = self.db.get_active_patch()
        (
            target,
            stratz_consts_gv_id,
            stratz_gv_ids,
            min_start_datetime,
            max_start_datetime,
            valve_ahead,
        ) = self._resolve_target_patch(versions, valve_patches, prev)

        new_id = int(target["id"])
        new_name = str(target["name"])
        new_as_of = int(target["asOfDateTime"])

        switched = prev is not None and (
            int(prev["id"]) != new_id
            and str(prev.get("name") or "").strip().lower() != new_name.strip().lower()
        )

        if switched and prev is not None:
            old_id = int(prev["id"])
            old_name = str(prev["name"])
            old_as_of = int(prev["asOfDateTime"])
            logger.warning(
                "Detected Dota 2 patch switch: %s (gv=%d) -> %s (gv=%d)",
                old_name,
                old_id,
                new_name,
                new_id,
            )
            # 1. Flush remainder of old patch buffer to HF under data/gv_<old_id>_<old_name>/
            if not skip_hf_upload:
                await self.flush_buffer_for_patch(
                    gv_id=old_id,
                    gv_name=old_name,
                    as_of_datetime=old_as_of,
                    is_active_patch=False,
                )
                # Also flush any unexpected older patch IDs lingering in buffer
                for stray_gv in self.db.buffered_game_versions():
                    if stray_gv != new_id:
                        stray_meta = next(
                            (v for v in versions if int(v["id"]) == stray_gv),
                            {"id": stray_gv, "name": f"gv{stray_gv}", "asOfDateTime": 0},
                        )
                        await self.flush_buffer_for_patch(
                            gv_id=stray_gv,
                            gv_name=str(stray_meta["name"]),
                            as_of_datetime=int(stray_meta.get("asOfDateTime") or 0),
                            is_active_patch=False,
                        )
            # 3. Reset match_queue and players_queue cursors for the new patch
            self.db.reset_for_new_patch()

        vocab_path = self._vocab_cache_path(new_id)
        local_vocab = self._load_valid_local_vocab(vocab_path, new_id, expected_name=new_name)

        hf_latest: dict[str, Any] | None = None
        hf_needs_update = False
        if not skip_hf_upload and (
            force or prev is None or switched or local_vocab is None or self.current is None
        ):
            hf_latest = await self.hf_storage.read_latest_manifest()
            expected_vocab_path = f"{patch_data_prefix(new_id, new_name)}/vocab.json"
            hf_needs_update = self._hf_manifest_needs_update(
                hf_latest, gv_id=new_id, expected_vocab_path=expected_vocab_path
            )

        needs_vocab_sync = (
            force or prev is None or switched or local_vocab is None or hf_needs_update
        )

        if needs_vocab_sync:
            if local_vocab is not None and not (force or prev is None or switched):
                vocab = local_vocab
            else:
                vocab = await self._build_and_save_vocab(
                    target, stratz_constants_gv_id=stratz_consts_gv_id
                )
            if not skip_hf_upload:
                await self.hf_storage.ensure_repo()
                remote_vocab_path = await self.hf_storage.upload_vocab(new_id, new_name, vocab)
                if switched or hf_needs_update:
                    await self.hf_storage.update_manifest_for_patch(
                        new_id, new_name, new_as_of, remote_vocab_path
                    )
        else:
            assert local_vocab is not None
            vocab = local_vocab

        self.db.set_active_patch(
            new_id,
            new_name,
            new_as_of,
            min_start_datetime=min_start_datetime if valve_ahead else None,
            valve_ahead=valve_ahead,
        )
        self.current = ActivePatchContext(
            gv_id=new_id,
            gv_name=new_name,
            as_of_datetime=new_as_of,
            vocab=vocab,
            switched=switched,
            stratz_gv_ids=stratz_gv_ids,
            min_start_datetime=min_start_datetime,
            max_start_datetime=max_start_datetime,
            valve_ahead=valve_ahead,
        )
        return self.current

    def should_check_patch(self, interval_seconds: int) -> bool:
        if self.current is None:
            return True
        return (time.monotonic() - self._last_checked_at) >= interval_seconds
