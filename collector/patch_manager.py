"""Patch detection, auto-switching, buffer flushing, and vocab synchronization."""

from __future__ import annotations

import gc
import json
import logging
import time
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
from shared.vocab import build_vocab

logger = logging.getLogger(__name__)


@dataclass
class ActivePatchContext:
    """Currently active Dota 2 patch metadata and built vocabulary."""

    gv_id: int
    gv_name: str
    as_of_datetime: int
    vocab: dict[str, Any]
    switched: bool = False


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
    ) -> None:
        self.target_game_version = target_game_version
        self.db = db
        self.stratz = stratz
        self.hf_storage = hf_storage
        self.http_client = http_client
        self.constants_cache_dir = constants_cache_dir
        self.parquet_dir = parquet_dir
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

    def _vocab_cache_path(self, gv_id: int) -> Path:
        return self.constants_cache_dir / f"vocab_{int(gv_id)}.json"

    @staticmethod
    def _load_valid_local_vocab(vocab_path: Path, gv_id: int) -> dict[str, Any] | None:
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
        self, target: dict[str, Any]
    ) -> dict[str, Any]:
        gv_id = int(target["id"])
        stratz_consts = await self.stratz.get_constants(gv_id)
        odota_consts = await fetch_dotaconstants(
            self.http_client, cache_dir=self.constants_cache_dir
        )
        vocab = build_vocab(target, stratz_consts, odota_consts)
        self._vocab_cache_path(gv_id).write_text(
            json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return vocab

    async def sync_patch(self, *, force: bool = False) -> ActivePatchContext:
        """Check STRATZ gameVersions and handle initial setup or live patch switch."""
        self._last_checked_at = time.monotonic()
        versions = await self.stratz.get_game_versions()
        target = self._select_target_version(versions)
        new_id = int(target["id"])
        new_name = str(target["name"])
        new_as_of = int(target["asOfDateTime"])

        prev = self.db.get_active_patch()
        switched = prev is not None and int(prev["id"]) != new_id

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
        local_vocab = self._load_valid_local_vocab(vocab_path, new_id)

        hf_latest: dict[str, Any] | None = None
        hf_needs_update = False
        if force or prev is None or switched or local_vocab is None or self.current is None:
            hf_latest = await self.hf_storage.read_latest_manifest()
            expected_vocab_path = f"{patch_data_prefix(new_id, new_name)}/vocab.json"
            hf_needs_update = self._hf_manifest_needs_update(
                hf_latest, gv_id=new_id, expected_vocab_path=expected_vocab_path
            )

        needs_vocab_sync = (
            force or prev is None or switched or local_vocab is None or hf_needs_update
        )

        if needs_vocab_sync:
            await self.hf_storage.ensure_repo()
            vocab = await self._build_and_save_vocab(target)
            remote_vocab_path = await self.hf_storage.upload_vocab(new_id, new_name, vocab)

            if switched or hf_needs_update:
                await self.hf_storage.update_manifest_for_patch(
                    new_id, new_name, new_as_of, remote_vocab_path
                )
        else:
            assert local_vocab is not None
            vocab = local_vocab

        self.db.set_active_patch(new_id, new_name, new_as_of)
        self.current = ActivePatchContext(
            gv_id=new_id,
            gv_name=new_name,
            as_of_datetime=new_as_of,
            vocab=vocab,
            switched=switched,
        )
        return self.current

    def should_check_patch(self, interval_seconds: int) -> bool:
        if self.current is None:
            return True
        return (time.monotonic() - self._last_checked_at) >= interval_seconds
