"""Hugging Face Datasets storage client for shards, vocab.json, and latest.json."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import (
    EntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from shared.constants import PARQUET_SCHEMA_VERSION, VOCAB_SCHEMA_VERSION

logger = logging.getLogger(__name__)


def sanitize_patch_name(name: str) -> str:
    """Normalize a patch name for safe directory paths (e.g., '7.41f' -> '7.41f')."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return cleaned.strip("_") or "unknown"


def patch_data_prefix(gv_id: int, gv_name: str) -> str:
    return f"data/gv_{int(gv_id)}_{sanitize_patch_name(gv_name)}"


class HFDatasetStorage:
    """Uploads Parquet shards, vocab.json, and latest.json to Hugging Face Datasets."""

    def __init__(self, repo_id: str, token: str) -> None:
        self.repo_id = repo_id
        self._api = HfApi(token=token)

    def _ensure_repo_sync(self) -> None:
        self._api.create_repo(
            repo_id=self.repo_id,
            repo_type="dataset",
            exist_ok=True,
        )

    async def ensure_repo(self) -> None:
        await asyncio.to_thread(self._ensure_repo_sync)

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1.0, max=10.0),
        reraise=True,
    )
    def _upload_file_sync(
        self, local_path: Path, path_in_repo: str, commit_message: str
    ) -> str:
        return self._api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1.0, max=10.0),
        reraise=True,
    )
    def _read_latest_manifest_sync(self) -> dict[str, Any] | None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                downloaded = self._api.hf_hub_download(
                    repo_id=self.repo_id,
                    filename="latest.json",
                    repo_type="dataset",
                    local_dir=tmpdir,
                    force_download=True,
                )
                return json.loads(Path(downloaded).read_text(encoding="utf-8"))
        except (EntryNotFoundError, RemoteEntryNotFoundError, RepositoryNotFoundError):
            return None

    async def read_latest_manifest(self) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._read_latest_manifest_sync)

    async def upload_vocab(
        self, gv_id: int, gv_name: str, vocab: dict[str, Any]
    ) -> str:
        prefix = patch_data_prefix(gv_id, gv_name)
        path_in_repo = f"{prefix}/vocab.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            local_vocab = Path(tmpdir) / "vocab.json"
            local_vocab.write_text(
                json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            await asyncio.to_thread(
                self._upload_file_sync,
                local_vocab,
                path_in_repo,
                f"Update vocab.json for patch {gv_name} (gv={gv_id})",
            )
        logger.info("Uploaded %s to HF dataset %s", path_in_repo, self.repo_id)
        return path_in_repo

    async def _write_and_upload_manifest(
        self, manifest: dict[str, Any], commit_message: str
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_manifest = Path(tmpdir) / "latest.json"
            local_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            await asyncio.to_thread(
                self._upload_file_sync,
                local_manifest,
                "latest.json",
                commit_message,
            )

    @staticmethod
    def _compatible_existing_shards(
        existing: dict[str, Any] | None, gv_id: int
    ) -> list[dict[str, Any]]:
        if not existing or int(existing.get("game_version_id") or -1) != int(gv_id):
            return []
        existing_schema = int(
            existing.get("parquet_schema_version") or existing.get("schema_version") or 1
        )
        if existing_schema != PARQUET_SCHEMA_VERSION:
            return []
        return list(existing.get("shards") or [])

    @staticmethod
    def _build_manifest(
        *,
        gv_id: int,
        gv_name: str,
        as_of_datetime: int,
        vocab_path: str,
        shards: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prefix = patch_data_prefix(gv_id, gv_name)
        return {
            "schema_version": PARQUET_SCHEMA_VERSION,
            "parquet_schema_version": PARQUET_SCHEMA_VERSION,
            "vocab_schema_version": VOCAB_SCHEMA_VERSION,
            "game_version_id": int(gv_id),
            "game_version_name": str(gv_name),
            "as_of_datetime": int(as_of_datetime),
            "updated_at": datetime.now(UTC).isoformat(),
            "data_prefix": f"{prefix}/",
            "vocab_path": vocab_path,
            "shards": shards,
            "total_matches": sum(int(s.get("num_rows") or 0) for s in shards),
        }

    async def update_manifest_for_patch(
        self,
        gv_id: int,
        gv_name: str,
        as_of_datetime: int,
        vocab_path: str,
    ) -> dict[str, Any]:
        """Update root latest.json when initializing or switching to a patch."""
        existing = await self.read_latest_manifest()
        shards = self._compatible_existing_shards(existing, gv_id)
        manifest = self._build_manifest(
            gv_id=gv_id,
            gv_name=gv_name,
            as_of_datetime=as_of_datetime,
            vocab_path=vocab_path,
            shards=shards,
        )
        await self._write_and_upload_manifest(
            manifest, f"Switch latest.json to patch {gv_name} (gv={gv_id})"
        )
        logger.info("Updated latest.json on HF for patch %s (gv=%s)", gv_name, gv_id)
        return manifest

    async def upload_shard_and_update_manifest(
        self,
        *,
        gv_id: int,
        gv_name: str,
        as_of_datetime: int,
        shard_path: Path,
        shard_meta: dict[str, Any],
        is_active_patch: bool = True,
    ) -> dict[str, Any] | None:
        """Upload a Parquet shard under data/gv_<id>_<name>/ and update latest.json if active."""
        prefix = patch_data_prefix(gv_id, gv_name)
        path_in_repo = f"{prefix}/{shard_path.name}"
        await asyncio.to_thread(
            self._upload_file_sync,
            shard_path,
            path_in_repo,
            f"Upload {shard_meta['num_rows']} matches for patch {gv_name} (gv={gv_id})",
        )
        logger.info("Uploaded shard %s (%d rows)", path_in_repo, shard_meta["num_rows"])

        if not is_active_patch:
            return None

        existing = await self.read_latest_manifest()
        shards = [
            s
            for s in self._compatible_existing_shards(existing, gv_id)
            if s.get("path") != path_in_repo
        ]
        shards.append({**shard_meta, "path": path_in_repo})

        vocab_path = (
            existing.get("vocab_path")
            if existing and int(existing.get("game_version_id") or -1) == int(gv_id)
            else f"{prefix}/vocab.json"
        )
        manifest = self._build_manifest(
            gv_id=gv_id,
            gv_name=gv_name,
            as_of_datetime=as_of_datetime,
            vocab_path=str(vocab_path),
            shards=shards,
        )
        await self._write_and_upload_manifest(
            manifest,
            f"Add shard {shard_path.name} ({shard_meta['num_rows']} matches) to latest.json",
        )
        return manifest
