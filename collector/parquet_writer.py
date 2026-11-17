"""Write validated match batches to zstd-compressed Parquet shards."""

from __future__ import annotations

import gc
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from shared.constants import PARQUET_ARROW_SCHEMA


def write_parquet_shard(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    timestamp: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Serialize validated match records into a zstd-compressed Parquet file."""
    if not records:
        raise ValueError("Cannot write an empty Parquet shard")

    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(records)
    match_ids = [int(r["match_id"]) for r in records if r.get("match_id") is not None]
    if match_ids:
        shard_path = output_dir / f"matches_{min(match_ids)}_{max(match_ids)}_{n}.parquet"
    else:
        ts = int(timestamp if timestamp is not None else time.time())
        shard_path = output_dir / f"matches_{ts}_{n}.parquet"
        if shard_path.exists():
            shard_path = output_dir / f"matches_{ts}_{n}_{time.monotonic_ns() % 100000}.parquet"

    table = pa.Table.from_pylist(records, schema=PARQUET_ARROW_SCHEMA)
    try:
        pq.write_table(table, shard_path, compression="zstd")
    finally:
        del table
        pa.default_memory_pool().release_unused()
        gc.collect()

    start_times = [int(r["start_datetime"]) for r in records]
    meta: dict[str, Any] = {
        "num_rows": n,
        "min_start_datetime": min(start_times),
        "max_start_datetime": max(start_times),
        "size_bytes": shard_path.stat().st_size,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return shard_path, meta
