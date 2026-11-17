"""Fetch and cache odota/dotaconstants metadata for UI icons, items, and abilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from shared.constants import DOTACONSTANTS_BASE_URL, STEAM_CDN_BASE_URL

logger = logging.getLogger(__name__)

DOTACONSTANTS_FILES: tuple[str, ...] = (
    "heroes.json",
    "items.json",
    "abilities.json",
    "hero_abilities.json",
    "ability_ids.json",
)


def normalize_steam_cdn_url(rel_path: str | None) -> str | None:
    """Convert an odota relative Steam CDN path into an absolute URL."""
    if not rel_path:
        return None
    if rel_path.startswith("http://") or rel_path.startswith("https://"):
        return rel_path
    clean = rel_path.split("?", 1)[0]
    if not clean.startswith("/"):
        clean = f"/{clean}"
    return f"{STEAM_CDN_BASE_URL}{clean}"


async def fetch_dotaconstants(
    client: httpx.AsyncClient,
    cache_dir: Path | None = None,
    base_url: str = DOTACONSTANTS_BASE_URL,
) -> dict[str, dict[str, Any]]:
    """Fetch required odota/dotaconstants JSON files and optionally cache them locally."""
    result: dict[str, dict[str, Any]] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for filename in DOTACONSTANTS_FILES:
        url = f"{base_url.rstrip('/')}/{filename}"
        key = filename.removesuffix(".json")
        try:
            resp = await client.get(url, timeout=30.0)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                raise ValueError(f"Expected dict from {url}, got {type(payload).__name__}")
            result[key] = payload
            if cache_dir is not None:
                (cache_dir / filename).write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
        except Exception as exc:
            if cache_dir is not None and (cache_dir / filename).exists():
                logger.warning(
                    "Failed to fetch %s (%s); falling back to cached file", url, exc
                )
                result[key] = json.loads((cache_dir / filename).read_text(encoding="utf-8"))
            else:
                raise

    return result
