"""Fetch and cache Valve official Dota 2 datafeed constants and patch notes."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from shared.constants import DOTA2_DATAFEED_BASE_URL

logger = logging.getLogger(__name__)


class ValveDatafeedTransientError(RuntimeError):
    """Retryable HTTP error from Valve Dota 2 datafeed."""


async def _get_datafeed_json(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    base_url: str = DOTA2_DATAFEED_BASE_URL,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type(
            (ValveDatafeedTransientError, httpx.TransportError, httpx.TimeoutException)
        ),
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(multiplier=0.5, max=10.0),
        reraise=True,
    ):
        with attempt:
            resp = await client.get(url, params=params, timeout=30.0)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise ValveDatafeedTransientError(
                    f"Valve datafeed {endpoint} returned HTTP {resp.status_code}"
                )
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected JSON object from {url}, got {type(payload).__name__}"
                )
            return payload

    raise AssertionError("Unreachable retry termination")


def _write_cache_json(cache_subdir: Path | None, filename: str, data: Any) -> None:
    if cache_subdir is None:
        return
    cache_subdir.mkdir(parents=True, exist_ok=True)
    (cache_subdir / filename).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def _read_cache_json(cache_subdir: Path | None, filename: str) -> Any | None:
    if cache_subdir is None:
        return None
    path = cache_subdir / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


async def fetch_valve_patchnotes_list(
    client: httpx.AsyncClient,
    *,
    cache_dir: Path | None = None,
    base_url: str = DOTA2_DATAFEED_BASE_URL,
) -> list[dict[str, Any]]:
    """Fetch official Dota 2 patch list (patch_number, patch_name, patch_timestamp)."""
    cache_subdir = cache_dir / "valve_datafeed" if cache_dir is not None else None
    try:
        payload = await _get_datafeed_json(
            client,
            "patchnoteslist",
            params={"language": "english"},
            base_url=base_url,
        )
        patches = payload.get("patches")
        if patches is None:
            patches = ((payload.get("result") or {}).get("data") or {}).get("patches")
        if not isinstance(patches, list):
            raise ValueError("Valve patchnoteslist response missing 'patches' list")
        clean = [p for p in patches if isinstance(p, dict)]
        _write_cache_json(cache_subdir, "patchnoteslist.json", clean)
        return clean
    except Exception as exc:
        cached = _read_cache_json(cache_subdir, "patchnoteslist.json")
        if isinstance(cached, list):
            logger.warning(
                "Failed to fetch Valve patchnoteslist (%s); falling back to cache", exc
            )
            return cached
        raise


async def fetch_valve_patchnotes(
    client: httpx.AsyncClient,
    version: str,
    *,
    cache_dir: Path | None = None,
    base_url: str = DOTA2_DATAFEED_BASE_URL,
) -> dict[str, Any]:
    """Fetch official Dota 2 patchnotes JSON for a specific patch version string."""
    cache_subdir = cache_dir / "valve_datafeed" if cache_dir is not None else None
    safe_ver = "".join(c if c.isalnum() or c in "._-" else "_" for c in version.strip())
    cache_file = f"patchnotes_{safe_ver}.json"
    try:
        payload = await _get_datafeed_json(
            client,
            "patchnotes",
            params={"version": version, "language": "english"},
            base_url=base_url,
        )
        _write_cache_json(cache_subdir, cache_file, payload)
        return payload
    except Exception as exc:
        cached = _read_cache_json(cache_subdir, cache_file)
        if isinstance(cached, dict):
            logger.warning(
                "Failed to fetch Valve patchnotes for %s (%s); falling back to cache",
                version,
                exc,
            )
            return cached
        raise


async def _fetch_hero_list(
    client: httpx.AsyncClient,
    cache_subdir: Path | None,
    base_url: str,
) -> list[dict[str, Any]]:
    try:
        payload = await _get_datafeed_json(
            client, "herolist", params={"language": "english"}, base_url=base_url
        )
        heroes = ((payload.get("result") or {}).get("data") or {}).get("heroes")
        if not isinstance(heroes, list):
            raise ValueError("Valve herolist response missing 'result.data.heroes'")
        clean = [h for h in heroes if isinstance(h, dict) and h.get("id") is not None]
        _write_cache_json(cache_subdir, "herolist.json", clean)
        return clean
    except Exception as exc:
        cached = _read_cache_json(cache_subdir, "herolist.json")
        if isinstance(cached, list):
            logger.warning("Failed to fetch Valve herolist (%s); using cache", exc)
            return cached
        raise


async def _fetch_all_herodata(
    client: httpx.AsyncClient,
    hero_list: list[dict[str, Any]],
    cache_subdir: Path | None,
    base_url: str,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def _one(hero_id: int) -> dict[str, Any] | None:
        async with sem:
            payload = await _get_datafeed_json(
                client,
                "herodata",
                params={"language": "english", "hero_id": int(hero_id)},
                base_url=base_url,
            )
            heroes = ((payload.get("result") or {}).get("data") or {}).get("heroes") or []
            return heroes[0] if heroes and isinstance(heroes[0], dict) else None

    try:
        raw_results = await asyncio.gather(*(_one(int(h["id"])) for h in hero_list))
        heroes_out = [h for h in raw_results if h is not None and h.get("id") is not None]
        if not heroes_out:
            raise ValueError("Valve herodata returned zero hero records")
        by_id = {str(int(h["id"])): h for h in heroes_out}
        _write_cache_json(cache_subdir, "herodata_by_id.json", by_id)
        return sorted(heroes_out, key=lambda h: int(h["id"]))
    except Exception as exc:
        cached = _read_cache_json(cache_subdir, "herodata_by_id.json")
        if isinstance(cached, dict) and cached:
            logger.warning("Failed to fetch Valve herodata (%s); using cache", exc)
            return sorted(cached.values(), key=lambda h: int(h["id"]))
        raise


async def _fetch_item_list(
    client: httpx.AsyncClient,
    cache_subdir: Path | None,
    base_url: str,
) -> list[dict[str, Any]]:
    try:
        payload = await _get_datafeed_json(
            client, "itemlist", params={"language": "english"}, base_url=base_url
        )
        items = ((payload.get("result") or {}).get("data") or {}).get("itemabilities")
        if not isinstance(items, list):
            raise ValueError("Valve itemlist response missing 'result.data.itemabilities'")
        clean = [it for it in items if isinstance(it, dict) and it.get("id") is not None]
        _write_cache_json(cache_subdir, "itemlist.json", clean)
        return clean
    except Exception as exc:
        cached = _read_cache_json(cache_subdir, "itemlist.json")
        if isinstance(cached, list):
            logger.warning("Failed to fetch Valve itemlist (%s); using cache", exc)
            return cached
        raise


async def _fetch_all_itemdata(
    client: httpx.AsyncClient,
    item_list: list[dict[str, Any]],
    cache_subdir: Path | None,
    base_url: str,
    max_concurrency: int,
) -> dict[int, dict[str, Any]]:
    candidates = [
        it
        for it in item_list
        if not str(it.get("name") or "").startswith("item_recipe_")
        and int(it.get("neutral_item_tier", -1)) == -1
    ]
    sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def _one(item_id: int) -> dict[str, Any] | None:
        async with sem:
            payload = await _get_datafeed_json(
                client,
                "itemdata",
                params={"language": "english", "item_id": int(item_id)},
                base_url=base_url,
            )
            items = ((payload.get("result") or {}).get("data") or {}).get("items") or []
            return items[0] if items and isinstance(items[0], dict) else None

    try:
        raw_results = await asyncio.gather(*(_one(int(it["id"])) for it in candidates))
        by_id_int: dict[int, dict[str, Any]] = {}
        for item in raw_results:
            if item is not None and item.get("id") is not None:
                by_id_int[int(item["id"])] = item
        if candidates and not by_id_int:
            raise ValueError("Valve itemdata returned zero item records")
        _write_cache_json(
            cache_subdir,
            "itemdata_by_id.json",
            {str(k): v for k, v in sorted(by_id_int.items())},
        )
        return by_id_int
    except Exception as exc:
        cached = _read_cache_json(cache_subdir, "itemdata_by_id.json")
        if isinstance(cached, dict) and cached:
            logger.warning("Failed to fetch Valve itemdata (%s); using cache", exc)
            return {int(k): v for k, v in cached.items()}
        raise


async def _fetch_ability_list(
    client: httpx.AsyncClient,
    cache_subdir: Path | None,
    base_url: str,
) -> list[dict[str, Any]]:
    try:
        payload = await _get_datafeed_json(
            client, "abilitylist", params={"language": "english"}, base_url=base_url
        )
        abilities = ((payload.get("result") or {}).get("data") or {}).get("itemabilities")
        if not isinstance(abilities, list):
            raise ValueError("Valve abilitylist response missing 'result.data.itemabilities'")
        clean = [ab for ab in abilities if isinstance(ab, dict) and ab.get("id") is not None]
        _write_cache_json(cache_subdir, "abilitylist.json", clean)
        return clean
    except Exception as exc:
        cached = _read_cache_json(cache_subdir, "abilitylist.json")
        if isinstance(cached, list):
            logger.warning("Failed to fetch Valve abilitylist (%s); using cache", exc)
            return cached
        raise


async def fetch_valve_constants(
    client: httpx.AsyncClient,
    *,
    patch_name: str | None = None,
    cache_dir: Path | None = None,
    base_url: str = DOTA2_DATAFEED_BASE_URL,
    max_concurrency: int = 12,
) -> dict[str, Any]:
    """Fetch complete Valve datafeed heroes, items, abilities, and optional patch item IDs."""
    cache_subdir = cache_dir / "valve_datafeed" if cache_dir is not None else None

    hero_list, item_list, ability_list = await asyncio.gather(
        _fetch_hero_list(client, cache_subdir, base_url),
        _fetch_item_list(client, cache_subdir, base_url),
        _fetch_ability_list(client, cache_subdir, base_url),
    )

    heroes, itemdata = await asyncio.gather(
        _fetch_all_herodata(client, hero_list, cache_subdir, base_url, max_concurrency),
        _fetch_all_itemdata(client, item_list, cache_subdir, base_url, max_concurrency),
    )

    patch_item_ids: list[int] = []
    if patch_name:
        try:
            pn = await fetch_valve_patchnotes(
                client, patch_name, cache_dir=cache_dir, base_url=base_url
            )
            seen_pids: set[int] = set()
            for entry in pn.get("items") or []:
                if isinstance(entry, dict) and entry.get("ability_id") is not None:
                    pid = int(entry["ability_id"])
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        patch_item_ids.append(pid)
        except Exception as exc:
            logger.warning(
                "Could not fetch Valve patchnotes for %s (%s); continuing without patch_item_ids",
                patch_name,
                exc,
            )

    return {
        "heroes": heroes,
        "itemlist": item_list,
        "itemdata": itemdata,
        "abilitylist": ability_list,
        "patch_item_ids": patch_item_ids,
    }
