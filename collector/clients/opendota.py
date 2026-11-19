"""Async OpenDota REST API client with header-aware rate limiting."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from collector.rate_limit import MultiWindowRateLimiter
from shared.constants import OPENDOTA_API_URL

logger = logging.getLogger(__name__)


class OpenDotaTransientError(RuntimeError):
    """Retryable OpenDota REST error (429, 5xx, transport)."""


class OpenDotaClient:
    """Async client for OpenDota REST API endpoints used in match discovery."""

    def __init__(
        self,
        rate_limiter: MultiWindowRateLimiter,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = OPENDOTA_API_URL,
    ) -> None:
        self.rate_limiter = rate_limiter
        self.base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=25.0)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(
                (OpenDotaTransientError, httpx.TransportError, httpx.TimeoutException)
            ),
            stop=stop_after_attempt(4),
            wait=wait_exponential_jitter(initial=1.0, max=15.0),
            reraise=True,
        ):
            with attempt:
                await self.rate_limiter.acquire()
                resp = await self._client.get(url, params=params, timeout=25.0)
                self.rate_limiter.update_from_headers(resp.headers)

                if resp.status_code == 429:
                    wait = self.rate_limiter.notify_rate_limited(resp.headers)
                    logger.warning("OpenDota returned 429; cooling down %.1fs", wait)
                    raise OpenDotaTransientError("OpenDota HTTP 429")
                if resp.status_code >= 500:
                    raise OpenDotaTransientError(f"OpenDota HTTP {resp.status_code}")
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()

        raise AssertionError("Unreachable retry termination")

    async def get_public_matches(
        self,
        *,
        min_rank: int = 80,
        less_than_match_id: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"min_rank": int(min_rank)}
        if less_than_match_id is not None:
            params["less_than_match_id"] = int(less_than_match_id)
        data = await self._get_json("publicMatches", params=params)
        return list(data) if isinstance(data, list) else []

    async def get_pro_matches(
        self, *, less_than_match_id: int | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if less_than_match_id is not None:
            params["less_than_match_id"] = int(less_than_match_id)
        data = await self._get_json("proMatches", params=params or None)
        return list(data) if isinstance(data, list) else []

    async def get_match(self, match_id: int) -> dict[str, Any] | None:
        """Fetch parsed match details from OpenDota (used on demand for new-patch item purchases)."""
        data = await self._get_json(f"matches/{int(match_id)}")
        return dict(data) if isinstance(data, dict) else None
