"""Header-aware multi-window async rate limiters for STRATZ and OpenDota."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class QuotaExhaustedError(RuntimeError):
    """Raised when a remote API daily/hourly quota is near hard exhaustion."""

    def __init__(self, limiter_name: str, window: str, retry_after_seconds: float) -> None:
        self.limiter_name = limiter_name
        self.window = window
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"{limiter_name} {window} quota exhausted; retry after {retry_after_seconds:.1f}s"
        )


@dataclass
class WindowBucket:
    """Tracks one sliding or UTC-calendar rate window and remote header counters."""

    name: str
    window_seconds: float
    safe_limit: int
    hard_limit: int
    soft_reserve: int
    hard_stop_reserve: int
    calendar_utc_day: bool = False
    timestamps: deque[float] = field(default_factory=deque)
    utc_day: int | None = None
    utc_day_count: int = 0
    last_request_at: float | None = None
    remote_remaining: int | None = None
    remote_updated_at: float | None = None
    remote_reset_seconds: float | None = None

    @staticmethod
    def seconds_until_utc_midnight(utc_now: float) -> float:
        cur_day = int(utc_now // 86400)
        next_midnight = (cur_day + 1) * 86400.0
        return max(1.0, next_midnight - utc_now)

    def prune(self, now: float, *, utc_now: float | None = None) -> None:
        if self.calendar_utc_day:
            u_now = time.time() if utc_now is None else float(utc_now)
            cur_day = int(u_now // 86400)
            if self.utc_day is None:
                self.utc_day = cur_day
            elif cur_day != self.utc_day:
                self.utc_day = cur_day
                self.utc_day_count = 0
                self.remote_remaining = None
                self.remote_updated_at = None
                self.remote_reset_seconds = None
            return

        cutoff = now - self.window_seconds
        while self.timestamps and self.timestamps[0] <= cutoff:
            self.timestamps.popleft()

    def probe_ttl_seconds(self) -> float:
        """Bound how long an exhausted remote quota header blocks requests before probing."""
        if self.name == "day":
            return min(self.window_seconds, 1800.0)
        if self.name == "hour":
            return min(self.window_seconds, 600.0)
        if self.name == "second" and self.remote_reset_seconds:
            return self.remote_reset_seconds
        return self.window_seconds

    def clear_stale_remote(self, now: float) -> None:
        if self.remote_updated_at is None:
            return
        if self.remote_remaining is not None and self.remote_remaining <= self.hard_stop_reserve:
            ttl = self.probe_ttl_seconds()
        elif self.name == "second" and self.remote_reset_seconds:
            ttl = self.remote_reset_seconds
        else:
            ttl = self.window_seconds
        if now - self.remote_updated_at >= ttl:
            self.remote_remaining = None
            self.remote_updated_at = None
            self.remote_reset_seconds = None


class MultiWindowRateLimiter:
    """Async multi-window rate limiter combining local caps, UTC day reset, and API headers."""

    def __init__(
        self,
        name: str,
        buckets: list[WindowBucket],
        *,
        clock: Callable[[], float] = time.monotonic,
        utc_clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        raise_on_long_exhaustion: bool = True,
        long_exhaustion_threshold_seconds: float = 300.0,
    ) -> None:
        self.name = name
        self.buckets: dict[str, WindowBucket] = {b.name: b for b in buckets}
        self._clock = clock
        self._utc_clock = utc_clock
        self._sleeper = sleeper
        self._lock = asyncio.Lock()
        self._cooldown_until: float = 0.0
        self._raise_on_long_exhaustion = raise_on_long_exhaustion
        self._long_exhaustion_threshold_seconds = long_exhaustion_threshold_seconds

    def _raise_if_long_exhaustion(self, bucket_name: str, wait_seconds: float) -> None:
        if (
            self._raise_on_long_exhaustion
            and bucket_name in ("hour", "day")
            and wait_seconds > self._long_exhaustion_threshold_seconds
        ):
            raise QuotaExhaustedError(self.name, bucket_name, wait_seconds)

    def _compute_wait_seconds(self, now: float, *, utc_now: float | None = None) -> float:
        u_now = self._utc_clock() if utc_now is None else float(utc_now)
        wait = 0.0
        if self._cooldown_until > now:
            wait = max(wait, self._cooldown_until - now)

        for bucket in self.buckets.values():
            bucket.prune(now, utc_now=u_now)
            bucket.clear_stale_remote(now)

            # 1. Local window cap (UTC calendar day or sliding window)
            if bucket.calendar_utc_day:
                if bucket.utc_day_count >= bucket.safe_limit:
                    local_wait = bucket.seconds_until_utc_midnight(u_now) + 1.0
                    self._raise_if_long_exhaustion(bucket.name, local_wait)
                    wait = max(wait, local_wait)
            elif len(bucket.timestamps) >= bucket.safe_limit:
                oldest = bucket.timestamps[0]
                local_wait = (oldest + bucket.window_seconds) - now + 0.01
                self._raise_if_long_exhaustion(bucket.name, local_wait)
                wait = max(wait, local_wait)

            # 2. Remote header awareness
            if bucket.remote_remaining is not None and bucket.remote_updated_at is not None:
                rem = bucket.remote_remaining
                elapsed = max(0.0, now - bucket.remote_updated_at)

                if rem <= bucket.hard_stop_reserve:
                    if bucket.name == "second":
                        reset = bucket.remote_reset_seconds or 1.0
                        rem_wait = max(0.05, reset - elapsed)
                    elif bucket.name == "minute":
                        rem_wait = max(1.0, 60.0 - elapsed)
                    elif bucket.name == "hour":
                        rem_wait = min(bucket.probe_ttl_seconds(), max(60.0, 3600.0 - elapsed))
                        self._raise_if_long_exhaustion(bucket.name, rem_wait)
                    else:  # day
                        day_remaining = (
                            bucket.seconds_until_utc_midnight(u_now) + 1.0
                            if bucket.calendar_utc_day
                            else 86400.0 - elapsed
                        )
                        rem_wait = min(bucket.probe_ttl_seconds(), max(60.0, day_remaining))
                        self._raise_if_long_exhaustion(bucket.name, rem_wait)
                    wait = max(wait, rem_wait)
                elif rem <= bucket.soft_reserve:
                    last_activity = (
                        bucket.last_request_at
                        if bucket.last_request_at is not None
                        else (bucket.timestamps[-1] if bucket.timestamps else bucket.remote_updated_at)
                    )
                    since_last = now - last_activity
                    if bucket.name == "second":
                        reset = bucket.remote_reset_seconds or 1.0
                        wait = max(wait, max(0.05, reset - elapsed))
                    elif bucket.name == "minute":
                        # Spread remaining minute budget smoothly
                        pace = max(0.5, (60.0 - elapsed) / max(1, rem))
                        wait = max(wait, max(0.0, pace - since_last))
                    elif bucket.name == "hour":
                        pace = min(2.0, max(0.5, (3600.0 - elapsed) / max(1, rem)))
                        wait = max(wait, max(0.0, pace - since_last))
                    elif bucket.name == "day":
                        day_remaining = (
                            bucket.seconds_until_utc_midnight(u_now)
                            if bucket.calendar_utc_day
                            else 86400.0 - elapsed
                        )
                        pace = min(10.0, max(0.5, day_remaining / max(1, rem)))
                        wait = max(wait, max(0.0, pace - since_last))

        return wait

    async def acquire(self) -> None:
        """Wait until all local and header-based rate windows permit one request."""
        while True:
            async with self._lock:
                now = self._clock()
                u_now = self._utc_clock()
                wait_seconds = self._compute_wait_seconds(now, utc_now=u_now)
                if wait_seconds <= 0.0:
                    for bucket in self.buckets.values():
                        bucket.last_request_at = now
                        if bucket.calendar_utc_day:
                            bucket.utc_day_count += 1
                        else:
                            bucket.timestamps.append(now)
                        if bucket.remote_remaining is not None and bucket.remote_remaining > 0:
                            bucket.remote_remaining -= 1
                    return
            logger.debug("[%s] rate limit sleeping %.2fs", self.name, wait_seconds)
            await self._sleeper(wait_seconds)

    def update_from_headers(self, headers: Mapping[str, str]) -> None:
        """Update remote remaining and reset metadata from HTTP response headers."""
        lowered = {k.lower(): v for k, v in headers.items()}
        now = self._clock()
        u_now = self._utc_clock()

        reset_sec: float | None = None
        if "ratelimit-reset" in lowered:
            reset_sec = _parse_float(lowered["ratelimit-reset"])
        elif "retry-after" in lowered:
            reset_sec = _parse_float(lowered["retry-after"])

        header_map = {
            "second": ("x-ratelimit-remaining-second", "x-ratelimit-limit-second"),
            "minute": ("x-ratelimit-remaining-minute", "x-ratelimit-limit-minute", "x-rate-limit-remaining-minute"),
            "hour": ("x-ratelimit-remaining-hour", "x-ratelimit-limit-hour"),
            "day": ("x-ratelimit-remaining-day", "x-ratelimit-limit-day", "x-rate-limit-remaining-day"),
        }

        for wname, keys in header_map.items():
            bucket = self.buckets.get(wname)
            if bucket is None:
                continue
            bucket.prune(now, utc_now=u_now)
            rem_val = None
            for k in keys:
                if "limit-" in k and "remaining" not in k:
                    lim = _parse_int(lowered.get(k))
                    if lim is not None and lim > 0:
                        bucket.hard_limit = lim
                elif k in lowered:
                    rem_val = _parse_int(lowered.get(k))
            if rem_val is not None:
                bucket.remote_remaining = rem_val
                bucket.remote_updated_at = now
                if wname == "second" and reset_sec is not None:
                    bucket.remote_reset_seconds = max(0.1, reset_sec)

    def notify_rate_limited(self, headers: Mapping[str, str] | None = None, default_wait: float = 5.0) -> float:
        """Apply explicit cooldown on HTTP 429 responses."""
        wait = default_wait
        if headers:
            self.update_from_headers(headers)
            lowered = {k.lower(): v for k, v in headers.items()}
            parsed = _parse_float(lowered.get("retry-after")) or _parse_float(lowered.get("ratelimit-reset"))
            if parsed is not None and parsed > 0:
                wait = max(wait, parsed)
        now = self._clock()
        self._cooldown_until = max(self._cooldown_until, now + wait)
        return wait


def build_stratz_rate_limiter(
    *,
    rps: int = 6,
    rpm: int = 130,
    rph: int = 1350,
    rpd: int = 14000,
    clock: Callable[[], float] = time.monotonic,
    utc_clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> MultiWindowRateLimiter:
    """Create a STRATZ GraphQL limiter with safe ceilings below 8/s, 150/m, 1500/h, 15000/d."""
    return MultiWindowRateLimiter(
        name="stratz",
        buckets=[
            WindowBucket("second", 1.0, rps, 8, soft_reserve=max(1, 8 - rps), hard_stop_reserve=0),
            WindowBucket("minute", 60.0, rpm, 150, soft_reserve=min(25, max(5, 150 - rpm)), hard_stop_reserve=2),
            WindowBucket("hour", 3600.0, rph, 1500, soft_reserve=min(30, max(10, 1500 - rph)), hard_stop_reserve=10),
            WindowBucket(
                "day",
                86400.0,
                rpd,
                15000,
                soft_reserve=250,
                hard_stop_reserve=50,
                calendar_utc_day=True,
            ),
        ],
        clock=clock,
        utc_clock=utc_clock,
        sleeper=sleeper,
    )


def build_opendota_rate_limiter(
    *,
    rpm: int = 50,
    rpd: int = 2800,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> MultiWindowRateLimiter:
    """Create an OpenDota REST limiter with safe ceilings below 60/m and 3000/d."""
    return MultiWindowRateLimiter(
        name="opendota",
        buckets=[
            WindowBucket("minute", 60.0, rpm, 60, soft_reserve=max(5, 60 - rpm), hard_stop_reserve=2),
            WindowBucket("day", 86400.0, rpd, 3000, soft_reserve=max(50, 3000 - rpd), hard_stop_reserve=25),
        ],
        clock=clock,
        sleeper=sleeper,
    )


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value.strip()))
    except (ValueError, TypeError):
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.strip())
    except (ValueError, TypeError):
        return None
