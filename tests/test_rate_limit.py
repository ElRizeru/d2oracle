"""Unit tests for header-aware multi-window async rate limiting."""

from __future__ import annotations

import pytest

from collector.rate_limit import (
    QuotaExhaustedError,
    build_opendota_rate_limiter,
    build_stratz_rate_limiter,
)


class FakeClock:
    def __init__(self, initial: float = 1000.0) -> None:
        self.now = initial
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


async def test_stratz_local_second_window_throttles_above_safe_limit() -> None:
    clock = FakeClock()
    limiter = build_stratz_rate_limiter(
        rps=6, rpm=130, rph=1350, rpd=14000, clock=clock, sleeper=clock.sleep
    )

    for _ in range(6):
        await limiter.acquire()
    assert clock.sleeps == []

    # 7th request within same second must sleep ~1.0s
    await limiter.acquire()
    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] >= 0.99


async def test_stratz_reads_remote_headers_and_throttles_near_zero() -> None:
    clock = FakeClock()
    limiter = build_stratz_rate_limiter(
        rps=6, rpm=130, rph=1350, rpd=14000, clock=clock, sleeper=clock.sleep
    )

    limiter.update_from_headers(
        {
            "x-ratelimit-remaining-second": "1",
            "x-ratelimit-remaining-minute": "140",
            "x-ratelimit-remaining-hour": "1450",
            "x-ratelimit-remaining-day": "13500",
            "ratelimit-reset": "1",
        }
    )
    await limiter.acquire()
    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] >= 0.9


async def test_stratz_minute_soft_reserve_paces_requests() -> None:
    clock = FakeClock()
    limiter = build_stratz_rate_limiter(
        rps=6, rpm=130, rph=1350, rpd=14000, clock=clock, sleeper=clock.sleep
    )
    limiter.update_from_headers(
        {
            "x-ratelimit-remaining-second": "7",
            "x-ratelimit-remaining-minute": "10",  # <= soft_reserve (20)
            "x-ratelimit-remaining-hour": "1400",
            "x-ratelimit-remaining-day": "13000",
        }
    )
    await limiter.acquire()
    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] >= 5.0


async def test_opendota_day_quota_near_zero_raises_exhausted() -> None:
    clock = FakeClock()
    limiter = build_opendota_rate_limiter(rpm=50, rpd=2800, clock=clock, sleeper=clock.sleep)
    limiter.update_from_headers(
        {
            "X-Rate-Limit-Remaining-Minute": "40",
            "X-Rate-Limit-Remaining-Day": "10",  # <= hard_stop_reserve (25)
        }
    )
    with pytest.raises(QuotaExhaustedError) as exc_info:
        await limiter.acquire()
    assert exc_info.value.window == "day"


async def test_http_429_cooldown_applies_retry_after() -> None:
    clock = FakeClock()
    limiter = build_stratz_rate_limiter(clock=clock, sleeper=clock.sleep)
    wait = limiter.notify_rate_limited({"Retry-After": "12"})
    assert wait == 12.0
    await limiter.acquire()
    assert clock.sleeps == [12.0]


async def test_stratz_compact_day_soft_reserve_does_not_throttle_early_when_rpd_lowered() -> None:
    clock = FakeClock()
    limiter = build_stratz_rate_limiter(rpd=9200, clock=clock, sleeper=clock.sleep)
    # With 5000 requests remaining today, compact soft_reserve (250) must NOT pace early.
    limiter.update_from_headers(
        {
            "x-ratelimit-remaining-second": "7",
            "x-ratelimit-remaining-minute": "140",
            "x-ratelimit-remaining-hour": "1400",
            "x-ratelimit-remaining-day": "5000",
            "x-ratelimit-limit-day": "15000",
        }
    )
    await limiter.acquire()
    assert clock.sleeps == []


async def test_stratz_day_exhaustion_bounds_probe_ttl_and_clears_stale_remote() -> None:
    clock = FakeClock(initial=10_000.0)
    limiter = build_stratz_rate_limiter(
        clock=clock, utc_clock=lambda: 1000.0, sleeper=clock.sleep
    )
    limiter.update_from_headers(
        {
            "x-ratelimit-remaining-second": "7",
            "x-ratelimit-remaining-minute": "140",
            "x-ratelimit-remaining-hour": "1400",
            "x-ratelimit-remaining-day": "2",
        }
    )
    with pytest.raises(QuotaExhaustedError) as exc_info:
        await limiter.acquire()
    assert exc_info.value.window == "day"
    assert exc_info.value.retry_after_seconds <= 1800.0

    # Advancing past day probe_ttl_seconds (1800s) clears stale remote exhaustion
    clock.now += 1805.0
    await limiter.acquire()
    assert clock.sleeps == []


async def test_stratz_day_bucket_resets_at_utc_midnight_while_opendota_stays_sliding() -> None:
    mono = FakeClock(initial=100.0)
    # 2026-11-16 23:00:00 UTC
    utc = FakeClock(initial=1794870000.0)
    stratz_rl = build_stratz_rate_limiter(
        rps=100, rpm=1000, rph=5000, rpd=3, clock=mono, utc_clock=utc, sleeper=mono.sleep
    )
    for _ in range(3):
        await stratz_rl.acquire()
    stratz_rl.update_from_headers({"x-ratelimit-remaining-day": "2"})

    # Next request same UTC day must raise QuotaExhaustedError (~3601s until 00:00 UTC)
    with pytest.raises(QuotaExhaustedError) as exc_info:
        await stratz_rl.acquire()
    assert exc_info.value.window == "day"
    assert 3500.0 <= exc_info.value.retry_after_seconds <= 3700.0

    # Crossing 00:00 UTC immediately clears local utc_day_count and remote_remaining
    utc.now += 3605.0
    await stratz_rl.acquire()
    assert mono.sleeps == []

    # OpenDota day bucket still uses 24h sliding window and keeps blocking after UTC midnight
    od_rl = build_opendota_rate_limiter(rpm=100, rpd=2, clock=mono, sleeper=mono.sleep)
    od_rl._utc_clock = utc  # type: ignore[attr-defined]
    await od_rl.acquire()
    await od_rl.acquire()
    with pytest.raises(QuotaExhaustedError) as od_exc:
        await od_rl.acquire()
    assert od_exc.value.window == "day"


async def test_local_hour_cap_raises_quota_exhausted_above_threshold() -> None:
    clock = FakeClock(initial=500.0)
    limiter = build_stratz_rate_limiter(
        rps=100,
        rpm=1000,
        rph=3,
        rpd=10000,
        clock=clock,
        utc_clock=lambda: 1000.0,
        sleeper=clock.sleep,
    )
    for _ in range(3):
        await limiter.acquire()
    with pytest.raises(QuotaExhaustedError) as exc_info:
        await limiter.acquire()
    assert exc_info.value.window == "hour"
    assert exc_info.value.retry_after_seconds > 300.0
    assert clock.sleeps == []


async def test_stratz_hour_soft_reserve_and_pace_cap() -> None:
    clock = FakeClock(initial=500.0)
    limiter = build_stratz_rate_limiter(
        rps=6,
        rpm=130,
        rph=1350,
        rpd=14000,
        clock=clock,
        utc_clock=lambda: 1794870000.0,
        sleeper=clock.sleep,
    )
    limiter.update_from_headers(
        {
            "x-ratelimit-remaining-second": "7",
            "x-ratelimit-remaining-minute": "140",
            "x-ratelimit-remaining-hour": "31",
            "x-ratelimit-remaining-day": "14000",
        }
    )
    await limiter.acquire()
    assert clock.sleeps == []

    # Entering soft_reserve (<= 30) triggers gentle pacing capped at <= 2.0s
    limiter.update_from_headers(
        {
            "x-ratelimit-remaining-second": "7",
            "x-ratelimit-remaining-minute": "140",
            "x-ratelimit-remaining-hour": "30",
            "x-ratelimit-remaining-day": "14000",
        }
    )
    await limiter.acquire()
    assert len(clock.sleeps) == 1
    assert 0.0 < clock.sleeps[0] <= 2.05

    limiter.update_from_headers(
        {
            "x-ratelimit-remaining-second": "7",
            "x-ratelimit-remaining-minute": "140",
            "x-ratelimit-remaining-hour": "20",
            "x-ratelimit-remaining-day": "14000",
        }
    )
    await limiter.acquire()
    assert len(clock.sleeps) == 2
    assert 0.0 < clock.sleeps[1] <= 2.05
