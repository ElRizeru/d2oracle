"""Unit tests for SteamClient coalescing, rate limiting, and secret redaction."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from collector.clients.steam import SteamAPIError, SteamClient, SteamTransientError
from collector.rate_limit import QuotaExhaustedError, build_steam_rate_limiter
from shared.constants import STEAM_API_BASE_URL


async def test_steam_seq_num_coalescing_and_miss_fallback() -> None:
    calls: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-webapi-key") == "dummy_steam_key"
        assert "key=" not in str(request.url)
        start = int(request.url.params["start_at_match_seq_num"])
        req_n = int(request.url.params["matches_requested"])
        calls.append((start, req_n))
        if start == 1000 and req_n > 1:
            # Omit match 1085 on the coalesced range to verify single-match fallback
            return httpx.Response(
                200,
                json={
                    "result": {
                        "status": 1,
                        "matches": [
                            {"match_id": 9001, "match_seq_num": 1000},
                            {"match_id": 9002, "match_seq_num": 1020},
                        ],
                    }
                },
            )
        if start == 1085 and req_n == 1:
            return httpx.Response(
                200,
                json={"result": {"status": 1, "matches": [{"match_id": 9003, "match_seq_num": 1085}]}},
            )
        if start == 2000 and req_n == 1:
            return httpx.Response(
                200,
                json={"result": {"status": 1, "matches": [{"match_id": 9004, "match_seq_num": 2000}]}},
            )
        raise AssertionError(f"Unexpected call start={start}, req={req_n}")

    with respx.mock:
        respx.get(
            f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetMatchHistoryBySequenceNum/v1/"
        ).mock(side_effect=handler)
        async with httpx.AsyncClient() as http_client:
            steam = SteamClient(
                api_key="dummy_steam_key",
                rate_limiter=build_steam_rate_limiter(rps=20, rpm=600, rpd=90000),
                http_client=http_client,
            )
            out = await steam.get_matches_for_seq_nums(
                {1085: 9003, 1000: 9001, 2000: 9004, 1020: 9002},
                max_concurrency=2,
                coalesce_span=90,
            )

    assert sorted(out.keys()) == [9001, 9002, 9003, 9004]
    assert (1000, 86) in calls
    assert (2000, 1) in calls
    assert (1085, 1) in calls


async def test_steam_private_history_and_live_endpoints() -> None:
    with respx.mock:
        respx.get(f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetMatchHistory/v1/").mock(
            return_value=httpx.Response(
                200,
                json={"result": {"status": 15, "statusDetail": "Cannot get match history for a user that hasn't allowed it"}},
            )
        )
        respx.get(f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetTopLiveGame/v1/").mock(
            return_value=httpx.Response(
                200,
                json={"game_list": [{"match_id": "9096001", "lobby_type": 7, "players": [{"account_id": 111}]}]},
            )
        )
        respx.get(f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetLiveLeagueGames/v1/").mock(
            return_value=httpx.Response(
                200,
                json={"result": {"games": [{"match_id": 9096002, "league_id": 20511, "players": [{"account_id": 222, "team": 0}]}]}},
            )
        )

        async with httpx.AsyncClient() as http_client:
            steam = SteamClient(
                api_key="dummy_steam_key",
                rate_limiter=build_steam_rate_limiter(rps=20, rpm=600, rpd=90000),
                http_client=http_client,
            )
            hist = await steam.get_match_history(account_id=12345)
            live = await steam.get_top_live_games(partner=0)
            league = await steam.get_live_league_games()

    assert hist == []
    assert len(live) == 1 and live[0]["match_id"] == "9096001"
    assert len(league) == 1 and league[0]["match_id"] == 9096002


async def test_steam_redacts_api_key_on_errors_and_respects_day_cap(tmp_path: Path) -> None:
    secret = "TEST_SECRET_STEAM_KEY_98765"

    def err_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"failed connecting with key={secret}", request=request
        )

    with respx.mock:
        respx.get(f"{STEAM_API_BASE_URL}/IGCVersion_570/GetServerVersion/v1/").mock(
            side_effect=err_handler
        )
        respx.get(f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetTopLiveGame/v1/").mock(
            return_value=httpx.Response(403, text=f"Forbidden for {secret}")
        )
        async with httpx.AsyncClient() as http_client:
            rl = build_steam_rate_limiter(rps=20, rpm=600, rpd=90000)
            steam = SteamClient(api_key=secret, rate_limiter=rl, http_client=http_client)
            steam._get_json.__wrapped__ if hasattr(steam._get_json, "__wrapped__") else None

            # Disable sleep jitter in transport retry by checking exception text after 403 first
            with pytest.raises(SteamAPIError) as api_exc:
                await steam.get_top_live_games()
            assert secret not in str(api_exc.value)
            assert "[REDACTED]" in str(api_exc.value)

    # Check Steam day rate limiter resets at UTC midnight
    now_mono = [100.0]
    now_utc = [1794870000.0]
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)
        now_mono[0] += s

    day_rl = build_steam_rate_limiter(
        rps=20,
        rpm=600,
        rpd=2,
        clock=lambda: now_mono[0],
        utc_clock=lambda: now_utc[0],
        sleeper=fake_sleep,
    )
    await day_rl.acquire()
    await day_rl.acquire()
    with pytest.raises(QuotaExhaustedError) as qexc:
        await day_rl.acquire()
    assert qexc.value.window == "day"

    # Also test SteamTransientError secret scrubbing without sleeping across retries
    with respx.mock:
        respx.get(f"{STEAM_API_BASE_URL}/IGCVersion_570/GetServerVersion/v1/").mock(
            side_effect=err_handler
        )
        async with httpx.AsyncClient() as http_client:
            rl = build_steam_rate_limiter(rps=20, rpm=600, rpd=90000)
            steam = SteamClient(api_key=secret, rate_limiter=rl, http_client=http_client)
            # Speed up tenacity wait in test
            import collector.clients.steam as steam_mod

            orig_wait = steam_mod.wait_exponential_jitter
            steam_mod.wait_exponential_jitter = lambda **kwargs: orig_wait(multiplier=0.001, max=0.001)
            try:
                with pytest.raises(SteamTransientError) as trans_exc:
                    await steam.get_server_version()
            finally:
                steam_mod.wait_exponential_jitter = orig_wait
            assert secret not in str(trans_exc.value)
            assert "[REDACTED]" in str(trans_exc.value)
