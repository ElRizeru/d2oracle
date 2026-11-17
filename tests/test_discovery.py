"""Unit tests for all 3 discovery sources using respx-mocked STRATZ and OpenDota APIs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import respx

from collector.clients.opendota import OpenDotaClient
from collector.clients.stratz import StratzClient
from collector.db import CollectorDB
from collector.discovery import DiscoveryEngine
from collector.rate_limit import build_opendota_rate_limiter, build_stratz_rate_limiter
from shared.constants import OPENDOTA_API_URL, STRATZ_GRAPHQL_URL


async def test_public_seeds_filters_ranked_delay_and_patch_cutoff(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "disc.sqlite3")
    now_ts = 1794600000
    patch_as_of = now_ts - 86400

    def public_handler(request: httpx.Request) -> httpx.Response:
        lt = request.url.params.get("less_than_match_id")
        if lt is None:
            return httpx.Response(
                200,
                json=[{"match_id": 9088800000, "start_time": now_ts, "duration": 2000, "lobby_type": 7}],
                headers={"X-Rate-Limit-Remaining-Minute": "58", "X-Rate-Limit-Remaining-Day": "2900"},
            )
        assert lt == "9088580000"
        return httpx.Response(
            200,
            json=[
                # Valid ranked Immortal match (avg_rank_tier >= 80) 2h ago
                {"match_id": 9088570000, "start_time": now_ts - 7200, "duration": 1800, "lobby_type": 7, "avg_rank_tier": 80},
                # avg_rank_tier < 80 -> excluded before match_queue
                {"match_id": 9088569996, "start_time": now_ts - 7250, "duration": 1800, "lobby_type": 7, "avg_rank_tier": 75},
                # Missing avg_rank_tier -> excluded before match_queue
                {"match_id": 9088569995, "start_time": now_ts - 7260, "duration": 1800, "lobby_type": 7},
                # Non-ranked lobby_type == 0 -> excluded
                {"match_id": 9088569999, "start_time": now_ts - 7300, "duration": 1800, "lobby_type": 0, "avg_rank_tier": 80},
                # Too short -> excluded
                {"match_id": 9088569998, "start_time": now_ts - 7400, "duration": 600, "lobby_type": 7, "avg_rank_tier": 80},
                # Before patch -> excluded
                {"match_id": 9088569997, "start_time": patch_as_of - 10, "duration": 2200, "lobby_type": 7, "avg_rank_tier": 80},
            ],
            headers={"X-Rate-Limit-Remaining-Minute": "57", "X-Rate-Limit-Remaining-Day": "2899"},
        )

    with respx.mock:
        respx.get(f"{OPENDOTA_API_URL}/publicMatches").mock(side_effect=public_handler)

        async with httpx.AsyncClient() as client:
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            eng = DiscoveryEngine(db=db, stratz=stratz, opendota=od)
            enqueued = await eng.discover_public_seeds(patch_as_of)

    assert enqueued == 1
    assert [m[0] for m in db.pop_pending_matches()] == [9088570000]
    db.close()


async def test_pro_matches_does_not_skip_too_fresh_match_on_watermark(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "pro.sqlite3")
    now_ts = int(time.time())
    patch_as_of = now_ts - 86400

    with respx.mock:
        respx.get(f"{OPENDOTA_API_URL}/proMatches").mock(
            return_value=httpx.Response(
                200,
                json=[
                    # Too fresh (<1800s): must not advance watermark past it
                    {"match_id": 900010, "start_time": now_ts - 300, "duration": 2000, "leagueid": 123},
                    # Ready pro match
                    {"match_id": 900005, "start_time": now_ts - 3600, "duration": 2400, "leagueid": 123},
                ],
            )
        )
        async with httpx.AsyncClient() as client:
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            eng = DiscoveryEngine(db=db, stratz=stratz, opendota=od, pro_min_delay_seconds=1800)
            enqueued = await eng.discover_pro_matches(patch_as_of, max_pages=1)

    assert enqueued == 1
    assert db.get_meta("opendota_pro_newest_id") == "900005"
    assert db.pop_pending_matches()[0][:2] == (900005, "opendota_pro")
    db.close()


async def test_bfs_batched_aliases_exclude_all_pick_and_playback_data(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "bfs.sqlite3")
    db.enqueue_players([111, 222], source="match_bfs", priority=10)
    captured_queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_queries.append(body["query"])
        return httpx.Response(
            200,
            json={
                "data": {
                    "p0": {
                        "matches": [
                            {"id": 7001, "durationSeconds": 2000, "gameMode": "ALL_PICK_RANKED"},
                            {"id": 7002, "durationSeconds": 2000, "gameMode": "ALL_PICK"},
                        ]
                    },
                    "p1": {
                        "matches": [
                            {"id": 7003, "durationSeconds": 2200, "gameMode": "CAPTAINS_MODE"},
                        ]
                    },
                }
            },
        )

    with respx.mock:
        respx.post(STRATZ_GRAPHQL_URL).mock(side_effect=handler)
        async with httpx.AsyncClient() as client:
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            eng = DiscoveryEngine(db=db, stratz=stratz, opendota=od)
            enqueued = await eng.discover_bfs_from_players(190)

    assert enqueued == 2
    assert "playbackData" not in captured_queries[0]
    assert "p0: player(steamAccountId: 111)" in captured_queries[0]
    assert "p1: player(steamAccountId: 222)" in captured_queries[0]
    queued_ids = sorted(m[0] for m in db.pop_pending_matches())
    assert queued_ids == [7001, 7003]
    db.close()


async def test_leaderboard_tail_cycles_divisions_and_adapts_skip(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "lb.sqlite3")
    with respx.mock:
        route = respx.post(STRATZ_GRAPHQL_URL).mock(
            side_effect=[
                httpx.Response(200, json={"data": {"leaderboard": {"season": {"players": [{"steamAccountId": 5001}]}}}}),
                httpx.Response(200, json={"data": {"leaderboard": {"season": {"players": []}}}}),
            ]
        )
        async with httpx.AsyncClient() as client:
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            eng = DiscoveryEngine(db=db, stratz=stratz, opendota=od)
            c1 = await eng.discover_leaderboard_tail()
            c2 = await eng.discover_leaderboard_tail()

    assert route.call_count == 2
    payload = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert "($skip: Long, $take: Long)" in payload["query"]
    assert "($skip: Int" not in payload["query"]
    assert payload["variables"] == {"skip": 4500, "take": 100}
    assert c1 == 1
    assert c2 == 0
    cursors = json.loads(db.get_meta("leaderboard_cursors") or "{}")
    assert cursors["skips"]["EUROPE"] == 4400
    assert cursors["skips"]["AMERICAS"] == 3500
    db.close()


async def test_public_seeds_uses_min_rank_80_and_bfs_skips_prepatch_or_unparsed(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "disc_filters.sqlite3")
    now_ts = 1794860000
    patch_as_of = now_ts - 30 * 86400
    db.enqueue_players([999], source="match_bfs", priority=10)
    captured_params: dict[str, str] = {}
    captured_gql: dict[str, object] = {}

    def public_handler(request: httpx.Request) -> httpx.Response:
        captured_params.update(dict(request.url.params))
        return httpx.Response(200, json=[])

    def bfs_handler(request: httpx.Request) -> httpx.Response:
        captured_gql.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "data": {
                    "p0": {
                        "matches": [
                            {"id": 8001, "durationSeconds": 2000, "gameMode": "ALL_PICK_RANKED", "startDateTime": now_ts - 3600, "parsedDateTime": now_ts - 1200},
                            {"id": 8002, "durationSeconds": 2000, "gameMode": "ALL_PICK_RANKED", "startDateTime": now_ts - 3600, "parsedDateTime": None},
                            {"id": 8003, "durationSeconds": 2000, "gameMode": "ALL_PICK_RANKED", "startDateTime": patch_as_of - 500, "parsedDateTime": patch_as_of - 100},
                            # Within patch, but older than 14-day BFS window -> filtered out in Python
                            {"id": 8004, "durationSeconds": 2000, "gameMode": "ALL_PICK_RANKED", "startDateTime": now_ts - 15 * 86400, "parsedDateTime": now_ts - 15 * 86400 + 500},
                        ]
                    }
                }
            },
        )

    with respx.mock:
        respx.get(f"{OPENDOTA_API_URL}/publicMatches").mock(side_effect=public_handler)
        respx.post(STRATZ_GRAPHQL_URL).mock(side_effect=bfs_handler)
        async with httpx.AsyncClient() as client:
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            eng = DiscoveryEngine(
                db=db,
                stratz=stratz,
                opendota=od,
                wall_clock=lambda: float(now_ts),
            )
            await eng.discover_public_seeds(patch_as_of)
            enqueued = await eng.discover_bfs_from_players(190, active_gv_as_of=patch_as_of)

    assert captured_params.get("min_rank") == "80"
    assert "startDateTime: $since" in str(captured_gql.get("query") or "")
    assert captured_gql.get("variables") == {"gv": 190, "since": patch_as_of}
    assert enqueued == 2
    assert sorted(m[0] for m in db.pop_pending_matches()) == [8001, 8004]
    db.close()


async def test_bfs_queries_entire_patch_and_discovers_prior_matches(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "disc_take.sqlite3")
    now_ts = 1794860000
    patch_as_of = now_ts - 30 * 86400
    queries: list[str] = []

    def bfs_handler(request: httpx.Request) -> httpx.Response:
        q = str(json.loads(request.content.decode("utf-8"))["query"])
        queries.append(q)
        return httpx.Response(200, json={"data": {"p0": {"matches": []}}})

    db.enqueue_players([901], source="match_bfs", priority=10)
    with respx.mock:
        respx.post(STRATZ_GRAPHQL_URL).mock(side_effect=bfs_handler)
        async with httpx.AsyncClient() as client:
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            eng = DiscoveryEngine(
                db=db, stratz=stratz, opendota=od, wall_clock=lambda: float(now_ts)
            )
            await eng.discover_bfs_from_players(190, active_gv_as_of=patch_as_of)

            # Force cooldown expiry on already-fetched player 901 -> re-poll still uses take: 100 for full patch
            db.mark_players_fetched_batch([(901, 8800)])
            db._conn.execute(
                "UPDATE players_queue SET last_fetched_at = last_fetched_at - 15000 WHERE steam_account_id = 901"
            )
            db._conn.commit()
            await eng.discover_bfs_from_players(190, active_gv_as_of=patch_as_of)

    assert len(queries) == 2
    assert "take: 100" in queries[0]
    assert "take: 100" in queries[1]
    db.close()


async def test_pro_matches_backfills_historical_to_patch_start(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "pro_backfill.sqlite3")
    now_ts = 1794860000
    patch_as_of = now_ts - 10000

    def pro_handler(request: httpx.Request) -> httpx.Response:
        lt = request.url.params.get("less_than_match_id")
        if lt is None:
            return httpx.Response(
                200,
                json=[{"match_id": 900500, "start_time": now_ts - 3600, "duration": 2200, "leagueid": 10}],
            )
        if lt == "900500":
            return httpx.Response(
                200,
                json=[{"match_id": 900400, "start_time": now_ts - 5000, "duration": 2100, "leagueid": 10}],
            )
        if lt == "900400":
            return httpx.Response(
                200,
                json=[
                    {"match_id": 900300, "start_time": now_ts - 9000, "duration": 2000, "leagueid": 10},
                    {"match_id": 900100, "start_time": patch_as_of - 1000, "duration": 2300, "leagueid": 10},
                ],
            )
        return httpx.Response(200, json=[])

    with respx.mock:
        respx.get(f"{OPENDOTA_API_URL}/proMatches").mock(side_effect=pro_handler)
        async with httpx.AsyncClient() as client:
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            eng = DiscoveryEngine(db=db, stratz=stratz, opendota=od)

            # First run: forward scan discovers 900500, sets known_newest=900500
            e1 = await eng.discover_pro_matches(patch_as_of, max_pages=1)
            assert e1 == 1
            assert db.get_meta("opendota_pro_newest_id") == "900500"
            assert db.get_meta("opendota_pro_backfill_done") is None

            # Second run: backfill pages backwards from 900500 to patch_as_of
            e2 = await eng.discover_pro_matches(patch_as_of, max_pages=5)
            assert e2 == 2  # 900400 and 900300 (900100 is before patch_as_of)
            assert db.get_meta("opendota_pro_backfill_done") == "1"

    all_queued = sorted(m[0] for m in db.pop_pending_matches())
    assert all_queued == [900300, 900400, 900500]
    db.close()
