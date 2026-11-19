"""End-to-end collector pipeline test on mocked STRATZ, OpenDota, and HF storage."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import polars as pl
import pyarrow.parquet as pq
import pytest
import respx

from collector.clients.opendota import OpenDotaClient
from collector.clients.steam import SteamClient
from collector.clients.stratz import StratzClient
from collector.config import CollectorSettings
from collector.db import CollectorDB
from collector.discovery import DiscoveryEngine
from collector.main import CollectorRunner
from collector.patch_manager import ActivePatchContext, PatchManager
from collector.rate_limit import (
    build_opendota_rate_limiter,
    build_steam_rate_limiter,
    build_stratz_rate_limiter,
)
from shared.constants import (
    DOTA2_DATAFEED_BASE_URL,
    DOTACONSTANTS_BASE_URL,
    OPENDOTA_API_URL,
    PARQUET_ARROW_SCHEMA,
    PARQUET_SCHEMA_VERSION,
    STEAM_API_BASE_URL,
    STRATZ_GRAPHQL_URL,
)
from shared.dotaconstants import DOTACONSTANTS_FILES
from tests.conftest import (
    InMemoryHFStorage,
    make_dotaconstants_payload,
    make_raw_stratz_match,
    make_stratz_constants_payload,
)


async def test_collector_dry_run_e2e_produces_valid_parquet_and_hf_manifest(
    tmp_path: Path,
) -> None:
    db = CollectorDB(tmp_path / "e2e.sqlite3")
    hf = InMemoryHFStorage()
    consts = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    odota = make_dotaconstants_payload()

    now_ts = 1794600000
    seed_match = make_raw_stratz_match(match_id=9088001, gv_id=190, start_datetime=now_ts - 7200)
    bfs_match_1 = make_raw_stratz_match(match_id=9088002, gv_id=190, start_datetime=now_ts - 6000)
    bfs_match_2 = make_raw_stratz_match(match_id=9088003, gv_id=190, start_datetime=now_ts - 5000)
    pro_match = make_raw_stratz_match(
        match_id=9088004,
        gv_id=190,
        start_datetime=now_ts - 4000,
        game_mode="CAPTAINS_MODE",
        lobby_type="PRACTICE",
        league_id=20500,
    )
    matches_db = {
        9088001: seed_match,
        9088002: bfs_match_1,
        9088003: bfs_match_2,
        9088004: pro_match,
    }

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        q = body["query"]
        if "GetGameVersions" in q:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts["gameVersions"]}}})
        if "GetConstants" in q:
            return httpx.Response(200, json={"data": {"constants": consts}})
        if "GetMatchesBatch" in q:
            aliases = re.findall(r"(m\d+):\s*match\(id:\s*(\d+)\)", q)
            return httpx.Response(
                200,
                json={"data": {alias: matches_db.get(int(mid)) for alias, mid in aliases}},
            )
        if "GetMatch" in q:
            mid = int(body["variables"]["id"])
            return httpx.Response(200, json={"data": {"match": matches_db.get(mid)}})
        if "GetPlayersMatches" in q:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "p0": {
                            "matches": [
                                {"id": 9088002, "durationSeconds": 2100, "gameMode": "ALL_PICK_RANKED"},
                                {"id": 9088003, "durationSeconds": 2200, "gameMode": "CAPTAINS_MODE"},
                            ]
                        }
                    }
                },
            )
        raise AssertionError(f"Unexpected query: {q}")

    settings = CollectorSettings(
        stratz_token="fake_stratz",
        hf_token="fake_hf",
        batch_size=5000,
        sqlite_path=tmp_path / "e2e.sqlite3",
        parquet_dir=tmp_path / "shards",
        constants_cache_dir=tmp_path / "constants",
    )

    def public_handler(request: httpx.Request) -> httpx.Response:
        lt = request.url.params.get("less_than_match_id")
        if lt is None:
            return httpx.Response(
                200,
                json=[{"match_id": 9088999, "start_time": now_ts, "duration": 2100, "lobby_type": 7, "avg_rank_tier": 80}],
            )
        return httpx.Response(
            200,
            json=[{"match_id": 9088001, "start_time": now_ts - 7200, "duration": 2100, "lobby_type": 7, "avg_rank_tier": 80}],
        )

    with respx.mock:
        respx.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        for fname in DOTACONSTANTS_FILES:
            respx.get(f"{DOTACONSTANTS_BASE_URL}/{fname}").mock(
                return_value=httpx.Response(200, json=odota[fname.removesuffix(".json")])
            )
        respx.get(f"{OPENDOTA_API_URL}/publicMatches").mock(side_effect=public_handler)
        respx.get(f"{OPENDOTA_API_URL}/proMatches").mock(
            return_value=httpx.Response(
                200,
                json=[{"match_id": 9088004, "start_time": 1790000000, "duration": 2100, "leagueid": 20500}],
            )
        )

        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=settings.constants_cache_dir,
                parquet_dir=settings.parquet_dir,
            )
            discovery = DiscoveryEngine(
                db=db,
                stratz=stratz,
                opendota=od,
                pro_min_delay_seconds=0,
                wall_clock=lambda: float(now_ts),
            )
            runner = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=discovery,
            )
            summary = await runner.run(limit=4, dry_run=True)

    assert summary.collected == 4
    assert summary.flushed_shards == 1
    assert db.count_buffered_matches() == 0
    assert list((tmp_path / "shards").glob("*.parquet")) == []

    latest = json.loads(hf.files["latest.json"].decode("utf-8"))
    assert latest["schema_version"] == PARQUET_SCHEMA_VERSION
    assert latest["parquet_schema_version"] == PARQUET_SCHEMA_VERSION
    assert latest["total_matches"] == 4
    shard_key = latest["shards"][0]["path"]
    local_verify = tmp_path / "verify.parquet"
    local_verify.write_bytes(hf.files[shard_key])

    table = pq.read_table(local_verify)
    assert table.num_rows == 4
    assert "steam_account_ids" not in table.column_names
    assert table.schema.equals(PARQUET_ARROW_SCHEMA)

    df = pl.read_parquet(local_verify)
    assert df.height == 4
    assert df.filter(pl.col("is_pro")).height == 1
    assert df["radiant_gold_lead_10m"].to_list() == [1250] * 4
    assert df["radiant_xp_lead_10m"].to_list() == [840] * 4
    db.close()


async def test_periodic_pro_discovery_runs_while_public_skips_when_bfs_queue_sufficient(
    tmp_path: Path,
) -> None:
    db = CollectorDB(tmp_path / "sched.sqlite3")
    hf = InMemoryHFStorage()
    consts = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    odota = make_dotaconstants_payload()

    # Pre-fill queue with 30 BFS matches (>= 15 threshold)
    db.enqueue_matches([(8000000 + i, "bfs") for i in range(30)])
    now_ts = 1794600000
    fake_now = [100.0]
    pro_calls = 0
    public_calls = 0

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        q = body["query"]
        if "GetGameVersions" in q:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts["gameVersions"]}}})
        if "GetConstants" in q:
            return httpx.Response(200, json={"data": {"constants": consts}})
        if "GetMatchesBatch" in q:
            aliases = re.findall(r"(m\d+):\s*match\(id:\s*(\d+)\)", q)
            out: dict[str, object] = {}
            for alias, raw_mid in aliases:
                mid = int(raw_mid)
                is_pro = mid == 9999999
                out[alias] = make_raw_stratz_match(
                    match_id=mid,
                    gv_id=190,
                    start_datetime=now_ts - 4000,
                    game_mode="CAPTAINS_MODE" if is_pro else "ALL_PICK_RANKED",
                    lobby_type="PRACTICE" if is_pro else "RANKED",
                    league_id=20500 if is_pro else None,
                )
            return httpx.Response(200, json={"data": out})
        if "GetMatch" in q:
            mid = int(body["variables"]["id"])
            is_pro = mid == 9999999
            return httpx.Response(
                200,
                json={
                    "data": {
                        "match": make_raw_stratz_match(
                            match_id=mid,
                            gv_id=190,
                            start_datetime=now_ts - 4000,
                            game_mode="CAPTAINS_MODE" if is_pro else "ALL_PICK_RANKED",
                            lobby_type="PRACTICE" if is_pro else "RANKED",
                            league_id=20500 if is_pro else None,
                        )
                    }
                },
            )
        raise AssertionError(f"Unexpected query: {q}")

    def public_handler(request: httpx.Request) -> httpx.Response:
        nonlocal public_calls
        lt = request.url.params.get("less_than_match_id")
        if lt is None:
            public_calls += 1
            return httpx.Response(200, json=[{"match_id": 9088999, "start_time": now_ts, "duration": 2100, "lobby_type": 7}])
        return httpx.Response(200, json=[{"match_id": 9088001, "start_time": now_ts - 7200, "duration": 2100, "lobby_type": 7}])

    def pro_handler(request: httpx.Request) -> httpx.Response:
        nonlocal pro_calls
        if request.url.params.get("less_than_match_id") is not None:
            return httpx.Response(200, json=[])
        pro_calls += 1
        return httpx.Response(200, json=[{"match_id": 9999999, "start_time": 1790000000, "duration": 2100, "leagueid": 20500}])

    settings = CollectorSettings(
        stratz_token="fake_stratz",
        hf_token="fake_hf",
        sqlite_path=tmp_path / "sched.sqlite3",
        parquet_dir=tmp_path / "shards",
        constants_cache_dir=tmp_path / "constants",
        pro_discovery_interval_seconds=1800,
        public_discovery_interval_seconds=1800,
    )

    with respx.mock:
        respx.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        for fname in DOTACONSTANTS_FILES:
            respx.get(f"{DOTACONSTANTS_BASE_URL}/{fname}").mock(
                return_value=httpx.Response(200, json=odota[fname.removesuffix(".json")])
            )
        respx.get(f"{OPENDOTA_API_URL}/publicMatches").mock(side_effect=public_handler)
        respx.get(f"{OPENDOTA_API_URL}/proMatches").mock(side_effect=pro_handler)

        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=settings.constants_cache_dir,
                parquet_dir=settings.parquet_dir,
            )
            discovery = DiscoveryEngine(
                db=db,
                stratz=stratz,
                opendota=od,
                pro_min_delay_seconds=0,
                wall_clock=lambda: float(now_ts),
            )
            runner = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=discovery,
                clock=lambda: fake_now[0],
            )
            # With 30 BFS matches queued, pro discovery MUST still run while periodic public skips,
            # and pro match 9999999 MUST be popped first!
            summary = await runner.run(limit=1, dry_run=False, once=True)

    assert pro_calls == 1
    assert public_calls == 0
    assert summary.collected == 1
    buffered = db.fetch_buffered_matches(190)
    assert buffered[0]["match_id"] == 9999999
    assert buffered[0]["is_pro"] is True
    db.close()


async def test_collector_recovers_from_transient_httpx_and_stratz_errors(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "recover.sqlite3")
    hf = InMemoryHFStorage()
    consts = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    odota = make_dotaconstants_payload()
    now_ts = 1794600000
    valid_match = make_raw_stratz_match(match_id=9099001, gv_id=190, start_datetime=now_ts - 3600)
    db.enqueue_matches([(9099001, "bfs")])

    gv_calls = 0

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        nonlocal gv_calls
        body = json.loads(request.content.decode("utf-8"))
        q = body["query"]
        if "GetGameVersions" in q:
            gv_calls += 1
            if gv_calls == 1:
                raise httpx.ConnectError("transient dns failure", request=request)
            if gv_calls == 2:
                return httpx.Response(200, json={"errors": [{"message": "temporary upstream error"}]})
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts["gameVersions"]}}})
        if "GetConstants" in q:
            return httpx.Response(200, json={"data": {"constants": consts}})
        if "GetMatch" in q:
            return httpx.Response(200, json={"data": {"match": valid_match}})
        if "GetPlayersMatches" in q:
            return httpx.Response(200, json={"data": {}})
        raise AssertionError(f"Unexpected query: {q}")

    settings = CollectorSettings(
        stratz_token="fake_stratz",
        hf_token="fake_hf",
        batch_size=10,
        sqlite_path=tmp_path / "recover.sqlite3",
        parquet_dir=tmp_path / "shards",
        constants_cache_dir=tmp_path / "constants",
    )

    with respx.mock(assert_all_called=False) as router:
        router.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        for fname in DOTACONSTANTS_FILES:
            router.get(f"{DOTACONSTANTS_BASE_URL}/{fname}").mock(
                return_value=httpx.Response(200, json=odota[fname.removesuffix(".json")])
            )
        router.get(f"{OPENDOTA_API_URL}/publicMatches").mock(return_value=httpx.Response(200, json=[]))
        router.get(f"{OPENDOTA_API_URL}/proMatches").mock(return_value=httpx.Response(200, json=[]))

        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=settings.constants_cache_dir,
                parquet_dir=settings.parquet_dir,
            )
            discovery = DiscoveryEngine(
                db=db,
                stratz=stratz,
                opendota=od,
                pro_min_delay_seconds=0,
                wall_clock=lambda: float(now_ts),
            )
            runner = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=discovery,
            )
            summary = await runner.run(limit=1, dry_run=True, once=False, error_backoff_seconds=0.01)

    assert gv_calls >= 3
    assert summary.collected == 1
    db.close()


async def test_collector_marks_single_match_graphql_error_for_retry_and_continues(
    tmp_path: Path,
) -> None:
    db = CollectorDB(tmp_path / "gql_match.sqlite3")
    hf = InMemoryHFStorage()
    consts = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    odota = make_dotaconstants_payload()
    now_ts = 1794600000
    good_match = make_raw_stratz_match(match_id=9099102, gv_id=190, start_datetime=now_ts - 3600)
    db.enqueue_matches([(9099101, "bfs"), (9099102, "bfs")])

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        q = body["query"]
        if "GetGameVersions" in q:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts["gameVersions"]}}})
        if "GetConstants" in q:
            return httpx.Response(200, json={"data": {"constants": consts}})
        if "GetMatchesBatch" in q:
            aliases = re.findall(r"(m\d+):\s*match\(id:\s*(\d+)\)", q)
            errors = []
            data_out: dict[str, object] = {}
            for alias, raw_mid in aliases:
                mid = int(raw_mid)
                if mid == 9099101:
                    data_out[alias] = None
                    errors.append({"message": "match resolver error", "path": [alias]})
                else:
                    data_out[alias] = good_match
            return httpx.Response(200, json={"errors": errors, "data": data_out})
        if "GetMatch" in q:
            mid = int(body["variables"]["id"])
            if mid == 9099101:
                return httpx.Response(200, json={"errors": [{"message": "match resolver error"}]})
            return httpx.Response(200, json={"data": {"match": good_match}})
        if "GetPlayersMatches" in q:
            return httpx.Response(200, json={"data": {}})
        raise AssertionError(f"Unexpected query: {q}")

    settings = CollectorSettings(
        stratz_token="fake_stratz",
        hf_token="fake_hf",
        batch_size=10,
        sqlite_path=tmp_path / "gql_match.sqlite3",
        parquet_dir=tmp_path / "shards",
        constants_cache_dir=tmp_path / "constants",
    )

    with respx.mock(assert_all_called=False) as router:
        router.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        for fname in DOTACONSTANTS_FILES:
            router.get(f"{DOTACONSTANTS_BASE_URL}/{fname}").mock(
                return_value=httpx.Response(200, json=odota[fname.removesuffix(".json")])
            )
        router.get(f"{OPENDOTA_API_URL}/publicMatches").mock(return_value=httpx.Response(200, json=[]))
        router.get(f"{OPENDOTA_API_URL}/proMatches").mock(return_value=httpx.Response(200, json=[]))

        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=settings.constants_cache_dir,
                parquet_dir=settings.parquet_dir,
            )
            discovery = DiscoveryEngine(
                db=db,
                stratz=stratz,
                opendota=od,
                pro_min_delay_seconds=0,
                wall_clock=lambda: float(now_ts),
            )
            runner = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=discovery,
            )
            summary = await runner.run(limit=2, dry_run=True, once=True, error_backoff_seconds=0.01)

    assert summary.collected == 1
    assert summary.retried == 1
    row = db._conn.execute(
        "SELECT status, attempts, reject_reason FROM match_queue WHERE match_id = ?",
        (9099101,),
    ).fetchone()
    assert row is not None
    assert row["status"] == "pending"
    assert row["attempts"] == 1
    assert row["reject_reason"] == "stratz_graphql_error"
    db.close()


async def test_opendota_public_unparsed_rejects_after_one_attempt_while_pro_retries(
    tmp_path: Path,
) -> None:
    db = CollectorDB(tmp_path / "unparsed_retry.sqlite3")
    hf = InMemoryHFStorage()
    consts = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    odota = make_dotaconstants_payload()
    now_ts = 1794860000

    db.enqueue_matches([(9099201, "opendota_public"), (9099202, "opendota_pro")])

    unparsed_pub = make_raw_stratz_match(match_id=9099201, gv_id=190, start_datetime=now_ts - 3600)
    unparsed_pub["parsedDateTime"] = None
    unparsed_pro = make_raw_stratz_match(
        match_id=9099202,
        gv_id=190,
        start_datetime=now_ts - 3600,
        game_mode="CAPTAINS_MODE",
        lobby_type="PRACTICE",
        league_id=20500,
    )
    unparsed_pro["parsedDateTime"] = None
    by_id = {9099201: unparsed_pub, 9099202: unparsed_pro}

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        q = body["query"]
        if "GetGameVersions" in q:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts["gameVersions"]}}})
        if "GetConstants" in q:
            return httpx.Response(200, json={"data": {"constants": consts}})
        if "GetMatchesBatch" in q:
            aliases = re.findall(r"(m\d+):\s*match\(id:\s*(\d+)\)", q)
            return httpx.Response(200, json={"data": {a: by_id[int(mid)] for a, mid in aliases}})
        if "GetMatch" in q:
            mid = int(body["variables"]["id"])
            return httpx.Response(200, json={"data": {"match": by_id.get(mid)}})
        if "GetPlayersMatches" in q:
            return httpx.Response(200, json={"data": {}})
        raise AssertionError(f"Unexpected query: {q}")

    settings = CollectorSettings(
        stratz_token="fake_stratz",
        hf_token="fake_hf",
        sqlite_path=tmp_path / "unparsed_retry.sqlite3",
        parquet_dir=tmp_path / "shards",
        constants_cache_dir=tmp_path / "constants",
    )

    with respx.mock(assert_all_called=False) as router:
        router.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        for fname in DOTACONSTANTS_FILES:
            router.get(f"{DOTACONSTANTS_BASE_URL}/{fname}").mock(
                return_value=httpx.Response(200, json=odota[fname.removesuffix(".json")])
            )
        router.get(f"{OPENDOTA_API_URL}/publicMatches").mock(return_value=httpx.Response(200, json=[]))
        router.get(f"{OPENDOTA_API_URL}/proMatches").mock(return_value=httpx.Response(200, json=[]))

        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=settings.constants_cache_dir,
                parquet_dir=settings.parquet_dir,
            )
            discovery = DiscoveryEngine(
                db=db, stratz=stratz, opendota=od, pro_min_delay_seconds=0, wall_clock=lambda: float(now_ts)
            )
            runner = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=discovery,
            )
            summary = await runner.run(limit=5, dry_run=False, once=True)

    assert summary.collected == 0
    assert summary.rejected == 1
    assert summary.retried == 1
    pub_row = db._conn.execute("SELECT status, attempts, reject_reason FROM match_queue WHERE match_id = 9099201").fetchone()
    pro_row = db._conn.execute("SELECT status, attempts, reject_reason FROM match_queue WHERE match_id = 9099202").fetchone()
    assert tuple(pub_row) == ("rejected", 1, "unparsed")
    assert tuple(pro_row) == ("pending", 1, "unparsed")
    db.close()


async def test_get_matches_batch_handles_partial_outcomes_and_rebuilt_fallback(
    tmp_path: Path,
) -> None:
    db = CollectorDB(tmp_path / "batch_fallback.sqlite3")
    hf = InMemoryHFStorage()
    consts = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    odota = make_dotaconstants_payload()
    now_ts = 1794860000

    # First chunk of 20 (STRATZ_MATCH_BATCH_SIZE=20): 18 valid, 1 None, 1 unparsed;
    # second chunk of 3 (9099401..9099403) triggers batch fallback -> 3 single GetMatch calls.
    # Higher IDs are popped first by match_id DESC: 9099520..9099501 (first 20), then 9099403..9099401 (next 3).
    first_chunk_ids = list(range(9099501, 9099521))
    second_chunk_ids = [9099401, 9099402, 9099403]
    db.enqueue_matches([(mid, "bfs") for mid in (*first_chunk_ids, *second_chunk_ids)])
    unparsed_503 = make_raw_stratz_match(match_id=9099503, gv_id=190, start_datetime=now_ts - 3600)
    unparsed_503["parsedDateTime"] = None
    payloads: dict[int, dict[str, object] | None] = {
        mid: make_raw_stratz_match(match_id=mid, gv_id=190, start_datetime=now_ts - 3600)
        for mid in (*first_chunk_ids, *second_chunk_ids)
    }
    payloads[9099503] = unparsed_503
    payloads[9099502] = None
    single_fallback_ids: list[int] = []
    batch_calls = 0

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        nonlocal batch_calls
        body = json.loads(request.content.decode("utf-8"))
        q = body["query"]
        if "GetGameVersions" in q:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts["gameVersions"]}}})
        if "GetConstants" in q:
            return httpx.Response(200, json={"data": {"constants": consts}})
        if "GetMatchesBatch" in q:
            batch_calls += 1
            aliases = re.findall(r"(m\d+):\s*match\(id:\s*(\d+)\)", q)
            mids = [int(mid) for _, mid in aliases]
            if any(mid in (9099401, 9099402, 9099403) for mid in mids):
                return httpx.Response(
                    200,
                    json={
                        "errors": [
                            {
                                "message": "Playback data was requested for 3 matches that are not cached but at most 1 may be rebuilt at a time.",
                                "path": ["m0", "players", 0, "playbackData"],
                            }
                        ],
                        "data": {a: payloads[int(mid)] for a, mid in aliases},
                    },
                )
            return httpx.Response(200, json={"data": {a: payloads[int(mid)] for a, mid in aliases}})
        if "GetMatch" in q:
            mid = int(body["variables"]["id"])
            single_fallback_ids.append(mid)
            return httpx.Response(200, json={"data": {"match": payloads[mid]}})
        if "GetPlayersMatches" in q:
            return httpx.Response(200, json={"data": {}})
        raise AssertionError(f"Unexpected query: {q}")

    settings = CollectorSettings(
        stratz_token="fake_stratz",
        hf_token="fake_hf",
        sqlite_path=tmp_path / "batch_fallback.sqlite3",
        parquet_dir=tmp_path / "shards",
        constants_cache_dir=tmp_path / "constants",
    )

    with respx.mock(assert_all_called=False) as router:
        router.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        for fname in DOTACONSTANTS_FILES:
            router.get(f"{DOTACONSTANTS_BASE_URL}/{fname}").mock(
                return_value=httpx.Response(200, json=odota[fname.removesuffix(".json")])
            )
        router.get(f"{OPENDOTA_API_URL}/publicMatches").mock(return_value=httpx.Response(200, json=[]))
        router.get(f"{OPENDOTA_API_URL}/proMatches").mock(return_value=httpx.Response(200, json=[]))

        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=settings.constants_cache_dir,
                parquet_dir=settings.parquet_dir,
            )
            discovery = DiscoveryEngine(
                db=db, stratz=stratz, opendota=od, pro_min_delay_seconds=0, wall_clock=lambda: float(now_ts)
            )
            runner = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=discovery,
            )
            summary = await runner.run(limit=25, dry_run=False, once=True)

    assert batch_calls == 2
    assert sorted(single_fallback_ids) == [9099401, 9099402, 9099403]
    assert summary.collected == 21
    assert summary.retried == 2
    buffered_ids = sorted(r["match_id"] for r in db.fetch_buffered_matches(190))
    assert buffered_ids == [
        9099401, 9099402, 9099403,
        9099501, *range(9099504, 9099521),
    ]
    # Rebuilt fallback must NOT consume attempts (all 3 collected on attempt 1)
    att_rows = db._conn.execute(
        "SELECT match_id, attempts FROM match_queue WHERE match_id IN (9099401, 9099402, 9099403)"
    ).fetchall()
    assert all(r["attempts"] == 1 for r in att_rows)
    db.close()


async def test_hf_flush_error_isolated_with_backoff_preserves_buffer(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "hf_backoff.sqlite3")
    consts = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    odota = make_dotaconstants_payload()
    now_ts = 1794860000

    class FlakyHFStorage(InMemoryHFStorage):
        def __init__(self, fail_attempts: int = 1) -> None:
            super().__init__()
            self.fail_attempts = fail_attempts
            self.shard_upload_attempts = 0

        async def upload_shard_and_update_manifest(self, *args, **kwargs):
            self.shard_upload_attempts += 1
            if self.shard_upload_attempts <= self.fail_attempts:
                raise RuntimeError("HF 503 Service Unavailable")
            return await super().upload_shard_and_update_manifest(*args, **kwargs)

    hf = FlakyHFStorage(fail_attempts=1)
    # Seed 2 matches initially; after pass 1 flushes/fails, pass 2 discovers 9100003 & 9100004 via BFS
    db.enqueue_matches([(9100001, "bfs"), (9100002, "bfs")])
    payloads = {
        mid: make_raw_stratz_match(match_id=mid, gv_id=190, start_datetime=now_ts - 3600)
        for mid in (9100001, 9100002, 9100003, 9100004)
    }
    bfs_calls = 0

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        nonlocal bfs_calls
        body = json.loads(request.content.decode("utf-8"))
        q = body["query"]
        if "GetGameVersions" in q:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts["gameVersions"]}}})
        if "GetConstants" in q:
            return httpx.Response(200, json={"data": {"constants": consts}})
        if "GetMatchesBatch" in q:
            aliases = re.findall(r"(m\d+):\s*match\(id:\s*(\d+)\)", q)
            return httpx.Response(200, json={"data": {a: payloads[int(mid)] for a, mid in aliases}})
        if "GetPlayersMatches" in q:
            bfs_calls += 1
            aliases = re.findall(r"(p\d+):\s*player\(steamAccountId:", q)
            return httpx.Response(
                200,
                json={
                    "data": {
                        aliases[0]: {
                            "matches": [
                                {
                                    "id": 9100003,
                                    "durationSeconds": 2000,
                                    "gameMode": "ALL_PICK_RANKED",
                                    "startDateTime": now_ts - 1800,
                                    "parsedDateTime": now_ts - 600,
                                },
                                {
                                    "id": 9100004,
                                    "durationSeconds": 2000,
                                    "gameMode": "ALL_PICK_RANKED",
                                    "startDateTime": now_ts - 1800,
                                    "parsedDateTime": now_ts - 600,
                                },
                            ]
                        },
                        **{a: {"matches": []} for a in aliases[1:]},
                    }
                },
            )
        raise AssertionError(f"Unexpected query: {q}")

    settings = CollectorSettings(
        stratz_token="fake_stratz",
        hf_token="fake_hf",
        batch_size=2,
        pro_discovery_interval_seconds=99999,
        public_discovery_interval_seconds=99999,
        sqlite_path=tmp_path / "hf_backoff.sqlite3",
        parquet_dir=tmp_path / "shards",
        constants_cache_dir=tmp_path / "constants",
    )
    sleeps: list[float] = []

    with respx.mock(assert_all_called=False) as router:
        router.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        for fname in DOTACONSTANTS_FILES:
            router.get(f"{DOTACONSTANTS_BASE_URL}/{fname}").mock(
                return_value=httpx.Response(200, json=odota[fname.removesuffix(".json")])
            )
        router.get(f"{OPENDOTA_API_URL}/publicMatches").mock(return_value=httpx.Response(200, json=[]))
        router.get(f"{OPENDOTA_API_URL}/proMatches").mock(return_value=httpx.Response(200, json=[]))

        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=settings.constants_cache_dir,
                parquet_dir=settings.parquet_dir,
            )
            discovery = DiscoveryEngine(
                db=db, stratz=stratz, opendota=od, pro_min_delay_seconds=0, wall_clock=lambda: float(now_ts)
            )
            runner = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=discovery,
            )

            async def fake_sleep(seconds: float) -> None:
                sleeps.append(seconds)

            runner._sleep_interruptible = fake_sleep  # type: ignore[method-assign]
            summary = await runner.run(
                limit=4,
                dry_run=False,
                once=False,
                error_backoff_seconds=2.0,
            )

            assert summary.collected == 4
            assert summary.flushed_shards == 1
            assert hf.shard_upload_attempts == 2
            assert sleeps == [2.0]
            assert db.count_buffered_matches(190) == 2

            # Direct check: second consecutive failure doubles backoff (2.0 * 2**1 = 4.0) and once=True raises
            hf.fail_attempts = hf.shard_upload_attempts + 10
            ctx = await pm.sync_patch()
            flushed, streak = await runner._flush_buffer_if_ready(
                ctx,
                dry_run=False,
                once=False,
                no_upload=False,
                hf_error_streak=1,
                error_backoff_seconds=2.0,
            )
            assert (flushed, streak) == (0, 2)
            assert sleeps == [2.0, 4.0]
            with pytest.raises(RuntimeError, match="HF 503"):
                await runner._flush_buffer_if_ready(
                    ctx,
                    dry_run=False,
                    once=True,
                    no_upload=False,
                    hf_error_streak=0,
                    error_backoff_seconds=2.0,
                )

    db.close()


async def test_collector_e2e_with_steam_valve_datafeed_and_opendota_new_item_recovery(
    tmp_path: Path,
) -> None:
    db = CollectorDB(tmp_path / "steam_e2e.sqlite3")
    hf = InMemoryHFStorage()
    consts_190 = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    odota = make_dotaconstants_payload()

    now_ts = 1795043000
    pro_mid = 9096100
    pub_mid = 9096101

    def _make_stratz_no_pb(mid: int, seq: int, *, is_pro: bool) -> dict[str, object]:
        m = make_raw_stratz_match(
            match_id=mid,
            gv_id=190,  # STRATZ lags behind Valve 7.42
            start_datetime=now_ts - 1200,
            game_mode="CAPTAINS_MODE" if is_pro else "ALL_PICK_RANKED",
            lobby_type="PRACTICE" if is_pro else "RANKED",
            league_id=20511 if is_pro else None,
        )
        m["sequenceNum"] = seq
        for p in m["players"]:
            p.pop("playbackData")
            p["stats"]["itemPurchases"] = [{"time": 350, "itemId": 63}]
            p["abilities"] = [
                {"abilityId": 5003, "time": -80, "level": 0, "isTalent": False},
                {"abilityId": 0, "time": 900, "level": 0, "isTalent": False},  # STRATZ zero bug
            ]
        return m

    stratz_matches = {
        pro_mid: _make_stratz_no_pb(pro_mid, 7636820010, is_pro=True),
        pub_mid: _make_stratz_no_pb(pub_mid, 7636820020, is_pro=False),
    }

    def _make_steam_payload(mid: int, seq: int, *, is_pro: bool) -> dict[str, object]:
        sm = stratz_matches[mid]
        steam_players = []
        for idx, p in enumerate(sm["players"]):  # type: ignore[arg-type]
            is_carry = p["isRadiant"] and p["position"] == "POSITION_1"
            steam_players.append(
                {
                    "account_id": 500000 + idx,
                    "player_slot": p["playerSlot"],
                    "hero_id": p["heroId"],
                    "hero_variant": 0,
                    "kills": 7,
                    "deaths": 2,
                    "assists": 9,
                    "last_hits": 220,
                    "denies": 10,
                    "gold_per_min": 610,
                    "xp_per_min": 740,
                    "net_worth": 21000,
                    "level": 22,
                    "hero_damage": 25000,
                    "tower_damage": 4000,
                    "hero_healing": 0,
                    "leaver_status": 0,
                    "item_0": 63,
                    "item_1": 1953 if (is_pro and is_carry) else 0,
                    "item_2": 0,
                    "item_3": 0,
                    "item_4": 0,
                    "item_5": 0,
                    "item_neutral": 1720,
                    "ability_upgrades": [
                        {"ability": 5003, "time": 200, "level": 1},
                        {"ability": 1495, "time": 1180, "level": 2},
                    ],
                }
            )
        return {
            "match_id": mid,
            "match_seq_num": seq,
            "start_time": now_ts - 1200,
            "duration": 2100,
            "pre_game_duration": 90,
            "radiant_win": True,
            "lobby_type": 1 if is_pro else 7,
            "game_mode": 2 if is_pro else 22,
            "leagueid": 20511 if is_pro else 0,
            "players": steam_players,
        }

    steam_payloads = [
        _make_steam_payload(pro_mid, 7636820010, is_pro=True),
        _make_steam_payload(pub_mid, 7636820020, is_pro=False),
    ]

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        q = body["query"]
        if "GetGameVersions" in q:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts_190["gameVersions"]}}})
        if "GetConstants" in q:
            return httpx.Response(200, json={"data": {"constants": consts_190}})
        if "GetPlayersMatches" in q:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "p0": {
                            "matches": [
                                {"id": pub_mid, "durationSeconds": 2100, "gameMode": "ALL_PICK_RANKED", "startDateTime": now_ts - 1200, "parsedDateTime": now_ts - 100}
                            ]
                        }
                    }
                },
            )
        if "GetMatchesBatch" in q:
            aliases = re.findall(r"(m\d+):\s*match\(id:\s*(\d+)\)", q)
            return httpx.Response(200, json={"data": {a: stratz_matches.get(int(mid)) for a, mid in aliases}})
        if "GetMatch" in q:
            mid = int(body["variables"]["id"])
            return httpx.Response(200, json={"data": {"match": stratz_matches.get(mid)}})
        raise AssertionError(f"Unexpected STRATZ query: {q}")

    def valve_handler(request: httpx.Request) -> httpx.Response:
        ep = request.url.path.rsplit("/", 1)[-1]
        if ep == "patchnoteslist":
            return httpx.Response(
                200,
                json={
                    "patches": [
                        {"patch_number": "7.41f", "patch_timestamp": 1789455600},
                        {"patch_number": "7.42", "patch_timestamp": 1794988800},
                    ]
                },
            )
        if ep == "herolist":
            return httpx.Response(
                200,
                json={"result": {"data": {"heroes": [{"id": 1, "name": "npc_dota_hero_antimage", "name_loc": "Anti-Mage"}]}}},
            )
        if ep == "herodata":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "data": {
                            "heroes": [
                                {
                                    "id": 1,
                                    "name": "npc_dota_hero_antimage",
                                    "name_loc": "Anti-Mage",
                                    "primary_attr": 1,
                                    "attack_capability": 1,
                                    "abilities": [
                                        {"id": 5003, "name": "antimage_mana_break", "name_loc": "Mana Break", "max_level": 4, "ability_is_innate": False, "ability_is_granted_by_scepter": False, "ability_is_granted_by_shard": False, "behavior": "2"},
                                        {"id": 5004, "name": "antimage_blink", "name_loc": "Blink", "max_level": 4, "ability_is_innate": False, "ability_is_granted_by_scepter": False, "ability_is_granted_by_shard": False, "behavior": "16"},
                                        {"id": 7314, "name": "antimage_counterspell", "name_loc": "Counterspell", "max_level": 4, "ability_is_innate": False, "ability_is_granted_by_scepter": False, "ability_is_granted_by_shard": False, "behavior": "4"},
                                        {"id": 5006, "name": "antimage_mana_void", "name_loc": "Mana Void", "max_level": 3, "ability_is_innate": False, "ability_is_granted_by_scepter": False, "ability_is_granted_by_shard": False, "behavior": "8"},
                                    ],
                                    "talents": [{"id": 1495, "name": "special_bonus_hp_regen_3", "name_loc": "+3 Health Regen"}],
                                }
                            ]
                        }
                    }
                },
            )
        if ep == "itemlist":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "data": {
                            "itemabilities": [
                                {"id": 63, "name": "item_power_treads", "neutral_item_tier": -1, "recipes": []},
                                {"id": 563, "name": "item_recipe_power_treads", "neutral_item_tier": -1, "recipes": [{"items": [29, 25]}]},
                                {"id": 1953, "name": "item_berserkers_sabatons", "neutral_item_tier": -1, "recipes": []},
                                {"id": 1952, "name": "item_recipe_berserkers_sabatons", "neutral_item_tier": -1, "recipes": [{"items": [50, 151]}]},
                            ]
                        }
                    }
                },
            )
        if ep == "itemdata":
            iid = int(request.url.params["item_id"])
            if iid == 63:
                return httpx.Response(200, json={"result": {"data": {"items": [{"id": 63, "name": "item_power_treads", "name_loc": "Power Treads", "item_cost": 1400, "item_quality": 2, "item_neutral_tier": 4294967295}]}}})
            return httpx.Response(200, json={"result": {"data": {"items": [{"id": 1953, "name": "item_berserkers_sabatons", "name_loc": "Sabatons of Mordiggian", "item_cost": 5950, "item_quality": 1, "item_neutral_tier": 4294967295}]}}})
        if ep == "abilitylist":
            return httpx.Response(200, json={"result": {"data": {"itemabilities": []}}})
        if ep == "patchnotes":
            return httpx.Response(200, json={"items": [{"ability_id": 1953}]})
        raise AssertionError(f"Unexpected Valve path: {ep}")

    settings = CollectorSettings(
        stratz_token="fake_stratz",
        steam_api_key="fake_steam",
        hf_token="fake_hf",
        batch_size=5000,
        sqlite_path=tmp_path / "steam_e2e.sqlite3",
        parquet_dir=tmp_path / "shards",
        constants_cache_dir=tmp_path / "constants",
    )

    with respx.mock(assert_all_called=False) as router:
        router.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        router.get(url__startswith=DOTA2_DATAFEED_BASE_URL).mock(side_effect=valve_handler)
        for fname in DOTACONSTANTS_FILES:
            router.get(f"{DOTACONSTANTS_BASE_URL}/{fname}").mock(
                return_value=httpx.Response(200, json=odota[fname.removesuffix(".json")])
            )
        router.get(f"{OPENDOTA_API_URL}/publicMatches").mock(return_value=httpx.Response(200, json=[]))
        router.get(f"{OPENDOTA_API_URL}/proMatches").mock(
            return_value=httpx.Response(
                200,
                json=[{"match_id": pro_mid, "start_time": now_ts - 1200, "duration": 2100, "leagueid": 20511}],
            )
        )
        router.get(f"{OPENDOTA_API_URL}/matches/{pro_mid}").mock(
            return_value=httpx.Response(
                200,
                json={
                    "match_id": pro_mid,
                    "players": [
                        {
                            "player_slot": 2,
                            "purchase_log": [{"time": 1620, "key": "berserkers_sabatons"}],
                        }
                    ],
                },
            )
        )
        router.get(f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetTopLiveGame/v1/").mock(
            return_value=httpx.Response(
                200,
                json={"game_list": [{"match_id": "9096999", "lobby_type": 7, "players": [{"account_id": 500001}]}]},
            )
        )
        router.get(f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetLiveLeagueGames/v1/").mock(
            return_value=httpx.Response(200, json={"result": {"games": []}})
        )
        router.get(
            f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetMatchHistoryBySequenceNum/v1/"
        ).mock(
            return_value=httpx.Response(200, json={"result": {"status": 1, "matches": steam_payloads}})
        )

        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            steam = SteamClient(
                "fake_steam",
                build_steam_rate_limiter(rps=20, rpm=600, rpd=90000),
                http_client=client,
            )
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=settings.constants_cache_dir,
                parquet_dir=settings.parquet_dir,
                wall_clock=lambda: float(now_ts),
            )
            discovery = DiscoveryEngine(
                db=db,
                stratz=stratz,
                opendota=od,
                steam=steam,
                pro_min_delay_seconds=0,
                wall_clock=lambda: float(now_ts),
            )
            runner = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=discovery,
                steam=steam,
            )
            summary = await runner.run(limit=2, dry_run=True)

    assert summary.gv_id == 191
    assert summary.gv_name == "7.42"
    assert summary.collected == 2
    assert summary.flushed_shards == 1

    latest = json.loads(hf.files["latest.json"].decode("utf-8"))
    assert latest["game_version_name"] == "7.42"
    shard_bytes = hf.files[latest["shards"][0]["path"]]
    pq_path = tmp_path / "verify_742.parquet"
    pq_path.write_bytes(shard_bytes)
    tbl = pq.read_table(pq_path)
    assert tbl.schema.equals(PARQUET_ARROW_SCHEMA)
    rows = tbl.to_pylist()
    assert len(rows) == 2
    for row in rows:
        for slot_aids, slot_ranks in zip(row["ability_learn_ids"], row["ability_learn_ranks"], strict=True):
            assert 0 not in slot_aids
            assert all(r >= 1 for r in slot_ranks)
    pro_row = next(r for r in rows if r["match_id"] == pro_mid)
    assert pro_row["key_item_ids"][0] == [63, 1953]
    assert pro_row["key_item_times"][0] == [350, 1620]
    db.close()


async def test_opendota_missing_item_recovery_quota_tiers_for_pub_and_pro(
    tmp_path: Path,
) -> None:
    db = CollectorDB(tmp_path / "quota_tiers.sqlite3")
    hf = InMemoryHFStorage()
    now_ts = 1795043000
    pub_mid = 9096501
    pro_mid = 9096502

    vocab = {
        "items": {
            "by_id": {
                "63": {"short_name": "power_treads"},
                "1953": {"short_name": "berserkers_sabatons"},
            },
            "aliases": {},
        }
    }

    def _raw_stratz(mid: int, *, is_pro: bool) -> dict[str, object]:
        m = make_raw_stratz_match(
            match_id=mid,
            gv_id=191,
            start_datetime=now_ts - 1200,
            game_mode="CAPTAINS_MODE" if is_pro else "ALL_PICK_RANKED",
            lobby_type="PRACTICE" if is_pro else "RANKED",
            league_id=20511 if is_pro else None,
        )
        for p in m["players"]:  # type: ignore[attr-defined]
            p["stats"]["itemPurchases"] = [{"time": 350, "itemId": 63}]
        return m

    def _steam_with_new_item() -> dict[str, object]:
        return {
            "players": [
                {"player_slot": 0, "item_0": 63, "item_1": 1953},
            ]
        }

    settings = CollectorSettings(
        stratz_token="fake_stratz",
        hf_token="fake_hf",
        sqlite_path=tmp_path / "quota_tiers.sqlite3",
        parquet_dir=tmp_path / "shards",
        constants_cache_dir=tmp_path / "constants",
    )

    od_calls: list[int] = []
    with respx.mock(assert_all_called=False) as router:
        for mid in (pub_mid, pro_mid):
            def _make_handler(m_id: int):
                def _handler(request: httpx.Request) -> httpx.Response:
                    od_calls.append(m_id)
                    return httpx.Response(
                        200,
                        json={
                            "match_id": m_id,
                            "players": [{"player_slot": 0, "purchase_log": [{"time": 1500, "key": "berserkers_sabatons"}]}],
                        },
                    )
                return _handler

            router.get(f"{OPENDOTA_API_URL}/matches/{mid}").mock(side_effect=_make_handler(mid))

        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=settings.constants_cache_dir,
                parquet_dir=settings.parquet_dir,
            )

            # 1. Plenty of quota (>= 400/day, >= 15/min) -> pub match recovers missing new item
            od_full = OpenDotaClient(build_opendota_rate_limiter(rpm=50, rpd=1000), http_client=client)
            runner_full = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od_full,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=DiscoveryEngine(db=db, stratz=stratz, opendota=od_full),
            )
            pub_res = await runner_full._maybe_fetch_opendota_for_missing_items(
                pub_mid, _raw_stratz(pub_mid, is_pro=False), _steam_with_new_item(), vocab
            )
            assert pub_res is not None
            assert od_calls == [pub_mid]

            # 2. Tighter daily budget (< 400/day, >= 100/day) -> pub skips, pro still fetches
            od_mid_rl = OpenDotaClient(build_opendota_rate_limiter(rpm=50, rpd=200), http_client=client)
            runner_mid = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od_mid_rl,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=DiscoveryEngine(db=db, stratz=stratz, opendota=od_mid_rl),
            )
            pub_skipped = await runner_mid._maybe_fetch_opendota_for_missing_items(
                pub_mid, _raw_stratz(pub_mid, is_pro=False), _steam_with_new_item(), vocab
            )
            pro_fetched = await runner_mid._maybe_fetch_opendota_for_missing_items(
                pro_mid, _raw_stratz(pro_mid, is_pro=True), _steam_with_new_item(), vocab
            )
            assert pub_skipped is None
            assert pro_fetched is not None
            assert od_calls == [pub_mid, pro_mid]

            # 3. Exhausted budget (< 100/day) -> both pub and pro skip OpenDota cleanly
            od_low = OpenDotaClient(build_opendota_rate_limiter(rpm=50, rpd=50), http_client=client)
            runner_low = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od_low,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=DiscoveryEngine(db=db, stratz=stratz, opendota=od_low),
            )
            assert (
                await runner_low._maybe_fetch_opendota_for_missing_items(
                    pro_mid, _raw_stratz(pro_mid, is_pro=True), _steam_with_new_item(), vocab
                )
                is None
            )
            assert od_calls == [pub_mid, pro_mid]

    db.close()


async def test_ensure_queue_has_matches_respects_discovery_intervals_when_queue_low(
    tmp_path: Path,
) -> None:
    db = CollectorDB(tmp_path / "intervals.sqlite3")
    hf = InMemoryHFStorage()
    now_mono = [10000.0]
    now_wall = 1795043000

    # Keep 1 pending match (< PUBLIC_DISCOVERY_PENDING_THRESHOLD) + 1 fetched player so BFS sees empty
    db.enqueue_matches([(9096701, "seed")])
    db.enqueue_players([500001], source="seed", priority=1)
    db.mark_player_fetched(500001, 9096701)

    settings = CollectorSettings(
        stratz_token="fake_stratz",
        hf_token="fake_hf",
        sqlite_path=tmp_path / "intervals.sqlite3",
        parquet_dir=tmp_path / "shards",
        constants_cache_dir=tmp_path / "constants",
        pro_discovery_interval_seconds=1800,
        public_discovery_interval_seconds=1800,
        steam_live_discovery_interval_seconds=600,
    )

    counts = {"public": 0, "pro": 0, "steam_live": 0}
    with respx.mock(assert_all_called=False) as router:
        def _count_public(_req: httpx.Request) -> httpx.Response:
            counts["public"] += 1
            return httpx.Response(200, json=[])

        def _count_pro(_req: httpx.Request) -> httpx.Response:
            counts["pro"] += 1
            return httpx.Response(200, json=[])

        def _count_steam_live(_req: httpx.Request) -> httpx.Response:
            counts["steam_live"] += 1
            return httpx.Response(200, json={"game_list": []})

        router.get(f"{OPENDOTA_API_URL}/publicMatches").mock(side_effect=_count_public)
        router.get(f"{OPENDOTA_API_URL}/proMatches").mock(side_effect=_count_pro)
        router.get(f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetTopLiveGame/v1/").mock(
            side_effect=_count_steam_live
        )
        router.get(f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetLiveLeagueGames/v1/").mock(
            return_value=httpx.Response(200, json={"result": {"games": []}})
        )
        router.post(STRATZ_GRAPHQL_URL).mock(return_value=httpx.Response(200, json={"data": {"p0": {"matches": []}}}))
        router.get(f"{STEAM_API_BASE_URL}/IDOTA2Match_570/GetMatchHistory/v1/").mock(
            return_value=httpx.Response(200, json={"result": {"status": 1, "matches": []}})
        )

        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            od = OpenDotaClient(build_opendota_rate_limiter(), http_client=client)
            steam = SteamClient("fake_steam", build_steam_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=settings.constants_cache_dir,
                parquet_dir=settings.parquet_dir,
            )
            discovery = DiscoveryEngine(
                db=db,
                stratz=stratz,
                opendota=od,
                steam=steam,
                wall_clock=lambda: float(now_wall),
            )
            runner = CollectorRunner(
                settings=settings,
                db=db,
                stratz=stratz,
                opendota=od,
                hf_storage=hf,  # type: ignore[arg-type]
                patch_manager=pm,
                discovery=discovery,
                steam=steam,
                clock=lambda: now_mono[0],
            )
            ctx = ActivePatchContext(
                gv_id=191,
                gv_name="7.42",
                as_of_datetime=1794988800,
                vocab={},
                min_start_datetime=1795032000,
            )

            # 1st call runs initial periodic discovery once
            await runner._ensure_queue_has_matches(ctx)
            assert counts == {"public": 1, "pro": 1, "steam_live": 1}

            # Subsequent call 10s later with queue still below threshold must NOT spam endpoints
            now_mono[0] = 10010.0
            await runner._ensure_queue_has_matches(ctx)
            assert counts == {"public": 1, "pro": 1, "steam_live": 1}

            # After 605s (>= steam_live 600s, < public/pro 1800s), only Steam live refreshes
            now_mono[0] = 10605.0
            await runner._ensure_queue_has_matches(ctx)
            assert counts == {"public": 1, "pro": 1, "steam_live": 2}

            # After 1805s (>= 1800s), OpenDota public & pro refresh once again
            now_mono[0] = 11805.0
            await runner._ensure_queue_has_matches(ctx)
            assert counts == {"public": 2, "pro": 2, "steam_live": 3}

    db.close()
