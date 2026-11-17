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
from collector.clients.stratz import StratzClient
from collector.config import CollectorSettings
from collector.db import CollectorDB
from collector.discovery import DiscoveryEngine
from collector.main import CollectorRunner
from collector.patch_manager import PatchManager
from collector.rate_limit import build_opendota_rate_limiter, build_stratz_rate_limiter
from shared.constants import (
    DOTACONSTANTS_BASE_URL,
    OPENDOTA_API_URL,
    PARQUET_ARROW_SCHEMA,
    PARQUET_SCHEMA_VERSION,
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
