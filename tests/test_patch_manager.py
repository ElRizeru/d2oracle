"""Unit tests for automatic patch detection, buffer flushing, vocab rebuild, and queue reset."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import respx

from collector.clients.stratz import StratzClient
from collector.db import CollectorDB
from collector.patch_manager import PatchManager
from collector.rate_limit import build_stratz_rate_limiter
from collector.validator import validate_and_transform_match
from shared.constants import DOTACONSTANTS_BASE_URL, STRATZ_GRAPHQL_URL
from shared.dotaconstants import DOTACONSTANTS_FILES
from tests.conftest import (
    InMemoryHFStorage,
    make_dotaconstants_payload,
    make_raw_stratz_match,
    make_stratz_constants_payload,
)


def _mock_dotaconstants(odota_payload: dict) -> None:
    for fname in DOTACONSTANTS_FILES:
        key = fname.removesuffix(".json")
        respx.get(f"{DOTACONSTANTS_BASE_URL}/{fname}").mock(
            return_value=httpx.Response(200, json=odota_payload[key])
        )


async def test_patch_auto_switch_flushes_old_buffer_rebuilds_vocab_and_resets_queues(
    tmp_path: Path, sample_vocab
) -> None:
    db = CollectorDB(tmp_path / "patch.sqlite3")
    hf = InMemoryHFStorage()
    odota_payload = make_dotaconstants_payload()

    # Seed state on patch 190 (7.41f) with 2 buffered matches, pending queue, and player cursor
    db.set_active_patch(190, "7.41f", 1789430400)
    for mid in (1001, 1002):
        res = validate_and_transform_match(
            make_raw_stratz_match(match_id=mid, gv_id=190),
            active_gv_id=190,
            active_gv_name="7.41f",
            source="bfs",
            vocab=sample_vocab,
        )
        assert res.valid and res.record is not None
        db.enqueue_matches([(mid, "bfs")])
        db.mark_match_collected(mid, 190, res.record)
    db.enqueue_matches([(1003, "bfs")])  # pending on old patch
    db.enqueue_players([77777], source="match_bfs", priority=10)
    db.mark_player_fetched(77777, 1002)
    db.set_meta("opendota_public_cursor", "9000")

    # Seed existing latest.json on HF for patch 190
    await hf.upload_vocab(190, "7.41f", sample_vocab)
    await hf.update_manifest_for_patch(190, "7.41f", 1789430400, "data/gv_190_7.41f/vocab.json")

    # Now STRATZ announces new patch 191 (7.42)
    new_consts = make_stratz_constants_payload(gv_id=191, gv_name="7.42", as_of=1795000000)

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if "GetGameVersions" in body["query"]:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": new_consts["gameVersions"]}}})
        if "GetConstants" in body["query"]:
            return httpx.Response(200, json={"data": {"constants": new_consts}})
        raise AssertionError(f"Unexpected query: {body['query']}")

    with respx.mock:
        respx.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        _mock_dotaconstants(odota_payload)
        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=tmp_path / "constants",
                parquet_dir=tmp_path / "shards",
            )
            ctx = await pm.sync_patch()

    assert ctx.switched is True
    assert ctx.gv_id == 191
    assert ctx.gv_name == "7.42"
    assert ctx.as_of_datetime == 1795000000

    # 1. Old patch buffer was flushed to data/gv_190_7.41f/*.parquet and cleared locally
    old_shards = [k for k in hf.files if k.startswith("data/gv_190_7.41f/") and k.endswith(".parquet")]
    assert len(old_shards) == 1
    assert db.count_buffered_matches(190) == 0
    assert list((tmp_path / "shards").glob("*.parquet")) == []

    # Verify old shard readable via PyArrow with 2 rows
    tmp_parquet = tmp_path / "verify_old.parquet"
    tmp_parquet.write_bytes(hf.files[old_shards[0]])
    assert pq.read_table(tmp_parquet).num_rows == 2

    # 2. New vocab.json uploaded & latest.json points to patch 191
    assert "data/gv_191_7.42/vocab.json" in hf.files
    latest = json.loads(hf.files["latest.json"].decode("utf-8"))
    assert latest["game_version_id"] == 191
    assert latest["game_version_name"] == "7.42"
    assert latest["vocab_path"] == "data/gv_191_7.42/vocab.json"
    assert latest["shards"] == []

    # 3. match_queue cleared, players_queue cursors reset (player kept!)
    assert db.pop_pending_matches() == []
    assert db.get_meta("opendota_public_cursor") is None
    players = db.pop_players_for_bfs(limit=10, cooldown_seconds=1800)
    assert players == [(77777, None)]
    db.close()


async def test_manual_target_game_version_pinning(tmp_path: Path) -> None:
    db = CollectorDB(tmp_path / "pin.sqlite3")
    hf = InMemoryHFStorage()
    consts = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    consts["gameVersions"].append({"id": 191, "name": "7.42", "asOfDateTime": 1795000000})

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if "GetGameVersions" in body["query"]:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts["gameVersions"]}}})
        return httpx.Response(200, json={"data": {"constants": consts}})

    with respx.mock:
        respx.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        _mock_dotaconstants(make_dotaconstants_payload())
        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="190",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=tmp_path / "constants",
                parquet_dir=tmp_path / "shards",
            )
            ctx = await pm.sync_patch()

    assert ctx.gv_id == 190
    assert ctx.gv_name == "7.41f"
    db.close()


def test_hf_storage_manifest_uses_v2_schema_and_ignores_v1_shards() -> None:
    from collector.clients.hf_storage import HFDatasetStorage
    from shared.constants import PARQUET_SCHEMA_VERSION, VOCAB_SCHEMA_VERSION

    old_manifest = {
        "schema_version": 1,
        "game_version_id": 190,
        "shards": [{"path": "data/gv_190_7.41f/old.parquet", "num_rows": 100}],
    }
    assert HFDatasetStorage._compatible_existing_shards(old_manifest, 190) == []

    v2_manifest = HFDatasetStorage._build_manifest(
        gv_id=190,
        gv_name="7.41f",
        as_of_datetime=1789430400,
        vocab_path="data/gv_190_7.41f/vocab.json",
        shards=[{"path": "data/gv_190_7.41f/new.parquet", "num_rows": 50}],
    )
    assert v2_manifest["schema_version"] == PARQUET_SCHEMA_VERSION == 2
    assert v2_manifest["parquet_schema_version"] == 2
    assert v2_manifest["vocab_schema_version"] == VOCAB_SCHEMA_VERSION
    assert HFDatasetStorage._compatible_existing_shards(v2_manifest, 190) == v2_manifest["shards"]


async def test_sync_patch_upgrades_stale_v1_hf_manifest_even_when_patch_unchanged(
    tmp_path: Path, sample_vocab
) -> None:
    from shared.constants import PARQUET_SCHEMA_VERSION, VOCAB_SCHEMA_VERSION

    db = CollectorDB(tmp_path / "upgrade.sqlite3")
    hf = InMemoryHFStorage()
    consts = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    odota = make_dotaconstants_payload()

    # Local SQLite and local vocab cache already say gv 190 + v2, while HF still hosts v1 manifest/shard.
    db.set_active_patch(190, "7.41f", 1789430400)
    constants_dir = tmp_path / "constants"
    constants_dir.mkdir(parents=True, exist_ok=True)
    (constants_dir / "vocab_190.json").write_text(json.dumps(sample_vocab), encoding="utf-8")

    stale_v1_manifest = {
        "schema_version": 1,
        "game_version_id": 190,
        "game_version_name": "7.41f",
        "as_of_datetime": 1789430400,
        "data_prefix": "data/gv_190_7.41f/",
        "vocab_path": "data/gv_190_7.41f/vocab.json",
        "shards": [{"path": "data/gv_190_7.41f/v1_shard.parquet", "num_rows": 500}],
        "total_matches": 500,
    }
    hf.files["latest.json"] = json.dumps(stale_v1_manifest).encode("utf-8")
    hf.files["data/gv_190_7.41f/vocab.json"] = json.dumps({"schema_version": 1}).encode("utf-8")

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if "GetGameVersions" in body["query"]:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts["gameVersions"]}}})
        if "GetConstants" in body["query"]:
            return httpx.Response(200, json={"data": {"constants": consts}})
        raise AssertionError(f"Unexpected query: {body['query']}")

    with respx.mock:
        respx.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        _mock_dotaconstants(odota)
        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=constants_dir,
                parquet_dir=tmp_path / "shards",
            )
            ctx = await pm.sync_patch()

    assert ctx.gv_id == 190
    uploaded_vocab = json.loads(hf.files["data/gv_190_7.41f/vocab.json"].decode("utf-8"))
    assert uploaded_vocab["schema_version"] == VOCAB_SCHEMA_VERSION == 2
    manifest = json.loads(hf.files["latest.json"].decode("utf-8"))
    assert manifest["schema_version"] == PARQUET_SCHEMA_VERSION == 2
    assert manifest["parquet_schema_version"] == 2
    assert manifest["vocab_schema_version"] == VOCAB_SCHEMA_VERSION
    assert manifest["shards"] == []
    assert manifest["total_matches"] == 0
    db.close()


async def test_sync_patch_rebuilds_stale_local_v1_vocab_cache(tmp_path: Path) -> None:
    from shared.constants import VOCAB_SCHEMA_VERSION

    db = CollectorDB(tmp_path / "stale_local.sqlite3")
    hf = InMemoryHFStorage()
    consts = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)
    odota = make_dotaconstants_payload()

    db.set_active_patch(190, "7.41f", 1789430400)
    constants_dir = tmp_path / "constants"
    constants_dir.mkdir(parents=True, exist_ok=True)
    (constants_dir / "vocab_190.json").write_text(
        json.dumps({"schema_version": 1, "game_version_id": 190}),
        encoding="utf-8",
    )

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if "GetGameVersions" in body["query"]:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": consts["gameVersions"]}}})
        if "GetConstants" in body["query"]:
            return httpx.Response(200, json={"data": {"constants": consts}})
        raise AssertionError(f"Unexpected query: {body['query']}")

    with respx.mock:
        respx.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        _mock_dotaconstants(odota)
        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=constants_dir,
                parquet_dir=tmp_path / "shards",
            )
            ctx = await pm.sync_patch()

    assert ctx.vocab["schema_version"] == VOCAB_SCHEMA_VERSION == 2
    saved_local = json.loads((constants_dir / "vocab_190.json").read_text(encoding="utf-8"))
    assert saved_local["schema_version"] == 2
    db.close()


async def test_hf_read_latest_manifest_no_retry_on_not_found_and_retries_transient(
    tmp_path: Path, sample_vocab
) -> None:
    from huggingface_hub.errors import EntryNotFoundError

    from collector.clients.hf_storage import HFDatasetStorage
    from collector.patch_manager import PatchManager

    storage = HFDatasetStorage(repo_id="rizeru/d2oracle-data", token="fake_hf")
    storage._read_latest_manifest_sync.retry.sleep = lambda _seconds: None  # type: ignore[attr-defined]

    class StubApi:
        def __init__(self, outcomes: list[object]) -> None:
            self._outcomes = list(outcomes)
            self.calls = 0

        def hf_hub_download(self, **kwargs) -> str:
            self.calls += 1
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            manifest_path = tmp_path / "latest.json"
            manifest_path.write_text(json.dumps(outcome), encoding="utf-8")
            return str(manifest_path)

    # EntryNotFoundError returns None immediately without tenacity retries
    not_found_api = StubApi([EntryNotFoundError("missing latest.json")])
    storage._api = not_found_api  # type: ignore[assignment]
    assert await storage.read_latest_manifest() is None
    assert not_found_api.calls == 1

    # Transient errors retry up to 3 attempts and succeed once HF recovers
    transient_api = StubApi(
        [
            RuntimeError("502 Bad Gateway"),
            RuntimeError("503 Service Unavailable"),
            {"schema_version": 2, "total_matches": 5000},
        ]
    )
    storage._api = transient_api  # type: ignore[assignment]
    manifest = await storage.read_latest_manifest()
    assert manifest == {"schema_version": 2, "total_matches": 5000}
    assert transient_api.calls == 3

    # flush_buffer_for_patch also cleans records / Parquet file cleanly with Arrow pool release
    db = CollectorDB(tmp_path / "flush_release.sqlite3")
    hf = InMemoryHFStorage()
    res = validate_and_transform_match(
        make_raw_stratz_match(match_id=95001, gv_id=190),
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
    )
    assert res.valid and res.record is not None
    db.enqueue_matches([(95001, "bfs")])
    db.mark_match_collected(95001, 190, res.record)
    async with httpx.AsyncClient() as client:
        pm = PatchManager(
            target_game_version="auto",
            db=db,
            stratz=StratzClient("tok", build_stratz_rate_limiter(), http_client=client),
            hf_storage=hf,  # type: ignore[arg-type]
            http_client=client,
            constants_cache_dir=tmp_path / "constants",
            parquet_dir=tmp_path / "shards",
        )
        out = await pm.flush_buffer_for_patch(
            gv_id=190, gv_name="7.41f", as_of_datetime=1789430400, is_active_patch=True
        )
    assert out is not None
    assert db.count_buffered_matches(190) == 0
    assert list((tmp_path / "shards").glob("*.parquet")) == []
    db.close()


async def test_valve_ahead_patch_switch_resets_pro_backfill_and_continues_on_stratz_catchup(
    tmp_path: Path, sample_vocab
) -> None:
    from shared.constants import DOTA2_DATAFEED_BASE_URL

    db = CollectorDB(tmp_path / "valve_ahead.sqlite3")
    hf = InMemoryHFStorage()
    odota_payload = make_dotaconstants_payload()

    db.set_active_patch(190, "7.41f", 1789430400)
    db.set_meta("opendota_pro_backfill_done", "1")
    db.set_meta("opendota_pro_backfill_cursor", "9090000")
    await hf.upload_vocab(190, "7.41f", sample_vocab)
    await hf.update_manifest_for_patch(190, "7.41f", 1789430400, "data/gv_190_7.41f/vocab.json")

    stratz_versions = [
        {"id": 190, "name": "7.41f", "asOfDateTime": 1789430400},
    ]
    consts_190 = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        q = body["query"]
        if "GetGameVersions" in q:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": stratz_versions}}})
        if "GetConstants" in q:
            assert body["variables"]["gv"] == 190
            return httpx.Response(200, json={"data": {"constants": consts_190}})
        raise AssertionError(f"Unexpected query: {q}")

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
                json={"result": {"data": {"heroes": [{"id": 155, "name": "npc_dota_hero_largo", "name_loc": "Largo"}]}}},
            )
        if ep == "herodata":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "data": {
                            "heroes": [
                                {
                                    "id": 155,
                                    "name": "npc_dota_hero_largo",
                                    "name_loc": "Largo",
                                    "primary_attr": 0,
                                    "attack_capability": 1,
                                    "abilities": [],
                                    "talents": [{"id": 8001 + i, "name": f"special_bonus_unique_largo_{i+1}", "name_loc": f"Tal {i+1}"} for i in range(8)],
                                }
                            ]
                        }
                    }
                },
            )
        if ep == "itemlist":
            return httpx.Response(
                200,
                json={"result": {"data": {"itemabilities": [{"id": 1, "name": "item_blink", "neutral_item_tier": -1, "recipes": []}]}}},
            )
        if ep == "itemdata":
            return httpx.Response(
                200,
                json={"result": {"data": {"items": [{"id": 1, "name": "item_blink", "name_loc": "Blink Dagger", "item_cost": 2250, "item_quality": 1, "item_neutral_tier": 4294967295}]}}},
            )
        if ep == "abilitylist":
            return httpx.Response(200, json={"result": {"data": {"itemabilities": []}}})
        if ep == "patchnotes":
            return httpx.Response(200, json={"items": []})
        raise AssertionError(f"Unexpected Valve path: {ep}")

    with respx.mock(assert_all_called=False) as router:
        router.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        router.get(url__startswith=DOTA2_DATAFEED_BASE_URL).mock(side_effect=valve_handler)
        _mock_dotaconstants(odota_payload)
        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=tmp_path / "constants",
                parquet_dir=tmp_path / "shards",
                wall_clock=lambda: 1795043000.0,
            )
            ctx = await pm.sync_patch()

            assert ctx.switched is True
            assert ctx.gv_id == 191
            assert ctx.gv_name == "7.42"
            assert ctx.valve_ahead is True
            assert ctx.stratz_gv_ids == (190, 191)
            assert ctx.min_start_datetime == 1794988800 + 43200
            # Pro backfill flags cleared on patch switch
            assert db.get_meta("opendota_pro_backfill_done") is None
            assert db.get_meta("opendota_pro_backfill_cursor") is None

            # Enqueue a 7.42 match, then simulate STRATZ catching up and publishing gv 191
            db.enqueue_matches([(9096009999, "bfs")])
            stratz_versions.append({"id": 191, "name": "7.42", "asOfDateTime": 1795050000})
            ctx2 = await pm.sync_patch()

            assert ctx2.switched is False
            assert ctx2.gv_id == 191
            assert ctx2.valve_ahead is False
            assert ctx2.stratz_gv_ids == (191,)
            assert [m[0] for m in db.pop_pending_matches()] == [9096009999]

    db.close()


async def test_sync_patch_skip_hf_upload_and_pre_deploy_lookback(tmp_path: Path) -> None:
    from shared.constants import DOTA2_DATAFEED_BASE_URL

    db = CollectorDB(tmp_path / "no_upload.sqlite3")
    hf = InMemoryHFStorage()
    odota_payload = make_dotaconstants_payload()

    stratz_versions = [{"id": 190, "name": "7.41f", "asOfDateTime": 1789430400}]
    consts_190 = make_stratz_constants_payload(gv_id=190, gv_name="7.41f", as_of=1789430400)

    def stratz_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        q = body["query"]
        if "GetGameVersions" in q:
            return httpx.Response(200, json={"data": {"constants": {"gameVersions": stratz_versions}}})
        if "GetConstants" in q:
            return httpx.Response(200, json={"data": {"constants": consts_190}})
        raise AssertionError(f"Unexpected query: {q}")

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
                json={"result": {"data": {"heroes": [{"id": 155, "name": "npc_dota_hero_largo", "name_loc": "Largo"}]}}},
            )
        if ep == "herodata":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "data": {
                            "heroes": [
                                {
                                    "id": 155,
                                    "name": "npc_dota_hero_largo",
                                    "name_loc": "Largo",
                                    "primary_attr": 0,
                                    "attack_capability": 1,
                                    "abilities": [],
                                    "talents": [{"id": 8001 + i, "name": f"special_bonus_unique_largo_{i+1}", "name_loc": f"Tal {i+1}"} for i in range(8)],
                                }
                            ]
                        }
                    }
                },
            )
        if ep == "itemlist":
            return httpx.Response(
                200,
                json={"result": {"data": {"itemabilities": [{"id": 1, "name": "item_blink", "neutral_item_tier": -1, "recipes": []}]}}},
            )
        if ep == "itemdata":
            return httpx.Response(
                200,
                json={"result": {"data": {"items": [{"id": 1, "name": "item_blink", "name_loc": "Blink Dagger", "item_cost": 2250, "item_quality": 1, "item_neutral_tier": 4294967295}]}}},
            )
        if ep == "abilitylist":
            return httpx.Response(200, json={"result": {"data": {"itemabilities": []}}})
        if ep == "patchnotes":
            return httpx.Response(200, json={"items": []})
        raise AssertionError(f"Unexpected Valve path: {ep}")

    # Wall clock is 10000s after patch_timestamp (< 43200s deploy floor -> 7200s lookback applies)
    now_ts = 1794988800 + 10000
    with respx.mock(assert_all_called=False) as router:
        router.post(STRATZ_GRAPHQL_URL).mock(side_effect=stratz_handler)
        router.get(url__startswith=DOTA2_DATAFEED_BASE_URL).mock(side_effect=valve_handler)
        _mock_dotaconstants(odota_payload)
        async with httpx.AsyncClient() as client:
            stratz = StratzClient("tok", build_stratz_rate_limiter(), http_client=client)
            pm = PatchManager(
                target_game_version="auto",
                db=db,
                stratz=stratz,
                hf_storage=hf,  # type: ignore[arg-type]
                http_client=client,
                constants_cache_dir=tmp_path / "constants",
                parquet_dir=tmp_path / "shards",
                wall_clock=lambda: float(now_ts),
            )
            ctx = await pm.sync_patch(skip_hf_upload=True)

            assert ctx.gv_id == 191
            assert ctx.gv_name == "7.42"
            assert ctx.min_start_datetime == now_ts - 7200
            assert hf.files == {}
            assert (tmp_path / "constants" / "vocab_191.json").exists()

    db.close()
