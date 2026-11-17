"""Unit tests for patch-aware vocabulary building and key item filtering."""

from __future__ import annotations

import copy

from shared.constants import VOCAB_SCHEMA_VERSION
from shared.vocab import build_vocab, extract_key_item_purchases
from tests.conftest import make_dotaconstants_payload, make_stratz_constants_payload


def test_build_vocab_filters_key_items_and_maps_aliases() -> None:
    vocab = build_vocab(
        {"id": 190, "name": "7.41f", "asOfDateTime": 1789430400},
        make_stratz_constants_payload(),
        make_dotaconstants_payload(),
    )

    items_by_short = {v["short_name"]: v for v in vocab["items"]["by_id"].values()}
    assert set(items_by_short.keys()) == {
        "blink",
        "power_treads",
        "hand_of_midas",
        "dagon",
        "black_king_bar",
        "tranquil_boots",
        "aghanims_shard",
    }
    # Consumables, recipes, secret shop raw orbs, intermediate raw stat parts, neutral items excluded
    for excluded in ("tango", "recipe_black_king_bar", "hyperstone", "pers", "tier1_neutral", "dagon_5"):
        assert excluded not in items_by_short

    # Special tokens reserved at indices 0, 1, 2
    assert vocab["items"]["special_tokens"] == {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}
    assert vocab["items"]["aliases"]["204"] == 104  # dagon_5 -> dagon


def test_build_vocab_excludes_generic_hidden_and_backfills_talents() -> None:
    vocab = build_vocab(
        {"id": 190, "name": "7.41f", "asOfDateTime": 1789430400},
        make_stratz_constants_payload(),
        make_dotaconstants_payload(),
    )

    am = vocab["heroes"]["by_id"]["1"]
    assert am["skill_ability_ids"] == [5003, 5004, 7314, 5006]  # 6251 generic_hidden excluded
    assert am["talent_tiers"]["10"] == [1495, 966]
    assert am["img"].startswith("https://")

    # Hero 155 (Largo) had empty STRATZ talents -> backfilled from dotaconstants
    largo = vocab["heroes"]["by_id"]["155"]
    assert largo["talent_tiers"]["10"] == [8001, 8002]
    assert "8001" in vocab["abilities"]["by_id"]


def test_extract_key_item_purchases_chronological_and_deduplicated(sample_vocab) -> None:
    events = [
        {"time": 1500, "itemId": 204},  # dagon_5 -> dagon (104)
        {"time": -60, "itemId": 44},  # tango -> ignored
        {"time": 900, "itemId": 104},  # dagon bought earlier -> keeps 900s, drops 1500s dup
        {"time": 420, "itemId": 63},  # power_treads
    ]
    ids, times = extract_key_item_purchases(events, sample_vocab, deduplicate=True)
    assert ids == [63, 104]
    assert times == [420, 900]


def test_v2_vocab_moon_shard_built_from_abilities_and_instant_upgrade_collapse() -> None:
    stratz = copy.deepcopy(make_stratz_constants_payload())
    dota = copy.deepcopy(make_dotaconstants_payload())

    stratz["items"].extend(
        [
            {
                "id": 247,
                "shortName": "moon_shard",
                "displayName": "Moon Shard",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 4000,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 98,
                "shortName": "orchid",
                "displayName": "Orchid Malevolence",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 3275,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 250,
                "shortName": "bloodthorn",
                "displayName": "Bloodthorn",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 6625,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
        ]
    )
    dota["items"]["moon_shard"] = {
        "id": 247,
        "dname": "Moon Shard",
        "qual": "consumable",
        "created": False,
        "charges": False,
        "img": "/apps/dota2/images/dota_react/items/moon_shard.png",
    }
    dota["items"]["orchid"] = {
        "id": 98,
        "dname": "Orchid Malevolence",
        "qual": "epic",
        "created": True,
        "charges": False,
        "img": "/apps/dota2/images/dota_react/items/orchid.png",
    }
    dota["items"]["bloodthorn"] = {
        "id": 250,
        "dname": "Bloodthorn",
        "qual": "epic",
        "created": True,
        "charges": False,
        "components": ["orchid"],
        "img": "/apps/dota2/images/dota_react/items/bloodthorn.png",
    }

    # Add linked paired ability + innate flag coverage
    stratz["heroes"][0]["abilities"].extend(
        [
            {"slot": 7, "abilityId": 5059},
            {"slot": 8, "abilityId": 5060},
        ]
    )
    dota["ability_ids"]["5059"] = "nevermore_shadowraze1"
    dota["ability_ids"]["5060"] = "nevermore_shadowraze2"
    dota["ability_ids"]["5006"] = "antimage_mana_void"
    dota["abilities"]["nevermore_shadowraze1"] = {"dname": "Shadowraze", "behavior": "No Target", "img": "/a.png"}
    dota["abilities"]["nevermore_shadowraze2"] = {"dname": "Shadowraze", "behavior": "No Target", "img": "/b.png"}
    dota["abilities"]["antimage_mana_void"] = {"dname": "Mana Void", "cd": [70, 70, 70], "img": "/c.png"}

    vocab = build_vocab({"id": 190, "name": "7.41f", "asOfDateTime": 1789430400}, stratz, dota)
    assert vocab["schema_version"] == VOCAB_SCHEMA_VERSION == 2
    assert "247" in vocab["items"]["by_id"]
    assert vocab["items"]["by_id"]["250"]["built_from_ids"] == [98]
    assert vocab["abilities"]["aliases"]["5060"] == 5059
    assert vocab["abilities"]["by_id"]["5060"]["is_levelable"] is False
    assert vocab["abilities"]["by_id"]["5006"]["is_ultimate"] is True

    # Auto-assembled component <= 5s before finished upgrade is collapsed
    instant_ids, instant_times = extract_key_item_purchases(
        [{"time": 1200, "itemId": 98}, {"time": 1202, "itemId": 250}],
        vocab,
        deduplicate=True,
    )
    assert instant_ids == [250]
    assert instant_times == [1202]

    # Separate Orchid kept when > 5s before Bloodthorn
    manual_ids, manual_times = extract_key_item_purchases(
        [{"time": 900, "itemId": 98}, {"time": 1500, "itemId": 250}],
        vocab,
        deduplicate=True,
    )
    assert manual_ids == [98, 250]
    assert manual_times == [900, 1500]


def test_v2_vocab_in_slot_subability_aliases() -> None:
    stratz = copy.deepcopy(make_stratz_constants_payload())
    dota = copy.deepcopy(make_dotaconstants_payload())

    stratz["heroes"].append(
        {
            "id": 91,
            "shortName": "wisp",
            "displayName": "Io",
            "talents": [],
            "abilities": [
                {"slot": 0, "abilityId": 5485},
                {"slot": 1, "abilityId": 5489},
                {"slot": 2, "abilityId": 5252},
                {"slot": 3, "abilityId": 5253},
            ],
        }
    )
    for aid, name, is_ult in (
        (5485, "wisp_tether", False),
        (5486, "wisp_spirits", False),
        (5489, "wisp_tether_break", False),
        (5490, "wisp_spirits_in", False),
        (5493, "wisp_spirits_out", False),
        (5252, "life_stealer_infest", True),
        (5253, "life_stealer_consume", False),
    ):
        stratz["abilities"].append(
            {
                "id": aid,
                "name": name,
                "isTalent": False,
                "language": {"displayName": name},
            }
        )
        dota["ability_ids"][str(aid)] = name
        dota["abilities"][name] = {
            "dname": name,
            "behavior": "No Target",
            "cd": [30, 30, 30] if is_ult else [12, 12, 12, 12],
            "img": f"/apps/dota2/images/dota_react/abilities/{name}.png",
        }

    stratz["heroes"][-1]["abilities"].extend(
        [
            {"slot": 4, "abilityId": 5486},
            {"slot": 5, "abilityId": 5490},
            {"slot": 6, "abilityId": 5493},
        ]
    )
    vocab = build_vocab({"id": 190, "name": "7.41f", "asOfDateTime": 1789430400}, stratz, dota)
    assert vocab["abilities"]["aliases"]["5489"] == 5485
    assert vocab["abilities"]["aliases"]["5490"] == 5486
    assert vocab["abilities"]["aliases"]["5493"] == 5486
    assert vocab["abilities"]["aliases"]["5253"] == 5252
    assert vocab["abilities"]["by_id"]["5489"]["is_levelable"] is False
    assert vocab["abilities"]["by_id"]["5490"]["is_levelable"] is False
    assert vocab["abilities"]["by_id"]["5493"]["is_levelable"] is False
    assert vocab["abilities"]["by_id"]["5253"]["is_levelable"] is False
    assert vocab["abilities"]["by_id"]["5485"]["is_levelable"] is True
    assert vocab["abilities"]["by_id"]["5486"]["is_levelable"] is True
    assert vocab["abilities"]["by_id"]["5252"]["is_levelable"] is True
