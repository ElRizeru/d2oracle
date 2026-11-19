"""Unit tests for Valve Dota 2 Datafeed client and Valve-first vocab building."""

from __future__ import annotations

import copy
from pathlib import Path

import httpx
import respx

from shared.constants import ATTRIBUTE_BONUS_ABILITY_ID, DOTA2_DATAFEED_BASE_URL
from shared.valve_datafeed import fetch_valve_constants, fetch_valve_patchnotes_list
from shared.vocab import _format_valve_talent_display_name, build_vocab
from tests.conftest import make_dotaconstants_payload, make_stratz_constants_payload


async def test_valve_datafeed_fetch_and_cache_fallback(tmp_path: Path) -> None:
    def router_side_effect(request: httpx.Request) -> httpx.Response:
        path = request.url.path.rsplit("/", 1)[-1]
        if path == "patchnoteslist":
            return httpx.Response(
                200,
                json={"patches": [{"patch_number": "7.42", "patch_timestamp": 1794988800}], "success": True},
            )
        if path == "herolist":
            return httpx.Response(
                200,
                json={"result": {"data": {"heroes": [{"id": 145, "name": "npc_dota_hero_kez", "name_loc": "Kez"}]}}},
            )
        if path == "herodata":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "data": {
                            "heroes": [
                                {
                                    "id": 145,
                                    "name": "npc_dota_hero_kez",
                                    "name_loc": "Kez",
                                    "primary_attr": 1,
                                    "attack_capability": 1,
                                    "abilities": [],
                                    "talents": [],
                                }
                            ]
                        }
                    }
                },
            )
        if path == "itemlist":
            return httpx.Response(
                200,
                json={
                    "result": {
                        "data": {
                            "itemabilities": [
                                {"id": 1953, "name": "item_berserkers_sabatons", "neutral_item_tier": -1, "recipes": []},
                                {"id": 1952, "name": "item_recipe_berserkers_sabatons", "neutral_item_tier": -1, "recipes": [{"items": [50, 151]}]},
                                {"id": 1859, "name": "item_spellslinger", "neutral_item_tier": 1, "recipes": []},
                            ]
                        }
                    }
                },
            )
        if path == "itemdata":
            assert request.url.params["item_id"] == "1953"
            return httpx.Response(
                200,
                json={
                    "result": {
                        "data": {
                            "items": [
                                {
                                    "id": 1953,
                                    "name": "item_berserkers_sabatons",
                                    "name_loc": "Sabatons of Mordiggian",
                                    "item_cost": 5950,
                                    "item_quality": 1,
                                    "item_neutral_tier": 4294967295,
                                }
                            ]
                        }
                    }
                },
            )
        if path == "abilitylist":
            return httpx.Response(
                200,
                json={"result": {"data": {"itemabilities": [{"id": 1498, "name": "kez_echo_slash", "name_loc": "Echo Slash"}]}}},
            )
        if path == "patchnotes":
            return httpx.Response(200, json={"items": [{"ability_id": 1953}]})
        raise AssertionError(f"Unexpected path: {path}")

    with respx.mock:
        respx.get(url__startswith=DOTA2_DATAFEED_BASE_URL).mock(side_effect=router_side_effect)
        async with httpx.AsyncClient() as client:
            patches = await fetch_valve_patchnotes_list(client, cache_dir=tmp_path)
            consts = await fetch_valve_constants(client, patch_name="7.42", cache_dir=tmp_path)

    assert patches == [{"patch_number": "7.42", "patch_timestamp": 1794988800}]
    assert consts["heroes"][0]["id"] == 145
    assert 1953 in consts["itemdata"]
    assert consts["patch_item_ids"] == [1953]

    # Verify cache fallback works on HTTP 404/error
    with respx.mock:
        respx.get(url__startswith=DOTA2_DATAFEED_BASE_URL).mock(
            return_value=httpx.Response(404, text="not found")
        )
        async with httpx.AsyncClient() as client:
            cached_patches = await fetch_valve_patchnotes_list(client, cache_dir=tmp_path)
            cached_consts = await fetch_valve_constants(client, patch_name="7.42", cache_dir=tmp_path)

    assert cached_patches == patches
    assert cached_consts["heroes"][0]["id"] == 145
    assert 1953 in cached_consts["itemdata"]


def test_valve_talent_placeholder_formatting_and_valve_first_vocab() -> None:
    # Single-bonus fallback (Invoker 1747 style) + standard bonus_sv_name + own special_values
    hero_abs = [
        {
            "name": "invoker_forge_spirit",
            "special_values": [
                {
                    "name": "armor_per_attack",
                    "bonuses": [
                        {"name": "special_bonus_unique_invoker_forged_spirit_armor_reduction", "value": 50}
                    ],
                },
                {
                    "name": "duration",
                    "bonuses": [
                        {"name": "special_bonus_unique_kez_falcon_rush_duration", "value": 2.5}
                    ],
                },
            ],
        }
    ]
    assert (
        _format_valve_talent_display_name(
            {
                "id": 1747,
                "name": "special_bonus_unique_invoker_forged_spirit_armor_reduction",
                "name_loc": "+{s:bonus_armor_removed}% Forged Spirit Armor Reduction",
            },
            hero_abs,
        )
        == "+50% Forged Spirit Armor Reduction"
    )
    assert (
        _format_valve_talent_display_name(
            {
                "id": 1511,
                "name": "special_bonus_unique_kez_falcon_rush_duration",
                "name_loc": "+{s:bonus_duration}s Falcon Rush Duration",
            },
            hero_abs,
        )
        == "+2.5s Falcon Rush Duration"
    )
    assert (
        _format_valve_talent_display_name(
            {
                "id": 7386,
                "name": "special_bonus_agility_12",
                "name_loc": "+{s:value} Agility",
                "special_values": [{"name": "value", "values_int": [12]}],
            },
            [],
        )
        == "+12 Agility"
    )

    # Full Valve-first vocab build:
    # - Attribute Bonus 730 must NOT have is_talent=True even if STRATZ claimed so
    # - Hero 155 (Largo) uses Valve's 8 talents exclusively (no stale STRATZ/dotaconstants pollution)
    # - New 7.42 item (1953) + recipe -> built_from_ids=[50]
    # - Ghost Scepter (37) included via ICONIC_NON_CREATED_ITEMS
    # - Removed item (911 Revenant's Brooch cost=0) and legacy unpurchasable item (106 Necronomicon) excluded
    stratz = copy.deepcopy(make_stratz_constants_payload())
    stratz["abilities"].append(
        {"id": ATTRIBUTE_BONUS_ABILITY_ID, "name": "special_bonus_attributes", "isTalent": True, "language": {"displayName": "+2 All Attributes"}}
    )
    stratz["items"].append(
        {
            "id": 106,
            "shortName": "necronomicon",
            "displayName": "Necronomicon",
            "isSupportFullItem": None,
            "stat": {"cost": 2050, "isRecipe": False, "isPurchasable": False, "isStackable": False, "neutralItemTier": None},
        }
    )
    dota = copy.deepcopy(make_dotaconstants_payload())
    # Inject a stale extra tier-10 talent in dotaconstants for Largo -> Valve must override cleanly
    dota["hero_abilities"]["npc_dota_hero_largo"]["talents"].append(
        {"name": "special_bonus_unique_largo_stale", "level": 1}
    )
    dota["ability_ids"]["9999"] = "special_bonus_unique_largo_stale"

    valve = {
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
                "talents": [
                    {"id": 1495, "name": "special_bonus_hp_regen_3", "name_loc": "+{s:value} Health Regen", "special_values": [{"name": "value", "values_float": [3]}]},
                    {"id": 966, "name": "special_bonus_unique_antimage_1", "name_loc": "+200 Mana Void Radius"},
                ],
            },
            {
                "id": 155,
                "name": "npc_dota_hero_largo",
                "name_loc": "Largo",
                "primary_attr": 0,
                "attack_capability": 1,
                "abilities": [
                    {"id": 1659, "name": "largo_catchy_lick", "name_loc": "Catchy Lick", "max_level": 4, "ability_is_innate": False, "ability_is_granted_by_scepter": False, "ability_is_granted_by_shard": False, "behavior": "8"},
                    {"id": 1660, "name": "largo_frogstomp", "name_loc": "Frogstomp", "max_level": 4, "ability_is_innate": False, "ability_is_granted_by_scepter": False, "ability_is_granted_by_shard": False, "behavior": "48", "special_values": [{"name": "damage_per_stomp", "bonuses": [{"name": "special_bonus_unique_largo_1", "value": 15}]}]},
                    {"id": 1661, "name": "largo_croak_of_genius", "name_loc": "Croak of Genius", "max_level": 4, "ability_is_innate": False, "ability_is_granted_by_scepter": False, "ability_is_granted_by_shard": False, "behavior": "8"},
                    {"id": 1666, "name": "largo_encore", "name_loc": "Encore", "max_level": 1, "ability_is_innate": True, "ability_is_granted_by_scepter": False, "ability_is_granted_by_shard": False, "behavior": "3"},
                    {"id": 1662, "name": "largo_amphibian_rhapsody", "name_loc": "Amphibian Rhapsody", "max_level": 3, "ability_is_innate": False, "ability_is_granted_by_scepter": False, "ability_is_granted_by_shard": False, "behavior": "4"},
                ],
                "talents": [
                    {"id": 1667 + i, "name": f"special_bonus_unique_largo_{i + 1}", "name_loc": "+{s:bonus_damage_per_stomp} Frogstomp Damage" if i == 0 else f"Largo Talent {i + 1}"}
                    for i in range(8)
                ],
            },
        ],
        "itemlist": [
            {"id": 1, "name": "item_blink", "neutral_item_tier": -1, "recipes": []},
            {"id": 37, "name": "item_ghost", "neutral_item_tier": -1, "recipes": []},
            {"id": 50, "name": "item_phase_boots", "neutral_item_tier": -1, "recipes": []},
            {"id": 550, "name": "item_recipe_phase_boots", "neutral_item_tier": -1, "recipes": [{"items": [29, 4]}]},
            {"id": 609, "name": "item_aghanims_shard", "neutral_item_tier": -1, "recipes": []},
            {"id": 1953, "name": "item_berserkers_sabatons", "neutral_item_tier": -1, "recipes": []},
            {"id": 1952, "name": "item_recipe_berserkers_sabatons", "neutral_item_tier": -1, "recipes": [{"items": [50, 151]}]},
            {"id": 106, "name": "item_necronomicon", "neutral_item_tier": -1, "recipes": []},
            {"id": 1060, "name": "item_recipe_necronomicon", "neutral_item_tier": -1, "recipes": [{"items": [23, 27]}]},
            {"id": 911, "name": "item_revenants_brooch", "neutral_item_tier": -1, "recipes": []},
            {"id": 1720, "name": "item_tier1_neutral", "neutral_item_tier": 1, "recipes": []},
        ],
        "itemdata": {
            1: {"id": 1, "name": "item_blink", "name_loc": "Blink Dagger", "item_cost": 2250, "item_quality": 1, "item_neutral_tier": 4294967295},
            37: {"id": 37, "name": "item_ghost", "name_loc": "Ghost Scepter", "item_cost": 1500, "item_quality": 1, "item_neutral_tier": 4294967295},
            50: {"id": 50, "name": "item_phase_boots", "name_loc": "Phase Boots", "item_cost": 1450, "item_quality": 2, "item_neutral_tier": 4294967295},
            609: {"id": 609, "name": "item_aghanims_shard", "name_loc": "Aghanim's Shard", "item_cost": 1400, "item_quality": 3, "item_neutral_tier": 4294967295},
            1953: {"id": 1953, "name": "item_berserkers_sabatons", "name_loc": "Sabatons of Mordiggian", "item_cost": 5950, "item_quality": 1, "item_neutral_tier": 4294967295},
            106: {"id": 106, "name": "item_necronomicon", "name_loc": "Necronomicon", "item_cost": 2700, "item_quality": 3, "item_neutral_tier": 4294967295},
            911: {"id": 911, "name": "item_revenants_brooch", "name_loc": "Revenant's Brooch", "item_cost": 0, "item_quality": 1, "item_neutral_tier": 4294967295},
        },
        "abilitylist": [],
        "patch_item_ids": [1953],
    }

    vocab = build_vocab(
        {"id": 191, "name": "7.42", "asOfDateTime": 1794988800},
        stratz,
        dota,
        valve_datafeed=valve,
    )

    assert vocab["game_version_id"] == 191
    assert vocab["game_version_name"] == "7.42"
    # Attribute Bonus 730 must NOT be a talent
    assert vocab["abilities"]["by_id"][str(ATTRIBUTE_BONUS_ABILITY_ID)]["is_talent"] is False
    assert ATTRIBUTE_BONUS_ABILITY_ID not in vocab["abilities"]["talent_ids"]

    # Largo has exact [2, 2, 2, 2] talents from Valve and Encore marked innate/non-levelable
    largo = vocab["heroes"]["by_id"]["155"]
    assert largo["talent_tiers"] == {
        "10": [1667, 1668],
        "15": [1669, 1670],
        "20": [1671, 1672],
        "25": [1673, 1674],
    }
    assert vocab["abilities"]["by_id"]["1667"]["display_name"] == "+15 Frogstomp Damage"
    assert vocab["abilities"]["by_id"]["1666"]["is_innate"] is True
    assert vocab["abilities"]["by_id"]["1666"]["is_levelable"] is False
    assert vocab["abilities"]["by_id"]["1662"]["is_ultimate"] is True

    # Items: 1953 + built_from_ids=[50], ghost=37 kept, 106 & 911 excluded
    items_by_id = vocab["items"]["by_id"]
    assert "1953" in items_by_id
    assert items_by_id["1953"]["built_from_ids"] == [50]
    assert "37" in items_by_id
    assert "106" not in items_by_id
    assert "911" not in items_by_id
    assert items_by_id["1953"]["img"].endswith("/items/berserkers_sabatons.png")
