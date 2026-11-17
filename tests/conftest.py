"""Shared test fixtures and mock builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def make_stratz_constants_payload(
    *,
    gv_id: int = 190,
    gv_name: str = "7.41f",
    as_of: int = 1789430400,
) -> dict[str, Any]:
    return {
        "gameVersions": [
            {"id": gv_id - 1, "name": "7.41e", "asOfDateTime": as_of - 4000000},
            {"id": gv_id, "name": gv_name, "asOfDateTime": as_of},
        ],
        "heroes": [
            {
                "id": 1,
                "shortName": "antimage",
                "displayName": "Anti-Mage",
                "abilities": [
                    {"slot": 1, "abilityId": 5003},
                    {"slot": 2, "abilityId": 5004},
                    {"slot": 3, "abilityId": 7314},
                    {"slot": 4, "abilityId": 6251},  # generic_hidden
                    {"slot": 6, "abilityId": 5006},
                ],
                "talents": [
                    {"abilityId": 1495, "slot": 0},
                    {"abilityId": 966, "slot": 1},
                    {"abilityId": 6800, "slot": 2},
                    {"abilityId": 665, "slot": 3},
                    {"abilityId": 6606, "slot": 4},
                    {"abilityId": 735, "slot": 5},
                    {"abilityId": 6012, "slot": 6},
                    {"abilityId": 6353, "slot": 7},
                ],
            },
            {
                "id": 155,
                "shortName": "largo",
                "displayName": "Largo",
                "abilities": [
                    {"slot": 1, "abilityId": 1659},
                    {"slot": 2, "abilityId": 1660},
                    {"slot": 3, "abilityId": 1661},
                    {"slot": 6, "abilityId": 1662},
                ],
                "talents": [],  # Intentionally empty in STRATZ to verify dotaconstants backfill
            },
        ],
        "items": [
            {
                "id": 1,
                "shortName": "blink",
                "displayName": "Blink Dagger",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 2250,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 44,
                "shortName": "tango",
                "displayName": "Tango",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 90,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": True,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 63,
                "shortName": "power_treads",
                "displayName": "Power Treads",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 1400,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 65,
                "shortName": "hand_of_midas",
                "displayName": "Hand of Midas",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 2200,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 116,
                "shortName": "black_king_bar",
                "displayName": "Black King Bar",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 4050,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 115,
                "shortName": "recipe_black_king_bar",
                "displayName": "Black King Bar Recipe",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 1450,
                    "isRecipe": True,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 55,
                "shortName": "hyperstone",
                "displayName": "Hyperstone",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 2000,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 69,
                "shortName": "pers",
                "displayName": "Perseverance",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 1400,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 104,
                "shortName": "dagon",
                "displayName": "Dagon",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 3000,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 204,
                "shortName": "dagon_5",
                "displayName": "Dagon (Level 5)",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 7400,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 214,
                "shortName": "tranquil_boots",
                "displayName": "Tranquil Boots",
                "isSupportFullItem": True,
                "stat": {
                    "cost": 900,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 609,
                "shortName": "aghanims_shard",
                "displayName": "Aghanim's Shard",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 1400,
                    "isRecipe": False,
                    "isPurchasable": True,
                    "isStackable": False,
                    "neutralItemTier": None,
                },
            },
            {
                "id": 1720,
                "shortName": "tier1_neutral",
                "displayName": "Neutral Tier 1",
                "isSupportFullItem": None,
                "stat": {
                    "cost": 0,
                    "isRecipe": False,
                    "isPurchasable": False,
                    "isStackable": False,
                    "neutralItemTier": 1,
                },
            },
        ],
        "abilities": [
            {"id": 5003, "name": "antimage_mana_break", "isTalent": False, "language": {"displayName": "Mana Break"}},
            {"id": 5004, "name": "antimage_blink", "isTalent": False, "language": {"displayName": "Blink"}},
            {"id": 7314, "name": "antimage_counterspell", "isTalent": False, "language": {"displayName": "Counterspell"}},
            {"id": 6251, "name": "generic_hidden", "isTalent": False, "language": {"displayName": None}},
            {"id": 5006, "name": "antimage_mana_void", "isTalent": False, "language": {"displayName": "Mana Void"}},
            {"id": 1495, "name": "special_bonus_hp_regen_3", "isTalent": True, "language": {"displayName": "+3 Health Regen"}},
            {"id": 966, "name": "special_bonus_unique_antimage_1", "isTalent": True, "language": {"displayName": "+200 Mana Void Radius"}},
            {"id": 1659, "name": "largo_catchy_lick", "isTalent": False, "language": {"displayName": "Catchy Lick"}},
            {"id": 1660, "name": "largo_frogstomp", "isTalent": False, "language": {"displayName": "Frogstomp"}},
            {"id": 1661, "name": "largo_croak_of_genius", "isTalent": False, "language": {"displayName": "Croak of Genius"}},
            {"id": 1662, "name": "largo_amphibian_rhapsody", "isTalent": False, "language": {"displayName": "Amphibian Rhapsody"}},
            {"id": 8001, "name": "special_bonus_unique_largo_1", "isTalent": True, "language": {"displayName": "+50 Frogstomp Damage"}},
            {"id": 8002, "name": "special_bonus_unique_largo_2", "isTalent": True, "language": {"displayName": "-2s Lick Cooldown"}},
        ],
    }


def make_dotaconstants_payload() -> dict[str, dict[str, Any]]:
    return {
        "heroes": {
            "1": {
                "id": 1,
                "name": "npc_dota_hero_antimage",
                "primary_attr": "agi",
                "attack_type": "Melee",
                "roles": ["Carry", "Escape", "Nuker"],
                "img": "/apps/dota2/images/dota_react/heroes/antimage.png",
                "icon": "/apps/dota2/images/dota_react/heroes/icons/antimage.png",
            },
            "155": {
                "id": 155,
                "name": "npc_dota_hero_largo",
                "primary_attr": "str",
                "attack_type": "Melee",
                "roles": ["Support", "Disabler"],
                "img": "/apps/dota2/images/dota_react/heroes/largo.png",
                "icon": "/apps/dota2/images/dota_react/heroes/icons/largo.png",
            },
        },
        "items": {
            "blink": {"id": 1, "dname": "Blink Dagger", "qual": "component", "created": False, "charges": False, "img": "/apps/dota2/images/dota_react/items/blink.png"},
            "tango": {"id": 44, "dname": "Tango", "qual": "consumable", "created": False, "charges": 3, "img": "/apps/dota2/images/dota_react/items/tango.png"},
            "power_treads": {"id": 63, "dname": "Power Treads", "qual": "common", "created": True, "charges": False, "img": "/apps/dota2/images/dota_react/items/power_treads.png"},
            "hand_of_midas": {"id": 65, "dname": "Hand of Midas", "qual": "common", "created": True, "charges": 1, "img": "/apps/dota2/images/dota_react/items/hand_of_midas.png"},
            "black_king_bar": {"id": 116, "dname": "Black King Bar", "qual": "epic", "created": True, "charges": False, "img": "/apps/dota2/images/dota_react/items/black_king_bar.png"},
            "hyperstone": {"id": 55, "dname": "Hyperstone", "qual": "secret_shop", "created": False, "charges": False, "img": "/apps/dota2/images/dota_react/items/hyperstone.png"},
            "pers": {"id": 69, "dname": "Perseverance", "qual": "common", "created": True, "charges": False, "img": "/apps/dota2/images/dota_react/items/pers.png"},
            "dagon": {"id": 104, "dname": "Dagon", "qual": "rare", "created": True, "charges": False, "img": "/apps/dota2/images/dota_react/items/dagon.png"},
            "dagon_5": {"id": 204, "dname": "Dagon (Level 5)", "qual": "rare", "created": True, "charges": False, "img": "/apps/dota2/images/dota_react/items/dagon_5.png"},
            "tranquil_boots": {"id": 214, "dname": "Tranquil Boots", "qual": "rare", "created": True, "charges": False, "img": "/apps/dota2/images/dota_react/items/tranquil_boots.png"},
            "aghanims_shard": {"id": 609, "dname": "Aghanim's Shard", "qual": "rare", "created": False, "charges": False, "img": "/apps/dota2/images/dota_react/items/aghanims_shard.png"},
        },
        "abilities": {
            "antimage_mana_break": {"dname": "Mana Break", "img": "/apps/dota2/images/dota_react/abilities/antimage_mana_break.png"},
            "largo_catchy_lick": {"dname": "Catchy Lick", "img": "/apps/dota2/images/dota_react/abilities/largo_catchy_lick.png"},
        },
        "hero_abilities": {
            "npc_dota_hero_antimage": {
                "abilities": ["antimage_mana_break", "antimage_blink", "antimage_counterspell", "antimage_mana_void"],
                "talents": [],
            },
            "npc_dota_hero_largo": {
                "abilities": ["largo_catchy_lick", "largo_frogstomp", "largo_croak_of_genius", "largo_amphibian_rhapsody"],
                "talents": [
                    {"name": "special_bonus_unique_largo_1", "level": 1},
                    {"name": "special_bonus_unique_largo_2", "level": 1},
                ],
            },
        },
        "ability_ids": {
            "5003": "antimage_mana_break",
            "5004": "antimage_blink",
            "7314": "antimage_counterspell",
            "5006": "antimage_mana_void",
            "8001": "special_bonus_unique_largo_1",
            "8002": "special_bonus_unique_largo_2",
        },
    }


def make_raw_stratz_match(
    *,
    match_id: int = 9088306979,
    gv_id: int = 190,
    start_datetime: int = 1794580000,
    duration_seconds: int = 2100,
    did_radiant_win: bool = True,
    game_mode: str = "ALL_PICK_RANKED",
    lobby_type: str = "RANKED",
    bracket: int = 8,
    league_id: int | None = None,
    parsed_datetime: int | None = 1794584000,
) -> dict[str, Any]:
    """Build a valid 10-player STRATZ GetMatch payload with deliberately scrambled player order."""
    positions = ["POSITION_5", "POSITION_3", "POSITION_1", "POSITION_4", "POSITION_2"]
    players: list[dict[str, Any]] = []
    for team_idx, is_rad in enumerate((True, False)):
        for slot_i in range(5):
            pslot = slot_i if is_rad else 128 + slot_i
            hero_id = (team_idx * 5) + slot_i + 1
            players.append(
                {
                    "steamAccountId": 100000 + team_idx * 10 + slot_i,
                    "playerSlot": pslot,
                    "isRadiant": is_rad,
                    "heroId": hero_id,
                    "variant": 0,
                    "position": positions[slot_i],
                    "lane": "SAFE_LANE",
                    "role": "CORE",
                    "leaverStatus": "NONE",
                    "kills": 5 + slot_i,
                    "deaths": 3,
                    "assists": 10,
                    "numLastHits": 180 - slot_i * 30,
                    "numDenies": 12 - slot_i * 2,
                    "goldPerMinute": 500 + slot_i * 20,
                    "experiencePerMinute": 600 + slot_i * 20,
                    "networth": 18000 + slot_i * 1000,
                    "level": 22,
                    "heroDamage": 22000 - slot_i * 2500,
                    "towerDamage": 4000 - slot_i * 700,
                    "heroHealing": 1500 if slot_i in (3, 4) else 0,
                    "item0Id": 63,
                    "item1Id": 1,
                    "item2Id": 116,
                    "item3Id": None,
                    "item4Id": None,
                    "item5Id": None,
                    "neutral0Id": 1720,
                    "stats": {
                        "heroDamageReport": {
                            "receivedTotal": {
                                "physicalDamage": 8000 + slot_i * 500,
                                "magicalDamage": 6000 + slot_i * 300,
                                "pureDamage": 1200,
                            }
                        }
                    },
                    "playbackData": {
                        "purchaseEvents": [
                            {"time": -60, "itemId": 44},  # Tango (should be filtered from key items)
                            {"time": 420, "itemId": 63},  # Power Treads
                            {"time": 780, "itemId": 1},  # Blink
                            {"time": 1200, "itemId": 116},  # BKB
                            {"time": 1500, "itemId": 204},  # Dagon 5 -> aliased to Dagon (104)
                        ],
                        "abilityLearnEvents": [
                            {"time": -40, "abilityId": 5003, "level": 0, "levelObtained": 1, "isUltimate": False, "isTalent": False},
                            {"time": 110, "abilityId": 5004, "level": 0, "levelObtained": 2, "isUltimate": False, "isTalent": False},
                            {"time": 600, "abilityId": 5006, "level": 0, "levelObtained": 6, "isUltimate": True, "isTalent": False},
                            {"time": 1000, "abilityId": 1495, "level": 0, "levelObtained": 10, "isUltimate": False, "isTalent": True},
                        ],
                    },
                }
            )

    return {
        "id": match_id,
        "didRadiantWin": did_radiant_win,
        "durationSeconds": duration_seconds,
        "startDateTime": start_datetime,
        "gameVersionId": gv_id,
        "gameMode": game_mode,
        "lobbyType": lobby_type,
        "bracket": bracket,
        "rank": 80,
        "averageRank": 80,
        "actualRank": 80,
        "parsedDateTime": parsed_datetime,
        "leagueId": league_id,
        "topLaneOutcome": "RADIANT_VICTORY",
        "midLaneOutcome": "TIE",
        "bottomLaneOutcome": "DIRE_VICTORY",
        "radiantNetworthLeads": [0, 0, 120, 210, 180, -40, -110, 50, 200, 100, -300, 1250, 1400, 1800, 2100, 2400],
        "radiantExperienceLeads": [0, 0, -50, -90, 40, 120, 90, 150, 220, 80, -150, 840, 980, 1200, 1350, 1600],
        "players": players,
    }


class InMemoryHFStorage:
    """Fake HFDatasetStorage capturing uploaded files and latest.json in memory."""

    def __init__(self) -> None:
        self.repo_ensured = False
        self.files: dict[str, bytes] = {}
        self.manifests: list[dict[str, Any]] = []

    async def ensure_repo(self) -> None:
        self.repo_ensured = True

    async def read_latest_manifest(self) -> dict[str, Any] | None:
        import json

        raw = self.files.get("latest.json")
        return json.loads(raw.decode("utf-8")) if raw else None

    async def upload_vocab(self, gv_id: int, gv_name: str, vocab: dict[str, Any]) -> str:
        import json

        path = f"data/gv_{gv_id}_{gv_name}/vocab.json"
        self.files[path] = json.dumps(vocab, ensure_ascii=False).encode("utf-8")
        return path

    async def update_manifest_for_patch(
        self, gv_id: int, gv_name: str, as_of_datetime: int, vocab_path: str
    ) -> dict[str, Any]:
        import json

        from shared.constants import PARQUET_SCHEMA_VERSION, VOCAB_SCHEMA_VERSION

        existing = await self.read_latest_manifest()
        shards = []
        if (
            existing
            and int(existing.get("game_version_id") or -1) == int(gv_id)
            and int(existing.get("parquet_schema_version") or existing.get("schema_version") or 1)
            == PARQUET_SCHEMA_VERSION
        ):
            shards = list(existing.get("shards") or [])
        manifest = {
            "schema_version": PARQUET_SCHEMA_VERSION,
            "parquet_schema_version": PARQUET_SCHEMA_VERSION,
            "vocab_schema_version": VOCAB_SCHEMA_VERSION,
            "game_version_id": int(gv_id),
            "game_version_name": str(gv_name),
            "as_of_datetime": int(as_of_datetime),
            "data_prefix": f"data/gv_{gv_id}_{gv_name}/",
            "vocab_path": vocab_path,
            "shards": shards,
            "total_matches": sum(int(s["num_rows"]) for s in shards),
        }
        self.files["latest.json"] = json.dumps(manifest).encode("utf-8")
        self.manifests.append(manifest)
        return manifest

    async def upload_shard_and_update_manifest(
        self,
        *,
        gv_id: int,
        gv_name: str,
        as_of_datetime: int,
        shard_path: Path,
        shard_meta: dict[str, Any],
        is_active_patch: bool = True,
    ) -> dict[str, Any] | None:
        import json

        from shared.constants import PARQUET_SCHEMA_VERSION, VOCAB_SCHEMA_VERSION

        path = f"data/gv_{gv_id}_{gv_name}/{shard_path.name}"
        self.files[path] = shard_path.read_bytes()
        if not is_active_patch:
            return None

        existing = await self.read_latest_manifest()
        shards = []
        if (
            existing
            and int(existing.get("game_version_id") or -1) == int(gv_id)
            and int(existing.get("parquet_schema_version") or existing.get("schema_version") or 1)
            == PARQUET_SCHEMA_VERSION
        ):
            shards = [s for s in (existing.get("shards") or []) if s.get("path") != path]
        shards.append({**shard_meta, "path": path})
        vocab_path = (
            existing.get("vocab_path")
            if existing and int(existing.get("game_version_id") or -1) == int(gv_id)
            else f"data/gv_{gv_id}_{gv_name}/vocab.json"
        )
        manifest = {
            "schema_version": PARQUET_SCHEMA_VERSION,
            "parquet_schema_version": PARQUET_SCHEMA_VERSION,
            "vocab_schema_version": VOCAB_SCHEMA_VERSION,
            "game_version_id": int(gv_id),
            "game_version_name": str(gv_name),
            "as_of_datetime": int(as_of_datetime),
            "data_prefix": f"data/gv_{gv_id}_{gv_name}/",
            "vocab_path": vocab_path,
            "shards": shards,
            "total_matches": sum(int(s["num_rows"]) for s in shards),
        }
        self.files["latest.json"] = json.dumps(manifest).encode("utf-8")
        self.manifests.append(manifest)
        return manifest


@pytest.fixture
def sample_vocab() -> dict[str, Any]:
    from shared.vocab import build_vocab

    return build_vocab(
        {"id": 190, "name": "7.41f", "asOfDateTime": 1789430400},
        make_stratz_constants_payload(),
        make_dotaconstants_payload(),
    )
