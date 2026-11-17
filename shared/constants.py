"""Shared constants, enums, URLs, and canonical Parquet Arrow schema for d2oracle."""

from __future__ import annotations

import pyarrow as pa

STRATZ_GRAPHQL_URL = "https://api.stratz.com/graphql"
OPENDOTA_API_URL = "https://api.opendota.com/api"
DOTACONSTANTS_BASE_URL = "https://raw.githubusercontent.com/odota/dotaconstants/master/build"
STEAM_CDN_BASE_URL = "https://cdn.cloudflare.steamstatic.com"

POSITION_TO_INT: dict[str, int] = {
    "POSITION_1": 1,
    "POSITION_2": 2,
    "POSITION_3": 3,
    "POSITION_4": 4,
    "POSITION_5": 5,
}
INT_TO_POSITION: dict[int, str] = {v: k for k, v in POSITION_TO_INT.items()}

VALID_PUB_GAME_MODES: frozenset[str] = frozenset({"ALL_PICK_RANKED", "CAPTAINS_MODE"})
PRO_GAME_MODE = "CAPTAINS_MODE"
IMMORTAL_BRACKET = 8
MIN_DURATION_SECONDS = 900
RANKED_LOBBY_TYPE_ID = 7

LEADERBOARD_DIVISIONS: tuple[str, ...] = ("EUROPE", "AMERICAS", "SE_ASIA", "CHINA")

ATTRIBUTE_BONUS_ABILITY_ID = 730

# STRATZ radiantNetworthLeads / radiantExperienceLeads prepend a pre-horn 0 at index 0,
# so index 1 corresponds to 0:00 (networthPerMinute[0]) and index 11 is 10:00.
STRATZ_LEADS_10M_INDEX = 11
MISSING_STAT_SENTINEL = -1

LANE_OUTCOME_TO_ID: dict[str, int] = {
    "RADIANT_VICTORY": 0,
    "RADIANT_STOMP": 0,
    "TIE": 1,
    "DIRE_VICTORY": 2,
    "DIRE_STOMP": 2,
}
LANE_OUTCOME_LABELS: tuple[str, ...] = ("RADIANT_WIN", "EVEN", "DIRE_WIN")

VOCAB_SCHEMA_VERSION = 2
PARQUET_SCHEMA_VERSION = 2

# Canonical Arrow schema for collected match shards (zstd-compressed Parquet).
# Every per-player list has length 10 ordered canonically:
# Radiant POSITION_1..5 followed by Dire POSITION_1..5.
PARQUET_ARROW_SCHEMA = pa.schema(
    [
        pa.field("match_id", pa.int64(), nullable=False),
        pa.field("start_datetime", pa.int64(), nullable=False),
        pa.field("duration_seconds", pa.int32(), nullable=False),
        pa.field("radiant_win", pa.bool_(), nullable=False),
        pa.field("game_version_id", pa.int32(), nullable=False),
        pa.field("game_version_name", pa.string(), nullable=False),
        pa.field("game_mode", pa.string(), nullable=False),
        pa.field("lobby_type", pa.string(), nullable=False),
        pa.field("bracket", pa.int8(), nullable=False),
        pa.field("rank", pa.int16(), nullable=True),
        pa.field("average_rank", pa.int16(), nullable=True),
        pa.field("actual_rank", pa.int16(), nullable=True),
        pa.field("league_id", pa.int32(), nullable=True),
        pa.field("is_pro", pa.bool_(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("top_lane_outcome", pa.string(), nullable=True),
        pa.field("mid_lane_outcome", pa.string(), nullable=True),
        pa.field("bottom_lane_outcome", pa.string(), nullable=True),
        pa.field("radiant_gold_lead_10m", pa.int32(), nullable=True),
        pa.field("radiant_xp_lead_10m", pa.int32(), nullable=True),
        pa.field("player_slots", pa.list_(pa.int16()), nullable=False),
        pa.field("is_radiant", pa.list_(pa.bool_()), nullable=False),
        pa.field("hero_ids", pa.list_(pa.int16()), nullable=False),
        pa.field("positions", pa.list_(pa.int8()), nullable=False),
        pa.field("variant", pa.list_(pa.int8()), nullable=False),
        pa.field("kills", pa.list_(pa.int16()), nullable=False),
        pa.field("deaths", pa.list_(pa.int16()), nullable=False),
        pa.field("assists", pa.list_(pa.int16()), nullable=False),
        pa.field("last_hits", pa.list_(pa.int16()), nullable=False),
        pa.field("denies", pa.list_(pa.int16()), nullable=False),
        pa.field("gpm", pa.list_(pa.int16()), nullable=False),
        pa.field("xpm", pa.list_(pa.int16()), nullable=False),
        pa.field("networth", pa.list_(pa.int32()), nullable=False),
        pa.field("levels", pa.list_(pa.int8()), nullable=False),
        pa.field("hero_damage", pa.list_(pa.int32()), nullable=False),
        pa.field("tower_damage", pa.list_(pa.int32()), nullable=False),
        pa.field("hero_healing", pa.list_(pa.int32()), nullable=False),
        pa.field("hero_damage_received", pa.list_(pa.int32()), nullable=False),
        pa.field("final_items", pa.list_(pa.list_(pa.int32())), nullable=False),
        pa.field("purchase_item_ids", pa.list_(pa.list_(pa.int32())), nullable=False),
        pa.field("purchase_times", pa.list_(pa.list_(pa.int16())), nullable=False),
        pa.field("key_item_ids", pa.list_(pa.list_(pa.int32())), nullable=False),
        pa.field("key_item_times", pa.list_(pa.list_(pa.int16())), nullable=False),
        pa.field("ability_learn_ids", pa.list_(pa.list_(pa.int32())), nullable=False),
        pa.field("ability_learn_times", pa.list_(pa.list_(pa.int16())), nullable=False),
        pa.field("ability_learn_levels", pa.list_(pa.list_(pa.int8())), nullable=False),
        pa.field("ability_learn_ranks", pa.list_(pa.list_(pa.int8())), nullable=False),
        pa.field("ability_learn_is_talent", pa.list_(pa.list_(pa.bool_())), nullable=False),
    ]
)
