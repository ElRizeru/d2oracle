"""Unit tests for STRATZ match validation and canonical record transformation."""

from __future__ import annotations

from collector.validator import (
    find_missing_key_items_from_finals,
    prevalidate_stratz_match,
    validate_and_transform_match,
)
from shared.constants import (
    ATTRIBUTE_BONUS_ABILITY_ID,
    LANE_OUTCOME_TO_ID,
    MISSING_STAT_SENTINEL,
    STEAM_ANONYMOUS_ACCOUNT_ID,
)
from tests.conftest import make_raw_stratz_match


def test_valid_ranked_pub_canonical_ordering_and_fields(sample_vocab) -> None:
    raw = make_raw_stratz_match(game_mode="ALL_PICK_RANKED", lobby_type="RANKED", bracket=8)
    res = validate_and_transform_match(
        raw,
        active_gv_id=190,
        active_gv_name="7.41f",
        source="opendota_public",
        vocab=sample_vocab,
    )

    assert res.valid is True
    assert res.record is not None
    rec = res.record
    assert "steam_account_ids" not in rec
    assert rec["is_pro"] is False
    assert rec["is_radiant"] == [True] * 5 + [False] * 5
    assert rec["positions"] == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    assert rec["variant"] == [0] * 10
    assert len(res.player_steam_ids) == 10
    assert rec["key_item_ids"][0] == [63, 1, 116, 104]
    assert rec["key_item_times"][0] == [420, 780, 1200, 1500]
    # @10m leads must read STRATZ index 11 (not index 10)
    assert rec["radiant_gold_lead_10m"] == 1250
    assert rec["radiant_xp_lead_10m"] == 840
    # Canonical slot 0 is POSITION_1 (which was slot_i == 2 in scrambled input)
    assert rec["last_hits"][0] == 120
    assert rec["denies"][0] == 8
    assert rec["hero_damage"][0] == 17000
    assert rec["tower_damage"][0] == 2600
    assert rec["hero_healing"][0] == 0
    assert rec["hero_healing"][3] == 1500
    assert rec["hero_damage_received"][0] == (9000 + 6600 + 1200)


def test_handles_lane_stomps_short_leads_and_missing_received_damage(sample_vocab) -> None:
    raw = make_raw_stratz_match()
    raw["topLaneOutcome"] = "RADIANT_STOMP"
    raw["midLaneOutcome"] = "TIE"
    raw["bottomLaneOutcome"] = "DIRE_STOMP"
    raw["radiantNetworthLeads"] = [0, 0, 100]  # shorter than index 11
    raw["radiantExperienceLeads"] = None
    # Strip stats on one player to test MISSING_STAT_SENTINEL (-1)
    raw["players"][0]["stats"] = None

    res = validate_and_transform_match(
        raw,
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
    )
    assert res.valid is True
    assert res.record is not None
    rec = res.record
    assert rec["top_lane_outcome"] == "RADIANT_STOMP"
    assert rec["bottom_lane_outcome"] == "DIRE_STOMP"
    assert LANE_OUTCOME_TO_ID[rec["top_lane_outcome"]] == 0
    assert LANE_OUTCOME_TO_ID[rec["mid_lane_outcome"]] == 1
    assert LANE_OUTCOME_TO_ID[rec["bottom_lane_outcome"]] == 2
    assert rec["radiant_gold_lead_10m"] is None
    assert rec["radiant_xp_lead_10m"] is None
    assert MISSING_STAT_SENTINEL in rec["hero_damage_received"]
    assert sum(1 for v in rec["hero_damage_received"] if v == MISSING_STAT_SENTINEL) == 1


def test_valid_pro_captains_mode_with_practice_lobby_sets_is_pro(sample_vocab) -> None:
    raw = make_raw_stratz_match(
        game_mode="CAPTAINS_MODE",
        lobby_type="PRACTICE",
        bracket=8,
        league_id=20486,
    )
    res = validate_and_transform_match(
        raw,
        active_gv_id=190,
        active_gv_name="7.41f",
        source="opendota_pro",
        vocab=sample_vocab,
    )
    assert res.valid is True
    assert res.record is not None
    assert res.record["is_pro"] is True
    assert res.record["league_id"] == 20486


def test_unparsed_or_missing_match_is_retryable(sample_vocab) -> None:
    res_none = validate_and_transform_match(
        None, active_gv_id=190, active_gv_name="7.41f", source="bfs", vocab=sample_vocab
    )
    assert res_none.valid is False
    assert res_none.retryable is True
    assert res_none.reason == "not_found"

    raw = make_raw_stratz_match(parsed_datetime=None)
    res_unparsed = validate_and_transform_match(
        raw, active_gv_id=190, active_gv_name="7.41f", source="bfs", vocab=sample_vocab
    )
    assert res_unparsed.valid is False
    assert res_unparsed.retryable is True
    assert res_unparsed.reason == "unparsed"


def test_rejects_non_ranked_all_pick_and_wrong_bracket_and_leavers(sample_vocab) -> None:
    # ALL_PICK (unranked/turbo style) rejected
    res_mode = validate_and_transform_match(
        make_raw_stratz_match(game_mode="ALL_PICK", lobby_type="RANKED"),
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
    )
    assert res_mode.valid is False
    assert res_mode.retryable is False
    assert res_mode.reason == "invalid_mode_or_lobby"

    # Non-pro PRACTICE lobby rejected
    res_lobby = validate_and_transform_match(
        make_raw_stratz_match(game_mode="CAPTAINS_MODE", lobby_type="PRACTICE", league_id=None),
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
    )
    assert res_lobby.valid is False
    assert res_lobby.reason == "invalid_mode_or_lobby"

    # Bracket != 8 rejected
    res_bracket = validate_and_transform_match(
        make_raw_stratz_match(bracket=7),
        active_gv_id=190,
        active_gv_name="7.41f",
        source="opendota_public",
        vocab=sample_vocab,
    )
    assert res_bracket.valid is False
    assert res_bracket.reason == "wrong_bracket"

    # Duration < 900s rejected
    res_short = validate_and_transform_match(
        make_raw_stratz_match(duration_seconds=899),
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
    )
    assert res_short.valid is False
    assert res_short.reason == "short_duration"

    # Leaver rejected
    raw_leaver = make_raw_stratz_match()
    raw_leaver["players"][3]["leaverStatus"] = "DISCONNECTED_TOO_LONG"
    res_leaver = validate_and_transform_match(
        raw_leaver,
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
    )
    assert res_leaver.valid is False
    assert res_leaver.reason == "leaver_detected"


def test_rejects_duplicate_positions_within_team(sample_vocab) -> None:
    raw = make_raw_stratz_match()
    # Two Radiant players assigned POSITION_1 -> missing POSITION_5
    radiant_players = [p for p in raw["players"] if p["isRadiant"]]
    for p in radiant_players:
        if p["position"] == "POSITION_5":
            p["position"] = "POSITION_1"
            break
    res = validate_and_transform_match(
        raw,
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
    )
    assert res.valid is False
    assert res.retryable is False
    assert res.reason == "duplicate_positions"


def test_accepts_post_game_disconnected_and_bounds_playback_retries(sample_vocab) -> None:
    from collector.validator import MAX_PLAYBACK_RETRY_AGE_SECONDS

    raw = make_raw_stratz_match()
    raw["players"][0]["leaverStatus"] = "DISCONNECTED"
    raw["players"][7]["leaverStatus"] = "DISCONNECTED"
    res = validate_and_transform_match(
        raw,
        active_gv_id=190,
        active_gv_name="7.41f",
        source="opendota_pro",
        vocab=sample_vocab,
    )
    assert res.valid is True
    assert res.record is not None

    # Disconnected too long (abandon) still rejected
    raw_abandon = make_raw_stratz_match()
    raw_abandon["players"][0]["leaverStatus"] = "DISCONNECTED_TOO_LONG"
    res_abandon = validate_and_transform_match(
        raw_abandon,
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
    )
    assert res_abandon.valid is False
    assert res_abandon.reason == "leaver_detected"

    # missing_playback_data stops retrying after MAX_PLAYBACK_RETRY_AGE_SECONDS
    raw_no_pb = make_raw_stratz_match(start_datetime=900, parsed_datetime=1000)
    for p in raw_no_pb["players"]:
        p["playbackData"] = None
    fresh_res = validate_and_transform_match(
        raw_no_pb,
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
        now_ts=1000 + MAX_PLAYBACK_RETRY_AGE_SECONDS - 60,
    )
    assert fresh_res.valid is False
    assert fresh_res.reason == "missing_playback_data"
    assert fresh_res.retryable is True

    stale_res = validate_and_transform_match(
        raw_no_pb,
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
        now_ts=1000 + MAX_PLAYBACK_RETRY_AGE_SECONDS + 60,
    )
    assert stale_res.valid is False
    assert stale_res.reason == "missing_playback_data"
    assert stale_res.retryable is False


def test_validates_abilities_and_item_purchases_without_playback_data(sample_vocab) -> None:
    raw = make_raw_stratz_match(game_mode="ALL_PICK_RANKED", lobby_type="RANKED", bracket=8)
    for p in raw["players"]:
        pb = p.pop("playbackData")
        p["abilities"] = [
            {
                "abilityId": ev["abilityId"],
                "time": ev["time"],
                "level": ev["level"],
                "isTalent": ev["isTalent"],
            }
            for ev in pb["abilityLearnEvents"]
        ]
        p["stats"]["itemPurchases"] = list(pb["purchaseEvents"])

    # Customize Radiant POSITION_1 (standard hero=1, 26 events w/ same-second tie + overflow guard),
    # Radiant POSITION_2 (Invoker=74, 25 events), and Radiant POSITION_3 (Meepo=82, 22 events).
    for p in raw["players"]:
        if not p["isRadiant"]:
            continue
        if p["position"] == "POSITION_1":
            p["heroId"] = 1
            events = [
                {"abilityId": 5003 if i % 2 == 0 else 5004, "time": 120 if i in (1, 2) else i * 100, "level": 1, "isTalent": i in (9, 14, 17, 18)}
                for i in range(26)
            ]
            # Swap tie event ids to verify order preservation at equal time=120
            events[1]["abilityId"] = 5006
            events[2]["abilityId"] = 7314
            p["abilities"] = events
        elif p["position"] == "POSITION_2":
            p["heroId"] = 74
            p["abilities"] = [
                {"abilityId": 5376 + (i % 3), "time": i * 80, "level": i // 3, "isTalent": i in (9, 14, 19, 24)}
                for i in range(25)
            ]
        elif p["position"] == "POSITION_3":
            p["heroId"] = 82
            p["abilities"] = [
                {"abilityId": 5426 + (i % 4), "time": i * 90, "level": i // 4, "isTalent": i in (9, 14, 17, 18)}
                for i in range(22)
            ]

    res = validate_and_transform_match(
        raw,
        active_gv_id=190,
        active_gv_name="7.41f",
        source="bfs",
        vocab=sample_vocab,
    )
    assert res.valid is True
    assert res.record is not None
    rec = res.record
    assert rec["key_item_ids"][0] == [63, 1, 116, 104]
    assert rec["key_item_times"][0] == [420, 780, 1200, 1500]

    # Under 7.40+ rules, talents use decoupled points (10, 15, 20, 25) and skills advance 1..22 continuously,
    # keeping same-second tie order intact across indices 1..2.
    assert rec["ability_learn_levels"][0] == [
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 10, 11, 12, 13,
        15, 14, 15, 20, 25,
        16, 17, 18, 19, 20, 21, 22,
    ]
    assert rec["ability_learn_ids"][0][1:3] == [5006, 7314]

    # Invoker (74): 21 non-talent orbs across 1..21 + 4 separate talents at 10, 15, 20, 25
    assert rec["ability_learn_levels"][1] == [
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 10, 11, 12, 13,
        15, 14, 15, 16, 17, 20,
        18, 19, 20, 21, 25,
    ]

    # Meepo (82): 18 non-talent skills across 1..18 + 4 separate talents at 10, 15, 20, 25
    assert rec["ability_learn_levels"][2] == [
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 10, 11, 12, 13,
        15, 14, 15, 20, 25,
        16, 17, 18,
    ]


def test_rejects_empty_item_purchases_or_abilities_without_playback_data(sample_vocab) -> None:
    raw = make_raw_stratz_match()
    for p in raw["players"]:
        pb = p.pop("playbackData")
        p["abilities"] = pb["abilityLearnEvents"]
        p["stats"]["itemPurchases"] = list(pb["purchaseEvents"])

    raw_empty_items = make_raw_stratz_match()
    for p in raw_empty_items["players"]:
        pb = p.pop("playbackData")
        p["abilities"] = pb["abilityLearnEvents"]
        p["stats"]["itemPurchases"] = list(pb["purchaseEvents"])
    raw_empty_items["players"][0]["stats"]["itemPurchases"] = []
    res_items = validate_and_transform_match(
        raw_empty_items, active_gv_id=190, active_gv_name="7.41f", source="bfs", vocab=sample_vocab
    )
    assert res_items.valid is False
    assert res_items.retryable is False
    assert res_items.reason == "empty_purchase_events"

    raw["players"][0]["abilities"] = []
    res_ab = validate_and_transform_match(
        raw, active_gv_id=190, active_gv_name="7.41f", source="bfs", vocab=sample_vocab
    )
    assert res_ab.valid is False
    assert res_ab.retryable is False
    assert res_ab.reason == "empty_ability_events"


def test_steam_merge_fixes_zero_talents_truncation_ranks_and_anon_ids(sample_vocab) -> None:
    raw = make_raw_stratz_match(gv_id=190)
    for p in raw["players"]:
        p.pop("playbackData")
        p["stats"]["itemPurchases"] = [
            {"time": 420, "itemId": 63},
            {"time": 1200, "itemId": 116},
        ]
        p["abilities"] = [
            {"abilityId": 5003, "time": -80, "level": 0, "isTalent": False},
            {"abilityId": 5004, "time": 110, "level": 0, "isTalent": False},
            {"abilityId": 0, "time": 900, "level": 0, "isTalent": False},
        ]

    # Customize Radiant pos 1 (Largo=155, slot=2) with STRATZ zero talents + truncated at 18
    rad_p1 = next(p for p in raw["players"] if p["isRadiant"] and p["position"] == "POSITION_1")
    rad_p1["heroId"] = 155
    rad_p1["abilities"] = [
        {"abilityId": 1659, "time": -89, "level": 0, "isTalent": False},
        {"abilityId": 1660, "time": 95, "level": 0, "isTalent": False},
        {"abilityId": 0, "time": 700, "level": 0, "isTalent": False},
        {"abilityId": ATTRIBUTE_BONUS_ABILITY_ID, "time": 800, "level": 0, "isTalent": True},
    ]

    steam_players = []
    for p in raw["players"]:
        slot = p["playerSlot"]
        is_largo = p["heroId"] == 155
        steam_players.append(
            {
                "account_id": STEAM_ANONYMOUS_ACCOUNT_ID if is_largo else 200000 + slot,
                "player_slot": slot,
                "hero_id": p["heroId"],
                "hero_variant": 2,
                "kills": 12 if is_largo else 4,
                "deaths": 2,
                "assists": 15,
                "last_hits": 250,
                "denies": 14,
                "gold_per_min": 640,
                "xp_per_min": 780,
                "net_worth": 22000,
                "level": 24,
                "hero_damage": 30000,
                "tower_damage": 5000,
                "hero_healing": 4200,
                "leaver_status": 0,
                "item_0": 63,
                "item_1": 116,
                "item_2": 0,
                "item_3": 0,
                "item_4": 0,
                "item_5": 0,
                "item_neutral": 1720,
                "ability_upgrades": (
                    [
                        {"ability": 1659, "time": 100, "level": 1},
                        {"ability": 1660, "time": 284, "level": 2},
                        {"ability": 8001, "time": 889, "level": 3},
                        {"ability": ATTRIBUTE_BONUS_ABILITY_ID, "time": 989, "level": 4},
                        {"ability": 1659, "time": 1289, "level": 5},
                    ]
                    if is_largo
                    else [
                        {"ability": 5003, "time": 109, "level": 1},
                        {"ability": 5004, "time": 299, "level": 2},
                        {"ability": 1495, "time": 1089, "level": 3},
                    ]
                ),
            }
        )
    steam_match = {
        "match_id": raw["id"],
        "start_time": raw["startDateTime"] + 15,
        "duration": 2222,
        "pre_game_duration": 90,
        "radiant_win": False,
        "lobby_type": 7,
        "game_mode": 22,
        "leagueid": 0,
        "players": steam_players,
    }

    res = validate_and_transform_match(
        raw,
        active_gv_id=191,
        active_gv_name="7.42",
        source="bfs",
        vocab=sample_vocab,
        steam_match=steam_match,
        allowed_gv_ids=(190, 191),
    )

    assert res.valid is True and res.record is not None
    rec = res.record
    assert rec["duration_seconds"] == 2222
    assert rec["radiant_win"] is False
    assert rec["kills"][0] == 12
    assert rec["gpm"][0] == 640
    assert rec["variant"][0] == 2
    assert STEAM_ANONYMOUS_ACCOUNT_ID not in res.player_steam_ids
    # Largo zero-talent replaced by 8001 and tail event extrapolated (+300s after 800 -> 1100)
    assert rec["ability_learn_ids"][0] == [1659, 1660, 8001, ATTRIBUTE_BONUS_ABILITY_ID, 1659]
    assert rec["ability_learn_times"][0] == [-89, 95, 700, 800, 1100]
    assert rec["ability_learn_ranks"][0] == [1, 1, 1, 1, 2]
    assert rec["ability_learn_is_talent"][0] == [False, False, True, False, False]


def test_opendota_purchase_log_recovers_missing_new_patch_key_item_and_prevalidation_bounds(
    sample_vocab,
) -> None:
    import copy

    vocab = copy.deepcopy(sample_vocab)
    vocab["items"]["by_id"]["1953"] = {
        "idx": vocab["items"]["vocab_size"],
        "id": 1953,
        "short_name": "berserkers_sabatons",
        "display_name": "Sabatons of Mordiggian",
        "cost": 5950,
        "img": "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/items/berserkers_sabatons.png",
    }
    vocab["items"]["id_to_idx"]["1953"] = vocab["items"]["vocab_size"]
    vocab["items"]["idx_to_id"].append(1953)
    vocab["items"]["vocab_size"] += 1

    raw = make_raw_stratz_match(
        gv_id=190,
        start_datetime=1795040000,
        game_mode="CAPTAINS_MODE",
        lobby_type="PRACTICE",
        league_id=20511,
    )
    for p in raw["players"]:
        p.pop("playbackData")
        p["stats"]["itemPurchases"] = [{"time": 350, "itemId": 63}]
        p["abilities"] = [{"abilityId": 5003, "time": -80, "level": 0, "isTalent": False}]

    steam_players = []
    od_players = []
    for p in raw["players"]:
        slot = p["playerSlot"]
        is_carry = p["isRadiant"] and p["position"] == "POSITION_1"
        steam_players.append(
            {
                "account_id": 300000 + slot,
                "player_slot": slot,
                "hero_id": p["heroId"],
                "kills": 5,
                "deaths": 1,
                "assists": 8,
                "last_hits": 200,
                "denies": 10,
                "gold_per_min": 600,
                "xp_per_min": 700,
                "net_worth": 20000,
                "level": 20,
                "hero_damage": 20000,
                "tower_damage": 3000,
                "hero_healing": 0,
                "leaver_status": 0,
                "item_0": 63,
                "item_1": 1953 if is_carry else 0,
                "item_2": 0,
                "item_3": 0,
                "item_4": 0,
                "item_5": 0,
                "item_neutral": 0,
                "ability_upgrades": [{"ability": 5003, "time": 200, "level": 1}],
            }
        )
        od_players.append(
            {
                "player_slot": slot,
                "purchase_log": [{"time": 1480, "key": "berserkers_sabatons"}] if is_carry else [],
            }
        )
    steam_match = {
        "match_id": raw["id"],
        "start_time": raw["startDateTime"],
        "duration": 2100,
        "radiant_win": True,
        "lobby_type": 1,
        "game_mode": 2,
        "leagueid": 20511,
        "players": steam_players,
    }
    od_match = {"match_id": raw["id"], "players": od_players}

    assert find_missing_key_items_from_finals(raw, steam_match, vocab) == {1953}
    res = validate_and_transform_match(
        raw,
        active_gv_id=191,
        active_gv_name="7.42",
        source="opendota_pro",
        vocab=vocab,
        steam_match=steam_match,
        opendota_match=od_match,
        allowed_gv_ids=(190, 191),
        min_start_datetime=1795035000,
    )
    assert res.valid and res.record is not None
    assert res.record["key_item_ids"][0] == [63, 1953]
    assert res.record["key_item_times"][0] == [350, 1480]

    # prevalidate_stratz_match enforces min/max start_datetime bounds
    pre_before = prevalidate_stratz_match(
        raw, active_gv_id=191, allowed_gv_ids=(190, 191), min_start_datetime=1795050000
    )
    assert not pre_before.valid and pre_before.reason == "before_patch_start"
    pre_after = prevalidate_stratz_match(
        raw, active_gv_id=190, max_start_datetime=1795035000
    )
    assert not pre_after.valid and pre_after.reason == "after_patch_end"
