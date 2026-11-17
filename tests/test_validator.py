"""Unit tests for STRATZ match validation and canonical record transformation."""

from __future__ import annotations

from collector.validator import validate_and_transform_match
from shared.constants import LANE_OUTCOME_TO_ID, MISSING_STAT_SENTINEL
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

    # Standard hero: 1..16, 18, 20, 25..30, bounded at 30 without IndexError, tie at index 1..2 preserved
    assert rec["ability_learn_levels"][0] == [*range(1, 17), 18, 20, 25, 26, 27, 28, 29, 30, 30, 30]
    assert rec["ability_learn_ids"][0][1:3] == [5006, 7314]

    # Invoker (74): contiguous 1..25 (includes 17..24)
    assert rec["ability_learn_levels"][1] == list(range(1, 26))

    # Meepo (82): includes 17, skips 18, jumps to 20, 25..28
    assert rec["ability_learn_levels"][2] == [*range(1, 18), 20, 25, 26, 27, 28]


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
