"""Match validation and canonical transformation into Parquet-ready records."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from shared.constants import (
    IMMORTAL_BRACKET,
    MIN_DURATION_SECONDS,
    MISSING_STAT_SENTINEL,
    POSITION_TO_INT,
    PRO_GAME_MODE,
    STRATZ_LEADS_10M_INDEX,
    VALID_PUB_GAME_MODES,
)
from shared.vocab import extract_key_item_purchases

ALLOWED_LEAVER_STATUSES: frozenset[str] = frozenset({"NONE", "DISCONNECTED"})
MAX_PLAYBACK_RETRY_AGE_SECONDS: int = 6 * 3600

# Dota 2 patch 7.41f ability/talent point grant levels (verify when new major patches like 7.42 drop).
# STRATZ PlayerAbilityType (`match.players[].abilities`) omits `levelObtained`, so we reconstruct
# the hero level at each 0-based chronological upgrade index:
# - Invoker (74): receives an ability/talent/stat point on every level 1..30.
# - Meepo (82): skips level 18 (points at 1..17, 20, 25, 26..30).
# - All other heroes: skip levels 17, 19, 21..24 (points at 1..16, 18, 20, 25, 26..30).
_INVOKER_HERO_ID: int = 74
_MEEPO_HERO_ID: int = 82
_INVOKER_POINT_LEVELS: tuple[int, ...] = tuple(range(1, 31))
_MEEPO_POINT_LEVELS: tuple[int, ...] = (*range(1, 18), 20, 25, 26, 27, 28, 29, 30)
_STANDARD_POINT_LEVELS: tuple[int, ...] = (*range(1, 17), 18, 20, 25, 26, 27, 28, 29, 30)


def _infer_point_level(hero_id: int, point_idx: int) -> int:
    """Infer hero levelObtained for 0-based ability upgrade index (patch 7.41f rules)."""
    if hero_id == _INVOKER_HERO_ID:
        table = _INVOKER_POINT_LEVELS
    elif hero_id == _MEEPO_HERO_ID:
        table = _MEEPO_POINT_LEVELS
    else:
        table = _STANDARD_POINT_LEVELS
    if 0 <= point_idx < len(table):
        return table[point_idx]
    if not table:
        return min(30, max(1, point_idx + 1))
    return min(30, table[-1] + (point_idx - len(table) + 1))


def _extract_player_timelines(
    player: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, bool, str | None]:
    """Extract purchase and ability timelines from playbackData (legacy) or abilities+stats.itemPurchases."""
    pb = player.get("playbackData")
    if isinstance(pb, dict):
        purchases = pb.get("purchaseEvents")
        abilities = pb.get("abilityLearnEvents")
        if not purchases:
            return None, None, False, "empty_purchase_events"
        if not abilities:
            return None, None, False, "empty_ability_events"
        return purchases, abilities, False, None

    stats = player.get("stats")
    purchases = stats.get("itemPurchases") if isinstance(stats, dict) else None
    abilities = player.get("abilities")
    if purchases is None and abilities is None:
        return None, None, False, "missing_playback_data"
    if not purchases:
        return None, None, False, "empty_purchase_events"
    if not abilities:
        return None, None, False, "empty_ability_events"
    return purchases, abilities, True, None


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a raw STRATZ match payload."""

    valid: bool
    retryable: bool = False
    reason: str | None = None
    record: dict[str, Any] | None = None
    player_steam_ids: list[int] = field(default_factory=list)


def _clamp(val: int | None, low: int, high: int, default: int = 0) -> int:
    if val is None:
        return default
    return max(low, min(high, int(val)))


def _extract_lead_at_10m(leads: Any) -> int | None:
    """Extract Radiant team lead at 10:00 (STRATZ index 11), or None if unavailable."""
    if not isinstance(leads, list) or len(leads) <= STRATZ_LEADS_10M_INDEX:
        return None
    raw = leads[STRATZ_LEADS_10M_INDEX]
    if raw is None:
        return None
    try:
        return _clamp(int(raw), -2_000_000_000, 2_000_000_000)
    except (TypeError, ValueError):
        return None


def _extract_hero_damage_received(player: dict[str, Any]) -> int:
    """Sum physical/magical/pure hero damage received, returning MISSING_STAT_SENTINEL if absent."""
    stats = player.get("stats")
    if not isinstance(stats, dict):
        return MISSING_STAT_SENTINEL
    hd_report = stats.get("heroDamageReport")
    if not isinstance(hd_report, dict):
        return MISSING_STAT_SENTINEL
    rec = hd_report.get("receivedTotal")
    if not isinstance(rec, dict):
        return MISSING_STAT_SENTINEL
    parts = (rec.get("physicalDamage"), rec.get("magicalDamage"), rec.get("pureDamage"))
    if all(v is None for v in parts):
        return MISSING_STAT_SENTINEL
    total = sum(int(v or 0) for v in parts)
    return _clamp(total, 0, 2_000_000_000)


def _is_playback_retryable(match: dict[str, Any], now_ts: float | None = None) -> bool:
    """Retry missing playbackData only if the match was parsed recently."""
    current = time.time() if now_ts is None else float(now_ts)
    parsed_dt = int(match.get("parsedDateTime") or 0)
    if parsed_dt > 0 and current >= parsed_dt:
        return (current - parsed_dt) <= MAX_PLAYBACK_RETRY_AGE_SECONDS
    start_dt = int(match.get("startDateTime") or 0)
    if start_dt > 0 and current >= start_dt:
        return (current - start_dt) <= MAX_PLAYBACK_RETRY_AGE_SECONDS
    return True


def validate_and_transform_match(
    match: dict[str, Any] | None,
    *,
    active_gv_id: int,
    active_gv_name: str,
    source: str,
    vocab: dict[str, Any],
    now_ts: float | None = None,
) -> ValidationResult:
    """Validate a STRATZ GetMatch payload and transform it into a Parquet-ready dict."""
    if match is None:
        return ValidationResult(valid=False, retryable=True, reason="not_found")

    if match.get("parsedDateTime") is None:
        return ValidationResult(valid=False, retryable=True, reason="unparsed")

    gv_id = match.get("gameVersionId")
    if gv_id is None or int(gv_id) != int(active_gv_id):
        return ValidationResult(valid=False, retryable=False, reason="wrong_game_version")

    bracket = match.get("bracket")
    if bracket is None or int(bracket) != IMMORTAL_BRACKET:
        return ValidationResult(valid=False, retryable=False, reason="wrong_bracket")

    duration = match.get("durationSeconds")
    if duration is None or int(duration) < MIN_DURATION_SECONDS:
        return ValidationResult(valid=False, retryable=False, reason="short_duration")

    did_radiant_win = match.get("didRadiantWin")
    if did_radiant_win is None:
        return ValidationResult(valid=False, retryable=False, reason="missing_outcome")

    game_mode = str(match.get("gameMode") or "")
    lobby_type = str(match.get("lobbyType") or "")
    league_id = match.get("leagueId")
    is_pro = (league_id is not None) and (game_mode == PRO_GAME_MODE)

    if not is_pro and not (lobby_type == "RANKED" and game_mode in VALID_PUB_GAME_MODES):
        return ValidationResult(valid=False, retryable=False, reason="invalid_mode_or_lobby")

    players = match.get("players")
    if not isinstance(players, list) or len(players) != 10:
        return ValidationResult(valid=False, retryable=False, reason="invalid_player_count")

    radiant_count = sum(1 for p in players if p.get("isRadiant") is True)
    dire_count = sum(1 for p in players if p.get("isRadiant") is False)
    if radiant_count != 5 or dire_count != 5:
        return ValidationResult(valid=False, retryable=False, reason="invalid_team_split")

    for p in players:
        if p.get("leaverStatus") not in (None, *ALLOWED_LEAVER_STATUSES):
            return ValidationResult(valid=False, retryable=False, reason="leaver_detected")
        if p.get("position") not in POSITION_TO_INT:
            return ValidationResult(valid=False, retryable=False, reason="invalid_position")
        hero_id = p.get("heroId")
        if hero_id is None or int(hero_id) <= 0:
            return ValidationResult(valid=False, retryable=False, reason="invalid_hero")
        _purchases, _abilities, _inferred, err_reason = _extract_player_timelines(p)
        if err_reason is not None:
            retryable = (
                _is_playback_retryable(match, now_ts=now_ts)
                if err_reason == "missing_playback_data"
                else False
            )
            return ValidationResult(valid=False, retryable=retryable, reason=err_reason)

    radiant_positions = {
        POSITION_TO_INT[p["position"]] for p in players if p.get("isRadiant") is True
    }
    dire_positions = {
        POSITION_TO_INT[p["position"]] for p in players if p.get("isRadiant") is False
    }
    if len(radiant_positions) != 5 or len(dire_positions) != 5:
        return ValidationResult(valid=False, retryable=False, reason="duplicate_positions")

    ordered = sorted(
        players,
        key=lambda p: (
            0 if p["isRadiant"] else 1,
            POSITION_TO_INT[p["position"]],
            int(p.get("playerSlot") or 0),
        ),
    )

    player_slots: list[int] = []
    is_radiant: list[bool] = []
    hero_ids: list[int] = []
    positions: list[int] = []
    variant: list[int] = []
    kills: list[int] = []
    deaths: list[int] = []
    assists: list[int] = []
    last_hits: list[int] = []
    denies: list[int] = []
    gpm: list[int] = []
    xpm: list[int] = []
    networth: list[int] = []
    levels: list[int] = []
    hero_damage: list[int] = []
    tower_damage: list[int] = []
    hero_healing: list[int] = []
    hero_damage_received: list[int] = []
    final_items: list[list[int]] = []
    purchase_item_ids: list[list[int]] = []
    purchase_times: list[list[int]] = []
    key_item_ids: list[list[int]] = []
    key_item_times: list[list[int]] = []
    ability_learn_ids: list[list[int]] = []
    ability_learn_times: list[list[int]] = []
    ability_learn_levels: list[list[int]] = []
    ability_learn_ranks: list[list[int]] = []
    ability_learn_is_talent: list[list[bool]] = []

    bfs_player_ids: list[int] = []

    for p in ordered:
        sid = int(p.get("steamAccountId") or 0)
        if sid > 0:
            bfs_player_ids.append(sid)
        player_slots.append(_clamp(p.get("playerSlot"), 0, 255))
        is_radiant.append(bool(p["isRadiant"]))
        hero_ids.append(_clamp(p.get("heroId"), 1, 32767))
        positions.append(POSITION_TO_INT[p["position"]])
        variant.append(_clamp(p.get("variant"), -128, 127, default=0))
        kills.append(_clamp(p.get("kills"), 0, 32767))
        deaths.append(_clamp(p.get("deaths"), 0, 32767))
        assists.append(_clamp(p.get("assists"), 0, 32767))
        last_hits.append(_clamp(p.get("numLastHits"), 0, 32767))
        denies.append(_clamp(p.get("numDenies"), 0, 32767))
        gpm.append(_clamp(p.get("goldPerMinute"), 0, 32767))
        xpm.append(_clamp(p.get("experiencePerMinute"), 0, 32767))
        networth.append(max(0, int(p.get("networth") or 0)))
        levels.append(_clamp(p.get("level"), 1, 30, default=1))
        hero_damage.append(_clamp(p.get("heroDamage"), 0, 2_000_000_000))
        tower_damage.append(_clamp(p.get("towerDamage"), 0, 2_000_000_000))
        hero_healing.append(_clamp(p.get("heroHealing"), 0, 2_000_000_000))
        hero_damage_received.append(_extract_hero_damage_received(p))

        final_items.append(
            [
                int(p.get("item0Id") or 0),
                int(p.get("item1Id") or 0),
                int(p.get("item2Id") or 0),
                int(p.get("item3Id") or 0),
                int(p.get("item4Id") or 0),
                int(p.get("item5Id") or 0),
                int(p.get("neutral0Id") or 0),
            ]
        )

        raw_purchases_src, raw_abilities_src, infer_levels, _err = _extract_player_timelines(p)
        assert raw_purchases_src is not None and raw_abilities_src is not None
        raw_purchases = sorted(
            raw_purchases_src, key=lambda ev: int(ev.get("time") or 0)
        )
        p_ids: list[int] = []
        p_times: list[int] = []
        for ev in raw_purchases:
            if ev.get("itemId") is None:
                continue
            p_ids.append(int(ev["itemId"]))
            p_times.append(_clamp(ev.get("time"), -3600, 32000))
        purchase_item_ids.append(p_ids)
        purchase_times.append(p_times)

        k_ids, k_times = extract_key_item_purchases(raw_purchases, vocab, deduplicate=True)
        key_item_ids.append(k_ids)
        key_item_times.append(k_times)

        if infer_levels:
            # Python's Timsort is stable and preserves STRATZ order when two skills share the same second.
            raw_abilities = sorted(
                raw_abilities_src, key=lambda ev: int(ev.get("time") or 0)
            )
        else:
            raw_abilities = sorted(
                raw_abilities_src,
                key=lambda ev: (int(ev.get("levelObtained") or 0), int(ev.get("time") or 0)),
            )

        hero_id_int = int(p["heroId"])
        a_ids: list[int] = []
        a_times: list[int] = []
        a_levels: list[int] = []
        a_ranks: list[int] = []
        a_talents: list[bool] = []
        for ab_idx, ev in enumerate(raw_abilities):
            if ev.get("abilityId") is None:
                continue
            raw_lvl = ev.get("levelObtained")
            inferred_lvl = (
                int(raw_lvl)
                if raw_lvl is not None
                else _infer_point_level(hero_id_int, len(a_ids) if infer_levels else ab_idx)
            )
            a_ids.append(int(ev["abilityId"]))
            a_times.append(_clamp(ev.get("time"), -3600, 32000))
            a_levels.append(_clamp(inferred_lvl, 0, 30))
            a_ranks.append(_clamp(ev.get("level"), 0, 10))
            a_talents.append(bool(ev.get("isTalent")))
        ability_learn_ids.append(a_ids)
        ability_learn_times.append(a_times)
        ability_learn_levels.append(a_levels)
        ability_learn_ranks.append(a_ranks)
        ability_learn_is_talent.append(a_talents)

    record: dict[str, Any] = {
        "match_id": int(match["id"]),
        "start_datetime": int(match.get("startDateTime") or 0),
        "duration_seconds": int(duration),
        "radiant_win": bool(did_radiant_win),
        "game_version_id": int(active_gv_id),
        "game_version_name": str(active_gv_name),
        "game_mode": game_mode,
        "lobby_type": lobby_type,
        "bracket": int(bracket),
        "rank": int(match["rank"]) if match.get("rank") is not None else None,
        "average_rank": (
            int(match["averageRank"]) if match.get("averageRank") is not None else None
        ),
        "actual_rank": (
            int(match["actualRank"]) if match.get("actualRank") is not None else None
        ),
        "league_id": int(league_id) if league_id is not None else None,
        "is_pro": bool(is_pro),
        "source": str(source),
        "top_lane_outcome": str(match["topLaneOutcome"]) if match.get("topLaneOutcome") else None,
        "mid_lane_outcome": str(match["midLaneOutcome"]) if match.get("midLaneOutcome") else None,
        "bottom_lane_outcome": (
            str(match["bottomLaneOutcome"]) if match.get("bottomLaneOutcome") else None
        ),
        "radiant_gold_lead_10m": _extract_lead_at_10m(match.get("radiantNetworthLeads")),
        "radiant_xp_lead_10m": _extract_lead_at_10m(match.get("radiantExperienceLeads")),
        "player_slots": player_slots,
        "is_radiant": is_radiant,
        "hero_ids": hero_ids,
        "positions": positions,
        "variant": variant,
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "last_hits": last_hits,
        "denies": denies,
        "gpm": gpm,
        "xpm": xpm,
        "networth": networth,
        "levels": levels,
        "hero_damage": hero_damage,
        "tower_damage": tower_damage,
        "hero_healing": hero_healing,
        "hero_damage_received": hero_damage_received,
        "final_items": final_items,
        "purchase_item_ids": purchase_item_ids,
        "purchase_times": purchase_times,
        "key_item_ids": key_item_ids,
        "key_item_times": key_item_times,
        "ability_learn_ids": ability_learn_ids,
        "ability_learn_times": ability_learn_times,
        "ability_learn_levels": ability_learn_levels,
        "ability_learn_ranks": ability_learn_ranks,
        "ability_learn_is_talent": ability_learn_is_talent,
    }

    return ValidationResult(
        valid=True,
        record=record,
        player_steam_ids=bfs_player_ids,
    )
