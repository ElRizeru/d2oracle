"""Match validation and canonical transformation into Parquet-ready records."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from shared.constants import (
    ATTRIBUTE_BONUS_ABILITY_ID,
    IMMORTAL_BRACKET,
    MIN_DURATION_SECONDS,
    MISSING_STAT_SENTINEL,
    POSITION_TO_INT,
    PRO_GAME_MODE,
    STEAM_ALLOWED_LEAVER_STATUSES,
    STEAM_ANONYMOUS_ACCOUNT_ID,
    STEAM_GAME_MODE_TO_NAME,
    STEAM_LOBBY_TYPE_TO_NAME,
    STRATZ_LEADS_10M_INDEX,
    VALID_PUB_GAME_MODES,
    infer_skill_point_level,
    infer_talent_point_level,
)
from shared.vocab import ITEM_ALIAS_TARGETS, extract_key_item_purchases

ALLOWED_LEAVER_STATUSES: frozenset[str] = frozenset({"NONE", "DISCONNECTED"})
MAX_PLAYBACK_RETRY_AGE_SECONDS: int = 6 * 3600


def _resolve_upgrade_rank_and_level(
    *,
    hero_id: int,
    aid: int,
    canon_aid: int,
    is_tal: bool,
    raw_lvl: Any,
    ab_by_id: dict[str, Any],
    rank_counts: dict[int, int],
    skill_idx: int,
    talent_idx: int,
    last_skill_lvl: int,
) -> tuple[int, int, int, int, int]:
    """Compute 1-based rank, 7.40+ hero level, and updated skill/talent cursors."""
    if is_tal:
        rank = 1
        lvl = int(raw_lvl) if raw_lvl is not None else infer_talent_point_level(talent_idx)
        return rank, lvl, skill_idx, talent_idx + 1, last_skill_lvl

    if aid == ATTRIBUTE_BONUS_ABILITY_ID:
        rank = 1
        is_ult = False
    else:
        rank = rank_counts.get(canon_aid, 0) + 1
        rank_counts[canon_aid] = rank
        ab_meta = ab_by_id.get(str(canon_aid)) or ab_by_id.get(str(aid)) or {}
        is_ult = bool(ab_meta.get("is_ultimate")) and not bool(ab_meta.get("is_innate"))

    lvl = (
        int(raw_lvl)
        if raw_lvl is not None
        else infer_skill_point_level(
            hero_id,
            skill_idx,
            is_ultimate=is_ult,
            rank=rank,
            last_skill_level=last_skill_lvl,
        )
    )
    return rank, lvl, skill_idx + 1, talent_idx, lvl


def _extract_player_timelines(
    player: dict[str, Any],
    *,
    require_abilities: bool = True,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, bool, str | None]:
    """Extract purchase and ability timelines from playbackData (legacy) or abilities+stats.itemPurchases."""
    pb = player.get("playbackData")
    if isinstance(pb, dict):
        purchases = pb.get("purchaseEvents")
        abilities = pb.get("abilityLearnEvents")
        if not purchases:
            return None, None, False, "empty_purchase_events"
        if require_abilities and not abilities:
            return None, None, False, "empty_ability_events"
        return purchases, abilities if isinstance(abilities, list) else [], False, None

    stats = player.get("stats")
    purchases = stats.get("itemPurchases") if isinstance(stats, dict) else None
    abilities = player.get("abilities")
    if purchases is None and abilities is None:
        return None, None, False, "missing_playback_data"
    if not purchases:
        return None, None, False, "empty_purchase_events"
    if require_abilities and not abilities:
        return None, None, False, "empty_ability_events"
    return purchases, abilities if isinstance(abilities, list) else [], True, None


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a raw match payload."""

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


def prevalidate_stratz_match(
    match: dict[str, Any] | None,
    *,
    active_gv_id: int,
    allowed_gv_ids: tuple[int, ...] | set[int] | list[int] | None = None,
    min_start_datetime: int = 0,
    max_start_datetime: int | None = None,
    now_ts: float | None = None,
    require_stratz_abilities: bool = False,
) -> ValidationResult:
    """Validate STRATZ match metadata, roles, teams, and purchase timelines before Steam enrichment."""
    if match is None:
        return ValidationResult(valid=False, retryable=True, reason="not_found")

    if match.get("parsedDateTime") is None:
        return ValidationResult(valid=False, retryable=True, reason="unparsed")

    valid_gvs = {int(g) for g in allowed_gv_ids} if allowed_gv_ids else {int(active_gv_id)}
    gv_id = match.get("gameVersionId")
    if gv_id is None or int(gv_id) not in valid_gvs:
        return ValidationResult(valid=False, retryable=False, reason="wrong_game_version")

    start_dt = int(match.get("startDateTime") or 0)
    if min_start_datetime > 0 and start_dt < int(min_start_datetime):
        return ValidationResult(valid=False, retryable=False, reason="before_patch_start")
    if max_start_datetime is not None and start_dt >= int(max_start_datetime):
        return ValidationResult(valid=False, retryable=False, reason="after_patch_end")

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
        _purchases, _abilities, _inferred, err_reason = _extract_player_timelines(
            p, require_abilities=require_stratz_abilities
        )
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

    return ValidationResult(valid=True)


def _align_steam_ability_upgrades(
    steam_upgrades: list[dict[str, Any]],
    stratz_abilities: list[dict[str, Any]] | None,
    *,
    hero_id: int,
    vocab: dict[str, Any],
    pre_game_duration: int = 90,
) -> tuple[list[int], list[int], list[int], list[int], list[bool]]:
    """Convert Steam ability_upgrades into aligned horn-relative timelines with 1-based ranks."""
    abilities_sec = vocab.get("abilities") or {}
    ab_by_id = abilities_sec.get("by_id") or {}
    talent_id_set = {int(x) for x in (abilities_sec.get("talent_ids") or [])}
    ab_by_id = abilities_sec.get("by_id") or {}
    ab_aliases = {int(k): int(v) for k, v in (abilities_sec.get("aliases") or {}).items()}

    steam_sorted = sorted(
        (
            ev
            for ev in steam_upgrades
            if isinstance(ev, dict) and ev.get("ability") is not None and int(ev["ability"]) > 0
        ),
        key=lambda ev: (int(ev.get("level") or 0), int(ev.get("time") or 0)),
    )

    stratz_raw = [
        ev
        for ev in (stratz_abilities or [])
        if isinstance(ev, dict) and ev.get("abilityId") is not None
    ]
    if any(ev.get("levelObtained") is not None for ev in stratz_raw):
        stratz_sorted = sorted(
            stratz_raw,
            key=lambda ev: (int(ev.get("levelObtained") or 0), int(ev.get("time") or 0)),
        )
    else:
        stratz_sorted = sorted(stratz_raw, key=lambda ev: int(ev.get("time") or 0))

    s_idx = 0
    last_stratz_t: int | None = None
    last_steam_t: int | None = None
    rank_counts: dict[int, int] = {}
    skill_idx = 0
    talent_idx = 0
    last_skill_lvl = 0

    a_ids: list[int] = []
    a_times: list[int] = []
    a_levels: list[int] = []
    a_ranks: list[int] = []
    a_talents: list[bool] = []

    for sev in steam_sorted:
        aid = int(sev["ability"])
        steam_t = int(sev.get("time") or 0)
        canon_aid = ab_aliases.get(aid, aid)
        is_tal = aid != ATTRIBUTE_BONUS_ABILITY_ID and (
            aid in talent_id_set
            or bool((ab_by_id.get(str(aid)) or {}).get("is_talent"))
        )

        matched_j: int | None = None
        for j in range(s_idx, min(len(stratz_sorted), s_idx + 3)):
            s_aid = int(stratz_sorted[j].get("abilityId") or 0)
            s_canon = ab_aliases.get(s_aid, s_aid)
            if s_aid == aid or s_canon == canon_aid or (s_aid == 0 and is_tal):
                matched_j = j
                break

        if matched_j is not None:
            t = int(stratz_sorted[matched_j].get("time") or 0)
            last_stratz_t = t
            last_steam_t = steam_t
            s_idx = matched_j + 1
        elif last_stratz_t is not None and last_steam_t is not None:
            t = last_stratz_t + max(0, steam_t - last_steam_t)
        else:
            t = steam_t - int(pre_game_duration)

        rank, lvl, skill_idx, talent_idx, last_skill_lvl = _resolve_upgrade_rank_and_level(
            hero_id=hero_id,
            aid=aid,
            canon_aid=canon_aid,
            is_tal=is_tal,
            raw_lvl=None,
            ab_by_id=ab_by_id,
            rank_counts=rank_counts,
            skill_idx=skill_idx,
            talent_idx=talent_idx,
            last_skill_lvl=last_skill_lvl,
        )

        a_ids.append(aid)
        a_times.append(_clamp(t, -3600, 32000))
        a_levels.append(_clamp(lvl, 0, 30))
        a_ranks.append(_clamp(rank, 1, 10))
        a_talents.append(is_tal)

    return a_ids, a_times, a_levels, a_ranks, a_talents


def _steam_final_key_item_ids(steam_player: dict[str, Any], vocab: dict[str, Any]) -> set[int]:
    """Return canonical vocab key item IDs present in a Steam player's inventory or backpack."""
    items_sec = vocab.get("items") or {}
    valid_ids = items_sec.get("by_id") or {}
    aliases = {int(k): int(v) for k, v in (items_sec.get("aliases") or {}).items()}
    key_ids: set[int] = set()
    slots = (
        "item_0",
        "item_1",
        "item_2",
        "item_3",
        "item_4",
        "item_5",
        "backpack_0",
        "backpack_1",
        "backpack_2",
    )
    for key in slots:
        raw = steam_player.get(key)
        if not raw:
            continue
        canon = aliases.get(int(raw), int(raw))
        if str(canon) in valid_ids:
            key_ids.add(canon)
    return key_ids


def find_missing_key_items_from_finals(
    stratz_match: dict[str, Any],
    steam_match: dict[str, Any],
    vocab: dict[str, Any],
) -> set[int]:
    """Find key items held at match end in Steam that STRATZ omitted from purchase timelines."""
    items_sec = vocab.get("items") or {}
    aliases = {int(k): int(v) for k, v in (items_sec.get("aliases") or {}).items()}

    steam_by_slot = {
        int(sp["player_slot"]): sp
        for sp in (steam_match.get("players") or [])
        if isinstance(sp, dict) and sp.get("player_slot") is not None
    }
    missing: set[int] = set()
    for p in stratz_match.get("players") or []:
        slot = p.get("playerSlot")
        if slot is None or int(slot) not in steam_by_slot:
            continue
        sp = steam_by_slot[int(slot)]
        final_key_ids = _steam_final_key_item_ids(sp, vocab)
        if not final_key_ids:
            continue
        purchases, _abs, _inf, _err = _extract_player_timelines(p, require_abilities=False)
        bought_canon = {
            aliases.get(int(ev["itemId"]), int(ev["itemId"]))
            for ev in (purchases or [])
            if ev.get("itemId") is not None
        }
        missing |= (final_key_ids - bought_canon)
    return missing


def _merge_opendota_missing_key_purchases(
    stratz_purchases: list[dict[str, Any]],
    steam_player: dict[str, Any] | None,
    opendota_player: dict[str, Any] | None,
    vocab: dict[str, Any],
) -> list[dict[str, Any]]:
    """Supplement STRATZ purchase events with OpenDota purchase_log for new-patch items."""
    if steam_player is None or not isinstance(opendota_player, dict):
        return stratz_purchases
    purchase_log = opendota_player.get("purchase_log")
    if not isinstance(purchase_log, list) or not purchase_log:
        return stratz_purchases

    items_sec = vocab.get("items") or {}
    by_id = items_sec.get("by_id") or {}
    aliases = {int(k): int(v) for k, v in (items_sec.get("aliases") or {}).items()}
    bought_canon = {
        aliases.get(int(ev["itemId"]), int(ev["itemId"]))
        for ev in stratz_purchases
        if ev.get("itemId") is not None
    }
    missing_key_ids = _steam_final_key_item_ids(steam_player, vocab) - bought_canon
    if not missing_key_ids:
        return stratz_purchases

    short_to_canon: dict[str, int] = {}
    for iid_str, meta in by_id.items():
        if isinstance(meta, dict) and meta.get("short_name"):
            short_to_canon[str(meta["short_name"])] = int(iid_str)
    for src_short, dst_short in ITEM_ALIAS_TARGETS.items():
        if dst_short in short_to_canon:
            short_to_canon[src_short] = short_to_canon[dst_short]

    merged = list(stratz_purchases)
    for ev in purchase_log:
        if not isinstance(ev, dict):
            continue
        key_short = str(ev.get("key") or "").removeprefix("item_")
        mapped_id = short_to_canon.get(key_short)
        if mapped_id is not None and mapped_id in missing_key_ids:
            merged.append({"time": int(ev.get("time") or 0), "itemId": mapped_id})
    return merged


def validate_and_transform_match(
    match: dict[str, Any] | None,
    *,
    active_gv_id: int,
    active_gv_name: str,
    source: str,
    vocab: dict[str, Any],
    steam_match: dict[str, Any] | None = None,
    opendota_match: dict[str, Any] | None = None,
    allowed_gv_ids: tuple[int, ...] | set[int] | list[int] | None = None,
    min_start_datetime: int = 0,
    max_start_datetime: int | None = None,
    now_ts: float | None = None,
) -> ValidationResult:
    """Validate and merge STRATZ + Steam (+ optional OpenDota) match payloads into a Parquet row."""
    pre = prevalidate_stratz_match(
        match,
        active_gv_id=active_gv_id,
        allowed_gv_ids=allowed_gv_ids,
        min_start_datetime=min_start_datetime,
        max_start_datetime=max_start_datetime,
        now_ts=now_ts,
        require_stratz_abilities=(steam_match is None),
    )
    if not pre.valid or match is None:
        return pre

    players = match["players"]
    duration = int(match["durationSeconds"])
    did_radiant_win = bool(match["didRadiantWin"])
    game_mode = str(match.get("gameMode") or "")
    lobby_type = str(match.get("lobbyType") or "")
    league_id = match.get("leagueId")

    steam_by_slot: dict[int, dict[str, Any]] = {}
    if steam_match is not None:
        steam_mid = steam_match.get("match_id")
        if steam_mid is None or int(steam_mid) != int(match["id"]):
            return ValidationResult(valid=False, retryable=False, reason="steam_stratz_mismatch")
        steam_dur = steam_match.get("duration")
        if steam_dur is None or int(steam_dur) < MIN_DURATION_SECONDS:
            return ValidationResult(valid=False, retryable=False, reason="short_duration")
        steam_players = steam_match.get("players")
        if not isinstance(steam_players, list) or len(steam_players) != 10:
            return ValidationResult(valid=False, retryable=False, reason="steam_stratz_mismatch")
        for sp in steam_players:
            if not isinstance(sp, dict) or sp.get("player_slot") is None:
                return ValidationResult(valid=False, retryable=False, reason="steam_stratz_mismatch")
            steam_by_slot[int(sp["player_slot"])] = sp

        for p in players:
            slot = int(p.get("playerSlot") if p.get("playerSlot") is not None else -1)
            sp = steam_by_slot.get(slot)
            if sp is None or int(sp.get("hero_id") or 0) != int(p["heroId"]):
                return ValidationResult(valid=False, retryable=False, reason="steam_stratz_mismatch")
            if sp.get("leaver_status") not in (None, *STEAM_ALLOWED_LEAVER_STATUSES):
                return ValidationResult(valid=False, retryable=False, reason="leaver_detected")
            upgrades = sp.get("ability_upgrades")
            if not isinstance(upgrades, list) or not any(
                isinstance(ev, dict) and ev.get("ability") and int(ev["ability"]) > 0
                for ev in upgrades
            ):
                return ValidationResult(valid=False, retryable=False, reason="empty_ability_events")

        duration = int(steam_dur)
        if steam_match.get("radiant_win") is not None:
            did_radiant_win = bool(steam_match["radiant_win"])
        if steam_match.get("game_mode") is not None:
            game_mode = STEAM_GAME_MODE_TO_NAME.get(
                int(steam_match["game_mode"]), game_mode
            )
        if steam_match.get("lobby_type") is not None:
            lobby_type = STEAM_LOBBY_TYPE_TO_NAME.get(
                int(steam_match["lobby_type"]), lobby_type
            )
        steam_league = steam_match.get("leagueid")
        if steam_league is not None and int(steam_league) > 0:
            league_id = int(steam_league)

    is_pro = (league_id is not None) and (game_mode == PRO_GAME_MODE)

    od_by_slot: dict[int, dict[str, Any]] = {}
    if isinstance(opendota_match, dict):
        for odp in opendota_match.get("players") or []:
            if isinstance(odp, dict) and odp.get("player_slot") is not None:
                od_by_slot[int(odp["player_slot"])] = odp

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

    bfs_player_ids_set: set[int] = set()
    abilities_sec = vocab.get("abilities") or {}
    ab_by_id = abilities_sec.get("by_id") or {}
    talent_id_set = {int(x) for x in (abilities_sec.get("talent_ids") or [])}
    ab_aliases = {int(k): int(v) for k, v in (abilities_sec.get("aliases") or {}).items()}
    pre_game_dur = int(steam_match.get("pre_game_duration") or 90) if steam_match else 90

    for p in ordered:
        sid = int(p.get("steamAccountId") or 0)
        if 0 < sid != STEAM_ANONYMOUS_ACCOUNT_ID:
            bfs_player_ids_set.add(sid)

        stratz_slot = int(p.get("playerSlot") or 0)
        sp = steam_by_slot.get(stratz_slot) if steam_by_slot else None
        od_p = od_by_slot.get(stratz_slot)

        if sp is not None:
            steam_aid = int(sp.get("account_id") or 0)
            if 0 < steam_aid != STEAM_ANONYMOUS_ACCOUNT_ID:
                bfs_player_ids_set.add(steam_aid)

            pslot = _clamp(sp.get("player_slot"), 0, 255)
            rad_flag = pslot < 128
            hero_id_int = _clamp(sp.get("hero_id"), 1, 32767)
            raw_var = sp.get("hero_variant") if sp.get("hero_variant") is not None else p.get("variant")
            player_slots.append(pslot)
            is_radiant.append(rad_flag)
            hero_ids.append(hero_id_int)
            positions.append(POSITION_TO_INT[p["position"]])
            variant.append(_clamp(raw_var, -128, 127, default=0))
            kills.append(_clamp(sp.get("kills"), 0, 32767))
            deaths.append(_clamp(sp.get("deaths"), 0, 32767))
            assists.append(_clamp(sp.get("assists"), 0, 32767))
            last_hits.append(_clamp(sp.get("last_hits"), 0, 32767))
            denies.append(_clamp(sp.get("denies"), 0, 32767))
            gpm.append(_clamp(sp.get("gold_per_min"), 0, 32767))
            xpm.append(_clamp(sp.get("xp_per_min"), 0, 32767))
            networth.append(max(0, int(sp.get("net_worth") or 0)))
            levels.append(_clamp(sp.get("level"), 1, 30, default=1))
            hero_damage.append(_clamp(sp.get("hero_damage"), 0, 2_000_000_000))
            tower_damage.append(_clamp(sp.get("tower_damage"), 0, 2_000_000_000))
            hero_healing.append(_clamp(sp.get("hero_healing"), 0, 2_000_000_000))
            final_items.append(
                [
                    int(sp.get("item_0") or 0),
                    int(sp.get("item_1") or 0),
                    int(sp.get("item_2") or 0),
                    int(sp.get("item_3") or 0),
                    int(sp.get("item_4") or 0),
                    int(sp.get("item_5") or 0),
                    int(sp.get("item_neutral") or 0),
                ]
            )
        else:
            hero_id_int = _clamp(p.get("heroId"), 1, 32767)
            player_slots.append(_clamp(p.get("playerSlot"), 0, 255))
            is_radiant.append(bool(p["isRadiant"]))
            hero_ids.append(hero_id_int)
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

        hero_damage_received.append(_extract_hero_damage_received(p))

        raw_purchases_src, raw_abilities_src, infer_levels, _err = _extract_player_timelines(
            p, require_abilities=(sp is None)
        )
        assert raw_purchases_src is not None
        merged_purchases = _merge_opendota_missing_key_purchases(
            raw_purchases_src, sp, od_p, vocab
        )
        raw_purchases = sorted(
            merged_purchases, key=lambda ev: int(ev.get("time") or 0)
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

        if sp is not None:
            a_ids, a_times, a_levels, a_ranks, a_talents = _align_steam_ability_upgrades(
                sp["ability_upgrades"],
                raw_abilities_src,
                hero_id=hero_id_int,
                vocab=vocab,
                pre_game_duration=pre_game_dur,
            )
        else:
            assert raw_abilities_src is not None
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

            a_ids = []
            a_times = []
            a_levels = []
            a_ranks = []
            a_talents = []
            rank_counts: dict[int, int] = {}
            skill_idx = 0
            talent_idx = 0
            last_skill_lvl = 0
            for ev in raw_abilities:
                if ev.get("abilityId") is None:
                    continue
                aid = int(ev["abilityId"])
                canon_aid = ab_aliases.get(aid, aid)
                is_tal = aid != ATTRIBUTE_BONUS_ABILITY_ID and (
                    bool(ev.get("isTalent")) or aid in talent_id_set
                )
                raw_lvl = ev.get("levelObtained")
                rank, inferred_lvl, skill_idx, talent_idx, last_skill_lvl = (
                    _resolve_upgrade_rank_and_level(
                        hero_id=hero_id_int,
                        aid=aid,
                        canon_aid=canon_aid,
                        is_tal=is_tal,
                        raw_lvl=raw_lvl,
                        ab_by_id=ab_by_id,
                        rank_counts=rank_counts,
                        skill_idx=skill_idx,
                        talent_idx=talent_idx,
                        last_skill_lvl=last_skill_lvl,
                    )
                )
                a_ids.append(aid)
                a_times.append(_clamp(ev.get("time"), -3600, 32000))
                a_levels.append(_clamp(inferred_lvl, 0, 30))
                a_ranks.append(_clamp(rank, 1, 10))
                a_talents.append(is_tal)

        ability_learn_ids.append(a_ids)
        ability_learn_times.append(a_times)
        ability_learn_levels.append(a_levels)
        ability_learn_ranks.append(a_ranks)
        ability_learn_is_talent.append(a_talents)

    start_datetime = (
        int(steam_match.get("start_time") or match.get("startDateTime") or 0)
        if steam_match is not None
        else int(match.get("startDateTime") or 0)
    )

    record: dict[str, Any] = {
        "match_id": int(match["id"]),
        "start_datetime": start_datetime,
        "duration_seconds": duration,
        "radiant_win": did_radiant_win,
        "game_version_id": int(active_gv_id),
        "game_version_name": str(active_gv_name),
        "game_mode": game_mode,
        "lobby_type": lobby_type,
        "bracket": int(match["bracket"]),
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
        player_steam_ids=sorted(bfs_player_ids_set),
    )
