"""Patch-aware vocabulary builder for heroes, key items, abilities, and talents."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from shared.constants import ATTRIBUTE_BONUS_ABILITY_ID, VOCAB_SCHEMA_VERSION
from shared.dotaconstants import normalize_steam_cdn_url

SPECIAL_ITEM_TOKENS: dict[str, int] = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2}

# Intermediate pure-stat recipe components >= 800 gold that are never final build slots.
INTERMEDIATE_COMPONENT_NAMES: frozenset[str] = frozenset(
    {"pers", "oblivion_staff", "soul_booster"}
)

# Non-crafted or special-qual iconic slots that belong in hero key builds.
ICONIC_NON_CREATED_ITEMS: frozenset[str] = frozenset(
    {"blink", "aghanims_shard", "rapier", "moon_shard"}
)

# Canonical linked / stance-paired / temporary in-slot sub-ability aliases:
# STRATZ / Dota 2 event name -> canonical levelable slot name.
# Verified against live STRATZ abilityLearnEvents (e.g. SF emits nevermore_shadowraze1=5059,
# Io emits wisp_tether_break=5489 when leveling Tether while tethered, Morphling emits 5055/5058,
# Troll emits 5509, Dark Willow emits 6340, Kez emits 1498-1501 or 1502-1506).
LINKED_ABILITY_NAME_ALIASES: dict[str, str] = {
    "nevermore_shadowraze2": "nevermore_shadowraze1",
    "nevermore_shadowraze3": "nevermore_shadowraze1",
    "morphling_morph_str": "morphling_morph_agi",
    "morphling_morph_replicate": "morphling_replicate",
    "troll_warlord_whirling_axes_melee": "troll_warlord_whirling_axes_ranged",
    "dark_willow_terrorize": "dark_willow_bedlam",
    "kez_falcon_rush": "kez_echo_slash",
    "kez_talon_toss": "kez_grappling_claw",
    "kez_shodo_sai": "kez_kazurai_katana",
    "kez_shodo_sai_parry_cancel": "kez_kazurai_katana",
    "kez_ravens_veil": "kez_raptor_dance",
    "monkey_king_primal_spring": "monkey_king_tree_dance",
    "monkey_king_primal_spring_early": "monkey_king_tree_dance",
    "jakiro_liquid_ice": "jakiro_liquid_fire",
    "tusk_drinking_buddies": "tusk_tag_team",
    "tusk_launch_snowball": "tusk_snowball",
    "wisp_tether_break": "wisp_tether",
    "wisp_spirits_in": "wisp_spirits",
    "wisp_spirits_out": "wisp_spirits",
    "kunkka_return": "kunkka_x_marks_the_spot",
    "elder_titan_return_spirit": "elder_titan_ancestral_spirit",
    "alchemist_unstable_concoction_throw": "alchemist_unstable_concoction",
    "bane_nightmare_end": "bane_nightmare",
    "naga_siren_song_of_the_siren_cancel": "naga_siren_song_of_the_siren",
    "keeper_of_the_light_illuminate_end": "keeper_of_the_light_illuminate",
    "shredder_return_chakram": "shredder_chakram",
    "pangolier_gyroshell_stop": "pangolier_gyroshell",
    "hoodwink_sharpshooter_release": "hoodwink_sharpshooter",
    "phoenix_icarus_dive_stop": "phoenix_icarus_dive",
    "phoenix_sun_ray_stop": "phoenix_sun_ray",
    "phoenix_launch_fire_spirit": "phoenix_fire_spirits",
    "crystal_maiden_freezing_field_stop": "crystal_maiden_freezing_field",
    "ancient_apparition_ice_blast_release": "ancient_apparition_ice_blast",
    "life_stealer_consume": "life_stealer_infest",
    "primal_beast_onslaught_release": "primal_beast_onslaught",
    "dawnbreaker_converge": "dawnbreaker_celestial_hammer",
    "dawnbreaker_land": "dawnbreaker_solar_guardian",
    "dazzle_nothl_projection_end": "dazzle_nothl_projection",
    "rubick_telekinesis_land": "rubick_telekinesis",
    "tiny_toss_tree": "tiny_tree_grab",
    "techies_reactive_tazer_stop": "techies_reactive_tazer",
}

# ID fallbacks (src_id -> (dst_id, dst_name)) for linked sub-abilities missing from
# already-generated vocab abilities.by_id without bumping VOCAB_SCHEMA_VERSION.
MISSING_LINKED_ABILITY_ID_FALLBACKS: dict[int, tuple[int, str]] = {
    5490: (5486, "wisp_spirits"),  # wisp_spirits_in -> wisp_spirits
    5493: (5486, "wisp_spirits"),  # wisp_spirits_out -> wisp_spirits
}

# Abilities present in hero ability lists that cannot be leveled with hero skill points.
NON_LEVELABLE_ABILITY_NAMES: frozenset[str] = frozenset(
    {
        "generic_hidden",
        "special_bonus_attributes",
        "invoker_invoke",
        "invoker_empty1",
        "invoker_empty2",
        "kez_switch_weapons",
        "troll_warlord_switch_stance",
        "rubick_empty1",
        "rubick_empty2",
        "morphling_morph",
        "doom_bringer_empty1",
        "doom_bringer_empty2",
        "ringmaster_empty_souvenir",
        "ringmaster_spotlight",
        "drow_ranger_glacier",
        "juggernaut_swift_slash",
        "puck_ethereal_jaunt",
        "zuus_lightning_hands",
        "zuus_cloud",
        "lina_flame_cloak",
        "shadow_shaman_urnaconda",
        "tidehunter_dead_in_the_water",
        "tinker_keen_teleport",
        "tinker_warp_grenade",
        "sniper_concussive_grenade",
        "beastmaster_drums_of_slom",
        "templar_assassin_trap",
        "templar_assassin_trap_teleport",
        "viper_nose_dive",
        "rattletrap_jetpack_toggle",
        "rattletrap_overclocking",
        "leshrac_greater_lightning_storm",
        "furion_curse_of_the_forest",
        "clinkz_burning_barrage",
        "clinkz_burning_army",
        "enchantress_little_friends",
        "bounty_hunter_wind_walk_ally",
        "spectre_reality",
        "spirit_breaker_planar_pocket",
        "gyrocopter_side_gunner_spawn_ability",
        "shadow_demon_shadow_poison_release",
        "shadow_demon_demonic_cleanse",
        "meepo_megameepo",
        "meepo_megameepo_fling",
        "meepo_petrify",
        "naga_siren_reel_in",
        "centaur_work_horse",
        "centaur_mount",
        "tusk_walrus_kick",
        "tusk_launch_snowball",
        "terrorblade_terror_wave",
        "terrorblade_demon_zeal",
        "phoenix_sun_ray_toggle_move",
        "phoenix_icarus_dive_stop",
        "phoenix_sun_ray_stop",
        "phoenix_launch_fire_spirit",
        "oracle_rain_of_destiny",
        "oracle_diviners_deck",
        "snapfire_spit_creep",
        "snapfire_gobble_up",
        "muerta_spectral_slug",
        "visage_stone_form_self_cast",
        "visage_summon_familiars_stone_form",
        "ember_spirit_activate_fire_remnant",
        "monkey_king_primal_spring_early",
        "monkey_king_mischief",
        "monkey_king_untransform",
        "monkey_king_transfiguration",
        "bane_nightmare_end",
        "crystal_maiden_crystal_clone",
        "crystal_maiden_freezing_field_stop",
        "kunkka_tidal_wave",
        "kunkka_return",
        "wisp_tether_break",
        "tiny_tree_channel",
        "tiny_toss_tree",
        "windrunner_gale_force",
        "windrunner_focusfire_cancel",
        "ancient_apparition_ice_blast_release",
        "elder_titan_move_spirit",
        "elder_titan_return_spirit",
        "techies_minefield_sign",
        "techies_reactive_tazer_stop",
        "techies_focused_detonate",
        "earth_spirit_petrify",
        "pangolier_rollup",
        "pangolier_gyroshell_stop",
        "pangolier_rollup_stop",
        "grimstroke_dark_portrait",
        "hoodwink_decoy",
        "hoodwink_hunters_boomerang",
        "hoodwink_sharpshooter_release",
        "ringmaster_tame_the_beasts_crack",
        "dawnbreaker_converge",
        "dawnbreaker_land",
        "primal_beast_rock_throw",
        "primal_beast_onslaught_release",
        "largo_song_fight_song",
        "largo_song_double_time",
        "largo_song_good_vibrations",
        "kez_shodo_sai_parry_cancel",
        "keeper_of_the_light_radiant_bind",
        "keeper_of_the_light_will_o_wisp",
        "keeper_of_the_light_illuminate_end",
        "slark_depth_shroud",
        "medusa_cold_blooded",
        "magnataur_horn_toss",
        "shredder_flamethrower",
        "shredder_return_chakram",
        "bristleback_hairball",
        "lycan_wolf_bite",
        "dragon_knight_fireball",
        *LINKED_ABILITY_NAME_ALIASES.keys(),
    }
)

EXPLICIT_ULTIMATE_ABILITY_NAMES: frozenset[str] = frozenset(
    {
        "dragon_knight_elder_dragon_form",
        "brewmaster_primal_split",
        "meepo_divided_we_stand",
        "rattletrap_hookshot",
        "dark_willow_bedlam",
        "dark_willow_terrorize",
        "kez_raptor_dance",
        "kez_ravens_veil",
        "morphling_replicate",
        "troll_warlord_battle_trance",
        "clinkz_wind_walk",
        "enchantress_untouchable",
    }
)

MEEPO_HERO_ID = 82
INVOKER_HERO_ID = 74
INSTANT_COMPONENT_COMBINE_WINDOW_SEC = 5


def _ability_has_three_level_array(od_ab: dict[str, Any]) -> bool:
    """Return True when dotaconstants encodes strictly 3-level scaling (typical ultimate)."""
    arr_lens: set[int] = set()
    for key in ("cd", "mc", "dmg"):
        val = od_ab.get(key)
        if isinstance(val, list):
            arr_lens.add(len(val))
    for attr in od_ab.get("attrib") or []:
        if isinstance(attr, dict) and isinstance(attr.get("value"), list):
            arr_lens.add(len(attr["value"]))
    return 3 in arr_lens and 4 not in arr_lens


def _extract_behavior_tokens(od_ab: dict[str, Any]) -> set[str]:
    beh = od_ab.get("behavior")
    if isinstance(beh, str):
        return {beh}
    if isinstance(beh, list):
        return {str(tok) for tok in beh}
    return set()

# Upgrade tiers collapsed onto their canonical base key item to avoid build sequence spam.
ITEM_ALIAS_TARGETS: dict[str, str] = {
    "dagon_2": "dagon",
    "dagon_3": "dagon",
    "dagon_4": "dagon",
    "dagon_5": "dagon",
    "travel_boots_2": "travel_boots",
    "ultimate_scepter_2": "ultimate_scepter",
}

HIDDEN_ABILITY_PREFIXES: tuple[str, ...] = ("generic_hidden", "dota_base_ability", "bear_empty")
TALENT_LEVEL_TO_TIER: dict[int, str] = {1: "10", 2: "15", 3: "20", 4: "25"}


def _is_ignored_ability_name(name: str | None) -> bool:
    if not name:
        return True
    if name.startswith(HIDDEN_ABILITY_PREFIXES):
        return True
    return "_empty" in name


def _build_ability_lookup(
    stratz_abilities: list[dict[str, Any]],
    dotaconstants: dict[str, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    od_abilities = dotaconstants.get("abilities") or {}
    od_ability_ids = dotaconstants.get("ability_ids") or {}

    name_to_id: dict[str, int] = {}
    id_to_name_od: dict[int, str] = {}
    for raw_id, name in od_ability_ids.items():
        try:
            aid = int(raw_id)
        except (TypeError, ValueError):
            continue
        id_to_name_od[aid] = name
        name_to_id[name] = aid

    stratz_by_id: dict[int, dict[str, Any]] = {}
    for ab in stratz_abilities:
        aid = ab.get("id")
        if aid is None:
            continue
        aid_int = int(aid)
        stratz_by_id[aid_int] = ab
        if ab.get("name"):
            name_to_id.setdefault(ab["name"], aid_int)

    all_ids = set(stratz_by_id.keys()) | set(id_to_name_od.keys()) | {ATTRIBUTE_BONUS_ABILITY_ID}
    merged: dict[int, dict[str, Any]] = {}
    for aid in sorted(all_ids):
        sab = stratz_by_id.get(aid, {})
        name = sab.get("name") or id_to_name_od.get(aid) or f"ability_{aid}"
        od_ab = od_abilities.get(name) or {}
        lang_name = (sab.get("language") or {}).get("displayName")
        display_name = lang_name or od_ab.get("dname") or name
        is_talent = bool(sab.get("isTalent")) or name.startswith("special_bonus_")
        is_innate = bool(od_ab.get("is_innate")) or bool(sab.get("isInnate"))
        behaviors = _extract_behavior_tokens(od_ab)
        is_levelable = (
            not is_talent
            and not is_innate
            and name not in NON_LEVELABLE_ABILITY_NAMES
            and not ("Hidden" in behaviors and name not in LINKED_ABILITY_NAME_ALIASES)
        )
        is_ultimate = is_levelable and (
            name in EXPLICIT_ULTIMATE_ABILITY_NAMES
            or _ability_has_three_level_array(od_ab)
        )
        merged[aid] = {
            "id": aid,
            "name": name,
            "display_name": display_name,
            "is_talent": is_talent,
            "is_innate": is_innate,
            "is_levelable": is_levelable,
            "is_ultimate": is_ultimate,
            "img": normalize_steam_cdn_url(od_ab.get("img")),
        }

    return merged, name_to_id


def _build_heroes_section(
    stratz_heroes: list[dict[str, Any]],
    dotaconstants: dict[str, dict[str, Any]],
    ability_meta: dict[int, dict[str, Any]],
    ability_name_to_id: dict[str, int],
) -> tuple[dict[str, Any], set[int]]:
    od_heroes_raw = dotaconstants.get("heroes") or {}
    od_hero_abilities = dotaconstants.get("hero_abilities") or {}

    od_heroes_by_id: dict[int, dict[str, Any]] = {}
    for raw_key, hdata in od_heroes_raw.items():
        hid = hdata.get("id")
        if hid is None:
            try:
                hid = int(raw_key)
            except (TypeError, ValueError):
                continue
        od_heroes_by_id[int(hid)] = hdata

    sorted_heroes = sorted(stratz_heroes, key=lambda h: int(h["id"]))
    by_id: dict[str, dict[str, Any]] = {}
    id_to_idx: dict[str, int] = {}
    idx_to_id: list[int] = [0]  # 0 reserved for <PAD>
    referenced_ability_ids: set[int] = {ATTRIBUTE_BONUS_ABILITY_ID}

    for idx, hero in enumerate(sorted_heroes, start=1):
        hid = int(hero["id"])
        od_h = od_heroes_by_id.get(hid, {})
        npc_name = od_h.get("name") or f"npc_dota_hero_{hero['shortName']}"
        od_hab = od_hero_abilities.get(npc_name) or {}

        # Ordered learnable skill IDs from STRATZ hero.abilities (fallback to odota hero_abilities)
        skill_ids: list[int] = []
        seen_skills: set[int] = set()
        for ab in sorted(hero.get("abilities") or [], key=lambda a: int(a.get("slot") or 0)):
            aid = int(ab["abilityId"])
            ab_name = (ability_meta.get(aid) or {}).get("name")
            if _is_ignored_ability_name(ab_name) or aid in seen_skills:
                continue
            seen_skills.add(aid)
            skill_ids.append(aid)
            referenced_ability_ids.add(aid)

        if not skill_ids:
            for ab_name in od_hab.get("abilities") or []:
                if _is_ignored_ability_name(ab_name):
                    continue
                aid = ability_name_to_id.get(ab_name)
                if aid is not None and aid not in seen_skills:
                    seen_skills.add(aid)
                    skill_ids.append(aid)
                    referenced_ability_ids.add(aid)

        # Ensure any linked-alias ability IDs targeting/matching this hero's skills are retained in vocab
        reverse_aliases: dict[str, list[str]] = {}
        for src_nm, dst_nm in LINKED_ABILITY_NAME_ALIASES.items():
            reverse_aliases.setdefault(dst_nm, []).append(src_nm)
        for aid in list(skill_ids):
            ab_name = (ability_meta.get(aid) or {}).get("name") or ""
            for extra_nm in (
                *(reverse_aliases.get(ab_name) or []),
                *([LINKED_ABILITY_NAME_ALIASES[ab_name]] if ab_name in LINKED_ABILITY_NAME_ALIASES else []),
            ):
                extra_id = ability_name_to_id.get(extra_nm)
                if extra_id is not None:
                    referenced_ability_ids.add(extra_id)

        # Fallback for synthetic/minimal fixtures lacking 3-level scaling arrays:
        # treat the last levelable skill on non-Invoker heroes as the ultimate.
        levelable_ids = [
            aid for aid in skill_ids if (ability_meta.get(aid) or {}).get("is_levelable", True)
        ]
        if hid != INVOKER_HERO_ID and levelable_ids and not any(
            (ability_meta.get(aid) or {}).get("is_ultimate", False) for aid in levelable_ids
        ):
            ability_meta[levelable_ids[-1]]["is_ultimate"] = True

        # Talent tiers 10/15/20/25 merging STRATZ talents + dotaconstants hero_abilities.talents
        talent_tiers: dict[str, list[int]] = {"10": [], "15": [], "20": [], "25": []}
        for tal in sorted(hero.get("talents") or [], key=lambda t: int(t.get("slot") or 0)):
            aid = int(tal["abilityId"])
            slot = int(tal.get("slot") or 0)
            tier = str(10 + 5 * (slot // 2))
            if tier in talent_tiers and aid not in talent_tiers[tier]:
                talent_tiers[tier].append(aid)
                referenced_ability_ids.add(aid)

        for od_tal in od_hab.get("talents") or []:
            tier = TALENT_LEVEL_TO_TIER.get(int(od_tal.get("level") or 0))
            aid = ability_name_to_id.get(od_tal.get("name") or "")
            if tier and aid is not None and aid not in talent_tiers[tier]:
                talent_tiers[tier].append(aid)
                referenced_ability_ids.add(aid)

        by_id[str(hid)] = {
            "idx": idx,
            "id": hid,
            "short_name": hero["shortName"],
            "display_name": hero["displayName"],
            "primary_attr": od_h.get("primary_attr"),
            "attack_type": od_h.get("attack_type"),
            "roles": od_h.get("roles") or [],
            "img": normalize_steam_cdn_url(od_h.get("img")),
            "icon": normalize_steam_cdn_url(od_h.get("icon")),
            "skill_ability_ids": skill_ids,
            "talent_tiers": talent_tiers,
        }
        id_to_idx[str(hid)] = idx
        idx_to_id.append(hid)

    return (
        {
            "by_id": by_id,
            "id_to_idx": id_to_idx,
            "idx_to_id": idx_to_id,
            "num_heroes": len(sorted_heroes),
        },
        referenced_ability_ids,
    )


def _build_items_section(
    stratz_items: list[dict[str, Any]],
    dotaconstants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    od_items = dotaconstants.get("items") or {}
    od_by_id: dict[int, dict[str, Any]] = {
        int(v["id"]): v for v in od_items.values() if isinstance(v, dict) and v.get("id") is not None
    }
    stratz_by_short: dict[str, dict[str, Any]] = {
        it["shortName"]: it for it in stratz_items if it.get("shortName")
    }

    selected: list[dict[str, Any]] = []
    raw_components_by_id: dict[int, list[str]] = {}
    for it in sorted(stratz_items, key=lambda x: int(x["id"])):
        stat = it.get("stat") or {}
        if not stat.get("isPurchasable"):
            continue
        if stat.get("isRecipe") or stat.get("isStackable"):
            continue
        if stat.get("neutralItemTier") is not None:
            continue
        cost = int(stat.get("cost") or 0)
        if cost < 800:
            continue

        short = it["shortName"]
        if short in INTERMEDIATE_COMPONENT_NAMES or short in ITEM_ALIAS_TARGETS:
            continue

        item_id = int(it["id"])
        od_v = od_by_id.get(item_id) or od_items.get(short) or {}
        qual = od_v.get("qual")
        is_iconic = short in ICONIC_NON_CREATED_ITEMS
        if not is_iconic and qual in {"consumable", "secret_shop"}:
            continue

        created = bool(od_v.get("created"))
        is_support_full = bool(it.get("isSupportFullItem"))
        if not (created or is_support_full or is_iconic):
            continue

        raw_comps = [c for c in (od_v.get("components") or []) if isinstance(c, str)]
        raw_components_by_id[item_id] = raw_comps
        selected.append(
            {
                "id": item_id,
                "short_name": short,
                "display_name": it.get("displayName") or od_v.get("dname") or short,
                "cost": cost,
                "img": normalize_steam_cdn_url(od_v.get("img")),
            }
        )

    by_id: dict[str, dict[str, Any]] = {}
    id_to_idx: dict[str, int] = {}
    idx_to_id: list[int] = [0, 0, 0]  # <PAD>=0, <BOS>=1, <EOS>=2
    offset = len(SPECIAL_ITEM_TOKENS)

    for idx_offset, item in enumerate(selected):
        vocab_idx = offset + idx_offset
        iid = item["id"]
        by_id[str(iid)] = {"idx": vocab_idx, **item}
        id_to_idx[str(iid)] = vocab_idx
        idx_to_id.append(iid)

    aliases: dict[str, int] = {}
    for src_short, dst_short in ITEM_ALIAS_TARGETS.items():
        src_item = stratz_by_short.get(src_short)
        dst_item = stratz_by_short.get(dst_short)
        if src_item and dst_item and str(dst_item["id"]) in by_id:
            aliases[str(int(src_item["id"]))] = int(dst_item["id"])

    for iid_str, entry in by_id.items():
        built_from: list[int] = []
        seen_comps: set[int] = set()
        for comp_short in raw_components_by_id.get(int(iid_str), []):
            comp_item = od_items.get(comp_short) or stratz_by_short.get(comp_short)
            if not isinstance(comp_item, dict) or comp_item.get("id") is None:
                continue
            raw_cid = int(comp_item["id"])
            canon_cid = int(aliases.get(str(raw_cid), raw_cid))
            if str(canon_cid) in by_id and canon_cid != int(iid_str) and canon_cid not in seen_comps:
                seen_comps.add(canon_cid)
                built_from.append(canon_cid)
        if built_from:
            entry["built_from_ids"] = built_from

    return {
        "special_tokens": dict(SPECIAL_ITEM_TOKENS),
        "by_id": by_id,
        "id_to_idx": id_to_idx,
        "idx_to_id": idx_to_id,
        "aliases": aliases,
        "vocab_size": len(idx_to_id),
        "num_key_items": len(selected),
    }


def build_vocab(
    game_version: dict[str, Any],
    stratz_constants: dict[str, Any],
    dotaconstants: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build patch-specific vocab.json combining STRATZ constants and odota/dotaconstants."""
    ability_meta, ability_name_to_id = _build_ability_lookup(
        stratz_constants.get("abilities") or [],
        dotaconstants,
    )
    heroes_section, referenced_ability_ids = _build_heroes_section(
        stratz_constants.get("heroes") or [],
        dotaconstants,
        ability_meta,
        ability_name_to_id,
    )
    items_section = _build_items_section(
        stratz_constants.get("items") or [],
        dotaconstants,
    )

    abilities_by_id: dict[str, dict[str, Any]] = {}
    talent_ids: list[int] = []
    for aid in sorted(referenced_ability_ids):
        meta = ability_meta.get(aid)
        if not meta:
            continue
        abilities_by_id[str(aid)] = meta
        if meta["is_talent"]:
            talent_ids.append(aid)

    ability_aliases: dict[str, int] = {}
    for src_name, dst_name in LINKED_ABILITY_NAME_ALIASES.items():
        src_id = ability_name_to_id.get(src_name)
        dst_id = ability_name_to_id.get(dst_name)
        if src_id is not None and dst_id is not None:
            ability_aliases[str(int(src_id))] = int(dst_id)

    return {
        "schema_version": VOCAB_SCHEMA_VERSION,
        "game_version_id": int(game_version["id"]),
        "game_version_name": str(game_version["name"]),
        "as_of_datetime": int(game_version["asOfDateTime"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "heroes": heroes_section,
        "items": items_section,
        "abilities": {
            "by_id": abilities_by_id,
            "talent_ids": talent_ids,
            "aliases": ability_aliases,
            "attribute_bonus_id": ATTRIBUTE_BONUS_ABILITY_ID,
        },
    }


def extract_key_item_purchases(
    purchase_events: list[dict[str, Any]],
    vocab: dict[str, Any],
    *,
    deduplicate: bool = True,
) -> tuple[list[int], list[int]]:
    """Filter chronological purchase events to key build items defined in vocab."""
    items_section = vocab["items"]
    valid_ids = items_section["by_id"]
    aliases = items_section.get("aliases") or {}

    sorted_events = sorted(purchase_events, key=lambda ev: int(ev.get("time") or 0))
    seen: set[int] = set()
    key_ids: list[int] = []
    key_times: list[int] = []

    for ev in sorted_events:
        raw_id = ev.get("itemId")
        if raw_id is None:
            continue
        canonical_id = int(aliases.get(str(int(raw_id)), int(raw_id)))
        if str(canonical_id) not in valid_ids:
            continue
        if deduplicate and canonical_id in seen:
            continue
        seen.add(canonical_id)
        t = int(ev.get("time") or 0)
        t_clamped = max(-3600, min(32000, t))
        key_ids.append(canonical_id)
        key_times.append(t_clamped)

    # Collapse transient intermediate key components that auto-combined into a parent item within <= 5s.
    if len(key_ids) >= 2:
        drop_indices: set[int] = set()
        for j, parent_id in enumerate(key_ids):
            built_from = (valid_ids.get(str(parent_id)) or {}).get("built_from_ids") or []
            if not built_from:
                continue
            comp_set = {int(c) for c in built_from}
            t_parent = key_times[j]
            for i, comp_id in enumerate(key_ids):
                if i == j or i in drop_indices:
                    continue
                if comp_id in comp_set and abs(t_parent - key_times[i]) <= INSTANT_COMPONENT_COMBINE_WINDOW_SEC:
                    drop_indices.add(i)
        if drop_indices:
            key_ids = [kid for idx, kid in enumerate(key_ids) if idx not in drop_indices]
            key_times = [kt for idx, kt in enumerate(key_times) if idx not in drop_indices]

    return key_ids, key_times
