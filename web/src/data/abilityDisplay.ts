import type { AbilityMeta, HeroMeta, SkillLevelStep } from "../types/oracle";

const BASIC_SKILL_LABELS = ["Q", "W", "E", "D", "F", "G"] as const;

const INVOKER_HERO_ID = 74;

/**
 * Explicit ultimate ability names for multi-form or non-standard-slot heroes
 * where more than one ultimate entry appears or the ultimate is not uniquely last.
 */
export const EXPLICIT_ULTIMATE_ABILITY_NAMES: ReadonlySet<string> = new Set([
  "dark_willow_bedlam",
  "dark_willow_terrorize",
  "kez_raptor_dance",
  "kez_ravens_veil",
  "meepo_divided_we_stand",
  "morphling_replicate",
]);

/**
 * Explicit Dota hotkey slot labels for abilities whose positional filtered index
 * does not match their in-game hotkey (e.g., Dark Willow D/R paired ultimates,
 * Kez Katana/Sai stance mirror Q/W/E/R).
 */
const EXPLICIT_SLOT_LABEL_BY_NAME: Readonly<Record<string, string>> = {
  dark_willow_bedlam: "D",
  dark_willow_terrorize: "R",
  kez_echo_slash: "Q",
  kez_grappling_claw: "W",
  kez_kazurai_katana: "E",
  kez_raptor_dance: "R",
  kez_falcon_rush: "Q",
  kez_talon_toss: "W",
  kez_shodo_sai: "E",
  kez_ravens_veil: "R",
};

/**
 * Innate, passive auto-granted, toggle, cancel, and parent-scaled sub-abilities
 * present in hero skill_ability_ids inside vocab.json (sourced from Dota 2 constants)
 * that players never allocate skill points into on levels 1 to 18 and whose icons
 * are frequently absent on Steam's dota_react CDN.
 */
export const NON_LEVELABLE_ABILITY_NAMES: ReadonlySet<string> = new Set([
  "abaddon_withering_mist",
  "abyssal_underlord_raid_boss",
  "alchemist_berserk_potion",
  "alchemist_goblins_greed",
  "alchemist_unstable_concoction_throw",
  "ancient_apparition_bone_chill",
  "ancient_apparition_ice_blast_release",
  "antimage_persectur",
  "arc_warden_runic_infusion",
  "axe_one_man_army",
  "bane_ichor_of_nyctasha",
  "bane_nightmare_end",
  "batrider_smoldering_resin",
  "batrider_sticky_napalm_application_damage",
  "beastmaster_drums_of_slom",
  "beastmaster_inner_beast",
  "bloodseeker_sanguivore",
  "bounty_hunter_big_game_hunter",
  "bounty_hunter_wind_walk_ally",
  "brewmaster_liquid_courage",
  "bristleback_hairball",
  "bristleback_prickly",
  "broodmother_spiders_milk",
  "broodmother_sticky_snare",
  "centaur_horsepower",
  "centaur_mount",
  "centaur_work_horse",
  "chaos_knight_fundamental_forging",
  "chen_zealot",
  "clinkz_burning_army",
  "clinkz_burning_barrage",
  "clinkz_infernal_shred",
  "crystal_maiden_crystal_clone",
  "crystal_maiden_freezing_field_stop",
  "crystal_maiden_glacial_guard",
  "dark_seer_aggrandize",
  "dark_seer_normal_punch",
  "dark_willow_pixie_dust",
  "dawnbreaker_break_of_dawn",
  "dawnbreaker_converge",
  "dawnbreaker_land",
  "dazzle_innate_weave",
  "dazzle_nothl_projection_end",
  "death_prophet_witchcraft",
  "disruptor_electromagnetic_repulsion",
  "disruptor_kinetic_fence",
  "doom_bringer_lvl_pain",
  "dragon_knight_dragon_blood",
  "dragon_knight_fireball",
  "drow_ranger_glacier",
  "drow_ranger_trueshot",
  "earth_spirit_petrify",
  "earth_spirit_stone_caller",
  "earthshaker_slugger",
  "elder_titan_momentum",
  "elder_titan_move_spirit",
  "elder_titan_return_spirit",
  "ember_spirit_activate_fire_remnant",
  "ember_spirit_immolation",
  "enchantress_bunny_hop",
  "enchantress_little_friends",
  "enchantress_rabblerouser",
  "enigma_event_horizon",
  "faceless_void_distortion_field",
  "faceless_void_time_walk_reverse",
  "furion_curse_of_the_forest",
  "furion_spirit_of_the_forest",
  "grimstroke_dark_portrait",
  "grimstroke_ink_trail",
  "gyrocopter_afterburner",
  "gyrocopter_side_gunner_spawn_ability",
  "hoodwink_decoy",
  "hoodwink_hunters_boomerang",
  "hoodwink_mistwoods_wayfarer",
  "hoodwink_sharpshooter_release",
  "huskar_blood_magic",
  "invoker_alacrity",
  "invoker_chaos_meteor",
  "invoker_cold_snap",
  "invoker_deafening_blast",
  "invoker_emp",
  "invoker_forge_spirit",
  "invoker_ghost_walk",
  "invoker_ice_wall",
  "invoker_invoke",
  "invoker_sun_strike",
  "invoker_tornado",
  "jakiro_double_trouble",
  "jakiro_liquid_ice",
  "juggernaut_bladeform",
  "juggernaut_swift_slash",
  "keeper_of_the_light_bright_speed",
  "keeper_of_the_light_illuminate_end",
  "keeper_of_the_light_radiant_bind",
  "keeper_of_the_light_will_o_wisp",
  "kez_shodo_sai_parry_cancel",
  "kez_switch_weapons",
  "kunkka_admirals_rum",
  "kunkka_return",
  "kunkka_tidal_wave",
  "largo_encore",
  "largo_song_double_time",
  "largo_song_fight_song",
  "largo_song_good_vibrations",
  "legion_commander_outfight_them",
  "leshrac_defilement",
  "leshrac_greater_lightning_storm",
  "lich_death_charge",
  "lich_ice_spire",
  "life_stealer_consume",
  "life_stealer_ghoul_frenzy",
  "lina_flame_cloak",
  "lina_slow_burn",
  "lion_to_hell_and_back",
  "lone_druid_spirit_bear",
  "luna_lunar_blessing",
  "lycan_apex_predator",
  "lycan_wolf_bite",
  "magnataur_horn_toss",
  "magnataur_solid_core",
  "marci_special_delivery",
  "mars_dauntless",
  "medusa_cold_blooded",
  "medusa_mana_shield",
  "meepo_geomancy",
  "meepo_megameepo",
  "meepo_megameepo_fling",
  "meepo_petrify",
  "mirana_celestial_quiver",
  "monkey_king_mischief",
  "monkey_king_primal_spring",
  "monkey_king_primal_spring_early",
  "monkey_king_transfiguration",
  "monkey_king_untransform",
  "morphling_ebb_and_flow",
  "morphling_morph_replicate",
  "morphling_morph_str",
  "muerta_spectral_slug",
  "muerta_supernatural",
  "naga_siren_eelskin",
  "naga_siren_reel_in",
  "naga_siren_song_of_the_siren_cancel",
  "necrolyte_death_seeker",
  "necrolyte_sadist",
  "nevermore_necromastery",
  "nevermore_shadowraze2",
  "nevermore_shadowraze3",
  "night_stalker_hunter_in_the_night",
  "nyx_assassin_burrow",
  "nyx_assassin_neuro_sting",
  "nyx_assassin_unburrow",
  "obsidian_destroyer_equilibrium",
  "ogre_magi_dumb_luck",
  "ogre_magi_smash",
  "ogre_magi_unrefined_fireblast",
  "omniknight_degen_aura",
  "oracle_diviners_deck",
  "oracle_prognosticate",
  "oracle_rain_of_destiny",
  "pangolier_fortune_favors_the_bold",
  "pangolier_gyroshell_stop",
  "pangolier_rollup",
  "pangolier_rollup_stop",
  "phantom_assassin_blur",
  "phantom_assassin_fan_of_knives",
  "phantom_lancer_illusory_armaments",
  "phoenix_dying_light",
  "phoenix_icarus_dive_stop",
  "phoenix_launch_fire_spirit",
  "phoenix_sun_ray_stop",
  "phoenix_sun_ray_toggle_move",
  "primal_beast_colossal",
  "primal_beast_onslaught_release",
  "primal_beast_rock_throw",
  "puck_ethereal_jaunt",
  "puck_puckish",
  "pudge_innate_graft_flesh",
  "pugna_oblivion_savant",
  "queenofpain_succubus",
  "rattletrap_armor_power",
  "rattletrap_jetpack",
  "rattletrap_jetpack_toggle",
  "rattletrap_overclocking",
  "razor_unstable_current",
  "riki_innate_backstab",
  "ringmaster_spotlight",
  "ringmaster_tame_the_beasts_crack",
  "rubick_curiosity",
  "rubick_hidden1",
  "rubick_hidden2",
  "rubick_telekinesis_land",
  "rubick_telekinesis_land_self",
  "sandking_caustic_finale",
  "shadow_demon_demonic_cleanse",
  "shadow_demon_menace",
  "shadow_demon_shadow_poison_release",
  "shadow_shaman_fowl_play",
  "shadow_shaman_urnaconda",
  "shredder_exposure_therapy",
  "shredder_flamethrower",
  "shredder_return_chakram",
  "silencer_brain_drain",
  "skeleton_king_vampiric_spirit",
  "skywrath_mage_shield_of_the_scion",
  "slardar_seaborn_sentinel",
  "slark_depth_shroud",
  "slark_essence_shift",
  "snapfire_boomstick",
  "snapfire_gobble_up",
  "snapfire_spit_creep",
  "sniper_concussive_grenade",
  "sniper_keen_scope",
  "spectre_desolate",
  "spectre_reality",
  "spirit_breaker_bull_rush",
  "spirit_breaker_planar_pocket",
  "storm_spirit_galvanized",
  "sven_wrath_of_god",
  "techies_focused_detonate",
  "techies_minefield_sign",
  "techies_mutually_assured_destruction",
  "techies_reactive_tazer_stop",
  "templar_assassin_inner_peace",
  "templar_assassin_trap",
  "templar_assassin_trap_teleport",
  "terrorblade_dark_unity",
  "terrorblade_demon_zeal",
  "terrorblade_terror_wave",
  "tidehunter_dead_in_the_water",
  "tidehunter_leviathans_catch",
  "tinker_eureka",
  "tinker_keen_teleport",
  "tinker_warp_grenade",
  "tiny_insurmountable",
  "tiny_toss_tree",
  "tiny_tree_channel",
  "treant_eyes_in_the_forest",
  "treant_natures_guise",
  "troll_warlord_switch_stance",
  "troll_warlord_whirling_axes_melee",
  "tusk_bitter_chill",
  "tusk_drinking_buddies",
  "tusk_launch_snowball",
  "tusk_walrus_kick",
  "undying_ceaseless_dirge",
  "ursa_maul",
  "vengefulspirit_retribution",
  "venomancer_poison_sting",
  "viper_nose_dive",
  "viper_predator",
  "visage_silent_as_the_grave",
  "visage_stone_form_self_cast",
  "visage_summon_familiars_stone_form",
  "void_spirit_intrinsic_edge",
  "warlock_eldritch_summoning",
  "weaver_threads_of_fate",
  "windrunner_focusfire_cancel",
  "windrunner_gale_force",
  "windrunner_tailwind",
  "winter_wyvern_eldwurms_edda",
  "wisp_equilibrium",
  "wisp_spirits_in",
  "wisp_spirits_out",
  "wisp_tether_break",
  "witch_doctor_gris_gris",
  "witch_doctor_voodoo_switcheroo",
  "zuus_cloud",
  "zuus_lightning_hands",
  "zuus_static_field",
]);

export interface LevelableSkillEntry {
  abilityId: number;
  origSlotIdx: number;
  filteredIdx: number;
  ability: AbilityMeta | undefined;
  shortTag: string;
  isUltimate: boolean;
  colorKey: AbilitySlotColorKey;
}

const SUB_ABILITY_NAME_PATTERN = /(_stop|_cancel|_release|_hidden\d*)$/;

export type AbilitySlotColorKey = "q" | "w" | "e" | "r" | "extra";

/**
 * Return false for innate / non-levelable abilities known from vocab metadata.
 * Unknown abilities (undefined metadata in sparse unit-test fixtures) stay visible.
 */
export function isLevelableAbility(ability: AbilityMeta | undefined): boolean {
  if (!ability) {
    return true;
  }
  if (typeof ability.is_levelable === "boolean") {
    return ability.is_levelable;
  }
  if (ability.is_talent || ability.is_innate) {
    return false;
  }
  if (SUB_ABILITY_NAME_PATTERN.test(ability.name)) {
    return false;
  }
  return !NON_LEVELABLE_ABILITY_NAMES.has(ability.name);
}

/**
 * Map a Dota ability slot tag (Q/W/E/D/F/R) and optional ultimate flag
 * into a visual slot color class key.
 */
export function getAbilityColorKey(
  shortTag: string,
  isUltimate: boolean = false,
): AbilitySlotColorKey {
  if (isUltimate || shortTag === "R") return "r";
  if (shortTag === "Q") return "q";
  if (shortTag === "W") return "w";
  if (shortTag === "E") return "e";
  return "extra";
}

/**
 * Filter a hero's skill_ability_ids down to levelable active/passive abilities while
 * retaining origSlotIdx so model skill_logits indexing stays invariant.
 */
export function getLevelableSkillEntries(
  hero: HeroMeta,
  abilitiesById: Record<string, AbilityMeta>,
): LevelableSkillEntry[] {
  const raw: Array<{ abilityId: number; origSlotIdx: number; ability: AbilityMeta | undefined }> = [];
  const skills = hero.skill_ability_ids ?? [];
  for (let origSlotIdx = 0; origSlotIdx < skills.length; origSlotIdx++) {
    const abilityId = skills[origSlotIdx]!;
    const ability = abilitiesById[String(abilityId)];
    if (!isLevelableAbility(ability)) {
      continue;
    }
    raw.push({ abilityId, origSlotIdx, ability });
  }

  const isInvoker = hero.id === INVOKER_HERO_ID;
  const hasVocabUlt = raw.some((item) => item.ability?.is_ultimate === true);
  const explicitUltMatches = raw.filter(
    (item) => item.ability && EXPLICIT_ULTIMATE_ABILITY_NAMES.has(item.ability.name),
  );
  const hasExplicitUlt = explicitUltMatches.length > 0;

  let basicOrderIdx = 0;
  return raw.map((item, filteredIdx) => {
    const abName = item.ability?.name ?? "";
    const isUltimate =
      !isInvoker &&
      (hasVocabUlt
        ? item.ability?.is_ultimate === true
        : hasExplicitUlt
          ? EXPLICIT_ULTIMATE_ABILITY_NAMES.has(abName)
          : raw.length >= 4 && filteredIdx === raw.length - 1);

    let shortTag: string;
    const explicitTag = abName ? EXPLICIT_SLOT_LABEL_BY_NAME[abName] : undefined;
    if (explicitTag) {
      shortTag = explicitTag;
    } else if (isUltimate) {
      shortTag = "R";
    } else {
      shortTag = BASIC_SKILL_LABELS[basicOrderIdx] ?? "F";
      basicOrderIdx += 1;
    }

    const colorKey = getAbilityColorKey(shortTag, isUltimate);
    return {
      abilityId: item.abilityId,
      origSlotIdx: item.origSlotIdx,
      filteredIdx,
      ability: item.ability,
      shortTag,
      isUltimate,
      colorKey,
    };
  });
}

export function getMaxAbilityRanks(
  heroId: number,
  entry: LevelableSkillEntry,
  pickedCount: number,
): number {
  const baseCap =
    heroId === INVOKER_HERO_ID && entry.origSlotIdx < 3
      ? 7
      : entry.isUltimate
        ? 3
        : 4;
  return Math.max(baseCap, pickedCount);
}

/**
 * Collect the hero levels (1..18) where a given levelable skill entry was upgraded,
 * matching by original slot index or canonical hotkey tag for paired abilities.
 */
export function getAbilityPickedLevels(
  entry: LevelableSkillEntry,
  entries: readonly LevelableSkillEntry[],
  skills: readonly SkillLevelStep[],
): number[] {
  const knownSlots = new Set(entries.map((e) => e.origSlotIdx));
  const levels: number[] = [];

  for (const step of skills) {
    if (step.kind !== "ability") continue;
    if (knownSlots.has(step.slotClass)) {
      if (step.slotClass === entry.origSlotIdx) {
        levels.push(step.level);
      }
    } else if (step.shortTag === entry.shortTag) {
      levels.push(step.level);
    }
  }

  return levels;
}
