export type TeamSide = "radiant" | "dire";

export type RolePosition = 1 | 2 | 3 | 4 | 5;

export type MatchPaceOption = 0 | 25 | 35 | 50;

export type LaneKey = "top" | "mid" | "bot";

export type LaneVerdictSide = "radiant" | "even" | "dire";

export interface HeroMeta {
  idx: number;
  id: number;
  short_name: string;
  display_name: string;
  primary_attr: string | null;
  attack_type: string | null;
  roles: string[];
  img: string | null;
  icon: string | null;
  skill_ability_ids: number[];
  talent_tiers: Record<"10" | "15" | "20" | "25", number[]>;
}

export interface ItemMeta {
  idx: number;
  id: number;
  short_name: string;
  display_name: string;
  cost: number;
  img: string | null;
  built_from_ids?: number[];
}

export interface AbilityMeta {
  id: number;
  name: string;
  display_name: string;
  is_talent: boolean;
  is_innate?: boolean;
  is_levelable?: boolean;
  is_ultimate?: boolean;
  img: string | null;
}

export interface VocabJson {
  schema_version: number;
  game_version_id: number;
  game_version_name: string;
  as_of_datetime: number;
  generated_at: string;
  heroes: {
    by_id: Record<string, HeroMeta>;
    id_to_idx: Record<string, number>;
    idx_to_id: number[];
    num_heroes: number;
  };
  items: {
    special_tokens: Record<string, number>;
    by_id: Record<string, ItemMeta>;
    id_to_idx: Record<string, number>;
    idx_to_id: number[];
    aliases: Record<string, number>;
    vocab_size: number;
    num_key_items: number;
  };
  abilities: {
    by_id: Record<string, AbilityMeta>;
    talent_ids: number[];
    aliases?: Record<string, number>;
    attribute_bonus_id: number;
  };
}

export interface TargetScalesJson {
  duration_center_min: number;
  duration_scale_min: number;
  min_match_duration_min: number;
  max_duration_override_min: number;
  gold_xp_lead_scale: number;
  dpm_hpm_scale: number;
  gpm_xpm_scale: number;
  item_time_scale_min: number;
  heal_share_min_team_sum: number;
}

export interface ModelSummaryJson {
  trained_at_utc?: string | null;
  dataset_snapshot_utc?: string | null;
  total_matches?: number | null;
  ranked_matches?: number | null;
  pro_matches?: number | null;
  train_matches?: number | null;
  val_matches?: number | null;
  num_shards?: number | null;
  num_heroes?: number | null;
  num_key_items?: number | null;
  bundle_size_mb?: number | null;
  win_accuracy?: number | null;
  lane_accuracy?: number | null;
  duration_mae_min?: number | null;
  duration_window_coverage?: number | null;
  skill_accuracy?: number | null;
  talent_accuracy?: number | null;
  item_accuracy?: number | null;
  item_timing_mae_min?: number | null;
  gpm_mae?: number | null;
}

export interface ModelConfigJson {
  model_schema_version: number;
  encoder_arch_version?: number;
  parquet_schema_version: number;
  vocab_schema_version: number;
  game_version_id: number;
  game_version_name: string;
  num_heroes: number;
  item_vocab_size: number;
  max_item_len: number;
  max_hero_skills: number;
  num_skill_classes: number;
  slot_talent: number;
  slot_attr: number;
  d_model: number;
  nhead: number;
  num_encoder_layers: number;
  num_decoder_layers: number;
  dim_feedforward: number;
  dropout: number;
  stage1_pair_dim?: number;
  stage1_pair_dropout?: number;
  use_stage2_block?: boolean;
  eos_logit_penalty?: number;
  eos_stop_prob?: number;
  target_scales: TargetScalesJson;
  model_summary?: ModelSummaryJson;
}

export interface DraftSlot {
  slotIdx: number;
  side: TeamSide;
  position: RolePosition;
  heroId: number | null;
}

export interface DecodedItemToken {
  itemToken: number;
  expectedMinute: number;
}

export interface PredictedItem {
  itemToken: number;
  itemId: number;
  shortName: string;
  displayName: string;
  cost: number;
  img: string | null;
  expectedMinute: number;
}

export interface SkillLevelStep {
  level: number;
  kind: "ability" | "talent" | "attribute" | "empty";
  slotClass: number;
  abilityId: number | null;
  label: string;
  shortTag: string;
  rank?: number;
  isUltimate?: boolean;
  img: string | null;
}

export interface TalentOption {
  id: number;
  displayName: string;
  chosen: boolean;
}

export interface TalentChoice {
  tier: 10 | 15 | 20 | 25;
  chosenSide: 0 | 1;
  abilityId: number | null;
  displayName: string;
  options: TalentOption[];
}

export interface CombatShareStat {
  dpm: number;
  teamShare: number;
}

export interface HeroCombatStats {
  heroDamage: CombatShareStat;
  towerDamage: CombatShareStat;
  damageTaken: CombatShareStat;
  healing: CombatShareStat;
  showHealingShare: boolean;
}

export interface RangeMetric {
  low: number;
  mid: number;
  high: number;
}

export interface HeroEconomyStats {
  gpm: RangeMetric;
  xpm: RangeMetric;
  csPerMin: RangeMetric;
  csTotal: RangeMetric;
  kills: RangeMetric;
  deaths: RangeMetric;
  assists: RangeMetric;
}

export interface LanePrediction {
  lane: LaneKey;
  label: "Top" | "Mid" | "Bot";
  radiantProb: number;
  evenProb: number;
  direProb: number;
  radiantPct: number;
  evenPct: number;
  direPct: number;
  verdict: "Radiant favored" | "Even" | "Dire favored";
  verdictSide: LaneVerdictSide;
}

export interface HeroPrediction {
  slotIdx: number;
  side: TeamSide;
  position: RolePosition;
  hero: HeroMeta;
  items: PredictedItem[];
  skills: SkillLevelStep[];
  talents: TalentChoice[];
  combat: HeroCombatStats;
  economy: HeroEconomyStats;
}

export interface DraftPrediction {
  radiantWinProb: number;
  direWinProb: number;
  durationP25: number;
  durationP50: number;
  durationP75: number;
  lanes: LanePrediction[];
  goldLead10m: number;
  xpLead10m: number;
  matchPaceMinutes: MatchPaceOption;
  effectiveDurationMinutes: number;
  heroes: HeroPrediction[];
}
