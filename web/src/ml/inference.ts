import * as ort from "onnxruntime-web/wasm";

import type {
  DraftPrediction,
  DraftSlot,
  HeroMeta,
  HeroPrediction,
  MatchPaceOption,
  ModelConfigJson,
  VocabJson,
} from "../types/oracle";
import {
  BOS_ITEM_ID,
  EOS_ITEM_ID,
  decodeCombatStats,
  decodeEconomyStatsForDuration,
  decodeItemsGreedy,
  decodeLanes,
  decodeSkillBuild1To18,
  decodeTalentChoices,
  resolvePredictedItems,
  sanitizeFinite,
  type ItemDecoderRunner,
} from "./decoder";
import { configureOrtEnvironment, normalizeModelBaseUrl } from "./ortSetup";

export const EXPECTED_MODEL_SCHEMA_VERSION = 2;
export const EXPECTED_ENCODER_ARCH_VERSION = 4;
export const SUPPORTED_ENCODER_ARCH_VERSIONS = new Set<number>([3, 4]);
export const EXPECTED_PARQUET_SCHEMA_VERSION = 2;
export const EXPECTED_VOCAB_SCHEMA_VERSION = 2;

export interface LoadedOracleBundle {
  baseUrl: string;
  vocab: VocabJson;
  modelConfig: ModelConfigJson;
  encoderSession: ort.InferenceSession;
  decoderSession: ort.InferenceSession;
  heroesList: HeroMeta[];
}

const bundleCache = new Map<string, LoadedOracleBundle>();

async function fetchJsonChecked<T>(url: string, label: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${label} (${res.status} ${res.statusText}) at ${url}`);
  }
  return (await res.json()) as T;
}

async function fetchBinaryChecked(url: string, label: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${label} (${res.status} ${res.statusText}) at ${url}`);
  }
  const buf = await res.arrayBuffer();
  return new Uint8Array(buf);
}

export function validateBundleMetadata(
  vocab: VocabJson,
  modelConfig: ModelConfigJson,
): void {
  if (!vocab.heroes?.by_id || !vocab.items?.idx_to_id || !vocab.abilities?.by_id) {
    throw new Error("Loaded vocab.json is missing heroes, items, or abilities lookup tables.");
  }
  if (vocab.schema_version !== EXPECTED_VOCAB_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported vocab schema version ${String(vocab.schema_version)} (expected v${EXPECTED_VOCAB_SCHEMA_VERSION}).`,
    );
  }
  if (modelConfig.model_schema_version !== EXPECTED_MODEL_SCHEMA_VERSION) {
    throw new Error(
      `Incompatible model bundle version (${String(modelConfig.model_schema_version ?? 1)}); expected v${EXPECTED_MODEL_SCHEMA_VERSION}. Reload with updated v2 patch bundle files.`,
    );
  }
  if (
    typeof modelConfig.encoder_arch_version !== "number" ||
    !SUPPORTED_ENCODER_ARCH_VERSIONS.has(modelConfig.encoder_arch_version)
  ) {
    throw new Error(
      `Incompatible encoder architecture version (${String(modelConfig.encoder_arch_version)}); expected v3 or v${EXPECTED_ENCODER_ARCH_VERSION}. Reload with updated v2 patch bundle files.`,
    );
  }
  if (modelConfig.parquet_schema_version !== EXPECTED_PARQUET_SCHEMA_VERSION) {
    throw new Error(
      `Incompatible dataset schema version (${String(modelConfig.parquet_schema_version)}); expected v${EXPECTED_PARQUET_SCHEMA_VERSION}.`,
    );
  }
  if (modelConfig.vocab_schema_version !== EXPECTED_VOCAB_SCHEMA_VERSION) {
    throw new Error(
      `Incompatible bundle vocab schema version (${String(modelConfig.vocab_schema_version)}); expected v${EXPECTED_VOCAB_SCHEMA_VERSION}.`,
    );
  }
  if (
    typeof modelConfig.d_model !== "number" ||
    typeof modelConfig.max_item_len !== "number" ||
    typeof modelConfig.num_skill_classes !== "number" ||
    !modelConfig.target_scales ||
    typeof modelConfig.target_scales.heal_share_min_team_sum !== "number"
  ) {
    throw new Error("Loaded model_config.json has invalid architecture or target_scales metadata.");
  }
  if (vocab.game_version_id !== modelConfig.game_version_id) {
    throw new Error(
      `Patch mismatch: vocab is patch ${vocab.game_version_name}, model is patch ${modelConfig.game_version_name}.`,
    );
  }
}

export async function loadOracleBundle(
  rawBaseUrl: string = "./models",
  forceReload: boolean = false,
): Promise<LoadedOracleBundle> {
  configureOrtEnvironment(ort);
  const baseUrl = normalizeModelBaseUrl(rawBaseUrl);

  if (!forceReload) {
    const cached = bundleCache.get(baseUrl);
    if (cached) {
      return cached;
    }
  }

  const [vocab, modelConfig, encoderBytes, decoderBytes] = await Promise.all([
    fetchJsonChecked<VocabJson>(`${baseUrl}/vocab.json`, "vocab.json"),
    fetchJsonChecked<ModelConfigJson>(`${baseUrl}/model_config.json`, "model_config.json"),
    fetchBinaryChecked(`${baseUrl}/d2oracle_encoder.onnx`, "d2oracle_encoder.onnx"),
    fetchBinaryChecked(`${baseUrl}/d2oracle_item_decoder.onnx`, "d2oracle_item_decoder.onnx"),
  ]);

  validateBundleMetadata(vocab, modelConfig);

  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  };

  const [encoderSession, decoderSession] = await Promise.all([
    ort.InferenceSession.create(encoderBytes, sessionOptions),
    ort.InferenceSession.create(decoderBytes, sessionOptions),
  ]);

  const heroesList = Object.values(vocab.heroes.by_id).sort((a, b) =>
    a.display_name.localeCompare(b.display_name),
  );

  const bundle: LoadedOracleBundle = {
    baseUrl,
    vocab,
    modelConfig,
    encoderSession,
    decoderSession,
    heroesList,
  };
  bundleCache.set(baseUrl, bundle);
  return bundle;
}

export async function predictDraft(
  bundle: LoadedOracleBundle,
  slots: readonly DraftSlot[],
  matchPace: MatchPaceOption = 0,
): Promise<DraftPrediction> {
  if (slots.length !== 10) {
    throw new Error(`Expected 10 draft slots, received ${slots.length}.`);
  }

  const ordered = [...slots].sort((a, b) => a.slotIdx - b.slotIdx);
  const resolvedHeroes: HeroMeta[] = [];
  const heroIdxArr = new BigInt64Array(10);
  const isRadArr = new BigInt64Array(10);
  const posIdxArr = new BigInt64Array(10);

  for (let i = 0; i < 10; i++) {
    const slot = ordered[i]!;
    if (slot.heroId === null) {
      throw new Error(`Draft slot ${slot.side} Pos ${slot.position} has no hero selected.`);
    }
    const heroMeta = bundle.vocab.heroes.by_id[String(slot.heroId)];
    const mappedIdx = bundle.vocab.heroes.id_to_idx[String(slot.heroId)];
    if (!heroMeta || mappedIdx === undefined) {
      throw new Error(`Hero ID ${slot.heroId} was not found in active patch vocabulary.`);
    }
    resolvedHeroes.push(heroMeta);
    heroIdxArr[i] = BigInt(mappedIdx);
    isRadArr[i] = BigInt(slot.side === "radiant" ? 1 : 0);
    posIdxArr[i] = BigInt(slot.position - 1);
  }

  const durOverrideArr = new Float32Array([matchPace]);

  const encOutputs = await bundle.encoderSession.run({
    hero_idx: new ort.Tensor("int64", heroIdxArr, [1, 10]),
    is_radiant: new ort.Tensor("int64", isRadArr, [1, 10]),
    pos_idx: new ort.Tensor("int64", posIdxArr, [1, 10]),
    duration_override: new ort.Tensor("float32", durOverrideArr, [1]),
  });

  const winProbTensor = encOutputs.win_prob;
  const durTensor = encOutputs.duration_quantiles;
  const laneTensor = encOutputs.lane_probs;
  const lead10Tensor = encOutputs.lead10;
  const combatRatesTensor = encOutputs.combat_rates;
  const combatSharesTensor = encOutputs.combat_shares;
  const econTensor = encOutputs.econ_kda_quantiles;
  const skillTensor = encOutputs.skill_logits;
  const talentTensor = encOutputs.talent_logits;
  const memTensor = encOutputs.slot_memory;

  if (
    !winProbTensor ||
    !durTensor ||
    !laneTensor ||
    !lead10Tensor ||
    !combatRatesTensor ||
    !combatSharesTensor ||
    !econTensor ||
    !skillTensor ||
    !talentTensor ||
    !memTensor
  ) {
    throw new Error("Model outputs are incomplete for v2 draft bundle.");
  }

  const winProbArr = winProbTensor.data as Float32Array;
  const durArr = durTensor.data as Float32Array;
  const laneArr = laneTensor.data as Float32Array;
  const lead10Arr = lead10Tensor.data as Float32Array;
  const combatRatesArr = combatRatesTensor.data as Float32Array;
  const combatSharesArr = combatSharesTensor.data as Float32Array;
  const econArr = econTensor.data as Float32Array;
  const skillArr = skillTensor.data as Float32Array;
  const talentArr = talentTensor.data as Float32Array;
  const memArr = memTensor.data as Float32Array;

  const decoderRunner: ItemDecoderRunner = async (
    slotMemoryFlat,
    batchSize,
    dModel,
    targetSlotIdx,
    itemSeq,
    seqLen,
  ) => {
    const decOut = await bundle.decoderSession.run({
      slot_memory: new ort.Tensor("float32", slotMemoryFlat, [batchSize, 10, dModel]),
      target_slot_idx: new ort.Tensor("int64", targetSlotIdx, [batchSize]),
      item_seq: new ort.Tensor("int64", itemSeq, [batchSize, seqLen]),
    });
    const logitsT = decOut.next_item_logits;
    const minT = decOut.next_item_minutes;
    if (!logitsT || !minT) {
      throw new Error("Item decoder outputs are incomplete.");
    }
    return {
      logits: logitsT.data as Float32Array,
      minutes: minT.data as Float32Array,
    };
  };

  const maxSteps = Math.max(1, bundle.modelConfig.max_item_len - 1);
  const eosPenalty = bundle.modelConfig.eos_logit_penalty ?? 0;
  const eosStopProb = bundle.modelConfig.eos_stop_prob ?? 0.25;
  const maxMinute = matchPace > 0 ? matchPace : null;
  const decodedItemsPerSlot = await decodeItemsGreedy(
    decoderRunner,
    memArr,
    bundle.modelConfig.d_model,
    bundle.modelConfig.item_vocab_size,
    maxSteps,
    BOS_ITEM_ID,
    EOS_ITEM_ID,
    eosPenalty,
    maxMinute,
    eosStopProb,
  );

  const durP50 = Math.max(15, sanitizeFinite(durArr[1], 35));
  const durP25 = Math.max(15, Math.min(durP50, sanitizeFinite(durArr[0], 28)));
  const durP75 = Math.max(durP50, sanitizeFinite(durArr[2], 42));
  const effectiveDurationMinutes = matchPace > 0 ? matchPace : durP50;

  // Hide healing team share % when predicted team total healing is below threshold (avoids junk 100%).
  const minTeamHeal = bundle.modelConfig.target_scales.heal_share_min_team_sum;
  const teamHealTotal = [0, 0];
  for (let s = 0; s < 10; s++) {
    const hpm = Math.max(0, sanitizeFinite(combatRatesArr[s * 4 + 3], 0));
    teamHealTotal[s < 5 ? 0 : 1]! += hpm * effectiveDurationMinutes;
  }
  const showRadHealShare = teamHealTotal[0]! >= minTeamHeal;
  const showDireHealShare = teamHealTotal[1]! >= minTeamHeal;

  const numSkillClasses = bundle.modelConfig.num_skill_classes;
  const slotTalent = bundle.modelConfig.slot_talent;
  const slotAttr = bundle.modelConfig.slot_attr;

  const heroes: HeroPrediction[] = [];
  for (let s = 0; s < 10; s++) {
    const slot = ordered[s]!;
    const hero = resolvedHeroes[s]!;
    const items = resolvePredictedItems(
      decodedItemsPerSlot[s] ?? [],
      bundle.vocab.items.idx_to_id,
      bundle.vocab.items.by_id,
    );
    const skills = decodeSkillBuild1To18(
      hero,
      skillArr,
      s * 18 * numSkillClasses,
      numSkillClasses,
      slotTalent,
      slotAttr,
      bundle.vocab.abilities.by_id,
    );
    const talents = decodeTalentChoices(
      hero,
      talentArr,
      s * 4 * 2,
      bundle.vocab.abilities.by_id,
    );
    const combat = decodeCombatStats(
      combatRatesArr,
      combatSharesArr,
      s,
      s < 5 ? showRadHealShare : showDireHealShare,
    );
    const economy = decodeEconomyStatsForDuration(
      econArr,
      s,
      effectiveDurationMinutes,
    );

    heroes.push({
      slotIdx: slot.slotIdx,
      side: slot.side,
      position: slot.position,
      hero,
      items,
      skills,
      talents,
      combat,
      economy,
    });
  }

  const radWin = Math.min(1, Math.max(0, sanitizeFinite(winProbArr[0], 0.5)));

  return {
    radiantWinProb: radWin,
    direWinProb: 1 - radWin,
    durationP25: durP25,
    durationP50: durP50,
    durationP75: durP75,
    lanes: decodeLanes(laneArr),
    goldLead10m: sanitizeFinite(lead10Arr[0], 0),
    xpLead10m: sanitizeFinite(lead10Arr[1], 0),
    matchPaceMinutes: matchPace,
    effectiveDurationMinutes,
    heroes,
  };
}
