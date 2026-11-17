import { getLevelableSkillEntries } from "../data/abilityDisplay";
import type {
  AbilityMeta,
  DecodedItemToken,
  HeroCombatStats,
  HeroEconomyStats,
  HeroMeta,
  ItemMeta,
  LaneKey,
  LanePrediction,
  PredictedItem,
  RangeMetric,
  SkillLevelStep,
  TalentChoice,
  TalentOption,
} from "../types/oracle";

export const PAD_ITEM_ID = 0;
export const BOS_ITEM_ID = 1;
export const EOS_ITEM_ID = 2;

export const TALENT_TIERS: readonly (10 | 15 | 20 | 25)[] = [10, 15, 20, 25];
const MASKED_LOGIT_THRESHOLD = -5000;

export interface ItemDecoderStepResult {
  logits: Float32Array;
  minutes: Float32Array;
}

export type ItemDecoderRunner = (
  slotMemoryFlat: Float32Array,
  batchSize: number,
  dModel: number,
  targetSlotIdx: BigInt64Array,
  itemSeq: BigInt64Array,
  seqLen: number,
) => Promise<ItemDecoderStepResult>;

export function selectNextItemToken(
  stepLogits: Float32Array,
  vocabOffset: number,
  vocabSize: number,
  previousTokens: readonly number[],
  bosId: number = BOS_ITEM_ID,
  eosId: number = EOS_ITEM_ID,
  eosPenalty: number = 0,
  eosStopProb: number = 0,
): number {
  const picked = new Set<number>();
  for (const tok of previousTokens) {
    if (tok >= 3) {
      picked.add(tok);
    }
  }

  let allowEos = true;
  if (eosStopProb > 0) {
    let maxLogit = -Infinity;
    let hasCandidateItem = false;
    for (let tok = 0; tok < vocabSize; tok++) {
      if (tok === PAD_ITEM_ID || tok === bosId || picked.has(tok)) {
        continue;
      }
      const s = stepLogits[vocabOffset + tok] ?? -Infinity;
      if (tok >= 3 && s > -1e8) {
        hasCandidateItem = true;
      }
      if (s > maxLogit) {
        maxLogit = s;
      }
    }
    if (hasCandidateItem && Number.isFinite(maxLogit)) {
      let sumExp = 0;
      let eosExp = 0;
      for (let tok = 0; tok < vocabSize; tok++) {
        if (tok === PAD_ITEM_ID || tok === bosId || picked.has(tok)) {
          continue;
        }
        const s = stepLogits[vocabOffset + tok] ?? -Infinity;
        if (s > -1e8) {
          const e = Math.exp(Math.max(-80, s - maxLogit));
          sumExp += e;
          if (tok === eosId) {
            eosExp = e;
          }
        }
      }
      if (sumExp > 0 && eosExp / sumExp < eosStopProb) {
        allowEos = false;
      }
    }
  }

  let bestIdx = eosId;
  let bestScore = -Infinity;
  for (let tok = 0; tok < vocabSize; tok++) {
    if (tok === PAD_ITEM_ID || tok === bosId || picked.has(tok) || (tok === eosId && !allowEos)) {
      continue;
    }
    let score = stepLogits[vocabOffset + tok] ?? -Infinity;
    if (tok === eosId && eosPenalty !== 0) {
      score -= eosPenalty;
    }
    if (score > bestScore) {
      bestScore = score;
      bestIdx = tok;
    }
  }
  return bestIdx;
}

export async function decodeItemsGreedy(
  runner: ItemDecoderRunner,
  slotMemoryFlat: Float32Array,
  dModel: number,
  vocabSize: number,
  maxSteps: number,
  bosId: number = BOS_ITEM_ID,
  eosId: number = EOS_ITEM_ID,
  eosPenalty: number = 0,
  maxMinute: number | null = null,
  eosStopProb: number = 0,
): Promise<DecodedItemToken[][]> {
  const effMaxMin = maxMinute !== null && maxMinute > 0 ? maxMinute : null;
  const numSlots = 10;
  const slotFrameSize = numSlots * dModel;
  const expandedMemory = new Float32Array(numSlots * slotFrameSize);
  for (let stream = 0; stream < numSlots; stream++) {
    expandedMemory.set(
      slotMemoryFlat.subarray(0, slotFrameSize),
      stream * slotFrameSize,
    );
  }

  const targetSlots = new BigInt64Array(numSlots);
  for (let s = 0; s < numSlots; s++) {
    targetSlots[s] = BigInt(s);
  }

  const streamsHistory: number[][] = Array.from({ length: numSlots }, () => []);
  const streamsMinutes: number[][] = Array.from({ length: numSlots }, () => []);
  const finished: boolean[] = Array.from({ length: numSlots }, () => false);
  const lastMinute: number[] = Array.from({ length: numSlots }, () => 0);
  let seq = Array.from({ length: numSlots }, () => [bosId]);

  for (let step = 0; step < maxSteps; step++) {
    const seqLen = step + 1;
    const flatSeq = new BigInt64Array(numSlots * seqLen);
    for (let s = 0; s < numSlots; s++) {
      const row = seq[s]!;
      for (let t = 0; t < seqLen; t++) {
        flatSeq[s * seqLen + t] = BigInt(row[t]!);
      }
    }

    const { logits, minutes } = await runner(
      expandedMemory,
      numSlots,
      dModel,
      targetSlots,
      flatSeq,
      seqLen,
    );

    const stepTokens: number[] = [];
    for (let s = 0; s < numSlots; s++) {
      if (finished[s]) {
        streamsHistory[s]!.push(eosId);
        streamsMinutes[s]!.push(0);
        stepTokens.push(eosId);
        continue;
      }

      const logitOffset = (s * seqLen + (seqLen - 1)) * vocabSize;
      const prevRow = seq[s]!.slice(1);
      let nextTok = selectNextItemToken(
        logits,
        logitOffset,
        vocabSize,
        prevRow,
        bosId,
        eosId,
        eosPenalty,
        eosStopProb,
      );

      let predMin = 0;
      if (nextTok === eosId) {
        finished[s] = true;
      } else {
        // minute_head outputs [N, T, V]: index per-token minute for chosen nextTok
        // (with fallback if a flat [N, T] minute buffer is supplied by a test stub).
        const perTokIdx = logitOffset + nextTok;
        const flatIdx = s * seqLen + (seqLen - 1);
        const rawMin =
          minutes.length >= numSlots * seqLen * vocabSize
            ? (minutes[perTokIdx] ?? 0)
            : (minutes[flatIdx] ?? 0);
        const monoMin = Math.max(lastMinute[s]!, rawMin);
        if (step > 0 && effMaxMin !== null && monoMin > effMaxMin) {
          nextTok = eosId;
          finished[s] = true;
        } else {
          predMin = monoMin;
          lastMinute[s] = predMin;
        }
      }

      streamsHistory[s]!.push(nextTok);
      streamsMinutes[s]!.push(predMin);
      stepTokens.push(nextTok);
    }

    seq = seq.map((row, s) => [...row, stepTokens[s]!]);

    if (finished.every(Boolean)) {
      break;
    }
  }

  return streamsHistory.map((tokens, s) => {
    const mins = streamsMinutes[s]!;
    const out: DecodedItemToken[] = [];
    for (let i = 0; i < tokens.length; i++) {
      const tok = tokens[i]!;
      if (tok === eosId) {
        break;
      }
      if (tok >= 3) {
        out.push({
          itemToken: tok,
          expectedMinute: mins[i] ?? 0,
        });
      }
    }
    return out;
  });
}

export function resolvePredictedItems(
  decoded: readonly DecodedItemToken[],
  idxToId: readonly number[],
  itemsById: Record<string, ItemMeta>,
): PredictedItem[] {
  const out: PredictedItem[] = [];
  for (const step of decoded) {
    const itemId = idxToId[step.itemToken];
    if (!itemId) {
      continue;
    }
    const meta = itemsById[String(itemId)];
    if (!meta) {
      continue;
    }
    out.push({
      itemToken: step.itemToken,
      itemId,
      shortName: meta.short_name,
      displayName: meta.display_name,
      cost: meta.cost,
      img: meta.img,
      expectedMinute: step.expectedMinute,
    });
  }
  return out;
}

export function getMaxUltRankForLevel(heroId: number, level: number): number {
  if (heroId === 74) {
    // Invoker has no leveled ultimate ability (Invoke scales independently of skill points).
    return 0;
  }
  if (heroId === 82) {
    // In 7.41f, Meepo's Divided We Stand can unlock at level 2 and upgrades at 10 and 17 (4th rank is >18).
    return level >= 17 ? 3 : level >= 10 ? 2 : level >= 2 ? 1 : 0;
  }
  return level >= 18 ? 3 : level >= 12 ? 2 : level >= 6 ? 1 : 0;
}

export function getSkippedSkillLevel(heroId: number): number | null {
  if (heroId === 74) {
    // Invoker receives skill points across every level 1-18.
    return null;
  }
  if (heroId === 82) {
    // Meepo earns his 17th point at level 17 and skips level 18.
    return 18;
  }
  // Standard heroes earn 17 points across levels 1-16 and 18, with no point at level 17.
  return 17;
}

export function decodeSkillBuild1To18(
  hero: HeroMeta,
  skillLogits: Float32Array,
  slotOffset: number,
  numSkillClasses: number,
  slotTalent: number,
  slotAttr: number,
  abilitiesById: Record<string, AbilityMeta>,
): SkillLevelStep[] {
  const skills = hero.skill_ability_ids ?? [];
  const levelableEntries = getLevelableSkillEntries(hero, abilitiesById).filter(
    (entry) => entry.origSlotIdx < slotTalent,
  );
  const entryByOrigSlot = new Map(levelableEntries.map((entry) => [entry.origSlotIdx, entry]));
  const ultOrigSlots = levelableEntries
    .filter((entry) => entry.isUltimate)
    .map((entry) => entry.origSlotIdx);
  const skippedLevel = getSkippedSkillLevel(hero.id);
  const counts = new Array<number>(numSkillClasses).fill(0);
  const countsByTag: Record<string, number> = {};
  const steps: SkillLevelStep[] = [];

  for (let lvlIdx = 0; lvlIdx < 18; lvlIdx++) {
    const level = lvlIdx + 1;
    if (level === skippedLevel) {
      steps.push({
        level,
        kind: "empty",
        slotClass: -1,
        abilityId: null,
        label: `No skill point at level ${level}`,
        shortTag: "—",
        isUltimate: false,
        img: null,
      });
      continue;
    }
    const base = slotOffset + lvlIdx * numSkillClasses;

    const maxTalentsByLevel = level >= 15 ? 2 : level >= 10 ? 1 : 0;
    const canTakeTalent =
      (skillLogits[base + slotTalent] ?? -Infinity) > MASKED_LOGIT_THRESHOLD &&
      level >= 10 &&
      (counts[slotTalent] ?? 0) < Math.min(4, maxTalentsByLevel);

    const maxUltForLevel = getMaxUltRankForLevel(hero.id, level);
    const totalUltRank = ultOrigSlots.reduce((sum, idx) => sum + (counts[idx] ?? 0), 0);
    const legalCandidates: number[] = [];

    for (const entry of levelableEntries) {
      const s = entry.origSlotIdx;
      if ((skillLogits[base + s] ?? -Infinity) <= MASKED_LOGIT_THRESHOLD) {
        continue;
      }
      const rank = counts[s] ?? 0;
      const sharedTagRank =
        entry.shortTag === "Q" ||
        entry.shortTag === "W" ||
        entry.shortTag === "E" ||
        entry.shortTag === "R"
          ? (countsByTag[entry.shortTag] ?? 0)
          : 0;
      const effectiveRank = Math.max(rank, sharedTagRank);
      if (entry.isUltimate) {
        if (Math.max(totalUltRank, effectiveRank) < maxUltForLevel) {
          legalCandidates.push(s);
        }
      } else {
        // In 7.41f, Invoker orbs scale to rank 8; standard basic abilities cap at rank 4 and ceil(level / 2).
        const maxBasicCap = hero.id === 74 && entry.origSlotIdx < 3 ? 8 : 4;
        const maxBasic = Math.min(maxBasicCap, Math.ceil(level / 2));
        if (effectiveRank < maxBasic) {
          legalCandidates.push(s);
        }
      }
    }

    if (canTakeTalent) {
      legalCandidates.push(slotTalent);
    }

    const canTakeAttr =
      (skillLogits[base + slotAttr] ?? -Infinity) > MASKED_LOGIT_THRESHOLD &&
      (counts[slotAttr] ?? 0) < 7 &&
      (legalCandidates.length === 0 || level >= 6);
    if (canTakeAttr) {
      legalCandidates.push(slotAttr);
    }

    // Fallback: relax Dota 2 rank caps across encoder-unmasked levelable classes so build never stalls.
    let pool = legalCandidates;
    if (pool.length === 0) {
      pool = [];
      for (const entry of levelableEntries) {
        const s = entry.origSlotIdx;
        if ((skillLogits[base + s] ?? -Infinity) > MASKED_LOGIT_THRESHOLD) {
          pool.push(s);
        }
      }
      if (level >= 10 && (counts[slotTalent] ?? 0) < 4) {
        pool.push(slotTalent);
      }
      if ((skillLogits[base + slotAttr] ?? -Infinity) > MASKED_LOGIT_THRESHOLD) {
        pool.push(slotAttr);
      }
      if (pool.length === 0) {
        pool.push(levelableEntries[0]?.origSlotIdx ?? 0);
      }
    }

    let chosen = pool[0]!;
    let bestScore = skillLogits[base + chosen] ?? -Infinity;
    for (let i = 1; i < pool.length; i++) {
      const c = pool[i]!;
      const score = skillLogits[base + c] ?? -Infinity;
      if (score > bestScore) {
        bestScore = score;
        chosen = c;
      }
    }

    counts[chosen] = (counts[chosen] ?? 0) + 1;

    if (chosen === slotTalent) {
      steps.push({
        level,
        kind: "talent",
        slotClass: chosen,
        abilityId: null,
        label: "Talent",
        shortTag: "T",
        isUltimate: false,
        img: null,
      });
    } else if (chosen === slotAttr) {
      steps.push({
        level,
        kind: "attribute",
        slotClass: chosen,
        abilityId: null,
        label: "Attribute Bonus",
        shortTag: "+",
        isUltimate: false,
        img: null,
      });
    } else {
      const matchedEntry = entryByOrigSlot.get(chosen);
      const aid = matchedEntry?.abilityId ?? skills[chosen] ?? null;
      const ab = matchedEntry?.ability ?? (aid !== null ? abilitiesById[String(aid)] : undefined);
      const shortTag = matchedEntry?.shortTag ?? "?";
      if (shortTag === "Q" || shortTag === "W" || shortTag === "E" || shortTag === "R") {
        countsByTag[shortTag] = (countsByTag[shortTag] ?? 0) + 1;
      }
      steps.push({
        level,
        kind: "ability",
        slotClass: chosen,
        abilityId: aid,
        label: ab?.display_name || ab?.name || `Ability ${chosen + 1}`,
        shortTag,
        isUltimate: matchedEntry?.isUltimate ?? false,
        img: ab?.img ?? null,
      });
    }
  }

  return steps;
}

export function decodeTalentChoices(
  hero: HeroMeta,
  talentLogits: Float32Array,
  slotOffset: number,
  abilitiesById: Record<string, AbilityMeta>,
): TalentChoice[] {
  return TALENT_TIERS.map((tier, tierIdx) => {
    const left = talentLogits[slotOffset + tierIdx * 2] ?? 0;
    const right = talentLogits[slotOffset + tierIdx * 2 + 1] ?? 0;
    const chosenSide: 0 | 1 = right > left ? 1 : 0;

    // Deduplicate reworked talent IDs sharing the exact internal ability name (matches Python VocabIndex).
    const rawIds = hero.talent_tiers[String(tier) as "10" | "15" | "20" | "25"] ?? [];
    const deduped: Array<{ id: number; name: string; displayName: string }> = [];
    for (const rawId of rawIds) {
      const ab = abilitiesById[String(rawId)];
      const internalName = ab?.name || `tal_${rawId}`;
      const displayName = ab?.display_name || internalName;
      if (!deduped.some((entry) => entry.name === internalName)) {
        deduped.push({ id: rawId, name: internalName, displayName });
      }
      if (deduped.length >= 2) {
        break;
      }
    }

    const options: TalentOption[] = deduped.map((entry, idx) => ({
      id: entry.id,
      displayName: entry.displayName,
      chosen: idx === chosenSide,
    }));

    const selected = options[chosenSide] ?? options[0] ?? null;
    return {
      tier,
      chosenSide,
      abilityId: selected ? selected.id : null,
      displayName: selected ? selected.displayName : `Tier ${tier} talent`,
      options,
    };
  });
}

export function sanitizeFinite(value: number | undefined, fallback: number = 0): number {
  if (value === undefined || !Number.isFinite(value)) {
    return fallback;
  }
  return Object.is(value, -0) || value === 0 ? 0 : value;
}

export function roundPercentagesTo100(probs: readonly number[]): number[] {
  const safe = probs.map((p) => Math.max(0, sanitizeFinite(p, 0)));
  const total = safe.reduce((acc, v) => acc + v, 0);
  if (total <= 0) {
    return safe.map((_, i) => (i === 0 ? 100 : 0));
  }
  const scaled = safe.map((p) => (p / total) * 100);
  const floors = scaled.map((v) => Math.floor(v));
  let remainder = 100 - floors.reduce((acc, v) => acc + v, 0);
  const order = scaled
    .map((v, i) => ({ i, frac: v - floors[i]! }))
    .sort((a, b) => b.frac - a.frac);
  for (let k = 0; k < order.length && remainder > 0; k++) {
    floors[order[k]!.i]! += 1;
    remainder -= 1;
  }
  return floors.map((v) => sanitizeFinite(v, 0));
}

export function formatWinOddsPair(radProb: number): {
  radProb: number;
  direProb: number;
  radPctText: string;
  direPctText: string;
} {
  const clamped = Math.min(1, Math.max(0, sanitizeFinite(radProb, 0.5)));
  const radTenths = Math.round(clamped * 1000);
  const direTenths = 1000 - radTenths;
  return {
    radProb: radTenths / 1000,
    direProb: direTenths / 1000,
    radPctText: (radTenths / 10).toFixed(1),
    direPctText: (direTenths / 10).toFixed(1),
  };
}

const LANE_DEFS: ReadonlyArray<{ lane: LaneKey; label: "Top" | "Mid" | "Bot" }> = [
  { lane: "top", label: "Top" },
  { lane: "mid", label: "Mid" },
  { lane: "bot", label: "Bot" },
];

export function decodeLanes(laneProbsArr: Float32Array): LanePrediction[] {
  return LANE_DEFS.map((def, idx) => {
    const base = idx * 3;
    const rawRad = Math.max(0, sanitizeFinite(laneProbsArr[base], 1 / 3));
    const rawEven = Math.max(0, sanitizeFinite(laneProbsArr[base + 1], 1 / 3));
    const rawDire = Math.max(0, sanitizeFinite(laneProbsArr[base + 2], 1 / 3));
    const sum = rawRad + rawEven + rawDire;
    const radiantProb = sum > 0 ? rawRad / sum : 1 / 3;
    const evenProb = sum > 0 ? rawEven / sum : 1 / 3;
    const direProb = sum > 0 ? rawDire / sum : 1 / 3;
    const [radiantPct = 34, evenPct = 33, direPct = 33] = roundPercentagesTo100([
      radiantProb,
      evenProb,
      direProb,
    ]);

    const diff = radiantProb - direProb;
    const verdictSide = diff >= 0.07 ? "radiant" : diff <= -0.07 ? "dire" : "even";
    const verdict =
      verdictSide === "radiant"
        ? "Radiant favored"
        : verdictSide === "dire"
          ? "Dire favored"
          : "Even";

    return {
      lane: def.lane,
      label: def.label,
      radiantProb,
      evenProb,
      direProb,
      radiantPct,
      evenPct,
      direPct,
      verdict,
      verdictSide,
    };
  });
}

function makeMonotoneRange(
  rawLow: number,
  rawMid: number,
  rawHigh: number,
  multiplier: number,
  decimals: 0 | 1 = 0,
): RangeMetric {
  const safeRawLow = Math.max(0, sanitizeFinite(rawLow, 0));
  const safeRawMid = Math.max(safeRawLow, sanitizeFinite(rawMid, safeRawLow));
  const safeRawHigh = Math.max(safeRawMid, sanitizeFinite(rawHigh, safeRawMid));

  const factor = decimals === 1 ? 10 : 1;
  const roundVal = (x: number) => sanitizeFinite(Math.round(x * factor) / factor, 0);

  const low = roundVal(safeRawLow * multiplier);
  const mid = Math.max(low, roundVal(safeRawMid * multiplier));
  const high = Math.max(mid, roundVal(safeRawHigh * multiplier));
  return { low, mid, high };
}

export function decodeEconomyStatsForDuration(
  econArr: Float32Array,
  slotIdx: number,
  effectiveDurationMinutes: number,
): HeroEconomyStats {
  const effDur = Math.max(15, sanitizeFinite(effectiveDurationMinutes, 35));
  const per10m = effDur / 10;
  const slotBase = slotIdx * 6 * 3;

  const readTriplet = (metricIdx: number): [number, number, number] => {
    const o = slotBase + metricIdx * 3;
    return [
      econArr[o] ?? 0,
      econArr[o + 1] ?? 0,
      econArr[o + 2] ?? 0,
    ];
  };

  const [gpmLo, gpmMid, gpmHi] = readTriplet(0);
  const [xpmLo, xpmMid, xpmHi] = readTriplet(1);
  const [csLo, csMid, csHi] = readTriplet(2);
  const [kLo, kMid, kHi] = readTriplet(3);
  const [dLo, dMid, dHi] = readTriplet(4);
  const [aLo, aMid, aHi] = readTriplet(5);

  return {
    gpm: makeMonotoneRange(gpmLo, gpmMid, gpmHi, 1, 0),
    xpm: makeMonotoneRange(xpmLo, xpmMid, xpmHi, 1, 0),
    csPerMin: makeMonotoneRange(csLo, csMid, csHi, 1, 1),
    csTotal: makeMonotoneRange(csLo, csMid, csHi, effDur, 0),
    kills: makeMonotoneRange(kLo, kMid, kHi, per10m, 0),
    deaths: makeMonotoneRange(dLo, dMid, dHi, per10m, 0),
    assists: makeMonotoneRange(aLo, aMid, aHi, per10m, 0),
  };
}

export function decodeCombatStats(
  combatRatesArr: Float32Array,
  combatSharesArr: Float32Array,
  slotIdx: number,
  showHealingShare: boolean,
): HeroCombatStats {
  const rateBase = slotIdx * 4;
  const teamIdx = slotIdx < 5 ? 0 : 1;
  const posInTeam = slotIdx % 5;
  const shareBase = (teamIdx * 5 + posInTeam) * 4;

  const getRate = (statIdx: number) =>
    sanitizeFinite(Math.max(0, combatRatesArr[rateBase + statIdx] ?? 0), 0);
  const getShare = (statIdx: number) =>
    Math.min(1, Math.max(0, sanitizeFinite(combatSharesArr[shareBase + statIdx], 0.2)));

  return {
    heroDamage: { dpm: getRate(0), teamShare: getShare(0) },
    towerDamage: { dpm: getRate(1), teamShare: getShare(1) },
    damageTaken: { dpm: getRate(2), teamShare: getShare(2) },
    healing: { dpm: getRate(3), teamShare: getShare(3) },
    showHealingShare,
  };
}

export function formatRangeText(metric: RangeMetric, decimals: 0 | 1 = 0): string {
  const fmt = (n: number) =>
    decimals === 1 ? sanitizeFinite(n, 0).toFixed(1) : String(Math.round(sanitizeFinite(n, 0)));
  const lo = fmt(metric.low);
  const hi = fmt(metric.high);
  return lo === hi ? lo : `${lo}–${hi}`;
}

export function formatLead10m(leadValue: number): {
  side: "radiant" | "dire" | "even";
  label: string;
} {
  const safe = sanitizeFinite(leadValue, 0);
  const rounded = Math.round(safe / 10) * 10;
  const absVal = Math.abs(rounded);
  if (absVal < 100) {
    return { side: "even", label: "Even" };
  }
  const magText =
    absVal >= 1000
      ? `${(absVal / 1000).toFixed(1).replace(/\.0$/, "")}k`
      : String(absVal);
  const side = rounded > 0 ? "radiant" : "dire";
  const teamName = side === "radiant" ? "Radiant" : "Dire";
  return {
    side,
    label: `${teamName} +${magText}`,
  };
}

