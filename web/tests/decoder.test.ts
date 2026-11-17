import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import * as ort from "onnxruntime-web/wasm";
import { describe, expect, it } from "vitest";

import {
  NON_LEVELABLE_ABILITY_NAMES,
  getAbilityColorKey,
  getLevelableSkillEntries,
  isLevelableAbility,
} from "../src/data/abilityDisplay";
import {
  assignHeroToSlot,
  createEmptyDraft,
  createRandomDraft,
  parseDraftFromSearch,
  serializeDraftToSearch,
  swapSlotHeroes,
} from "../src/data/draftPresets";
import {
  BOS_ITEM_ID,
  EOS_ITEM_ID,
  PAD_ITEM_ID,
  decodeCombatStats,
  decodeEconomyStatsForDuration,
  decodeItemsGreedy,
  decodeLanes,
  decodeSkillBuild1To18,
  decodeTalentChoices,
  formatLead10m,
  formatRangeText,
  formatWinOddsPair,
  getMaxUltRankForLevel,
  getSkippedSkillLevel,
  roundPercentagesTo100,
  selectNextItemToken,
} from "../src/ml/decoder";
import {
  predictDraft,
  validateBundleMetadata,
  type LoadedOracleBundle,
} from "../src/ml/inference";
import { configureOrtEnvironment, normalizeModelBaseUrl } from "../src/ml/ortSetup";
import type { AbilityMeta, HeroMeta, ModelConfigJson, VocabJson } from "../src/types/oracle";

const THIS_DIR = path.dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = path.resolve(THIS_DIR, "../public/models");
const ORT_DIR = path.resolve(THIS_DIR, "../public/ort");

function makeStandardHero(overrides: Partial<HeroMeta> = {}): HeroMeta {
  return {
    idx: 0,
    id: 2,
    short_name: "axe",
    display_name: "Axe",
    primary_attr: "str",
    attack_type: "Melee",
    roles: ["Initiator"],
    img: null,
    icon: null,
    skill_ability_ids: [5007, 5008, 5009, 5010],
    talent_tiers: {
      "10": [601, 602],
      "15": [603, 604],
      "20": [605, 606],
      "25": [607, 608],
    },
    ...overrides,
  };
}

function makeAbilitiesMap(): Record<string, AbilityMeta> {
  return {
    "5007": { id: 5007, name: "axe_berserkers_call", display_name: "Berserker's Call", is_talent: false, img: null },
    "5008": { id: 5008, name: "axe_battle_hunger", display_name: "Battle Hunger", is_talent: false, img: null },
    "5009": { id: 5009, name: "axe_counter_helix", display_name: "Counter Helix", is_talent: false, img: null },
    "5010": { id: 5010, name: "axe_culling_blade", display_name: "Culling Blade", is_talent: false, img: null },
    "601": { id: 601, name: "tal_10_l", display_name: "+10 Str", is_talent: true, img: null },
    "602": { id: 602, name: "tal_10_r", display_name: "+25 MS", is_talent: true, img: null },
    "603": { id: 603, name: "tal_15_dup", display_name: "+30 Damage", is_talent: true, img: null },
    "604": { id: 604, name: "tal_15_dup", display_name: "+30 Damage", is_talent: true, img: null },
    "605": { id: 605, name: "tal_15_alt", display_name: "+120 Battle Hunger DPS", is_talent: true, img: null },
  };
}

describe("selectNextItemToken & decodeItemsGreedy", () => {
  it("blocks PAD, BOS, and already picked >=3 tokens", () => {
    const logits = new Float32Array([99, 88, 1.5, 10, 8, 5]);
    expect(selectNextItemToken(logits, 0, 6, [], BOS_ITEM_ID)).toBe(3);
    expect(selectNextItemToken(logits, 0, 6, [3], BOS_ITEM_ID)).toBe(4);
    expect(selectNextItemToken(logits, 0, 6, [3, 4, 5], BOS_ITEM_ID)).toBe(EOS_ITEM_ID);
    expect(selectNextItemToken(logits, 0, 6, [], PAD_ITEM_ID)).not.toBe(PAD_ITEM_ID);
  });

  it("stops streams at EOS while decoding unfinished streams cleanly", async () => {
    const dModel = 4;
    const vocabSize = 6;
    const mem = new Float32Array(10 * dModel).fill(0.1);

    const decoded = await decodeItemsGreedy(
      async (_mem, batchSize, _d, _target, _seq, seqLen) => {
        const logits = new Float32Array(batchSize * seqLen * vocabSize).fill(-10);
        const minutes = new Float32Array(batchSize * seqLen).fill(15);
        for (let s = 0; s < batchSize; s++) {
          const offset = (s * seqLen + (seqLen - 1)) * vocabSize;
          if (seqLen === 1) {
            logits[offset + (s === 0 ? 3 : 4)] = 5.0;
            minutes[s * seqLen + (seqLen - 1)] = 12.5 + s;
          } else if (seqLen === 2) {
            logits[offset + (s === 0 ? 3 : EOS_ITEM_ID)] = 6.0;
            logits[offset + (s === 0 ? 5 : -1)] = s === 0 ? 4.0 : -10;
            minutes[s * seqLen + (seqLen - 1)] = 21.0;
          } else {
            logits[offset + EOS_ITEM_ID] = 9.0;
          }
        }
        return { logits, minutes };
      },
      mem,
      dModel,
      vocabSize,
      4,
      BOS_ITEM_ID,
      EOS_ITEM_ID,
    );

    expect(decoded).toHaveLength(10);
    // Stream 0 tried to pick 3 again on step 2 -> duplicate masked -> picks 5, then EOS.
    expect(decoded[0]).toEqual([
      { itemToken: 3, expectedMinute: 12.5 },
      { itemToken: 5, expectedMinute: 21.0 },
    ]);
    // Streams 1..9 picked 4 on step 1 and EOS on step 2.
    expect(decoded[1]).toEqual([{ itemToken: 4, expectedMinute: 13.5 }]);
  });

  it("reads per-token [N, T, V] item minutes and clamps monotonically across steps", async () => {
    const dModel = 4;
    const vocabSize = 6;
    const mem = new Float32Array(10 * dModel).fill(0.1);

    const decoded = await decodeItemsGreedy(
      async (_mem, batchSize, _d, _target, _seq, seqLen) => {
        const logits = new Float32Array(batchSize * seqLen * vocabSize).fill(-10);
        // [N, T, V] flattened per-token minute predictions.
        const minutes = new Float32Array(batchSize * seqLen * vocabSize).fill(5);
        for (let s = 0; s < batchSize; s++) {
          const logitOffset = (s * seqLen + (seqLen - 1)) * vocabSize;
          const minOffset = logitOffset;
          if (seqLen === 1) {
            logits[logitOffset + 3] = 8.0;
            // Distractor token 4 has huge minute, selected token 3 has 14.0.
            minutes[minOffset + 3] = 14.0;
            minutes[minOffset + 4] = 55.0;
          } else if (seqLen === 2) {
            logits[logitOffset + 4] = 8.0;
            // Selected token 4 dips below 14.0 -> monotonic clamp lifts it to 14.0.
            minutes[minOffset + 4] = 11.2;
          } else if (seqLen === 3) {
            logits[logitOffset + 5] = 8.0;
            minutes[minOffset + 5] = 24.5;
          } else {
            logits[logitOffset + EOS_ITEM_ID] = 9.0;
          }
        }
        return { logits, minutes };
      },
      mem,
      dModel,
      vocabSize,
      5,
      BOS_ITEM_ID,
      EOS_ITEM_ID,
    );

    expect(decoded[0]).toEqual([
      { itemToken: 3, expectedMinute: 14.0 },
      { itemToken: 4, expectedMinute: 14.0 },
      { itemToken: 5, expectedMinute: 24.5 },
    ]);
  });

  it("applies eosPenalty, eosStopProb, and maxMinute cutoff after step 0", async () => {
    const logits = new Float32Array([-10, -10, 4.5, 4.0, 3.8, -5]);
    // Without penalty, EOS (4.5) wins over item 3 (4.0); with penalty 0.75, item 3 wins (4.0 > 3.75).
    expect(selectNextItemToken(logits, 0, 6, [], BOS_ITEM_ID, EOS_ITEM_ID, 0)).toBe(EOS_ITEM_ID);
    expect(selectNextItemToken(logits, 0, 6, [], BOS_ITEM_ID, EOS_ITEM_ID, 0.75)).toBe(3);

    // Diffuse support item logits: P(EOS) ~ 0.36 < 0.48 -> eosStopProb=0.48 suppresses premature EOS
    const diffuseLogits = new Float32Array([-100, -100, 1.0, 0.45, 0.4, 0.35]);
    expect(selectNextItemToken(diffuseLogits, 0, 6, [], BOS_ITEM_ID, EOS_ITEM_ID, 0, 0)).toBe(EOS_ITEM_ID);
    expect(selectNextItemToken(diffuseLogits, 0, 6, [], BOS_ITEM_ID, EOS_ITEM_ID, 0, 0.48)).toBe(3);

    const dModel = 4;
    const vocabSize = 6;
    const mem = new Float32Array(10 * dModel).fill(0.1);
    const decodedShort = await decodeItemsGreedy(
      async (_mem, batchSize, _d, _target, _seq, seqLen) => {
        const stepLogits = new Float32Array(batchSize * seqLen * vocabSize).fill(-10);
        const stepMins = new Float32Array(batchSize * seqLen * vocabSize).fill(0);
        for (let s = 0; s < batchSize; s++) {
          const offset = (s * seqLen + (seqLen - 1)) * vocabSize;
          if (seqLen === 1) {
            stepLogits[offset + 3] = 5.0;
            stepMins[offset + 3] = 14.0;
          } else if (seqLen === 2) {
            stepLogits[offset + 4] = 5.0;
            stepMins[offset + 4] = 22.5;
          } else if (seqLen === 3) {
            stepLogits[offset + 5] = 5.0;
            stepMins[offset + 5] = 28.0; // Exceeds maxMinute=25
          } else {
            stepLogits[offset + EOS_ITEM_ID] = 8.0;
          }
        }
        return { logits: stepLogits, minutes: stepMins };
      },
      mem,
      dModel,
      vocabSize,
      5,
      BOS_ITEM_ID,
      EOS_ITEM_ID,
      0.75,
      25,
    );

    expect(decodedShort[0]).toEqual([
      { itemToken: 3, expectedMinute: 14.0 },
      { itemToken: 4, expectedMinute: 22.5 },
    ]);
  });
});

describe("decodeSkillBuild1To18 & decodeTalentChoices", () => {
  it("enforces ult level 6/12/18 locks and ceil(level/2) basic caps on 4-skill heroes", () => {
    const hero = makeStandardHero();
    const numClasses = 16;
    const slotTalent = 14;
    const slotAttr = 15;
    const logits = new Float32Array(18 * numClasses).fill(-2);

    // Try spamming ult (3) and skill 0 at every level to test legal masking.
    for (let lvl = 0; lvl < 18; lvl++) {
      logits[lvl * numClasses + 3] = 10.0;
      logits[lvl * numClasses + 0] = 8.0;
      logits[lvl * numClasses + 1] = 6.0;
      logits[lvl * numClasses + 2] = 4.0;
      logits[lvl * numClasses + slotTalent] = 9.0;
      logits[lvl * numClasses + slotAttr] = 1.0;
    }

    const steps = decodeSkillBuild1To18(hero, logits, 0, numClasses, slotTalent, slotAttr, makeAbilitiesMap());
    expect(steps).toHaveLength(18);
    expect(getSkippedSkillLevel(hero.id)).toBe(17);
    // Standard hero skips level 17 cleanly without fabricating +2 Attributes.
    expect(steps[16]!.level).toBe(17);
    expect(steps[16]!.kind).toBe("empty");
    expect(steps[16]!.abilityId).toBeNull();
    expect(steps[16]!.shortTag).toBe("—");
    // First ult must occur exactly at level 6, second at level 12, third at level 18.
    expect(steps.filter((s) => s.slotClass === 3).map((s) => s.level)).toEqual([6, 12, 18]);
    // First talent occurs at level 10, second at level 15 (none before 10).
    expect(steps.filter((s) => s.kind === "talent").map((s) => s.level)).toEqual([10, 15]);
    // Level 1 cannot take Q rank 2 at level 2 because ceil(2/2) = 1.
    expect(steps[0]!.shortTag).toBe("Q");
    expect(steps[1]!.shortTag).toBe("W");

    // Even if slotAttr has a huge logit at level 17 on a standard hero, level 17 stays empty.
    const highAttrLogits = new Float32Array(logits);
    highAttrLogits[16 * numClasses + slotAttr] = 500.0;
    const stepsHighAttr = decodeSkillBuild1To18(
      hero,
      highAttrLogits,
      0,
      numClasses,
      slotTalent,
      slotAttr,
      makeAbilitiesMap(),
    );
    expect(stepsHighAttr[16]!.kind).toBe("empty");
    expect(stepsHighAttr[16]!.abilityId).toBeNull();
  });

  it("allows Attribute Bonus from level 6 when encoder logits prefer early stats, capped at 7", () => {
    const hero = makeStandardHero();
    const numClasses = 16;
    const slotTalent = 14;
    const slotAttr = 15;
    const logits = new Float32Array(18 * numClasses).fill(-2);

    for (let lvl = 0; lvl < 18; lvl++) {
      logits[lvl * numClasses + 3] = 8.0; // Ult
      logits[lvl * numClasses + 0] = 6.0;
      logits[lvl * numClasses + 1] = 5.0;
      logits[lvl * numClasses + 2] = 4.0;
      logits[lvl * numClasses + slotTalent] = 7.0;
      // Strongly prefer Attribute Bonus everywhere, should unlock at level 6+ up to 7 ranks.
      logits[lvl * numClasses + slotAttr] = 12.0;
    }

    const steps = decodeSkillBuild1To18(hero, logits, 0, numClasses, slotTalent, slotAttr, makeAbilitiesMap());
    const attrLevels = steps.filter((s) => s.kind === "attribute").map((s) => s.level);
    expect(attrLevels.every((lvl) => lvl >= 6 && lvl !== 17)).toBe(true);
    expect(attrLevels).toHaveLength(7);
    expect(attrLevels.some((lvl) => lvl < 15)).toBe(true);
  });

  it("supports non-4-skill heroes (Anti-Mage 5 skills & Invoker 8-rank orbs) safely", () => {
    const invoker = makeStandardHero({
      id: 74,
      short_name: "invoker",
      display_name: "Invoker",
      skill_ability_ids: Array.from({ length: 14 }, (_, i) => 5370 + i),
    });
    expect(getSkippedSkillLevel(invoker.id)).toBeNull();
    const numClasses = 16;
    const logits = new Float32Array(18 * numClasses).fill(-5);
    for (let lvl = 0; lvl < 18; lvl++) {
      // Mask invokable slots 3..13 below -5000, keep orbs 0,1,2 high.
      for (let s = 3; s < 14; s++) {
        logits[lvl * numClasses + s] = -10000;
      }
      logits[lvl * numClasses + 0] = 10;
      logits[lvl * numClasses + 1] = 8;
      logits[lvl * numClasses + 2] = 6;
    }

    const steps = decodeSkillBuild1To18(invoker, logits, 0, numClasses, 14, 15, {});
    expect(steps).toHaveLength(18);
    // Invoker learns an ability on every level 1..18 including level 17.
    expect(steps.some((s) => s.kind === "empty")).toBe(false);
    expect(steps[16]!.kind).toBe("ability");
    // In 7.41+, Invoker orbs (Q/W/E) can reach 8 ranks (8 + 8 + 2 = 18 levels).
    expect(steps.filter((s) => s.slotClass === 0)).toHaveLength(8);
    expect(steps.filter((s) => s.slotClass === 1)).toHaveLength(8);
    expect(steps.filter((s) => s.slotClass === 2)).toHaveLength(2);
  });

  it("filters innate abilities while keeping ult level locks and Q/W/E/R badges intact", () => {
    const underlord = makeStandardHero({
      id: 108,
      short_name: "abyssal_underlord",
      display_name: "Underlord",
      skill_ability_ids: [5613, 5614, 5615, 5617, 5616],
    });
    const underlordAbilities: Record<string, AbilityMeta> = {
      "5613": { id: 5613, name: "abyssal_underlord_firestorm", display_name: "Firestorm", is_talent: false, img: null },
      "5614": { id: 5614, name: "abyssal_underlord_pit_of_malice", display_name: "Pit of Malice", is_talent: false, img: null },
      "5615": { id: 5615, name: "abyssal_underlord_atrophy_aura", display_name: "Atrophy Aura", is_talent: false, img: null },
      "5617": { id: 5617, name: "abyssal_underlord_raid_boss", display_name: "Invading Force", is_talent: false, img: null },
      "5616": { id: 5616, name: "abyssal_underlord_dark_portal", display_name: "Fiend's Gate", is_talent: false, img: null },
    };

    const levelable = getLevelableSkillEntries(underlord, underlordAbilities);
    expect(levelable.map((e) => [e.origSlotIdx, e.shortTag])).toEqual([
      [0, "Q"],
      [1, "W"],
      [2, "E"],
      [4, "R"],
    ]);

    const numClasses = 16;
    const logits = new Float32Array(18 * numClasses).fill(-2);
    for (let lvl = 0; lvl < 18; lvl++) {
      // Give the innate ability (slot 3) the highest logit and ult (slot 4) next highest.
      logits[lvl * numClasses + 3] = 20.0;
      logits[lvl * numClasses + 4] = 10.0;
      logits[lvl * numClasses + 0] = 8.0;
      logits[lvl * numClasses + 1] = 6.0;
      logits[lvl * numClasses + 2] = 4.0;
      logits[lvl * numClasses + 14] = 9.0;
      logits[lvl * numClasses + 15] = 1.0;
    }

    const steps = decodeSkillBuild1To18(underlord, logits, 0, numClasses, 14, 15, underlordAbilities);
    expect(steps).toHaveLength(18);
    expect(steps.some((s) => s.abilityId === 5617)).toBe(false);
    expect(steps.filter((s) => s.abilityId === 5616).map((s) => s.level)).toEqual([6, 12, 18]);
    expect(steps.find((s) => s.level === 6)?.shortTag).toBe("R");
  });

  it("deduplicates reworked talent entries sharing internal ability name", () => {
    const hero = makeStandardHero({
      talent_tiers: {
        "10": [601, 602],
        "15": [603, 604, 605],
        "20": [601, 602],
        "25": [601, 602],
      },
    });
    const logits = new Float32Array([1.0, 0.2, -0.5, 2.0, 0.5, 0.5, 0.1, -0.1]);
    const talents = decodeTalentChoices(hero, logits, 0, makeAbilitiesMap());
    expect(talents[0]!.chosenSide).toBe(0);
    expect(talents[1]!.chosenSide).toBe(1);
    expect(talents[1]!.options.map((o) => o.id)).toEqual([603, 605]);
    expect(talents[1]!.abilityId).toBe(605);
  });

  it("audits vocab.json heroes and enforces slot/ultimate rules for Morphling, Dark Willow, Meepo, and 5-skill heroes", () => {
    const vocab = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, "vocab.json"), "utf8")) as VocabJson;
    const byId = vocab.abilities.by_id;

    // Only Invoker (3 orbs) differs from 4 levelable skills; paired skills (Dark Willow, Kez, Morphling, Troll, etc.) collapse cleanly.
    const nonStandard = Object.values(vocab.heroes.by_id)
      .map((hero) => ({ hero, entries: getLevelableSkillEntries(hero, byId) }))
      .filter(({ entries }) => entries.length !== 4)
      .map(({ hero, entries }) => [hero.id, entries.length]);
    expect(nonStandard).toEqual([[74, 3]]);

    for (const hid of [10, 11, 64, 95, 114, 145]) {
      const hero = vocab.heroes.by_id[String(hid)]!;
      const entries = getLevelableSkillEntries(hero, byId);
      expect(entries.map((e) => e.shortTag)).toEqual(["Q", "W", "E", "R"]);
      expect(entries.map((e) => e.isUltimate)).toEqual([false, false, false, true]);
    }

    // Regression: 5-skill hero (4 basic skills Q/W/E/D + ultimate R at index 4) must not mark D as ult
    // or unlock R before levels 6, 12, 18.
    const fiveSkillHero = makeStandardHero({
      id: 999,
      skill_ability_ids: [7101, 7102, 7103, 7104, 7105],
    });
    const fiveSkillAbilities: Record<string, AbilityMeta> = {
      "7101": { id: 7101, name: "custom_skill_q", display_name: "Wave", is_talent: false, img: null },
      "7102": { id: 7102, name: "custom_skill_w", display_name: "Strike", is_talent: false, img: null },
      "7103": { id: 7103, name: "custom_skill_e", display_name: "Shift", is_talent: false, img: null },
      "7104": { id: 7104, name: "custom_skill_d", display_name: "Stance", is_talent: false, img: null },
      "7105": { id: 7105, name: "custom_ult_r", display_name: "Morph", is_talent: false, img: null },
    };
    const fiveEntries = getLevelableSkillEntries(fiveSkillHero, fiveSkillAbilities);
    expect(fiveEntries.map((e) => e.shortTag)).toEqual(["Q", "W", "E", "D", "R"]);
    expect(fiveEntries.map((e) => e.isUltimate)).toEqual([false, false, false, false, true]);
    expect(fiveEntries.map((e) => e.colorKey)).toEqual(["q", "w", "e", "extra", "r"]);

    const numClasses = 16;
    const fiveLogits = new Float32Array(18 * numClasses).fill(-2);
    for (let lvl = 0; lvl < 18; lvl++) {
      // Force highest logit onto ult (slot 4) and second highest onto 4th basic skill (slot 3).
      fiveLogits[lvl * numClasses + 4] = 30.0;
      fiveLogits[lvl * numClasses + 3] = 20.0;
      fiveLogits[lvl * numClasses + 0] = 10.0;
      fiveLogits[lvl * numClasses + 1] = 8.0;
      fiveLogits[lvl * numClasses + 2] = 6.0;
      fiveLogits[lvl * numClasses + 14] = 9.0;
      fiveLogits[lvl * numClasses + 15] = 1.0;
    }
    const fiveSteps = decodeSkillBuild1To18(fiveSkillHero, fiveLogits, 0, numClasses, 14, 15, fiveSkillAbilities);
    expect(fiveSteps.filter((s) => s.abilityId === 7105).map((s) => s.level)).toEqual([6, 12, 18]);
    expect(fiveSteps.filter((s) => s.abilityId === 7105).every((s) => s.isUltimate && s.shortTag === "R")).toBe(true);
    const dSteps = fiveSteps.filter((s) => s.abilityId === 7104);
    expect(dSteps.every((s) => !s.isUltimate && s.shortTag === "D")).toBe(true);
    expect(dSteps.length).toBeLessThanOrEqual(4);
    expect(dSteps[0]?.level).toBe(1);

    // Dark Willow (id=119): Bedlam (D) is canonical ult (idx 3), paired Terrorize (idx 5) collapsed cleanly.
    const willow = vocab.heroes.by_id["119"]!;
    const willowEntries = getLevelableSkillEntries(willow, byId);
    expect(willowEntries.map((e) => e.shortTag)).toEqual(["Q", "W", "E", "D"]);
    expect(willowEntries.map((e) => e.isUltimate)).toEqual([false, false, false, true]);
    expect(willowEntries.map((e) => e.colorKey)).toEqual(["q", "w", "e", "r"]);

    const willowLogits = new Float32Array(18 * numClasses).fill(-2);
    for (let lvl = 0; lvl < 18; lvl++) {
      willowLogits[lvl * numClasses + 3] = 30.0; // Bedlam
      willowLogits[lvl * numClasses + 5] = 25.0; // Terrorize
      willowLogits[lvl * numClasses + 0] = 10.0;
      willowLogits[lvl * numClasses + 1] = 8.0;
      willowLogits[lvl * numClasses + 2] = 6.0;
      willowLogits[lvl * numClasses + 14] = 4.0;
      willowLogits[lvl * numClasses + 15] = 1.0;
    }
    const willowSteps = decodeSkillBuild1To18(willow, willowLogits, 0, numClasses, 14, 15, byId);
    expect(willowSteps.filter((s) => s.isUltimate).map((s) => s.level)).toEqual([6, 12, 18]);

    // Meepo (id=82): Divided We Stand unlocks at levels 2, 10, 17 in 7.41f and level 18 is skipped.
    expect(getSkippedSkillLevel(82)).toBe(18);
    expect(getMaxUltRankForLevel(82, 1)).toBe(0);
    expect(getMaxUltRankForLevel(82, 2)).toBe(1);
    expect(getMaxUltRankForLevel(82, 10)).toBe(2);
    expect(getMaxUltRankForLevel(82, 17)).toBe(3);
    const meepo = vocab.heroes.by_id["82"]!;
    const meepoLogits = new Float32Array(18 * numClasses).fill(-2);
    for (let lvl = 0; lvl < 18; lvl++) {
      meepoLogits[lvl * numClasses + 5] = 30.0; // meepo_divided_we_stand
      meepoLogits[lvl * numClasses + 0] = 10.0;
      meepoLogits[lvl * numClasses + 1] = 8.0;
      meepoLogits[lvl * numClasses + 2] = 6.0;
      meepoLogits[lvl * numClasses + 14] = 4.0;
      meepoLogits[lvl * numClasses + 15] = 1.0;
    }
    const meepoSteps = decodeSkillBuild1To18(meepo, meepoLogits, 0, numClasses, 14, 15, byId);
    expect(meepoSteps.filter((s) => s.isUltimate).map((s) => s.level)).toEqual([2, 10, 17]);
    expect(meepoSteps[17]!.level).toBe(18);
    expect(meepoSteps[17]!.kind).toBe("empty");
    expect(meepoSteps[17]!.abilityId).toBeNull();
  });
});

describe("draftPresets & ortSetup", () => {
  it("keeps 10 heroes unique across assignments, swaps, random drafts, and URL serialization", () => {
    const heroes = Array.from({ length: 12 }, (_, i) =>
      makeStandardHero({ id: i + 1, idx: i, short_name: `hero_${i + 1}`, display_name: `Hero ${i + 1}` }),
    );
    const heroesById = Object.fromEntries(heroes.map((h) => [String(h.id), h]));

    let draft = createEmptyDraft();
    draft = assignHeroToSlot(draft, 0, 5);
    draft = assignHeroToSlot(draft, 6, 5);
    // Re-assigning hero 5 to slot 6 removes hero 5 from slot 0.
    expect(draft[0]!.heroId).toBeNull();
    expect(draft[6]!.heroId).toBe(5);

    draft = swapSlotHeroes(draft, 0, 6);
    expect(draft[0]!.heroId).toBe(5);
    expect(draft[6]!.heroId).toBeNull();

    const full = createRandomDraft(heroes);
    const ids = full.map((s) => s.heroId);
    expect(new Set(ids).size).toBe(10);

    const searchDefault = serializeDraftToSearch(full, 0);
    expect(searchDefault).not.toContain("pace=");
    expect(searchDefault).not.toContain("pro=");
    const parsedDefault = parseDraftFromSearch(searchDefault, heroesById);
    expect(parsedDefault).not.toBeNull();
    expect(parsedDefault!.matchPace).toBe(0);

    const search50 = serializeDraftToSearch(full, 50);
    expect(search50).toContain("pace=50");
    const parsed = parseDraftFromSearch(search50, heroesById);
    expect(parsed).not.toBeNull();
    expect(parsed!.matchPace).toBe(50);
    expect(parsed!.slots.map((s) => s.heroId)).toEqual(ids);

    // Invalid pace parameter safely defaults to 0 (Auto)
    const parsedInvalidPace = parseDraftFromSearch(`${searchDefault}&pace=99`, heroesById);
    expect(parsedInvalidPace!.matchPace).toBe(0);
  });

  it("rounds win odds, lane probabilities, 10m leads, and duration-scaled KDA/CS cleanly", () => {
    const winPair = formatWinOddsPair(0.5004);
    expect(Number(winPair.radPctText) + Number(winPair.direPctText)).toBeCloseTo(100, 1);

    const rounded = roundPercentagesTo100([0.3334, 0.3334, 0.3332]);
    expect(rounded.reduce((a, b) => a + b, 0)).toBe(100);

    const laneArr = new Float32Array([
      0.52, 0.28, 0.20,
      0.34, 0.34, 0.32,
      0.18, 0.27, 0.55,
    ]);
    const lanes = decodeLanes(laneArr);
    expect(lanes.map((l) => l.verdictSide)).toEqual(["radiant", "even", "dire"]);
    for (const lane of lanes) {
      expect(lane.radiantPct + lane.evenPct + lane.direPct).toBe(100);
    }

    expect(formatLead10m(1842)).toEqual({ side: "radiant", label: "Radiant +1.8k" });
    expect(formatLead10m(-64)).toEqual({ side: "even", label: "Even" });
    expect(formatLead10m(-952)).toEqual({ side: "dire", label: "Dire +950" });

    // Slot 0 floats: [gpm q25/50/75, xpm, cs/min, k/10m, d/10m, a/10m]
    const econArr = new Float32Array(10 * 6 * 3);
    econArr.set([510, 560, 610, 580, 640, 700, 6.0, 7.0, 8.0, 1.5, 2.0, 2.5, 0.8, 1.0, 1.4, 2.2, 3.0, 4.0], 0);
    const econ25 = decodeEconomyStatsForDuration(econArr, 0, 25);
    const econ50 = decodeEconomyStatsForDuration(econArr, 0, 50);

    expect(econ25.csTotal).toEqual({ low: 150, mid: 175, high: 200 });
    expect(econ50.csTotal).toEqual({ low: 300, mid: 350, high: 400 });
    expect(econ50.csTotal.mid).toBe(econ25.csTotal.mid * 2);
    expect(econ50.kills.mid).toBeGreaterThan(econ25.kills.mid);
    expect(Number.isInteger(econ50.csTotal.low)).toBe(true);
    expect(Number.isInteger(econ50.kills.high)).toBe(true);
    expect(formatRangeText(econ25.gpm)).toBe("510–610");

    const rates = new Float32Array(10 * 4);
    rates.set([620, 190, 810, 0], 0);
    const shares = new Float32Array(2 * 5 * 4);
    shares.set([0.32, 0.45, 0.24, 1.0], 0);
    const combatHiddenHeal = decodeCombatStats(rates, shares, 0, false);
    expect(combatHiddenHeal.showHealingShare).toBe(false);
    expect(combatHiddenHeal.heroDamage.teamShare).toBeCloseTo(0.32, 4);
  });

  it("forces single-threaded WASM when crossOriginIsolated is unavailable", () => {
    const mockEnv = { env: { wasm: {} as { numThreads?: number; proxy?: boolean; wasmPaths?: string } } };
    configureOrtEnvironment(mockEnv);
    expect(mockEnv.env.wasm.numThreads).toBe(1);
    expect(mockEnv.env.wasm.proxy).toBe(false);
    expect(normalizeModelBaseUrl("  ")).toBe("./models");
    expect(normalizeModelBaseUrl("https://example.com/models///")).toBe("https://example.com/models");
  });

  it("maps ability color keys and excludes sub-abilities via suffix pattern", () => {
    expect(getAbilityColorKey("Q")).toBe("q");
    expect(getAbilityColorKey("W")).toBe("w");
    expect(getAbilityColorKey("E")).toBe("e");
    expect(getAbilityColorKey("R")).toBe("r");
    expect(getAbilityColorKey("D")).toBe("extra");
    expect(getAbilityColorKey("D", true)).toBe("r");
    expect(getAbilityColorKey("4")).toBe("extra");
    expect(getAbilityColorKey("5")).toBe("extra");

    expect(
      isLevelableAbility({ id: 1, name: "puck_illusory_orb", display_name: "Orb", is_talent: false, img: null }),
    ).toBe(true);
    expect(
      isLevelableAbility({ id: 2, name: "crystal_maiden_freezing_field_stop", display_name: "Stop", is_talent: false, img: null }),
    ).toBe(false);
    expect(
      isLevelableAbility({ id: 3, name: "phoenix_icarus_dive_stop", display_name: "Stop", is_talent: false, img: null }),
    ).toBe(false);
    expect(
      isLevelableAbility({ id: 4, name: "elder_titan_ancestral_spirit_cancel", display_name: "Cancel", is_talent: false, img: null }),
    ).toBe(false);
    expect(
      isLevelableAbility({ id: 5, name: "pangolier_gyroshell_release", display_name: "Release", is_talent: false, img: null }),
    ).toBe(false);
    expect(
      isLevelableAbility({ id: 6, name: "some_hero_skill_hidden1", display_name: "Hidden", is_talent: false, img: null }),
    ).toBe(false);
  });
});

describe("real v2 ONNX bundle inference in Node/WASM", () => {
  it("validates bundle metadata schema versions and rejects v1 config", () => {
    const vocab = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, "vocab.json"), "utf8")) as VocabJson;
    const modelConfig = JSON.parse(
      fs.readFileSync(path.join(MODELS_DIR, "model_config.json"), "utf8"),
    ) as ModelConfigJson;

    expect(() => validateBundleMetadata(vocab, modelConfig)).not.toThrow();
    expect(() =>
      validateBundleMetadata(vocab, { ...modelConfig, encoder_arch_version: 3 }),
    ).not.toThrow();
    expect(() =>
      validateBundleMetadata(vocab, { ...modelConfig, encoder_arch_version: 2 }),
    ).toThrow(/Incompatible encoder architecture version/);
    expect(() =>
      validateBundleMetadata(vocab, { ...modelConfig, encoder_arch_version: undefined }),
    ).toThrow(/Incompatible encoder architecture version/);
    expect(() =>
      validateBundleMetadata(vocab, { ...modelConfig, model_schema_version: 1 }),
    ).toThrow(/Incompatible model bundle version/);
  });

  it("loads exported v2 ONNX models and runs predictDraft with Auto vs Late 50m pace", async () => {
    const vocab = JSON.parse(fs.readFileSync(path.join(MODELS_DIR, "vocab.json"), "utf8")) as VocabJson;
    const modelConfig = JSON.parse(
      fs.readFileSync(path.join(MODELS_DIR, "model_config.json"), "utf8"),
    ) as ModelConfigJson;

    validateBundleMetadata(vocab, modelConfig);

    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = `${pathToFileURL(ORT_DIR).toString()}/`;

    const encBytes = new Uint8Array(fs.readFileSync(path.join(MODELS_DIR, "d2oracle_encoder.onnx")));
    const decBytes = new Uint8Array(fs.readFileSync(path.join(MODELS_DIR, "d2oracle_item_decoder.onnx")));

    const [encoderSession, decoderSession] = await Promise.all([
      ort.InferenceSession.create(encBytes, { executionProviders: ["wasm"] }),
      ort.InferenceSession.create(decBytes, { executionProviders: ["wasm"] }),
    ]);

    const bundle: LoadedOracleBundle = {
      baseUrl: MODELS_DIR,
      vocab,
      modelConfig,
      encoderSession,
      decoderSession,
      heroesList: Object.values(vocab.heroes.by_id).sort((a, b) =>
        a.display_name.localeCompare(b.display_name),
      ),
    };

    // Include Anti-Mage (id=1, 5 skills w/ innate), Invoker (id=74, 14 skills), Meepo (id=82), and Io (id=91).
    const fixedHeroIds = [1, 74, 82, 14, 91, 6, 11, 29, 84, 86];
    for (const hid of fixedHeroIds) {
      expect(vocab.heroes.by_id[String(hid)]).toBeDefined();
    }

    const slots = createEmptyDraft().map((s, i) => ({ ...s, heroId: fixedHeroIds[i]! }));
    const predAuto = await predictDraft(bundle, slots, 0);

    expect(predAuto.matchPaceMinutes).toBe(0);
    expect(predAuto.radiantWinProb).toBeGreaterThan(0);
    expect(predAuto.radiantWinProb).toBeLessThan(1);
    expect(predAuto.radiantWinProb + predAuto.direWinProb).toBeCloseTo(1, 5);
    expect(predAuto.durationP25).toBeLessThanOrEqual(predAuto.durationP50);
    expect(predAuto.durationP50).toBeLessThanOrEqual(predAuto.durationP75);
    expect(predAuto.effectiveDurationMinutes).toBeCloseTo(predAuto.durationP50, 4);

    // Laning predictions sum cleanly to 100% per lane
    expect(predAuto.lanes).toHaveLength(3);
    for (const lane of predAuto.lanes) {
      expect(lane.radiantPct + lane.evenPct + lane.direPct).toBe(100);
      expect(lane.radiantProb + lane.evenProb + lane.direProb).toBeCloseTo(1, 4);
    }
    expect(Number.isFinite(predAuto.goldLead10m)).toBe(true);
    expect(Number.isFinite(predAuto.xpLead10m)).toBe(true);

    // 10 hero predictions with valid skills, talents, items, combat & economy
    expect(predAuto.heroes).toHaveLength(10);
    for (const heroPred of predAuto.heroes) {
      expect(heroPred.skills).toHaveLength(18);
      const skippedLvl = getSkippedSkillLevel(heroPred.hero.id);
      if (skippedLvl === null) {
        expect(heroPred.skills.every((s) => s.kind !== "empty")).toBe(true);
      } else {
        expect(heroPred.skills[skippedLvl - 1]!.kind).toBe("empty");
        expect(heroPred.skills.filter((s) => s.kind !== "empty")).toHaveLength(17);
      }
      expect(heroPred.talents).toHaveLength(4);
      expect(heroPred.items.length).toBeGreaterThan(0);
      expect(heroPred.items.length).toBeLessThanOrEqual(modelConfig.max_item_len - 1);
      expect(heroPred.economy.gpm.low).toBeLessThanOrEqual(heroPred.economy.gpm.mid);
      expect(heroPred.economy.gpm.mid).toBeLessThanOrEqual(heroPred.economy.gpm.high);
      expect(heroPred.economy.csTotal.low).toBeLessThanOrEqual(heroPred.economy.csTotal.high);
      expect(heroPred.combat.heroDamage.dpm).toBeGreaterThanOrEqual(0);
      for (const step of heroPred.skills) {
        if (step.abilityId !== null) {
          const ab = vocab.abilities.by_id[String(step.abilityId)];
          if (ab) {
            expect(NON_LEVELABLE_ABILITY_NAMES.has(ab.name)).toBe(false);
          }
        }
      }
      if (heroPred.hero.id !== 74) {
        const abilitySteps = heroPred.skills.filter((s) => s.kind === "ability");
        expect(abilitySteps.every((s) => ["Q", "W", "E", "R"].includes(s.shortTag))).toBe(true);
        expect(abilitySteps.every((s) => s.isUltimate === (s.shortTag === "R"))).toBe(true);
      }
    }

    // Team shares across each 5-hero side sum to ~1.0
    for (const teamSlice of [predAuto.heroes.slice(0, 5), predAuto.heroes.slice(5, 10)]) {
      const heroDmgSum = teamSlice.reduce((s, h) => s + h.combat.heroDamage.teamShare, 0);
      const towerDmgSum = teamSlice.reduce((s, h) => s + h.combat.towerDamage.teamShare, 0);
      const takenSum = teamSlice.reduce((s, h) => s + h.combat.damageTaken.teamShare, 0);
      expect(heroDmgSum).toBeCloseTo(1.0, 2);
      expect(towerDmgSum).toBeCloseTo(1.0, 2);
      expect(takenSum).toBeCloseTo(1.0, 2);
    }

    // Switching pace to Late 50m preserves Stage 1 outputs identically and sets effective duration to 50m
    const predLate = await predictDraft(bundle, slots, 50);
    expect(predLate.matchPaceMinutes).toBe(50);
    expect(predLate.effectiveDurationMinutes).toBe(50);
    expect(predLate.radiantWinProb).toBeCloseTo(predAuto.radiantWinProb, 6);
    expect(predLate.durationP50).toBeCloseTo(predAuto.durationP50, 6);
    expect(predLate.goldLead10m).toBeCloseTo(predAuto.goldLead10m, 4);
    expect(predLate.xpLead10m).toBeCloseTo(predAuto.xpLead10m, 4);
    for (let i = 0; i < 3; i++) {
      expect(predLate.lanes[i]!.radiantProb).toBeCloseTo(predAuto.lanes[i]!.radiantProb, 5);
    }
    expect(predLate.heroes[0]!.economy.csTotal.mid).toBeGreaterThanOrEqual(
      predAuto.heroes[0]!.economy.csTotal.mid,
    );
  });
});
