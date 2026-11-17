import type { DraftSlot, HeroMeta, MatchPaceOption, RolePosition } from "../types/oracle";

const POSITIONS: readonly RolePosition[] = [1, 2, 3, 4, 5];

export const ROLE_NAMES: Record<RolePosition, { full: string; short: string }> = {
  1: { full: "Safe carry", short: "Carry" },
  2: { full: "Mid lane", short: "Mid" },
  3: { full: "Offlane", short: "Offlane" },
  4: { full: "Soft support", short: "Soft supp" },
  5: { full: "Hard support", short: "Hard supp" },
};

export const VALID_MATCH_PACES: readonly MatchPaceOption[] = [0, 25, 35, 50];

export function createEmptyDraft(): DraftSlot[] {
  const slots: DraftSlot[] = [];
  for (let i = 0; i < 5; i++) {
    slots.push({ slotIdx: i, side: "radiant", position: POSITIONS[i]!, heroId: null });
  }
  for (let i = 0; i < 5; i++) {
    slots.push({ slotIdx: 5 + i, side: "dire", position: POSITIONS[i]!, heroId: null });
  }
  return slots;
}

export function assignHeroToSlot(
  slots: readonly DraftSlot[],
  targetSlotIdx: number,
  heroId: number | null,
): DraftSlot[] {
  return slots.map((slot) => {
    if (slot.slotIdx === targetSlotIdx) {
      return { ...slot, heroId };
    }
    // Enforce uniqueness: move hero cleanly if selected in another slot.
    if (heroId !== null && slot.heroId === heroId) {
      return { ...slot, heroId: null };
    }
    return slot;
  });
}

export function swapSlotHeroes(
  slots: readonly DraftSlot[],
  slotA: number,
  slotB: number,
): DraftSlot[] {
  if (slotA === slotB) {
    return [...slots];
  }
  const a = slots.find((s) => s.slotIdx === slotA);
  const b = slots.find((s) => s.slotIdx === slotB);
  if (!a || !b) {
    return [...slots];
  }
  return slots.map((slot) => {
    if (slot.slotIdx === slotA) {
      return { ...slot, heroId: b.heroId };
    }
    if (slot.slotIdx === slotB) {
      return { ...slot, heroId: a.heroId };
    }
    return slot;
  });
}

export function createRandomDraft(heroes: readonly HeroMeta[]): DraftSlot[] {
  const base = createEmptyDraft();
  if (heroes.length < 10) {
    return base;
  }

  const pool = [...heroes];
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = pool[i]!;
    pool[i] = pool[j]!;
    pool[j] = tmp;
  }

  return base.map((slot, i) => ({
    ...slot,
    heroId: pool[i]!.id,
  }));
}

export function serializeDraftToSearch(
  slots: readonly DraftSlot[],
  matchPace: MatchPaceOption = 0,
): string {
  const rad = slots.slice(0, 5).map((s) => (s.heroId ?? 0)).join(",");
  const dire = slots.slice(5, 10).map((s) => (s.heroId ?? 0)).join(",");
  const params = new URLSearchParams();
  params.set("r", rad);
  params.set("d", dire);
  if (matchPace !== 0) {
    params.set("pace", String(matchPace));
  }
  return `?${params.toString()}`;
}

export function parseDraftFromSearch(
  search: string,
  heroesById: Record<string, HeroMeta>,
): { slots: DraftSlot[]; matchPace: MatchPaceOption } | null {
  const params = new URLSearchParams(search);
  const rRaw = params.get("r");
  const dRaw = params.get("d");
  if (!rRaw || !dRaw) {
    return null;
  }
  const rIds = rRaw.split(",").map((x) => Number.parseInt(x, 10));
  const dIds = dRaw.split(",").map((x) => Number.parseInt(x, 10));
  if (rIds.length !== 5 || dIds.length !== 5) {
    return null;
  }
  const combined = [...rIds, ...dIds];
  const seen = new Set<number>();
  const base = createEmptyDraft();
  for (let i = 0; i < 10; i++) {
    const hid = combined[i] ?? 0;
    if (Number.isFinite(hid) && hid > 0 && heroesById[String(hid)] && !seen.has(hid)) {
      seen.add(hid);
      base[i] = { ...base[i]!, heroId: hid };
    }
  }

  const paceRaw = Number.parseInt(params.get("pace") ?? "0", 10);
  const matchPace: MatchPaceOption = VALID_MATCH_PACES.includes(paceRaw as MatchPaceOption)
    ? (paceRaw as MatchPaceOption)
    : 0;

  return {
    slots: base,
    matchPace,
  };
}
