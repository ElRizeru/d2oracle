import type { PredictedItem } from "../types/oracle";

/**
 * Dota 2 item `short_name` keys representing boot-slot progression items,
 * separated into the dedicated Boots column in Builds > Summary.
 */
export const BOOT_ITEM_SHORT_NAMES: ReadonlySet<string> = new Set([
  "boots",
  "phase_boots",
  "power_treads",
  "arcane_boots",
  "tranquil_boots",
  "travel_boots",
  "travel_boots_2",
  "guardian_greaves",
  "boots_of_bearing",
]);

export function isBootItemShortName(shortName: string): boolean {
  return BOOT_ITEM_SHORT_NAMES.has(shortName);
}

/**
 * Partition predicted items into the primary Boots slot (earliest boot item)
 * and remaining chronological core items. Any secondary late boot upgrade
 * (e.g., Guardian Greaves after Arcane Boots) stays in `coreItems`.
 */
export function splitBootsAndCoreItems(items: readonly PredictedItem[]): {
  bootItem: PredictedItem | null;
  coreItems: PredictedItem[];
} {
  let bootItem: PredictedItem | null = null;
  const coreItems: PredictedItem[] = [];

  for (const item of items) {
    if (bootItem === null && isBootItemShortName(item.shortName)) {
      bootItem = item;
    } else {
      coreItems.push(item);
    }
  }

  return { bootItem, coreItems };
}

export function formatTimingShort(expectedMinute: number): string {
  const safeMin = Number.isFinite(expectedMinute) ? Math.max(0, expectedMinute) : 0;
  return `${Math.round(safeMin)}m`;
}
