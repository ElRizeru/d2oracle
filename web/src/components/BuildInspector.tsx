import {
  Fragment,
  useEffect,
  useMemo,
  useState,
  type FocusEvent,
  type PointerEvent,
  type ReactNode,
} from "react";
import { Check, ChevronDown, ChevronRight, ChevronUp, X } from "lucide-react";

import {
  getAbilityColorKey,
  getAbilityPickedLevels,
  getLevelableSkillEntries,
  getMaxAbilityRanks,
  type AbilitySlotColorKey,
  type LevelableSkillEntry,
} from "../data/abilityDisplay";
import { ROLE_NAMES } from "../data/draftPresets";
import { formatTimingShort, splitBootsAndCoreItems } from "../data/itemDisplay";
import {
  formatRangeText,
  summarizeEarlySkillBuild,
} from "../ml/decoder";
import type {
  CombatShareStat,
  DraftPrediction,
  HeroPrediction,
  MatchPaceOption,
  PredictedItem,
  TalentChoice,
  TeamSide,
  VocabJson,
} from "../types/oracle";
import { FactionEmblem, RoleIcon, TalentTreeGlyph } from "./DotaIcons";
import { HeroPortraitImg } from "./HeroPortraitImg";

interface BuildInspectorProps {
  prediction: DraftPrediction | null;
  vocab: VocabJson | null;
  matchPace: MatchPaceOption;
}

type TeamFilter = "all" | "radiant" | "dire";
type PrimaryTab = "builds" | "scoreboard" | "focus";
type BuildsSubTab = "summary" | "timeline" | "abilities";

interface TooltipData {
  x: number;
  yTop: number;
  yBottom: number;
  kicker?: string;
  title?: string;
  detail?: string;
  skillBadge?: string;
  skillColorKey?: AbilitySlotColorKey;
  skillLevels?: readonly number[];
  talents?: readonly TalentChoice[];
}

const ULT_LEVEL_CHECKPOINTS = new Set([6, 12, 18]);
const LEVELS_1_TO_18 = Array.from({ length: 18 }, (_, i) => i + 1);

function renderSlashBreakText(text: string): ReactNode {
  if (!text.includes("/")) {
    return text;
  }
  const parts = text.split("/");
  return parts.map((part, idx) => (
    <Fragment key={`${idx}-${part}`}>
      {part}
      {idx < parts.length - 1 ? (
        <>
          /<wbr />
        </>
      ) : null}
    </Fragment>
  ));
}

function isQuietHeal(stat: CombatShareStat, showShare: boolean): boolean {
  if (!showShare) {
    return true;
  }
  return Math.round(stat.dpm) < 5;
}

function formatTotalGold(costSum: number): string {
  if (costSum >= 1000) {
    return `${(costSum / 1000).toFixed(1).replace(/\.0$/, "")}k g`;
  }
  return `${costSum.toLocaleString()} g`;
}

function summarizeTeamCombat(heroes: readonly HeroPrediction[]) {
  const gpmSum = heroes.reduce((s, h) => s + h.economy.gpm.mid, 0);
  const xpmSum = heroes.reduce((s, h) => s + h.economy.xpm.mid, 0);
  const csSum = heroes.reduce((s, h) => s + h.economy.csTotal.mid, 0);
  const csPerMinSum = heroes.reduce((s, h) => s + h.economy.csPerMin.mid, 0);
  const kSum = heroes.reduce((s, h) => s + h.economy.kills.mid, 0);
  const dSum = heroes.reduce((s, h) => s + h.economy.deaths.mid, 0);
  const aSum = heroes.reduce((s, h) => s + h.economy.assists.mid, 0);
  const heroDmgSum = heroes.reduce((s, h) => s + h.combat.heroDamage.dpm, 0);
  const towerDmgSum = heroes.reduce((s, h) => s + h.combat.towerDamage.dpm, 0);
  const takenSum = heroes.reduce((s, h) => s + h.combat.damageTaken.dpm, 0);
  const healSum = heroes.reduce((s, h) => s + h.combat.healing.dpm, 0);
  const showHeal = heroes.some((h) => h.combat.showHealingShare) && Math.round(healSum) >= 10;

  return {
    gpmTotal: Math.round(gpmSum),
    gpmAvg: Math.round(gpmSum / Math.max(1, heroes.length)),
    xpmTotal: Math.round(xpmSum),
    xpmAvg: Math.round(xpmSum / Math.max(1, heroes.length)),
    csTotal: Math.round(csSum),
    csPerMin: csPerMinSum.toFixed(1),
    kills: Math.round(kSum),
    deaths: Math.round(dSum),
    assists: Math.round(aSum),
    heroDmg: Math.round(heroDmgSum),
    towerDmg: Math.round(towerDmgSum),
    taken: Math.round(takenSum),
    heal: Math.round(healSum),
    showHeal,
  };
}

interface PositionedTimelineItem {
  item: PredictedItem;
  leftPct: number;
  tierRow: 0 | 1;
}

function layoutTimelineItems(
  items: readonly PredictedItem[],
  axisMaxMin: number,
): PositionedTimelineItem[] {
  const safeMax = Math.max(25, axisMaxMin);
  let lastRow0Pct = -100;
  let lastRow1Pct = -100;

  return items.map((item) => {
    const rawPct = (Math.max(0, item.expectedMinute) / safeMax) * 100;
    const clampedPct = Math.min(96, Math.max(3, rawPct));
    let tierRow: 0 | 1 = 0;
    let adjustedPct = clampedPct;

    if (clampedPct - lastRow0Pct < 7.0) {
      if (clampedPct - lastRow1Pct >= 7.0) {
        tierRow = 1;
      } else {
        // Both upper and lower bands are crowded at this timestamp; advance gently on whichever band is freer.
        if (lastRow0Pct <= lastRow1Pct) {
          tierRow = 0;
          adjustedPct = Math.min(96, lastRow0Pct + 7.0);
        } else {
          tierRow = 1;
          adjustedPct = Math.min(96, lastRow1Pct + 7.0);
        }
      }
    }

    if (tierRow === 0) {
      lastRow0Pct = adjustedPct;
    } else {
      lastRow1Pct = adjustedPct;
    }

    return {
      item,
      leftPct: adjustedPct,
      tierRow,
    };
  });
}

function CombatCellContent({
  stat,
  side,
  showShare,
  unit = "DPM",
}: {
  stat: CombatShareStat;
  side: TeamSide;
  showShare: boolean;
  unit?: "DPM" | "HPM";
}) {
  const rate = Math.round(stat.dpm);
  const pct = Math.max(0, Math.min(100, Math.round(stat.teamShare * 100)));
  const quietHeal = unit === "HPM" && isQuietHeal(stat, showShare);

  if (quietHeal) {
    return <span className="combat-cell__quiet">—</span>;
  }

  return (
    <div className="combat-cell__wrap">
      <div className="combat-cell__top">
        <span className="combat-cell__rate">{rate}</span>
        <span className="combat-cell__unit">{unit}</span>
      </div>
      {showShare ? (
        <div className="combat-cell__share">
          <div className="share-bar">
            <span
              className={`share-bar__fill share-bar__fill--${side}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="combat-cell__pct">{pct}%</span>
        </div>
      ) : (
        <span className="combat-cell__quiet">—</span>
      )}
    </div>
  );
}

function CombatMeterRow({
  label,
  stat,
  side,
  unit,
  showShare,
}: {
  label: string;
  stat: CombatShareStat;
  side: TeamSide;
  unit: "DPM" | "HPM";
  showShare: boolean;
}) {
  const rate = Math.round(stat.dpm);
  const pct = Math.max(0, Math.min(100, Math.round(stat.teamShare * 100)));
  const quietHeal = unit === "HPM" && isQuietHeal(stat, showShare);

  return (
    <div className="combat-meter">
      <div className="combat-meter__head">
        <span className="combat-meter__label">{label}</span>
        {quietHeal ? (
          <span className="combat-meter__quiet">Minimal healing</span>
        ) : (
          <span className="combat-meter__vals measured">
            <strong>{rate}</strong> {unit}
            {showShare ? (
              <span className="combat-meter__share-tag"> · {pct}% of team</span>
            ) : null}
          </span>
        )}
      </div>
      {!quietHeal && showShare ? (
        <div className="share-bar share-bar--wide">
          <span
            className={`share-bar__fill share-bar__fill--${side}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      ) : null}
    </div>
  );
}

function AbilityPipSummary({
  hp,
  entries,
  bindTip,
  compact = false,
  iconSize,
}: {
  hp: HeroPrediction;
  entries: readonly LevelableSkillEntry[];
  bindTip: (data: Omit<TooltipData, "x" | "yTop" | "yBottom">) => Record<string, unknown>;
  compact?: boolean;
  iconSize?: number;
}) {
  const size = iconSize ?? (compact ? 28 : 34);
  return (
    <div className={`ability-pip-row ${compact ? "is-compact" : ""}`}>
      {entries.map((entry) => {
        const pickedLevels = getAbilityPickedLevels(entry, entries, hp.skills);
        const earlyRanks = pickedLevels.filter((lvl) => lvl <= 6).length;
        const totalRanks = getMaxAbilityRanks(hp.hero.id, entry, pickedLevels.length);
        const abName =
          entry.ability?.display_name || entry.ability?.name || `Ability ${entry.shortTag}`;
        const colorKey = getAbilityColorKey(entry.shortTag, entry.isUltimate);
        const levelList =
          pickedLevels.length > 0 ? `Leveled at ${pickedLevels.join(", ")}` : "Not leveled";

        return (
          <div
            key={entry.abilityId}
            tabIndex={0}
            className={`ability-pip-item ${entry.isUltimate ? "is-ult" : ""}`}
            aria-label={`${abName} (${entry.shortTag}): ${levelList}`}
            {...bindTip({
              kicker: entry.isUltimate ? "Ultimate" : undefined,
              title: abName,
              skillBadge: entry.shortTag,
              skillColorKey: colorKey,
              skillLevels: pickedLevels,
            })}
          >
            <span className="ability-pip-item__icon">
              <HeroPortraitImg
                src={entry.ability?.img}
                alt={abName}
                fallbackText={entry.shortTag}
                width={size}
                height={size}
                decorative
                priority
              />
            </span>
            <span className="ability-pip-item__dots" aria-hidden="true">
              {Array.from({ length: totalRanks }, (_, idx) => (
                <span
                  key={idx}
                  className={`pip-dot ${idx < earlyRanks ? "is-lit" : ""}`}
                />
              ))}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function HeroDetailContent({
  hp,
  prediction,
  vocab,
  bindTip,
  onCloseInline,
}: {
  hp: HeroPrediction;
  prediction: DraftPrediction;
  vocab: VocabJson;
  bindTip: (data: Omit<TooltipData, "x" | "yTop" | "yBottom">) => Record<string, unknown>;
  onCloseInline?: () => void;
}) {
  const skillEntries = getLevelableSkillEntries(hp.hero, vocab.abilities.by_id);
  const earlySummary = summarizeEarlySkillBuild(hp.hero, hp.skills, vocab.abilities.by_id);
  const totalItemsCost = hp.items.reduce((sum, it) => sum + it.cost, 0);
  const descendingTalents = [...hp.talents].sort((a, b) => b.tier - a.tier);
  const talentByLevelMap = useMemo(() => {
    const map = new Map<number, (typeof hp.talents)[number]>();
    for (const tal of hp.talents) {
      if (tal.tier === 10 || tal.tier === 15) {
        map.set(tal.tier, tal);
      }
    }
    const pickedTalents = hp.skills.filter((s) => s.kind === "talent");
    const ascTalents = [...hp.talents].sort((a, b) => a.tier - b.tier);
    pickedTalents.forEach((step, idx) => {
      const tal = ascTalents[idx];
      if (tal) {
        map.set(step.level, tal);
      }
    });
    return map;
  }, [hp.skills, hp.talents]);

  const showUltColumns = hp.hero.id !== 74;
  const hasBonusTrack =
    talentByLevelMap.size > 0 ||
    hp.skills.some((s) => s.kind === "talent" || s.kind === "attribute");

  return (
    <div className="hero-detail-panel">
      <div className={`hero-detail-panel__head hero-detail-panel__head--${hp.side}`}>
        <div className="hero-detail-panel__identity">
          <span className="hero-detail-panel__portrait">
            <HeroPortraitImg
              src={hp.hero.img}
              alt={hp.hero.display_name}
              fallbackText={hp.hero.display_name.slice(0, 2)}
              width={80}
              height={45}
              decorative
            />
          </span>
          <div className="hero-detail-panel__title-block">
            <div className="hero-detail-panel__name-row">
              <h3>{hp.hero.display_name}</h3>
              {earlySummary ? (
                <span
                  className="early-build-pill measured"
                  title="Skill ranks by level 6"
                >
                  {earlySummary} @ 6
                </span>
              ) : null}
            </div>
            <p className="hero-detail-panel__subline">
              <span className={`side-label--${hp.side}`}>
                {hp.side === "radiant" ? "Radiant" : "Dire"}
              </span>
              <span aria-hidden="true">·</span>
              <RoleIcon position={hp.position} size={13} />
              <span>{ROLE_NAMES[hp.position].full}</span>
            </p>
          </div>
        </div>

        <div className="hero-detail-panel__summary-chips">
          <span className="detail-kpi-chip measured">
            <span className="detail-kpi-chip__k">K/D/A</span>
            <strong>
              {Math.round(hp.economy.kills.mid)}/{Math.round(hp.economy.deaths.mid)}/
              {Math.round(hp.economy.assists.mid)}
            </strong>
          </span>
          <span className="detail-kpi-chip detail-kpi-chip--gold measured">
            <span className="detail-kpi-chip__k">GPM</span>
            <strong>{formatRangeText(hp.economy.gpm)}</strong>
          </span>
          <span className="detail-kpi-chip detail-kpi-chip--xp measured">
            <span className="detail-kpi-chip__k">XPM</span>
            <strong>{formatRangeText(hp.economy.xpm)}</strong>
          </span>
          {totalItemsCost > 0 ? (
            <span className="detail-kpi-chip measured">
              <span className="detail-kpi-chip__k">Items</span>
              <strong>{formatTotalGold(totalItemsCost)}</strong>
            </span>
          ) : null}
          {onCloseInline ? (
            <button
              type="button"
              className="btn btn--icon detail-inline-close"
              onClick={onCloseInline}
              aria-label={`Collapse ${hp.hero.display_name} details`}
              title="Collapse details"
            >
              <X size={15} aria-hidden="true" />
            </button>
          ) : null}
        </div>
      </div>

      <div className="focus-bento">
        {/* Focus Card 1: Ability & Skill Build (staircase + pip/talent summary) */}
        <div className="detail-block detail-block--staircase">
          <div className="detail-block__head">
            <h4>Ability &amp; Skill Build</h4>
            <span className="detail-block__sub">Levels 1–18 &amp; Talents</span>
          </div>

          <div className="focus-skill-split">
            <div className="focus-skill-summary">
              <div className="focus-skill-summary__icons">
                <div
                  className="talent-glyph-box"
                  tabIndex={0}
                  aria-label={`Talent choices: ${descendingTalents.map((t) => `Lvl ${t.tier} ${t.displayName}`).join(", ")}`}
                  {...bindTip({
                    talents: descendingTalents,
                  })}
                >
                  <TalentTreeGlyph talents={hp.talents} size={34} />
                </div>
                <AbilityPipSummary hp={hp} entries={skillEntries} bindTip={bindTip} />
              </div>

              <div className="talent-tree">
                {descendingTalents.map((tal) => {
                  const leftOpt = tal.options[0];
                  const rightOpt = tal.options[1];
                  return (
                    <div key={tal.tier} className="talent-tree__row">
                      {leftOpt ? (
                        <span
                          className={`talent-opt talent-opt--left ${leftOpt.chosen ? "is-chosen" : "is-unchosen"}`}
                        >
                          {leftOpt.chosen ? (
                            <Check size={12} className="talent-opt__check" aria-hidden="true" />
                          ) : null}
                          <span>{renderSlashBreakText(leftOpt.displayName)}</span>
                        </span>
                      ) : (
                        <span className="talent-opt is-unchosen">{tal.displayName}</span>
                      )}

                      <span
                        className="talent-tree__tier measured"
                        title={`Level ${tal.tier} talent`}
                      >
                        {tal.tier}
                      </span>

                      {rightOpt ? (
                        <span
                          className={`talent-opt talent-opt--right ${rightOpt.chosen ? "is-chosen" : "is-unchosen"}`}
                        >
                          {rightOpt.chosen ? (
                            <Check size={12} className="talent-opt__check" aria-hidden="true" />
                          ) : null}
                          <span>{renderSlashBreakText(rightOpt.displayName)}</span>
                        </span>
                      ) : (
                        <span className="talent-opt is-unchosen">—</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="staircase-wrap">
              <table className="staircase-grid" aria-label={`Level 1 to 18 skill grid for ${hp.hero.display_name}`}>
                <thead>
                  <tr>
                    <th scope="col" className="staircase-grid__head-label">
                      Skill
                    </th>
                    {hp.skills.map((st) => {
                      const isUltCol = showUltColumns && ULT_LEVEL_CHECKPOINTS.has(st.level);
                      const isSkippedCol = st.kind === "empty";
                      return (
                        <th
                          key={st.level}
                          scope="col"
                          className={`measured ${isUltCol ? "is-ult-col" : ""} ${isSkippedCol ? "is-skipped-col" : ""}`}
                        >
                          {st.level}
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {skillEntries.map((entry) => {
                    const abName =
                      entry.ability?.display_name ||
                      entry.ability?.name ||
                      `Ability ${entry.shortTag}`;
                    const colorKey = getAbilityColorKey(entry.shortTag, entry.isUltimate);
                    return (
                      <tr key={entry.abilityId}>
                        <th scope="row" className="staircase-grid__row-head">
                          <span className="staircase-label">
                            <span
                              className={`ability-matrix__key ability-matrix__key--${colorKey} measured`}
                            >
                              {entry.shortTag}
                            </span>
                            <span className="staircase-label__name">{abName}</span>
                          </span>
                        </th>
                        {hp.skills.map((st) => {
                          const hit =
                            st.kind === "ability" && st.slotClass === entry.origSlotIdx;
                          const isUltCol = showUltColumns && ULT_LEVEL_CHECKPOINTS.has(st.level);
                          return (
                            <td
                              key={st.level}
                              className={`${hit ? "is-picked" : ""} ${isUltCol ? "is-ult-col" : ""}`}
                            >
                              {hit ? (
                                <>
                                  <span
                                    tabIndex={0}
                                    className={`stair-icon ${entry.isUltimate ? "is-ult" : ""}`}
                                    aria-label={`${abName}${st.rank ? ` rank ${st.rank}` : ""} at level ${st.level}`}
                                    {...bindTip({
                                      kicker: `Level ${st.level} · ${entry.shortTag}`,
                                      title: abName,
                                      detail: st.rank ? `Rank ${st.rank}` : undefined,
                                    })}
                                  >
                                    <HeroPortraitImg
                                      src={entry.ability?.img}
                                      alt={abName}
                                      fallbackText={entry.shortTag}
                                      width={24}
                                      height={24}
                                      decorative
                                      priority
                                    />
                                  </span>
                                  <span
                                    className={`matrix-lvl-tag matrix-lvl-tag--slot-${colorKey} measured`}
                                  >
                                    {st.level}
                                  </span>
                                </>
                              ) : null}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}

                  {hasBonusTrack ? (
                    <tr>
                      <th scope="row" className="staircase-grid__row-head">
                        <span className="staircase-label">
                          <span className="ability-matrix__key ability-matrix__key--talent measured">
                            T
                          </span>
                          <span className="staircase-label__name">Talents / Stats</span>
                        </span>
                      </th>
                      {hp.skills.map((st) => {
                        const isAttr = st.kind === "attribute";
                        const matchedTal = talentByLevelMap.get(st.level);
                        const hasTalent = Boolean(matchedTal) || st.kind === "talent";
                        const isUltCol = showUltColumns && ULT_LEVEL_CHECKPOINTS.has(st.level);
                        return (
                          <td
                            key={st.level}
                            className={`${hasTalent ? "is-picked-talent" : ""} ${isAttr ? "is-picked-attr" : ""} ${isUltCol ? "is-ult-col" : ""}`}
                          >
                            {hasTalent && isAttr ? (
                              <div className="stair-dual-bonus">
                                <span
                                  tabIndex={0}
                                  className="stair-talent-badge"
                                  aria-label={`Level ${st.level} talent${matchedTal ? `: ${matchedTal.displayName}` : ""}`}
                                  {...bindTip({
                                    kicker: `Level ${st.level} Talent`,
                                    title: matchedTal?.displayName ?? "Talent choice",
                                  })}
                                >
                                  <TalentTreeGlyph
                                    talents={hp.talents}
                                    size={18}
                                    highlightTier={matchedTal?.tier}
                                  />
                                </span>
                                <span
                                  tabIndex={0}
                                  className="stair-attr-badge measured"
                                  aria-label={`Level ${st.level}: All attributes +2`}
                                  {...bindTip({
                                    kicker: `Level ${st.level}`,
                                    title: "Attribute Bonus (+2)",
                                  })}
                                >
                                  +2
                                </span>
                                <span className="matrix-lvl-tag matrix-lvl-tag--talent measured">
                                  {st.level}
                                </span>
                              </div>
                            ) : hasTalent ? (
                              <>
                                <span
                                  tabIndex={0}
                                  className="stair-talent-badge"
                                  aria-label={`Level ${st.level} talent${matchedTal ? `: ${matchedTal.displayName}` : ""}`}
                                  {...bindTip({
                                    kicker: `Level ${st.level} Talent`,
                                    title: matchedTal?.displayName ?? "Talent choice",
                                  })}
                                >
                                  <TalentTreeGlyph
                                    talents={hp.talents}
                                    size={20}
                                    highlightTier={matchedTal?.tier}
                                  />
                                </span>
                                <span className="matrix-lvl-tag matrix-lvl-tag--talent measured">
                                  {st.level}
                                </span>
                              </>
                            ) : isAttr ? (
                              <>
                                <span
                                  tabIndex={0}
                                  className="stair-attr-badge measured"
                                  aria-label={`Level ${st.level}: All attributes +2`}
                                  {...bindTip({
                                    kicker: `Level ${st.level}`,
                                    title: "Attribute Bonus (+2)",
                                  })}
                                >
                                  +2
                                </span>
                                <span className="matrix-lvl-tag matrix-lvl-tag--attr measured">
                                  {st.level}
                                </span>
                              </>
                            ) : null}
                          </td>
                        );
                      })}
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Focus Card 2: Item Progression (chevron sequence + dotted-leader purchase list) */}
        <div className="detail-block detail-block--items">
          <div className="detail-block__head">
            <h4>Item progression</h4>
            {totalItemsCost > 0 ? (
              <span className="detail-block__sub measured">
                {totalItemsCost.toLocaleString()} g total
              </span>
            ) : null}
          </div>
          {hp.items.length === 0 ? (
            <p className="muted-note">No key items predicted for this slot.</p>
          ) : (
            <>
              <ol
                className="item-chevron-strip"
                aria-label={`Predicted item sequence for ${hp.hero.display_name}`}
              >
                {hp.items.map((item, idx) => (
                  <li key={`${item.itemToken}-strip-${idx}`} className="item-chevron-strip__node">
                    {idx > 0 ? (
                      <ChevronRight
                        size={12}
                        className="item-chevron-strip__arrow"
                        aria-hidden="true"
                      />
                    ) : null}
                    <span
                      className="item-chevron-card"
                      tabIndex={0}
                      {...bindTip({
                        kicker: `#${idx + 1} Key item`,
                        title: item.displayName,
                        detail: `${item.cost.toLocaleString()} g · ${item.expectedMinute.toFixed(1)} min`,
                      })}
                    >
                      <span className="item-chevron-card__img">
                        <HeroPortraitImg
                          src={item.img}
                          alt={item.displayName}
                          fallbackText={item.displayName.slice(0, 2)}
                          width={34}
                          height={24}
                          decorative
                        />
                      </span>
                      <span className="item-chevron-card__min measured">
                        {formatTimingShort(item.expectedMinute)}
                      </span>
                    </span>
                  </li>
                ))}
              </ol>

              <ol className="detail-items-list detail-items-list--dotted">
                {hp.items.map((item, idx) => (
                  <li key={`${item.itemToken}-${idx}`} className="detail-item-row">
                    <span className="detail-item-row__step measured">{idx + 1}</span>
                    <span className="detail-item-row__img">
                      <HeroPortraitImg
                        src={item.img}
                        alt={item.displayName}
                        fallbackText={item.displayName.slice(0, 2)}
                        width={36}
                        height={26}
                        decorative
                      />
                    </span>
                    <span className="detail-item-row__name">{item.displayName}</span>
                    <span className="detail-item-row__leader" aria-hidden="true" />
                    <span className="detail-item-row__meta measured">
                      <span className="detail-item-row__cost">{item.cost.toLocaleString()} g</span>
                      <span className="detail-item-row__time">
                        {item.expectedMinute.toFixed(1)} min
                      </span>
                    </span>
                  </li>
                ))}
              </ol>
            </>
          )}
        </div>

        {/* Focus Card 3: Economy & Combat Impact */}
        <div className="detail-block detail-block--stats">
          <div className="detail-block__head">
            <h4>Economy &amp; combat impact</h4>
            <span className="detail-block__sub measured">
              {prediction.matchPaceMinutes === 0
                ? `Over ~${Math.round(prediction.effectiveDurationMinutes)} min`
                : `Over ${prediction.matchPaceMinutes} min`}
            </span>
          </div>

          <div className="econ-grid">
            <div className="econ-stat econ-stat--gold">
              <span className="econ-stat__label">GPM</span>
              <span className="econ-stat__val measured">
                {formatRangeText(hp.economy.gpm)}
              </span>
            </div>
            <div className="econ-stat econ-stat--xp">
              <span className="econ-stat__label">XPM</span>
              <span className="econ-stat__val measured">
                {formatRangeText(hp.economy.xpm)}
              </span>
            </div>
            <div className="econ-stat">
              <span className="econ-stat__label">CS</span>
              <span className="econ-stat__val measured">
                {formatRangeText(hp.economy.csTotal)}
              </span>
              <span className="econ-stat__sub measured">
                {formatRangeText(hp.economy.csPerMin, 1)}/min
              </span>
            </div>
            <div className="econ-stat">
              <span className="econ-stat__label">Kills</span>
              <span className="econ-stat__val econ-stat__val--k measured">
                {formatRangeText(hp.economy.kills)}
              </span>
            </div>
            <div className="econ-stat">
              <span className="econ-stat__label">Deaths</span>
              <span className="econ-stat__val econ-stat__val--d measured">
                {formatRangeText(hp.economy.deaths)}
              </span>
            </div>
            <div className="econ-stat">
              <span className="econ-stat__label">Assists</span>
              <span className="econ-stat__val econ-stat__val--a measured">
                {formatRangeText(hp.economy.assists)}
              </span>
            </div>
          </div>

          <div className="combat-meters">
            <CombatMeterRow
              label="Hero damage"
              stat={hp.combat.heroDamage}
              side={hp.side}
              unit="DPM"
              showShare
            />
            <CombatMeterRow
              label="Tower damage"
              stat={hp.combat.towerDamage}
              side={hp.side}
              unit="DPM"
              showShare
            />
            <CombatMeterRow
              label="Damage taken"
              stat={hp.combat.damageTaken}
              side={hp.side}
              unit="DPM"
              showShare
            />
            <CombatMeterRow
              label="Healing"
              stat={hp.combat.healing}
              side={hp.side}
              unit="HPM"
              showShare={hp.combat.showHealingShare}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export function BuildInspector({
  prediction,
  vocab,
  matchPace,
}: BuildInspectorProps) {
  const [primaryTab, setPrimaryTab] = useState<PrimaryTab>("builds");
  const [buildsSubTab, setBuildsSubTab] = useState<BuildsSubTab>("summary");
  const [teamFilter, setTeamFilter] = useState<TeamFilter>("all");
  const [selectedSlotIdx, setSelectedSlotIdx] = useState<number>(0);
  const [expandedSlotIdx, setExpandedSlotIdx] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);

  useEffect(() => {
    if (prediction && !prediction.heroes.some((h) => h.slotIdx === selectedSlotIdx)) {
      setSelectedSlotIdx(0);
    }
    setTooltip(null);
  }, [prediction, selectedSlotIdx]);

  useEffect(() => {
    if (!tooltip) return;
    const clear = () => setTooltip(null);
    window.addEventListener("scroll", clear, { passive: true, capture: true });
    window.addEventListener("resize", clear, { passive: true });
    return () => {
      window.removeEventListener("scroll", clear, true);
      window.removeEventListener("resize", clear);
    };
  }, [tooltip]);

  if (!prediction || !vocab) {
    return null;
  }

  const visibleHeroes = prediction.heroes.filter((h) => {
    if (teamFilter === "all") return true;
    return h.side === teamFilter;
  });

  const inspectedHero: HeroPrediction =
    prediction.heroes.find((h) => h.slotIdx === selectedSlotIdx) ??
    visibleHeroes[0] ??
    prediction.heroes[0]!;

  const maxHeroGpm = Math.max(
    1,
    ...prediction.heroes.map((h) => Math.round(h.economy.gpm.mid)),
  );
  const maxHeroXpm = Math.max(
    1,
    ...prediction.heroes.map((h) => Math.round(h.economy.xpm.mid)),
  );
  const maxItemMinute = Math.max(
    30,
    Math.ceil(
      Math.max(
        prediction.effectiveDurationMinutes,
        ...prediction.heroes.flatMap((h) => h.items.map((it) => it.expectedMinute)),
      ) / 10,
    ) * 10,
  );
  const timelineTicks = Array.from(
    { length: Math.floor(maxItemMinute / 10) + 1 },
    (_, i) => i * 10,
  );

  const handleFilterChange = (nextFilter: TeamFilter) => {
    setTeamFilter(nextFilter);
    setTooltip(null);
    const matching = prediction.heroes.filter((h) =>
      nextFilter === "all" ? true : h.side === nextFilter,
    );
    if (matching.length > 0 && !matching.some((h) => h.slotIdx === selectedSlotIdx)) {
      setSelectedSlotIdx(matching[0]!.slotIdx);
    }
    if (
      expandedSlotIdx !== null &&
      !matching.some((h) => h.slotIdx === expandedSlotIdx)
    ) {
      setExpandedSlotIdx(null);
    }
  };

  const toggleRowExpand = (slotIdx: number) => {
    setTooltip(null);
    setSelectedSlotIdx(slotIdx);
    setExpandedSlotIdx((prev) => (prev === slotIdx ? null : slotIdx));
  };

  const showTipFromEl = (
    el: HTMLElement,
    data: Omit<TooltipData, "x" | "yTop" | "yBottom">,
  ) => {
    const rect = el.getBoundingClientRect();
    setTooltip({
      ...data,
      x: rect.left + rect.width / 2,
      yTop: rect.top,
      yBottom: rect.bottom,
    });
  };

  const bindTip = (data: Omit<TooltipData, "x" | "yTop" | "yBottom">) => ({
    onPointerEnter: (e: PointerEvent<HTMLElement>) => showTipFromEl(e.currentTarget, data),
    onPointerLeave: () => setTooltip(null),
    onFocus: (e: FocusEvent<HTMLElement>) => showTipFromEl(e.currentTarget, data),
    onBlur: () => setTooltip(null),
  });

  const renderFloatingTooltip = () => {
    if (!tooltip || typeof window === "undefined") {
      return null;
    }
    const hasTalents = Boolean(tooltip.talents && tooltip.talents.length > 0);
    const placeBelow = tooltip.yTop < (hasTalents ? 150 : 90);
    const desiredPad = hasTalents ? 260 : 135;
    const halfPad = Math.min(desiredPad, window.innerWidth / 2);
    const clampedX = Math.max(halfPad, Math.min(window.innerWidth - halfPad, tooltip.x));
    const anchorY = placeBelow ? tooltip.yBottom + 8 : tooltip.yTop - 8;

    return (
      <div
        className={`floating-tip ${placeBelow ? "is-below" : "is-above"} ${hasTalents ? "floating-tip--talents" : ""}`}
        role="tooltip"
        style={{ left: `${clampedX}px`, top: `${anchorY}px` }}
      >
        {tooltip.kicker ? (
          <span className="floating-tip__kicker measured">{tooltip.kicker}</span>
        ) : null}
        {tooltip.title ? (
          <div className="floating-tip__head">
            {tooltip.skillBadge ? (
              <span
                className={`ability-matrix__key ability-matrix__key--${tooltip.skillColorKey ?? "q"} measured`}
              >
                {tooltip.skillBadge}
              </span>
            ) : null}
            <span className="floating-tip__title">{tooltip.title}</span>
          </div>
        ) : null}
        {tooltip.detail ? (
          <span className="floating-tip__detail measured">{tooltip.detail}</span>
        ) : null}
        {tooltip.skillLevels !== undefined ? (
          tooltip.skillLevels.length > 0 ? (
            <div className="floating-skill-levels">
              <span className="floating-skill-levels__label">Leveled at</span>
              <div className="floating-skill-levels__pills">
                {tooltip.skillLevels.map((lvl) => (
                  <span
                    key={lvl}
                    className={`floating-lvl-pill measured ${lvl <= 6 ? "is-early" : ""}`}
                  >
                    {lvl}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <span className="floating-tip__muted">Not leveled</span>
          )
        ) : null}
        {hasTalents && tooltip.talents ? (
          <div className="floating-talent-grid">
            {tooltip.talents.map((tal, idx) => {
              const leftOpt = tal.options[0];
              const rightOpt = tal.options[1];
              const rowStyle = { gridRow: idx + 1 };
              return (
                <Fragment key={tal.tier}>
                  <span
                    style={{ ...rowStyle, gridColumn: 1 }}
                    className={`floating-talent-text is-left ${leftOpt?.chosen ? "is-chosen" : "is-unchosen"}`}
                  >
                    {leftOpt ? renderSlashBreakText(leftOpt.displayName) : "—"}
                  </span>
                  <span
                    style={{ ...rowStyle, gridColumn: 3 }}
                    className={`floating-talent-text is-right ${rightOpt?.chosen ? "is-chosen" : "is-unchosen"}`}
                  >
                    {rightOpt ? renderSlashBreakText(rightOpt.displayName) : "—"}
                  </span>
                </Fragment>
              );
            })}
            <div className="floating-talent-center">
              <TalentTreeGlyph talents={tooltip.talents} size={74} />
            </div>
          </div>
        ) : null}
      </div>
    );
  };

  const teamSections: Array<{ side: TeamSide; label: string; heroes: HeroPrediction[] }> = [];
  if (teamFilter !== "dire") {
    const rad = visibleHeroes.filter((h) => h.side === "radiant");
    if (rad.length > 0) {
      teamSections.push({ side: "radiant", label: "Radiant", heroes: rad });
    }
  }
  if (teamFilter !== "radiant") {
    const dire = visibleHeroes.filter((h) => h.side === "dire");
    if (dire.length > 0) {
      teamSections.push({ side: "dire", label: "Dire", heroes: dire });
    }
  }

  const radHeroesAll = prediction.heroes.filter((h) => h.side === "radiant");
  const direHeroesAll = prediction.heroes.filter((h) => h.side === "dire");
  const radTotalsAll = summarizeTeamCombat(radHeroesAll);
  const direTotalsAll = summarizeTeamCombat(direHeroesAll);

  const renderItemChip = (item: PredictedItem, idx: number, kicker: string) => (
    <span
      key={`${item.itemToken}-${idx}`}
      tabIndex={0}
      className="item-tile"
      aria-label={`${item.displayName}, ${item.cost.toLocaleString()} gold, ${item.expectedMinute.toFixed(1)} min`}
      {...bindTip({
        kicker,
        title: item.displayName,
        detail: `${item.cost.toLocaleString()} g · ${item.expectedMinute.toFixed(1)} min`,
      })}
    >
      <span className="item-tile__icon">
        <HeroPortraitImg
          src={item.img}
          alt={item.displayName}
          fallbackText={item.displayName.slice(0, 2)}
          width={46}
          height={33}
          decorative
        />
      </span>
      <span className="item-tile__min measured">
        {formatTimingShort(item.expectedMinute)}
      </span>
    </span>
  );

  const renderAbilitiesLevelRow = (hp: HeroPrediction) => {
    const ascTalents = [...hp.talents].sort((a, b) => a.tier - b.tier);
    const talentByTier = new Map<number, (typeof hp.talents)[number]>(
      hp.talents.map((t) => [t.tier, t]),
    );
    let talentCursor = 0;

    return hp.skills.map((step) => {
      const isUltCol = hp.hero.id !== 74 && ULT_LEVEL_CHECKPOINTS.has(step.level);
      if (step.kind === "talent") {
        const tal = ascTalents[talentCursor];
        talentCursor += 1;
        return (
          <td key={step.level} className={`ab-grid__cell ${isUltCol ? "is-ult-col" : ""}`}>
            <span
              tabIndex={0}
              className="ab-grid__talent"
              aria-label={`Level ${step.level} talent${tal ? `: ${tal.displayName}` : ""}`}
              {...bindTip({
                kicker: `Level ${step.level} Talent`,
                title: tal?.displayName ?? "Talent choice",
              })}
            >
              <TalentTreeGlyph
                talents={hp.talents}
                size={20}
                highlightTier={tal?.tier}
              />
            </span>
          </td>
        );
      }

      const pairedTalent =
        step.level === 10 || step.level === 15 ? talentByTier.get(step.level) : undefined;
      const renderPairedTalent = () =>
        pairedTalent ? (
          <span
            tabIndex={0}
            className="ab-grid__talent ab-grid__talent--paired"
            aria-label={`Level ${step.level} talent: ${pairedTalent.displayName}`}
            {...bindTip({
              kicker: `Level ${step.level} Talent`,
              title: pairedTalent.displayName,
            })}
          >
            <TalentTreeGlyph
              talents={hp.talents}
              size={16}
              highlightTier={pairedTalent.tier}
            />
          </span>
        ) : null;

      if (step.kind === "attribute") {
        const attrNode = (
          <span
            tabIndex={0}
            className="ab-grid__attr measured"
            aria-label={`Level ${step.level}: Attribute Bonus +2`}
            {...bindTip({
              kicker: `Level ${step.level}`,
              title: "Attribute Bonus (+2)",
            })}
          >
            +2
          </span>
        );
        return (
          <td key={step.level} className={`ab-grid__cell ${isUltCol ? "is-ult-col" : ""}`}>
            {pairedTalent ? (
              <div className="ab-grid__stack">
                {attrNode}
                {renderPairedTalent()}
              </div>
            ) : (
              attrNode
            )}
          </td>
        );
      }
      if (step.kind === "empty") {
        return (
          <td
            key={step.level}
            className={`ab-grid__cell is-empty ${isUltCol ? "is-ult-col" : ""}`}
          >
            <span className="ab-grid__empty" aria-hidden="true">
              ·
            </span>
          </td>
        );
      }
      const colorKey = getAbilityColorKey(step.shortTag, step.isUltimate);
      const skillNode = (
        <span
          tabIndex={0}
          className={`ab-grid__skill ${step.isUltimate ? "is-ult" : ""}`}
          aria-label={`Level ${step.level}: ${step.label}${step.rank ? ` (Rank ${step.rank})` : ""}`}
          {...bindTip({
            kicker: `Level ${step.level} · ${step.shortTag}`,
            title: step.label,
            detail: step.rank ? `Rank ${step.rank}` : undefined,
          })}
        >
          <HeroPortraitImg
            src={step.img}
            alt={step.label}
            fallbackText={step.shortTag}
            width={26}
            height={26}
            decorative
            priority
          />
          <span className={`ab-grid__tag ab-grid__tag--${colorKey} measured`} aria-hidden="true">
            {step.shortTag}
          </span>
        </span>
      );
      return (
        <td key={step.level} className={`ab-grid__cell ${isUltCol ? "is-ult-col" : ""}`}>
          {pairedTalent ? (
            <div className="ab-grid__stack">
              {skillNode}
              {renderPairedTalent()}
            </div>
          ) : (
            skillNode
          )}
        </td>
      );
    });
  };

  const renderTeamCompareCards = () => {
    const comparisons = [
      {
        label: "Team Gold / min",
        radVal: radTotalsAll.gpmTotal,
        direVal: direTotalsAll.gpmTotal,
        unit: "GPM",
        accent: "gold",
      },
      {
        label: "Projected Kills",
        radVal: radTotalsAll.kills,
        direVal: direTotalsAll.kills,
        unit: "Kills",
        accent: "side",
      },
      {
        label: "Hero Damage / min",
        radVal: radTotalsAll.heroDmg,
        direVal: direTotalsAll.heroDmg,
        unit: "DPM",
        accent: "side",
      },
      {
        label: "Building Damage / min",
        radVal: radTotalsAll.towerDmg,
        direVal: direTotalsAll.towerDmg,
        unit: "DPM",
        accent: "side",
      },
    ];

    return (
      <div className="scoreboard-compare-grid" aria-label="Team totals comparison">
        {comparisons.map((cmp) => {
          const sum = Math.max(1, cmp.radVal + cmp.direVal);
          const radPct = (cmp.radVal / sum) * 100;
          const direPct = 100 - radPct;
          const radLeads = cmp.radVal >= cmp.direVal;
          return (
            <div key={cmp.label} className="compare-kpi-card">
              <div className="compare-kpi-card__head">
                <span className="compare-kpi-card__label">{cmp.label}</span>
                <span className="compare-kpi-card__unit measured">{cmp.unit}</span>
              </div>
              <div className="compare-kpi-card__nums measured">
                <span className={`compare-num compare-num--radiant ${radLeads ? "is-leader" : ""}`}>
                  {cmp.radVal.toLocaleString()}
                </span>
                <span className={`compare-num compare-num--dire ${!radLeads ? "is-leader" : ""}`}>
                  {cmp.direVal.toLocaleString()}
                </span>
              </div>
              <div className="compare-kpi-card__bar" aria-hidden="true">
                <span
                  className="compare-bar__rad"
                  style={{ width: `${radPct}%` }}
                />
                <span
                  className="compare-bar__dire"
                  style={{ width: `${direPct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <section className="builds-section" aria-labelledby="builds-heading">
      {/* Navigation Header with inline expanded sub-tabs */}
      <div className="forecast-nav-bar">
        <div className="forecast-nav-bar__left">
          <h2 id="builds-heading" className="visually-hidden">
            Hero forecast
          </h2>

          <nav className="forecast-tabs" aria-label="Forecast view">
            <div className={`forecast-tab-group ${primaryTab === "builds" ? "is-open" : ""}`}>
              <button
                type="button"
                className={`forecast-tab ${primaryTab === "builds" ? "is-active" : ""}`}
                aria-pressed={primaryTab === "builds"}
                onClick={() => {
                  setPrimaryTab("builds");
                  setTooltip(null);
                }}
              >
                <span>Builds</span>
                <ChevronRight size={13} className="forecast-tab__chev" aria-hidden="true" />
              </button>

              {primaryTab === "builds" ? (
                <div className="forecast-subtabs" role="group" aria-label="Builds view mode">
                  <button
                    type="button"
                    className={`forecast-subtab ${buildsSubTab === "summary" ? "is-active" : ""}`}
                    aria-pressed={buildsSubTab === "summary"}
                    onClick={() => {
                      setBuildsSubTab("summary");
                      setTooltip(null);
                    }}
                  >
                    Summary
                  </button>
                  <button
                    type="button"
                    className={`forecast-subtab ${buildsSubTab === "timeline" ? "is-active" : ""}`}
                    aria-pressed={buildsSubTab === "timeline"}
                    onClick={() => {
                      setBuildsSubTab("timeline");
                      setTooltip(null);
                    }}
                  >
                    Timeline
                  </button>
                  <button
                    type="button"
                    className={`forecast-subtab ${buildsSubTab === "abilities" ? "is-active" : ""}`}
                    aria-pressed={buildsSubTab === "abilities"}
                    onClick={() => {
                      setBuildsSubTab("abilities");
                      setTooltip(null);
                    }}
                  >
                    Abilities
                  </button>
                </div>
              ) : null}
            </div>

            <button
              type="button"
              className={`forecast-tab ${primaryTab === "scoreboard" ? "is-active" : ""}`}
              aria-pressed={primaryTab === "scoreboard"}
              onClick={() => {
                setPrimaryTab("scoreboard");
                setTooltip(null);
              }}
            >
              Scoreboard
            </button>

            <button
              type="button"
              className={`forecast-tab ${primaryTab === "focus" ? "is-active" : ""}`}
              aria-pressed={primaryTab === "focus"}
              onClick={() => {
                setPrimaryTab("focus");
                setTooltip(null);
              }}
            >
              Focus
            </button>
          </nav>

          {matchPace !== 0 ? (
            <span className="summary-badge summary-badge--pace measured nowrap">
              Builds @ {matchPace}m
            </span>
          ) : null}
        </div>

        <div className="forecast-nav-bar__right">
          {primaryTab !== "focus" ? (
            <div className="segmented" role="group" aria-label="Filter team builds">
              <button
                type="button"
                className={`segmented__btn ${teamFilter === "all" ? "is-active" : ""}`}
                aria-pressed={teamFilter === "all"}
                onClick={() => handleFilterChange("all")}
              >
                All 10
              </button>
              <button
                type="button"
                className={`segmented__btn ${teamFilter === "radiant" ? "is-active" : ""}`}
                aria-pressed={teamFilter === "radiant"}
                onClick={() => handleFilterChange("radiant")}
              >
                Radiant
              </button>
              <button
                type="button"
                className={`segmented__btn ${teamFilter === "dire" ? "is-active" : ""}`}
                aria-pressed={teamFilter === "dire"}
                onClick={() => handleFilterChange("dire")}
              >
                Dire
              </button>
            </div>
          ) : null}
        </div>
      </div>

      {/* VIEW 1: FOCUS (10-hero circular selector + deep-dive bento) */}
      {primaryTab === "focus" ? (
        <div className="hero-details-view">
          <div className="focus-hero-bar" role="tablist" aria-label="Select hero focus">
            <div className="focus-hero-bar__side focus-hero-bar__side--radiant">
              {radHeroesAll.map((hp) => {
                const isSelected = hp.slotIdx === inspectedHero.slotIdx;
                return (
                  <button
                    key={hp.slotIdx}
                    type="button"
                    role="tab"
                    aria-selected={isSelected}
                    className={`focus-circle-btn focus-circle-btn--radiant ${isSelected ? "is-selected" : ""}`}
                    onClick={() => {
                      setSelectedSlotIdx(hp.slotIdx);
                      setTooltip(null);
                    }}
                  >
                    <span className="focus-circle-btn__avatar">
                      <HeroPortraitImg
                        src={hp.hero.img}
                        alt={hp.hero.display_name}
                        fallbackText={hp.hero.display_name.slice(0, 2)}
                        width={46}
                        height={46}
                        decorative
                      />
                      <span className="focus-circle-btn__role" aria-hidden="true">
                        <RoleIcon position={hp.position} size={11} />
                      </span>
                    </span>
                    <span className="focus-circle-btn__name">{hp.hero.display_name}</span>
                  </button>
                );
              })}
            </div>

            <span className="focus-hero-bar__divider" aria-hidden="true" />

            <div className="focus-hero-bar__side focus-hero-bar__side--dire">
              {direHeroesAll.map((hp) => {
                const isSelected = hp.slotIdx === inspectedHero.slotIdx;
                return (
                  <button
                    key={hp.slotIdx}
                    type="button"
                    role="tab"
                    aria-selected={isSelected}
                    className={`focus-circle-btn focus-circle-btn--dire ${isSelected ? "is-selected" : ""}`}
                    onClick={() => {
                      setSelectedSlotIdx(hp.slotIdx);
                      setTooltip(null);
                    }}
                  >
                    <span className="focus-circle-btn__avatar">
                      <HeroPortraitImg
                        src={hp.hero.img}
                        alt={hp.hero.display_name}
                        fallbackText={hp.hero.display_name.slice(0, 2)}
                        width={46}
                        height={46}
                        decorative
                      />
                      <span className="focus-circle-btn__role" aria-hidden="true">
                        <RoleIcon position={hp.position} size={11} />
                      </span>
                    </span>
                    <span className="focus-circle-btn__name">{hp.hero.display_name}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <HeroDetailContent
            hp={inspectedHero}
            prediction={prediction}
            vocab={vocab}
            bindTip={bindTip}
          />
        </div>
      ) : null}

      {/* VIEW 2: BUILDS > SUMMARY */}
      {primaryTab === "builds" && buildsSubTab === "summary" ? (
        <>
          <div className="builds-table-wrap">
            <table className="builds-table builds-summary-table">
              <thead>
                <tr>
                  <th scope="col">Hero</th>
                  <th scope="col">Ability Build</th>
                  <th scope="col">Boots</th>
                  <th scope="col">Key Items</th>
                  <th scope="col" className="expand-th">
                    <span className="visually-hidden">Details</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {teamSections.map((section, sIdx) => (
                  <Fragment key={section.side}>
                    {sIdx > 0 ? (
                      <tr className="team-spacer-row" aria-hidden="true">
                        <td colSpan={5} />
                      </tr>
                    ) : null}
                    <tr className={`team-band-row team-band-row--${section.side}`}>
                      <th scope="colgroup" colSpan={5}>
                        <span className="team-band-row__inner">
                          <FactionEmblem side={section.side} size={15} />
                          <span className="team-band-row__label">{section.label}</span>
                        </span>
                      </th>
                    </tr>
                    {section.heroes.map((hp) => {
                      const isExpanded = expandedSlotIdx === hp.slotIdx;
                      const earlySummary = summarizeEarlySkillBuild(
                        hp.hero,
                        hp.skills,
                        vocab.abilities.by_id,
                      );
                      const skillEntries = getLevelableSkillEntries(
                        hp.hero,
                        vocab.abilities.by_id,
                      );
                      const { bootItem, coreItems } = splitBootsAndCoreItems(hp.items);
                      const descendingTalents = [...hp.talents].sort((a, b) => b.tier - a.tier);

                      return (
                        <Fragment key={hp.slotIdx}>
                          <tr
                            className={`row-side--${hp.side} ${isExpanded ? "is-expanded-row" : ""}`}
                          >
                            <th scope="row" className="hero-cell">
                              <button
                                type="button"
                                className="hero-row-trigger"
                                aria-expanded={isExpanded}
                                onClick={() => toggleRowExpand(hp.slotIdx)}
                              >
                                <span className="hero-cell__role-icon" aria-hidden="true">
                                  <RoleIcon position={hp.position} size={17} />
                                </span>
                                <span className="hero-cell__thumb">
                                  <HeroPortraitImg
                                    src={hp.hero.img}
                                    alt={hp.hero.display_name}
                                    fallbackText={hp.hero.display_name.slice(0, 2)}
                                    width={60}
                                    height={34}
                                    decorative
                                  />
                                </span>
                                <span className="hero-cell__meta">
                                  <span className="hero-cell__name">{hp.hero.display_name}</span>
                                  <span className="hero-cell__sub">
                                    <span className="hero-cell__role">
                                      {ROLE_NAMES[hp.position].short}
                                    </span>
                                    {earlySummary ? (
                                      <span className="hero-cell__build-code measured">
                                        {earlySummary}
                                      </span>
                                    ) : null}
                                  </span>
                                </span>
                              </button>
                            </th>

                            <td className="ability-build-cell">
                              <div className="ability-build-cluster">
                                <span
                                  tabIndex={0}
                                  className="talent-glyph-trigger"
                                  aria-label={`Talent choices: ${descendingTalents.map((t) => `Lvl ${t.tier} ${t.displayName}`).join(", ")}`}
                                  {...bindTip({
                                    talents: descendingTalents,
                                  })}
                                >
                                  <TalentTreeGlyph talents={hp.talents} size={36} />
                                </span>
                                <span className="ability-build-sep" aria-hidden="true" />
                                <AbilityPipSummary
                                  hp={hp}
                                  entries={skillEntries}
                                  bindTip={bindTip}
                                  iconSize={34}
                                />
                              </div>
                            </td>

                            <td className="boots-cell">
                              {bootItem ? (
                                renderItemChip(bootItem, 0, "Boots")
                              ) : (
                                <span
                                  className="empty-item-slot"
                                  title="No boots predicted"
                                  aria-label="No boots predicted"
                                />
                              )}
                            </td>

                            <td className="items-cell">
                              {coreItems.length === 0 ? (
                                <span className="muted-note">No key items</span>
                              ) : (
                                <ol
                                  className="items-row"
                                  aria-label={`Key items for ${hp.hero.display_name}`}
                                >
                                  {coreItems.map((item, idx) => (
                                    <li key={`${item.itemToken}-${idx}`}>
                                      {renderItemChip(item, idx, `#${idx + 1} Key item`)}
                                    </li>
                                  ))}
                                </ol>
                              )}
                            </td>

                            <td className="expand-cell">
                              <button
                                type="button"
                                className="btn btn--icon expand-row-btn"
                                aria-expanded={isExpanded}
                                aria-label={
                                  isExpanded
                                    ? `Hide ${hp.hero.display_name} details`
                                    : `Inspect ${hp.hero.display_name} details`
                                }
                                onClick={() => toggleRowExpand(hp.slotIdx)}
                              >
                                {isExpanded ? (
                                  <ChevronUp size={15} aria-hidden="true" />
                                ) : (
                                  <ChevronDown size={15} aria-hidden="true" />
                                )}
                              </button>
                            </td>
                          </tr>
                          {isExpanded ? (
                            <tr className="detail-drawer-row">
                              <td colSpan={5} className="detail-drawer-cell">
                                <HeroDetailContent
                                  hp={hp}
                                  prediction={prediction}
                                  vocab={vocab}
                                  bindTip={bindTip}
                                  onCloseInline={() => setExpandedSlotIdx(null)}
                                />
                              </td>
                            </tr>
                          ) : null}
                        </Fragment>
                      );
                    })}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mobile-forecast-cards">
            {teamSections.map((section) => (
              <div key={section.side} className="mobile-team-group">
                <div className={`mobile-team-group__head mobile-team-group__head--${section.side}`}>
                  <FactionEmblem side={section.side} size={14} />
                  <span>{section.label}</span>
                </div>

                {section.heroes.map((hp) => {
                  const isExpanded = expandedSlotIdx === hp.slotIdx;
                  const earlySummary = summarizeEarlySkillBuild(
                    hp.hero,
                    hp.skills,
                    vocab.abilities.by_id,
                  );
                  const skillEntries = getLevelableSkillEntries(hp.hero, vocab.abilities.by_id);
                  const descendingTalents = [...hp.talents].sort((a, b) => b.tier - a.tier);

                  return (
                    <article
                      key={hp.slotIdx}
                      className={`mobile-hero-card mobile-hero-card--${hp.side} ${isExpanded ? "is-expanded" : ""}`}
                    >
                      <button
                        type="button"
                        className="mobile-hero-card__head"
                        aria-expanded={isExpanded}
                        onClick={() => toggleRowExpand(hp.slotIdx)}
                      >
                        <span className="mobile-hero-card__id">
                          <RoleIcon position={hp.position} size={15} />
                          <span className="hero-cell__thumb">
                            <HeroPortraitImg
                              src={hp.hero.img}
                              alt={hp.hero.display_name}
                              fallbackText={hp.hero.display_name.slice(0, 2)}
                              width={48}
                              height={27}
                              decorative
                            />
                          </span>
                          <span className="hero-cell__meta">
                            <span className="hero-cell__name">{hp.hero.display_name}</span>
                            <span className="hero-cell__sub">
                              <span className="hero-cell__role">
                                {ROLE_NAMES[hp.position].short}
                              </span>
                              {earlySummary ? (
                                <span className="hero-cell__build-code measured">
                                  {earlySummary}
                                </span>
                              ) : null}
                            </span>
                          </span>
                        </span>

                        <span className="mobile-hero-card__chevron" aria-hidden="true">
                          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </span>
                      </button>

                      <div className="mobile-hero-card__body">
                        <div className="mobile-ability-summary">
                          <span
                            tabIndex={0}
                            className="talent-glyph-trigger"
                            aria-label={`Talent choices: ${descendingTalents.map((t) => `Lvl ${t.tier} ${t.displayName}`).join(", ")}`}
                            {...bindTip({
                              talents: descendingTalents,
                            })}
                          >
                            <TalentTreeGlyph talents={hp.talents} size={30} />
                          </span>
                          <AbilityPipSummary
                            hp={hp}
                            entries={skillEntries}
                            bindTip={bindTip}
                            compact
                          />
                        </div>
                        {hp.items.length === 0 ? (
                          <span className="muted-note">No key items</span>
                        ) : (
                          <ol className="items-row">
                            {hp.items.map((item, idx) => (
                              <li key={`${item.itemToken}-${idx}`}>
                                {renderItemChip(item, idx, `#${idx + 1} Key item`)}
                              </li>
                            ))}
                          </ol>
                        )}
                      </div>

                      {isExpanded ? (
                        <div className="mobile-hero-card__drawer">
                          <HeroDetailContent
                            hp={hp}
                            prediction={prediction}
                            vocab={vocab}
                            bindTip={bindTip}
                          />
                        </div>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            ))}
          </div>
        </>
      ) : null}

      {/* VIEW 3: BUILDS > TIMELINE (horizontal time-axis item spikes) */}
      {primaryTab === "builds" && buildsSubTab === "timeline" ? (
        <>
          <div className="timeline-panel builds-table-wrap">
            <div className="timeline-axis-row">
              <div className="timeline-axis-row__corner">Hero</div>
              <div className="timeline-axis-row__scale">
                {timelineTicks.map((tick) => {
                  const pct = (tick / maxItemMinute) * 100;
                  return (
                    <span
                      key={tick}
                      className={`timeline-axis-tick measured ${tick === 0 ? "is-start" : tick === maxItemMinute ? "is-end" : ""}`}
                      style={{ left: `${pct}%` }}
                    >
                      {tick}m
                    </span>
                  );
                })}
              </div>
            </div>

            {teamSections.map((section) => (
              <div key={section.side} className="timeline-team-group">
                <div className={`timeline-team-band timeline-team-band--${section.side}`}>
                  <FactionEmblem side={section.side} size={14} />
                  <span>{section.label}</span>
                </div>

                {section.heroes.map((hp) => {
                  const positioned = layoutTimelineItems(hp.items, maxItemMinute);
                  const isExpanded = expandedSlotIdx === hp.slotIdx;
                  return (
                    <div key={hp.slotIdx} className="timeline-hero-block">
                      <div className={`timeline-hero-row row-side--${hp.side}`}>
                        <button
                          type="button"
                          className="timeline-hero-row__id"
                          onClick={() => toggleRowExpand(hp.slotIdx)}
                          aria-expanded={isExpanded}
                        >
                          <RoleIcon position={hp.position} size={14} />
                          <span className="hero-cell__thumb">
                            <HeroPortraitImg
                              src={hp.hero.img}
                              alt={hp.hero.display_name}
                              fallbackText={hp.hero.display_name.slice(0, 2)}
                              width={44}
                              height={25}
                              decorative
                            />
                          </span>
                          <span className="hero-cell__meta">
                            <span className="hero-cell__name">{hp.hero.display_name}</span>
                            <span className="hero-cell__role">
                              {ROLE_NAMES[hp.position].short}
                            </span>
                          </span>
                        </button>

                        <div className="timeline-lane">
                          {timelineTicks.map((tick) => (
                            <span
                              key={tick}
                              className="timeline-lane__gridline"
                              style={{ left: `${(tick / maxItemMinute) * 100}%` }}
                              aria-hidden="true"
                            />
                          ))}
                          <span className="timeline-lane__baseline" aria-hidden="true" />
                          {positioned.map(({ item, leftPct, tierRow }, idx) => (
                            <span
                              key={`${item.itemToken}-tl-${idx}`}
                              tabIndex={0}
                              className={`timeline-node ${tierRow === 1 ? "is-staggered" : ""}`}
                              style={{ left: `${leftPct}%` }}
                              aria-label={`${hp.hero.display_name}: ${item.displayName} at ${item.expectedMinute.toFixed(1)} min`}
                              {...bindTip({
                                kicker: `${hp.hero.display_name} · #${idx + 1}`,
                                title: item.displayName,
                                detail: `${item.cost.toLocaleString()} g · ${item.expectedMinute.toFixed(1)} min`,
                              })}
                            >
                              <span className="timeline-node__img">
                                <HeroPortraitImg
                                  src={item.img}
                                  alt={item.displayName}
                                  fallbackText={item.displayName.slice(0, 2)}
                                  width={32}
                                  height={23}
                                  decorative
                                />
                              </span>
                              <span className="timeline-node__min measured">
                                {formatTimingShort(item.expectedMinute)}
                              </span>
                            </span>
                          ))}
                        </div>
                      </div>

                      {isExpanded ? (
                        <div className="timeline-drawer-wrap">
                          <HeroDetailContent
                            hp={hp}
                            prediction={prediction}
                            vocab={vocab}
                            bindTip={bindTip}
                            onCloseInline={() => setExpandedSlotIdx(null)}
                          />
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>

          <div className="mobile-forecast-cards">
            {teamSections.map((section) => (
              <div key={section.side} className="mobile-team-group">
                <div className={`mobile-team-group__head mobile-team-group__head--${section.side}`}>
                  <FactionEmblem side={section.side} size={14} />
                  <span>{section.label}</span>
                </div>
                {section.heroes.map((hp) => (
                  <article key={hp.slotIdx} className={`mobile-hero-card mobile-hero-card--${hp.side}`}>
                    <div className="mobile-hero-card__head mobile-hero-card__head--static">
                      <span className="mobile-hero-card__id">
                        <RoleIcon position={hp.position} size={14} />
                        <span className="hero-cell__thumb">
                          <HeroPortraitImg
                            src={hp.hero.img}
                            alt={hp.hero.display_name}
                            fallbackText={hp.hero.display_name.slice(0, 2)}
                            width={40}
                            height={22}
                            decorative
                          />
                        </span>
                        <span className="hero-cell__name">{hp.hero.display_name}</span>
                      </span>
                    </div>
                    <div className="mobile-hero-card__body">
                      <ol className="item-chevron-strip">
                        {hp.items.map((item, idx) => (
                          <li key={`${item.itemToken}-m-${idx}`} className="item-chevron-strip__node">
                            {idx > 0 ? (
                              <ChevronRight size={11} className="item-chevron-strip__arrow" aria-hidden="true" />
                            ) : null}
                            {renderItemChip(item, idx, `${hp.hero.display_name} · #${idx + 1}`)}
                          </li>
                        ))}
                      </ol>
                    </div>
                  </article>
                ))}
              </div>
            ))}
          </div>
        </>
      ) : null}

      {/* VIEW 4: BUILDS > ABILITIES (1..18 unified icon matrix) */}
      {primaryTab === "builds" && buildsSubTab === "abilities" ? (
        <>
          <div className="builds-table-wrap">
            <table className="builds-table abilities-grid-table">
              <thead>
                <tr>
                  <th scope="col">Hero</th>
                  {LEVELS_1_TO_18.map((lvl) => (
                    <th
                      key={lvl}
                      scope="col"
                      className={`ab-grid__th measured ${ULT_LEVEL_CHECKPOINTS.has(lvl) ? "is-ult-col" : ""}`}
                    >
                      {lvl}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {teamSections.map((section, sIdx) => (
                  <Fragment key={section.side}>
                    {sIdx > 0 ? (
                      <tr className="team-spacer-row" aria-hidden="true">
                        <td colSpan={19} />
                      </tr>
                    ) : null}
                    <tr className={`team-band-row team-band-row--${section.side}`}>
                      <th scope="colgroup" colSpan={19}>
                        <span className="team-band-row__inner">
                          <FactionEmblem side={section.side} size={15} />
                          <span className="team-band-row__label">{section.label}</span>
                        </span>
                      </th>
                    </tr>
                    {section.heroes.map((hp) => {
                      const earlySummary = summarizeEarlySkillBuild(
                        hp.hero,
                        hp.skills,
                        vocab.abilities.by_id,
                      );
                      return (
                        <tr key={hp.slotIdx} className={`row-side--${hp.side}`}>
                          <th scope="row" className="hero-cell">
                            <button
                              type="button"
                              className="hero-row-trigger"
                              onClick={() => {
                                setSelectedSlotIdx(hp.slotIdx);
                                setPrimaryTab("focus");
                                setTooltip(null);
                              }}
                              title={`Open ${hp.hero.display_name} in Focus`}
                            >
                              <span className="hero-cell__role-icon" aria-hidden="true">
                                <RoleIcon position={hp.position} size={14} />
                              </span>
                              <span className="hero-cell__thumb">
                                <HeroPortraitImg
                                  src={hp.hero.img}
                                  alt={hp.hero.display_name}
                                  fallbackText={hp.hero.display_name.slice(0, 2)}
                                  width={44}
                                  height={25}
                                  decorative
                                />
                              </span>
                              <span className="hero-cell__meta">
                                <span className="hero-cell__name">{hp.hero.display_name}</span>
                                <span className="hero-cell__sub">
                                  {earlySummary ? (
                                    <span className="hero-cell__build-code measured">
                                      {earlySummary}
                                    </span>
                                  ) : (
                                    <span className="hero-cell__role">
                                      {ROLE_NAMES[hp.position].short}
                                    </span>
                                  )}
                                </span>
                              </span>
                            </button>
                          </th>
                          {renderAbilitiesLevelRow(hp)}
                        </tr>
                      );
                    })}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mobile-forecast-cards">
            {teamSections.map((section) => (
              <div key={section.side} className="mobile-team-group">
                <div className={`mobile-team-group__head mobile-team-group__head--${section.side}`}>
                  <FactionEmblem side={section.side} size={14} />
                  <span>{section.label}</span>
                </div>
                {section.heroes.map((hp) => {
                  const earlySummary = summarizeEarlySkillBuild(
                    hp.hero,
                    hp.skills,
                    vocab.abilities.by_id,
                  );
                  return (
                    <article key={hp.slotIdx} className={`mobile-hero-card mobile-hero-card--${hp.side}`}>
                      <button
                        type="button"
                        className="mobile-hero-card__head"
                        onClick={() => {
                          setSelectedSlotIdx(hp.slotIdx);
                          setPrimaryTab("focus");
                        }}
                      >
                        <span className="mobile-hero-card__id">
                          <RoleIcon position={hp.position} size={14} />
                          <span className="hero-cell__thumb">
                            <HeroPortraitImg
                              src={hp.hero.img}
                              alt={hp.hero.display_name}
                              fallbackText={hp.hero.display_name.slice(0, 2)}
                              width={40}
                              height={22}
                              decorative
                            />
                          </span>
                          <span className="hero-cell__meta">
                            <span className="hero-cell__name">{hp.hero.display_name}</span>
                            {earlySummary ? (
                              <span className="hero-cell__build-code measured">{earlySummary}</span>
                            ) : null}
                          </span>
                        </span>
                        <ChevronRight size={15} className="mobile-hero-card__chevron" aria-hidden="true" />
                      </button>
                      <div className="mobile-hero-card__body">
                        <ol className="mobile-skill-sequence">
                          {hp.skills
                            .filter((s) => s.kind !== "empty")
                            .flatMap((step) => {
                              const colorKey =
                                step.kind === "ability"
                                  ? getAbilityColorKey(step.shortTag, step.isUltimate)
                                  : step.kind;
                              const nodes = [
                                <li
                                  key={step.level}
                                  className={`mobile-skill-chip mobile-skill-chip--${colorKey} measured`}
                                >
                                  <span className="mobile-skill-chip__lvl">{step.level}</span>
                                  <span>{step.kind === "talent" ? "T" : step.shortTag}</span>
                                </li>,
                              ];
                              if (
                                (step.level === 10 || step.level === 15) &&
                                step.kind !== "talent" &&
                                hp.talents.some((t) => t.tier === step.level)
                              ) {
                                nodes.push(
                                  <li
                                    key={`${step.level}-talent`}
                                    className="mobile-skill-chip mobile-skill-chip--talent measured"
                                  >
                                    <span className="mobile-skill-chip__lvl">{step.level}</span>
                                    <span>T</span>
                                  </li>,
                                );
                              }
                              return nodes;
                            })}
                        </ol>
                      </div>
                    </article>
                  );
                })}
              </div>
            ))}
          </div>
        </>
      ) : null}

      {/* VIEW 5: SCOREBOARD (team comparison + proportional GPM/XPM bars) */}
      {primaryTab === "scoreboard" ? (
        <>
          {teamFilter === "all" ? renderTeamCompareCards() : null}

          <div className="builds-table-wrap">
            <table className="builds-table combat-table">
              <thead>
                <tr>
                  <th scope="col">Hero</th>
                  <th scope="col">K / D / A</th>
                  <th scope="col">GPM</th>
                  <th scope="col">XPM</th>
                  <th scope="col">CS</th>
                  <th scope="col">Hero dmg</th>
                  <th scope="col">Tower dmg</th>
                  <th scope="col">Taken</th>
                  <th scope="col">Heal</th>
                  <th scope="col" className="expand-th">
                    <span className="visually-hidden">Details</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {teamSections.map((section, sIdx) => {
                  const totals = summarizeTeamCombat(section.heroes);
                  return (
                    <Fragment key={section.side}>
                      {sIdx > 0 ? (
                        <tr className="team-spacer-row" aria-hidden="true">
                          <td colSpan={10} />
                        </tr>
                      ) : null}
                      <tr className={`team-band-row team-band-row--${section.side}`}>
                        <th scope="colgroup" colSpan={10}>
                          <span className="team-band-row__inner">
                            <FactionEmblem side={section.side} size={15} />
                            <span className="team-band-row__label">{section.label}</span>
                            <span className="team-band-row__score measured">
                              {totals.kills} Kills · {totals.gpmTotal.toLocaleString()} Team GPM
                            </span>
                          </span>
                        </th>
                      </tr>
                      {section.heroes.map((hp) => {
                        const isExpanded = expandedSlotIdx === hp.slotIdx;
                        const gpmRatio = Math.min(
                          100,
                          Math.max(8, (hp.economy.gpm.mid / maxHeroGpm) * 100),
                        );
                        const xpmRatio = Math.min(
                          100,
                          Math.max(8, (hp.economy.xpm.mid / maxHeroXpm) * 100),
                        );
                        return (
                          <Fragment key={hp.slotIdx}>
                            <tr
                              className={`row-side--${hp.side} ${isExpanded ? "is-expanded-row" : ""}`}
                            >
                              <th scope="row" className="hero-cell">
                                <button
                                  type="button"
                                  className="hero-row-trigger"
                                  aria-expanded={isExpanded}
                                  onClick={() => toggleRowExpand(hp.slotIdx)}
                                >
                                  <span className="hero-cell__role-icon" aria-hidden="true">
                                    <RoleIcon position={hp.position} size={15} />
                                  </span>
                                  <span className="hero-cell__thumb">
                                    <HeroPortraitImg
                                      src={hp.hero.img}
                                      alt={hp.hero.display_name}
                                      fallbackText={hp.hero.display_name.slice(0, 2)}
                                      width={44}
                                      height={25}
                                      decorative
                                    />
                                  </span>
                                  <span className="hero-cell__meta">
                                    <span className="hero-cell__name">
                                      {hp.hero.display_name}
                                    </span>
                                    <span className="hero-cell__role">
                                      {ROLE_NAMES[hp.position].full}
                                    </span>
                                  </span>
                                </button>
                              </th>

                              <td className="metric-cell measured">
                                <span className="metric-cell__main kda-inline">
                                  <span className="kda-k">
                                    {Math.round(hp.economy.kills.mid)}
                                  </span>
                                  <span className="kda-sep">/</span>
                                  <span className="kda-d">
                                    {Math.round(hp.economy.deaths.mid)}
                                  </span>
                                  <span className="kda-sep">/</span>
                                  <span className="kda-a">
                                    {Math.round(hp.economy.assists.mid)}
                                  </span>
                                </span>
                                <span className="metric-cell__sub">
                                  {formatRangeText(hp.economy.kills)} ·{" "}
                                  {formatRangeText(hp.economy.deaths)} ·{" "}
                                  {formatRangeText(hp.economy.assists)}
                                </span>
                              </td>

                              <td className="metric-cell metric-cell--bar measured">
                                <span className="metric-cell__main metric-cell__main--gold">
                                  {formatRangeText(hp.economy.gpm)}
                                </span>
                                <div className="micro-bar" aria-hidden="true">
                                  <span
                                    className="micro-bar__fill micro-bar__fill--gold"
                                    style={{ width: `${gpmRatio}%` }}
                                  />
                                </div>
                              </td>

                              <td className="metric-cell metric-cell--bar measured">
                                <span className="metric-cell__main metric-cell__main--xp">
                                  {formatRangeText(hp.economy.xpm)}
                                </span>
                                <div className="micro-bar" aria-hidden="true">
                                  <span
                                    className="micro-bar__fill micro-bar__fill--xp"
                                    style={{ width: `${xpmRatio}%` }}
                                  />
                                </div>
                              </td>

                              <td className="metric-cell measured">
                                <span className="metric-cell__main">
                                  {formatRangeText(hp.economy.csTotal)}
                                </span>
                                <span className="metric-cell__sub">
                                  {formatRangeText(hp.economy.csPerMin, 1)}/min
                                </span>
                              </td>

                              <td className="combat-cell measured">
                                <CombatCellContent
                                  stat={hp.combat.heroDamage}
                                  side={hp.side}
                                  showShare
                                />
                              </td>
                              <td className="combat-cell measured">
                                <CombatCellContent
                                  stat={hp.combat.towerDamage}
                                  side={hp.side}
                                  showShare
                                />
                              </td>
                              <td className="combat-cell measured">
                                <CombatCellContent
                                  stat={hp.combat.damageTaken}
                                  side={hp.side}
                                  showShare
                                />
                              </td>
                              <td className="combat-cell measured">
                                <CombatCellContent
                                  stat={hp.combat.healing}
                                  side={hp.side}
                                  showShare={hp.combat.showHealingShare}
                                  unit="HPM"
                                />
                              </td>
                              <td className="expand-cell">
                                <button
                                  type="button"
                                  className="btn btn--icon expand-row-btn"
                                  aria-expanded={isExpanded}
                                  aria-label={
                                    isExpanded
                                      ? `Hide ${hp.hero.display_name} details`
                                      : `Inspect ${hp.hero.display_name} details`
                                  }
                                  onClick={() => toggleRowExpand(hp.slotIdx)}
                                >
                                  {isExpanded ? (
                                    <ChevronUp size={15} aria-hidden="true" />
                                  ) : (
                                    <ChevronDown size={15} aria-hidden="true" />
                                  )}
                                </button>
                              </td>
                            </tr>
                            {isExpanded ? (
                              <tr className="detail-drawer-row">
                                <td colSpan={10} className="detail-drawer-cell">
                                  <HeroDetailContent
                                    hp={hp}
                                    prediction={prediction}
                                    vocab={vocab}
                                    bindTip={bindTip}
                                    onCloseInline={() => setExpandedSlotIdx(null)}
                                  />
                                </td>
                              </tr>
                            ) : null}
                          </Fragment>
                        );
                      })}
                      <tr className={`team-totals-row team-totals-row--${section.side}`}>
                        <th scope="row" className="team-totals-row__head">
                          {section.label} totals
                        </th>
                        <td className="metric-cell measured">
                          <span className="metric-cell__main kda-inline">
                            <span className="kda-k">{totals.kills}</span>
                            <span className="kda-sep">/</span>
                            <span className="kda-d">{totals.deaths}</span>
                            <span className="kda-sep">/</span>
                            <span className="kda-a">{totals.assists}</span>
                          </span>
                        </td>
                        <td className="metric-cell measured">
                          <span className="metric-cell__main metric-cell__main--gold">
                            {totals.gpmTotal.toLocaleString()}
                          </span>
                          <span className="metric-cell__sub">{totals.gpmAvg} avg</span>
                        </td>
                        <td className="metric-cell measured">
                          <span className="metric-cell__main metric-cell__main--xp">
                            {totals.xpmTotal.toLocaleString()}
                          </span>
                          <span className="metric-cell__sub">{totals.xpmAvg} avg</span>
                        </td>
                        <td className="metric-cell measured">
                          <span className="metric-cell__main">
                            {totals.csTotal.toLocaleString()}
                          </span>
                          <span className="metric-cell__sub">{totals.csPerMin}/min</span>
                        </td>
                        <td className="metric-cell measured">
                          <span className="metric-cell__main">
                            {totals.heroDmg.toLocaleString()} DPM
                          </span>
                        </td>
                        <td className="metric-cell measured">
                          <span className="metric-cell__main">
                            {totals.towerDmg.toLocaleString()} DPM
                          </span>
                        </td>
                        <td className="metric-cell measured">
                          <span className="metric-cell__main">
                            {totals.taken.toLocaleString()} DPM
                          </span>
                        </td>
                        <td className="metric-cell measured">
                          <span className="metric-cell__main">
                            {totals.showHeal ? `${totals.heal.toLocaleString()} HPM` : "—"}
                          </span>
                        </td>
                        <td className="expand-cell" />
                      </tr>
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="mobile-forecast-cards">
            {teamSections.map((section) => (
              <div key={section.side} className="mobile-team-group">
                <div className={`mobile-team-group__head mobile-team-group__head--${section.side}`}>
                  <FactionEmblem side={section.side} size={14} />
                  <span>{section.label}</span>
                </div>

                {section.heroes.map((hp) => {
                  const isExpanded = expandedSlotIdx === hp.slotIdx;
                  return (
                    <article
                      key={hp.slotIdx}
                      className={`mobile-hero-card mobile-hero-card--${hp.side} ${isExpanded ? "is-expanded" : ""}`}
                    >
                      <button
                        type="button"
                        className="mobile-hero-card__head"
                        aria-expanded={isExpanded}
                        onClick={() => toggleRowExpand(hp.slotIdx)}
                      >
                        <span className="mobile-hero-card__id">
                          <RoleIcon position={hp.position} size={14} />
                          <span className="hero-cell__thumb">
                            <HeroPortraitImg
                              src={hp.hero.img}
                              alt={hp.hero.display_name}
                              fallbackText={hp.hero.display_name.slice(0, 2)}
                              width={44}
                              height={25}
                              decorative
                            />
                          </span>
                          <span className="hero-cell__meta">
                            <span className="hero-cell__name">{hp.hero.display_name}</span>
                            <span className="hero-cell__role">
                              {ROLE_NAMES[hp.position].full}
                            </span>
                          </span>
                        </span>
                        <span className="mobile-hero-card__chevron" aria-hidden="true">
                          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </span>
                      </button>

                      <div className="mobile-hero-card__combat measured">
                        <div className="mobile-stat-pill">
                          <span className="mobile-stat-pill__k">GPM</span>
                          <strong className="metric-cell__main--gold">
                            {formatRangeText(hp.economy.gpm)}
                          </strong>
                        </div>
                        <div className="mobile-stat-pill">
                          <span className="mobile-stat-pill__k">K/D/A</span>
                          <strong>
                            {Math.round(hp.economy.kills.mid)}/
                            {Math.round(hp.economy.deaths.mid)}/
                            {Math.round(hp.economy.assists.mid)}
                          </strong>
                        </div>
                        <div className="mobile-stat-pill">
                          <span className="mobile-stat-pill__k">Hero dmg</span>
                          <strong>
                            {Math.round(hp.combat.heroDamage.dpm)} DPM (
                            {Math.round(hp.combat.heroDamage.teamShare * 100)}%)
                          </strong>
                        </div>
                        <div className="mobile-stat-pill">
                          <span className="mobile-stat-pill__k">Tower dmg</span>
                          <strong>
                            {Math.round(hp.combat.towerDamage.dpm)} DPM (
                            {Math.round(hp.combat.towerDamage.teamShare * 100)}%)
                          </strong>
                        </div>
                      </div>

                      {isExpanded ? (
                        <div className="mobile-hero-card__drawer">
                          <HeroDetailContent
                            hp={hp}
                            prediction={prediction}
                            vocab={vocab}
                            bindTip={bindTip}
                          />
                        </div>
                      ) : null}
                    </article>
                  );
                })}
              </div>
            ))}
          </div>
        </>
      ) : null}

      {renderFloatingTooltip()}
    </section>
  );
}
