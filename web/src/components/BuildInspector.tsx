import {
  Fragment,
  useEffect,
  useState,
  type FocusEvent,
  type PointerEvent,
  type ReactNode,
} from "react";
import { Check, ChevronDown, ChevronUp, X } from "lucide-react";

import {
  getAbilityColorKey,
  getLevelableSkillEntries,
} from "../data/abilityDisplay";
import { ROLE_NAMES } from "../data/draftPresets";
import {
  formatRangeText,
  summarizeEarlySkillBuild,
} from "../ml/decoder";
import type {
  CombatShareStat,
  DraftPrediction,
  HeroPrediction,
  MatchPaceOption,
  TeamSide,
  VocabJson,
} from "../types/oracle";
import { HeroPortraitImg } from "./HeroPortraitImg";

interface BuildInspectorProps {
  prediction: DraftPrediction | null;
  vocab: VocabJson | null;
  matchPace: MatchPaceOption;
}

type TeamFilter = "all" | "radiant" | "dire";
type InspectorTab = "builds" | "combat" | "details";

interface TooltipData {
  x: number;
  yTop: number;
  yBottom: number;
  kicker?: string;
  title: string;
  detail?: string;
}

const ULT_LEVEL_CHECKPOINTS = new Set([6, 12, 18]);
const SKILL_GRID_COLS = [1, 2, 3, 4, 5, 6, 7, 8, 9] as const;

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

function HeroDetailContent({
  hp,
  prediction,
  vocab,
  onCloseInline,
}: {
  hp: HeroPrediction;
  prediction: DraftPrediction;
  vocab: VocabJson;
  onCloseInline?: () => void;
}) {
  const skillEntries = getLevelableSkillEntries(hp.hero, vocab.abilities.by_id);
  const earlySummary = summarizeEarlySkillBuild(hp.hero, hp.skills, vocab.abilities.by_id);
  const totalItemsCost = hp.items.reduce((sum, it) => sum + it.cost, 0);
  const descendingTalents = [...hp.talents].sort((a, b) => b.tier - a.tier);
  const showUltColumns = hp.hero.id !== 74;
  const hasAttributes = hp.skills.some((s) => s.kind === "attribute");

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
            <p className="muted-note">
              <span className={`side-label--${hp.side}`}>
                {hp.side === "radiant" ? "Radiant" : "Dire"}
              </span>
              {" · "}
              {ROLE_NAMES[hp.position].full}
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
          <span className="detail-kpi-chip measured">
            <span className="detail-kpi-chip__k">GPM</span>
            <strong>{formatRangeText(hp.economy.gpm)}</strong>
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

      <div className="hero-detail-panel__grid">
        <div className="detail-block detail-block--items">
          <div className="detail-block__head">
            <h4>Key items</h4>
            {totalItemsCost > 0 ? (
              <span className="detail-block__sub measured">
                {totalItemsCost.toLocaleString()} g total
              </span>
            ) : null}
          </div>
          {hp.items.length === 0 ? (
            <p className="muted-note">No key items predicted for this slot.</p>
          ) : (
            <ol className="detail-items-list">
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
                  <span className="detail-item-row__body">
                    <span className="detail-item-row__name">{item.displayName}</span>
                    <span className="detail-item-row__meta measured">
                      <span>{item.cost.toLocaleString()} g</span>
                      <span className="detail-item-row__time">
                        {item.expectedMinute.toFixed(1)} min
                      </span>
                    </span>
                  </span>
                </li>
              ))}
            </ol>
          )}
        </div>

        <div className="detail-block detail-block--talents">
          <div className="detail-block__head">
            <h4>Talent tree</h4>
            <span className="detail-block__sub">Lvl 25 → 10</span>
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

        <div className="detail-block detail-block--economy">
          <div className="detail-block__head">
            <h4>Economy &amp; KDA</h4>
            <span className="detail-block__sub measured">
              {prediction.matchPaceMinutes === 0
                ? `Over ~${Math.round(prediction.effectiveDurationMinutes)} min`
                : `Over ${prediction.matchPaceMinutes} min`}
            </span>
          </div>
          <div className="econ-grid">
            <div className="econ-stat">
              <span className="econ-stat__label">GPM</span>
              <span className="econ-stat__val measured">
                {formatRangeText(hp.economy.gpm)}
              </span>
            </div>
            <div className="econ-stat">
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
              <span className="econ-stat__val measured">
                {formatRangeText(hp.economy.kills)}
              </span>
            </div>
            <div className="econ-stat">
              <span className="econ-stat__label">Deaths</span>
              <span className="econ-stat__val measured">
                {formatRangeText(hp.economy.deaths)}
              </span>
            </div>
            <div className="econ-stat">
              <span className="econ-stat__label">Assists</span>
              <span className="econ-stat__val measured">
                {formatRangeText(hp.economy.assists)}
              </span>
            </div>
          </div>
        </div>

        <div className="detail-block detail-block--combat">
          <div className="detail-block__head">
            <h4>Combat impact</h4>
            <span className="detail-block__sub">Rate &amp; team share</span>
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

        <div className="detail-block detail-block--matrix">
          <div className="detail-block__head">
            <h4>Skill matrix (levels 1 to 18)</h4>
            <span className="detail-block__sub skill-matrix-rank-hint">
              Numbers inside grid show rank gained
            </span>
          </div>
          <div className="ability-matrix-wrap">
            <table className="ability-matrix">
              <thead>
                <tr>
                  <th scope="col" className="ability-matrix__head-label">
                    Ability
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
                      <th scope="row" className="ability-matrix__name">
                        <span className="ability-matrix__label-cell">
                          <span
                            className={`ability-matrix__key ability-matrix__key--${colorKey} measured`}
                          >
                            {entry.shortTag}
                          </span>
                          <span className="ability-matrix__icon">
                            <HeroPortraitImg
                              src={entry.ability?.img}
                              alt={abName}
                              fallbackText={entry.shortTag}
                              width={20}
                              height={20}
                              decorative
                            />
                          </span>
                          <span className="ability-matrix__title">{abName}</span>
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
                            aria-label={
                              hit
                                ? `${abName}${st.rank ? ` rank ${st.rank}` : ""} at level ${st.level}`
                                : undefined
                            }
                          >
                            {hit ? (
                              <>
                                <span
                                  className={`matrix-mark matrix-mark--slot-${colorKey} measured`}
                                  aria-hidden="true"
                                >
                                  {st.rank ?? ""}
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
                <tr>
                  <th scope="row" className="ability-matrix__name">
                    <span className="ability-matrix__label-cell">
                      <span className="ability-matrix__key ability-matrix__key--talent measured">
                        T
                      </span>
                      <span className="ability-matrix__title">Talent</span>
                    </span>
                  </th>
                  {hp.skills.map((st) => {
                    const hit = st.kind === "talent";
                    const isUltCol = showUltColumns && ULT_LEVEL_CHECKPOINTS.has(st.level);
                    return (
                      <td
                        key={st.level}
                        className={`${hit ? "is-picked-talent" : ""} ${isUltCol ? "is-ult-col" : ""}`}
                      >
                        {hit ? (
                          <>
                            <span
                              className="matrix-mark matrix-mark--talent measured"
                              aria-hidden="true"
                            >
                              T
                            </span>
                            <span className="matrix-lvl-tag matrix-lvl-tag--talent measured">
                              {st.level}
                            </span>
                          </>
                        ) : null}
                      </td>
                    );
                  })}
                </tr>
                {hasAttributes ? (
                  <tr>
                    <th scope="row" className="ability-matrix__name">
                      <span className="ability-matrix__label-cell">
                        <span className="ability-matrix__key ability-matrix__key--attr measured">
                          +
                        </span>
                        <span className="ability-matrix__title">Stats</span>
                      </span>
                    </th>
                    {hp.skills.map((st) => {
                      const hit = st.kind === "attribute";
                      const isUltCol = showUltColumns && ULT_LEVEL_CHECKPOINTS.has(st.level);
                      return (
                        <td
                          key={st.level}
                          className={`${hit ? "is-picked-attr" : ""} ${isUltCol ? "is-ult-col" : ""}`}
                        >
                          {hit ? (
                            <>
                              <span
                                className="matrix-mark matrix-mark--attr measured"
                                aria-hidden="true"
                              >
                                +
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
    </div>
  );
}

export function BuildInspector({
  prediction,
  vocab,
  matchPace,
}: BuildInspectorProps) {
  const [teamFilter, setTeamFilter] = useState<TeamFilter>("all");
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("builds");
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
    const placeBelow = tooltip.yTop < 78;
    const clampedX = Math.max(120, Math.min(window.innerWidth - 120, tooltip.x));
    const anchorY = placeBelow ? tooltip.yBottom + 8 : tooltip.yTop - 8;

    return (
      <div
        className={`floating-tip ${placeBelow ? "is-below" : "is-above"}`}
        role="tooltip"
        style={{ left: `${clampedX}px`, top: `${anchorY}px` }}
      >
        {tooltip.kicker ? (
          <span className="floating-tip__kicker measured">{tooltip.kicker}</span>
        ) : null}
        <span className="floating-tip__title">{tooltip.title}</span>
        {tooltip.detail ? (
          <span className="floating-tip__detail measured">{tooltip.detail}</span>
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

  const renderItemPills = (hp: HeroPrediction) =>
    hp.items.length === 0 ? (
      <span className="muted-note">No key items</span>
    ) : (
      <ol className="d2pt-items" aria-label={`Key items for ${hp.hero.display_name}`}>
        {hp.items.map((item, idx) => (
          <li key={`${item.itemToken}-${idx}`} className="d2pt-item">
            <span
              tabIndex={0}
              className="d2pt-item__trigger"
              aria-label={`${item.displayName}, ${item.cost.toLocaleString()} gold, ${item.expectedMinute.toFixed(1)} min`}
              {...bindTip({
                kicker: `#${idx + 1} Key item`,
                title: item.displayName,
                detail: `${item.cost.toLocaleString()} g · ${item.expectedMinute.toFixed(1)} min`,
              })}
            >
              <span className="d2pt-item__min measured">
                {Math.round(item.expectedMinute)}m
              </span>
              <span className="d2pt-item__icon">
                <HeroPortraitImg
                  src={item.img}
                  alt={item.displayName}
                  fallbackText={item.displayName.slice(0, 2)}
                  width={30}
                  height={22}
                  decorative
                />
              </span>
            </span>
          </li>
        ))}
      </ol>
    );

  const renderSkillGrid = (hp: HeroPrediction) => (
    <ol
      className="skill-strip"
      aria-label={`Level 1 to 18 skill order for ${hp.hero.display_name}`}
    >
      {hp.skills.map((step) => {
        const slotColor =
          step.kind === "ability"
            ? getAbilityColorKey(step.shortTag, step.isUltimate)
            : step.kind;
        const isUltPick = step.kind === "ability" && Boolean(step.isUltimate);
        const isCheckpoint =
          hp.hero.id !== 74 && ULT_LEVEL_CHECKPOINTS.has(step.level);
        return (
          <li
            key={step.level}
            className={`skill-cell skill-cell--${step.kind} skill-cell--slot-${slotColor} ${isUltPick ? "is-ult-pick" : ""} ${isCheckpoint ? "is-ult-checkpoint" : ""}`}
          >
            <span
              tabIndex={0}
              className="skill-cell__trigger"
              aria-label={
                step.kind === "empty"
                  ? `Level ${step.level}: No skill point`
                  : `Level ${step.level}: ${step.label}`
              }
              {...bindTip({
                kicker: `Level ${step.level}`,
                title: step.kind === "empty" ? "No skill point" : step.label,
                detail:
                  step.kind === "ability"
                    ? step.isUltimate
                      ? `Slot ${step.shortTag} · Ultimate${step.rank ? ` (Rank ${step.rank})` : ""}`
                      : `Slot ${step.shortTag}${step.rank ? ` (Rank ${step.rank})` : ""}`
                    : step.kind === "talent"
                      ? "Talent point"
                      : step.kind === "attribute"
                        ? "All attributes +2"
                        : "No skill point at this level",
              })}
            >
              {step.kind === "ability" && step.img ? (
                <span className="skill-cell__icon-wrap">
                  <HeroPortraitImg
                    src={step.img}
                    alt={step.label}
                    fallbackText={step.shortTag}
                    width={28}
                    height={28}
                    decorative
                  />
                  <span
                    className={`skill-cell__hotkey skill-cell__hotkey--${slotColor} measured`}
                    aria-hidden="true"
                  >
                    {step.shortTag}
                  </span>
                </span>
              ) : (
                <span className="skill-cell__badge">
                  {step.kind === "talent" ? "T" : step.shortTag}
                </span>
              )}
            </span>
          </li>
        );
      })}
    </ol>
  );

  return (
    <section className="builds-section" aria-labelledby="builds-heading">
      <div className="builds-header">
        <div className="builds-header__title-group">
          <div className="builds-header__heading-row">
            <h2 id="builds-heading">Hero forecast</h2>
            {matchPace !== 0 ? (
              <span className="summary-badge summary-badge--pace measured nowrap">
                Builds @ {matchPace}m
              </span>
            ) : null}
          </div>
          <p className="muted-note">
            Click any hero row to expand talents, item costs, and skill matrix.
          </p>
        </div>

        <div className="builds-header__controls">
          <div
            className="segmented builds-view-segmented"
            role="group"
            aria-label="Forecast view"
          >
            <button
              type="button"
              className={`segmented__btn ${inspectorTab === "builds" ? "is-active" : ""}`}
              aria-pressed={inspectorTab === "builds"}
              onClick={() => {
                setInspectorTab("builds");
                setTooltip(null);
              }}
            >
              Builds &amp; Skills
            </button>
            <button
              type="button"
              className={`segmented__btn ${inspectorTab === "combat" ? "is-active" : ""}`}
              aria-pressed={inspectorTab === "combat"}
              onClick={() => {
                setInspectorTab("combat");
                setTooltip(null);
              }}
            >
              Combat &amp; Economy
            </button>
            <button
              type="button"
              className={`segmented__btn ${inspectorTab === "details" ? "is-active" : ""}`}
              aria-pressed={inspectorTab === "details"}
              onClick={() => {
                setInspectorTab("details");
                setTooltip(null);
              }}
            >
              Hero Details
            </button>
          </div>

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
        </div>
      </div>

      {inspectorTab === "details" ? (
        <div className="hero-details-view">
          <div className="hero-strip-grid" role="tablist" aria-label="Select hero details">
            {teamSections.map((section) => (
              <div
                key={section.side}
                className={`hero-strip-group hero-strip-group--${section.side}`}
              >
                <span className={`hero-strip-group__label hero-strip-group__label--${section.side}`}>
                  {section.label}
                </span>
                <div className="hero-strip-group__cards">
                  {section.heroes.map((hp) => {
                    const isSelected = hp.slotIdx === inspectedHero.slotIdx;
                    return (
                      <button
                        key={hp.slotIdx}
                        type="button"
                        role="tab"
                        aria-selected={isSelected}
                        className={`hero-switch-chip hero-switch-chip--${hp.side} ${isSelected ? "is-selected" : ""}`}
                        onClick={() => {
                          setSelectedSlotIdx(hp.slotIdx);
                          setTooltip(null);
                        }}
                      >
                        <span className="hero-switch-chip__thumb">
                          <HeroPortraitImg
                            src={hp.hero.img}
                            alt={hp.hero.display_name}
                            fallbackText={hp.hero.display_name.slice(0, 2)}
                            width={40}
                            height={22}
                            decorative
                          />
                        </span>
                        <span className="hero-switch-chip__text">
                          <span className="hero-switch-chip__name">{hp.hero.display_name}</span>
                          <span className="hero-switch-chip__role">
                            {ROLE_NAMES[hp.position].short}
                          </span>
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          <HeroDetailContent
            hp={inspectedHero}
            prediction={prediction}
            vocab={vocab}
          />
        </div>
      ) : (
        <>
          <div className="builds-table-wrap">
            {inspectorTab === "builds" ? (
              <table className="builds-table">
                <thead>
                  <tr>
                    <th scope="col">Hero</th>
                    <th scope="col">Key items</th>
                    <th scope="col" className="skills-th">
                      <div className="skills-th__head">
                        <span>Skill build (1–18)</span>
                        <span className="skills-th__ruler measured" aria-hidden="true">
                          {SKILL_GRID_COLS.map((n) => (
                            <span
                              key={n}
                              className={n === 6 || n === 9 ? "is-milestone" : ""}
                            >
                              {n}
                            </span>
                          ))}
                        </span>
                      </div>
                    </th>
                    <th scope="col">Talents</th>
                    <th scope="col" className="expand-th">
                      <span className="visually-hidden">Details</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {teamSections.map((section) => (
                    <Fragment key={section.side}>
                      {teamFilter === "all" ? (
                        <tr className={`team-band-row team-band-row--${section.side}`}>
                          <th scope="colgroup" colSpan={5}>
                            <span className="team-band-row__label">{section.label}</span>
                          </th>
                        </tr>
                      ) : null}
                      {section.heroes.map((hp) => {
                        const isExpanded = expandedSlotIdx === hp.slotIdx;
                        const earlySummary = summarizeEarlySkillBuild(
                          hp.hero,
                          hp.skills,
                          vocab.abilities.by_id,
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
                                      <span className="hero-cell__role">
                                        {ROLE_NAMES[hp.position].full}
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

                              <td className="items-cell">{renderItemPills(hp)}</td>

                              <td className="skills-cell">{renderSkillGrid(hp)}</td>

                              <td className="talents-cell">
                                <ul className="talent-list">
                                  {hp.talents.map((tal) => (
                                    <li key={tal.tier} className="talent-row">
                                      <span className="talent-row__tier measured">{tal.tier}</span>
                                      <span
                                        className={`talent-row__branch talent-row__branch--${tal.chosenSide === 0 ? "left" : "right"}`}
                                        aria-hidden="true"
                                        title={
                                          tal.chosenSide === 0 ? "Left talent" : "Right talent"
                                        }
                                      >
                                        <span className="branch-dot" />
                                        <span className="branch-dot" />
                                      </span>
                                      <span className="talent-row__text" title={tal.displayName}>
                                        {tal.displayName}
                                      </span>
                                    </li>
                                  ))}
                                </ul>
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
            ) : (
              <table className="builds-table combat-table">
                <thead>
                  <tr>
                    <th scope="col">Hero</th>
                    <th scope="col">GPM</th>
                    <th scope="col">XPM</th>
                    <th scope="col">CS</th>
                    <th scope="col">K / D / A</th>
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
                  {teamSections.map((section) => {
                    const totals = summarizeTeamCombat(section.heroes);
                    return (
                      <Fragment key={section.side}>
                        {teamFilter === "all" ? (
                          <tr className={`team-band-row team-band-row--${section.side}`}>
                            <th scope="colgroup" colSpan={10}>
                              <span className="team-band-row__label">{section.label}</span>
                            </th>
                          </tr>
                        ) : null}
                        {section.heroes.map((hp) => {
                          const isExpanded = expandedSlotIdx === hp.slotIdx;
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
                                  <span className="metric-cell__main">
                                    {formatRangeText(hp.economy.gpm)}
                                  </span>
                                </td>
                                <td className="metric-cell measured">
                                  <span className="metric-cell__main">
                                    {formatRangeText(hp.economy.xpm)}
                                  </span>
                                </td>
                                <td className="metric-cell measured">
                                  <span className="metric-cell__main">
                                    {formatRangeText(hp.economy.csTotal)}
                                  </span>
                                  <span className="metric-cell__sub">
                                    {formatRangeText(hp.economy.csPerMin, 1)}/min
                                  </span>
                                </td>
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
                            <span className="metric-cell__main">
                              {totals.gpmTotal.toLocaleString()}
                            </span>
                            <span className="metric-cell__sub">{totals.gpmAvg} avg</span>
                          </td>
                          <td className="metric-cell measured">
                            <span className="metric-cell__main">
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
                            <span className="metric-cell__main kda-inline">
                              <span className="kda-k">{totals.kills}</span>
                              <span className="kda-sep">/</span>
                              <span className="kda-d">{totals.deaths}</span>
                              <span className="kda-sep">/</span>
                              <span className="kda-a">{totals.assists}</span>
                            </span>
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
            )}
          </div>

          <div className="mobile-forecast-cards">
            {teamSections.map((section) => (
              <div key={section.side} className="mobile-team-group">
                {teamFilter === "all" ? (
                  <div className={`mobile-team-group__head mobile-team-group__head--${section.side}`}>
                    <span>{section.label}</span>
                  </div>
                ) : null}

                {section.heroes.map((hp) => {
                  const isExpanded = expandedSlotIdx === hp.slotIdx;
                  const earlySummary = summarizeEarlySkillBuild(
                    hp.hero,
                    hp.skills,
                    vocab.abilities.by_id,
                  );

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
                              <span className="hero-cell__role">
                                {ROLE_NAMES[hp.position].full}
                              </span>
                              {inspectorTab === "builds" && earlySummary ? (
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

                      {inspectorTab === "builds" ? (
                        <div className="mobile-hero-card__body">
                          {renderItemPills(hp)}
                        </div>
                      ) : (
                        <div className="mobile-hero-card__combat measured">
                          <div className="mobile-stat-pill">
                            <span className="mobile-stat-pill__k">GPM</span>
                            <strong>{formatRangeText(hp.economy.gpm)}</strong>
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
                      )}

                      {isExpanded ? (
                        <div className="mobile-hero-card__drawer">
                          <HeroDetailContent
                            hp={hp}
                            prediction={prediction}
                            vocab={vocab}
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
      )}

      {renderFloatingTooltip()}
    </section>
  );
}
