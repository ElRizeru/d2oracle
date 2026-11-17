import { useEffect, useState, type FocusEvent, type PointerEvent } from "react";

import {
  getAbilityColorKey,
  getLevelableSkillEntries,
} from "../data/abilityDisplay";
import { formatRangeText } from "../ml/decoder";
import type {
  CombatShareStat,
  DraftPrediction,
  HeroPrediction,
  VocabJson,
} from "../types/oracle";
import { ROLE_NAMES } from "./DraftWorkspace";
import { HeroPortraitImg } from "./HeroPortraitImg";

interface BuildInspectorProps {
  prediction: DraftPrediction | null;
  vocab: VocabJson | null;
}

type TeamFilter = "all" | "radiant" | "dire";
type InspectorTab = "builds" | "combat";

interface TooltipData {
  x: number;
  yTop: number;
  yBottom: number;
  kicker?: string;
  title: string;
  detail?: string;
  accent?: string;
}

const ULT_LEVEL_CHECKPOINTS = new Set([6, 12, 18]);

function CombatCellContent({
  stat,
  side,
  showShare,
  unit = "DPM",
}: {
  stat: CombatShareStat;
  side: "radiant" | "dire";
  showShare: boolean;
  unit?: "DPM" | "HPM";
}) {
  const rate = Math.round(stat.dpm);
  const pct = Math.max(0, Math.min(100, Math.round(stat.teamShare * 100)));
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
  side: "radiant" | "dire";
  unit: "DPM" | "HPM";
  showShare: boolean;
}) {
  const rate = Math.round(stat.dpm);
  const pct = Math.max(0, Math.min(100, Math.round(stat.teamShare * 100)));
  return (
    <div className="combat-meter">
      <div className="combat-meter__head">
        <span className="combat-meter__label">{label}</span>
        <span className="combat-meter__vals measured">
          <strong>{rate}</strong> {unit}
          {showShare ? (
            <span className="combat-meter__share-tag"> · {pct}% of team</span>
          ) : null}
        </span>
      </div>
      {showShare ? (
        <div className="share-bar share-bar--wide">
          <span
            className={`share-bar__fill share-bar__fill--${side}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      ) : (
        <p className="combat-meter__quiet">Minimal team heal</p>
      )}
    </div>
  );
}

export function BuildInspector({ prediction, vocab }: BuildInspectorProps) {
  const [teamFilter, setTeamFilter] = useState<TeamFilter>("all");
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("builds");
  const [selectedSlotIdx, setSelectedSlotIdx] = useState<number>(0);
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

  const inspectedSkillEntries = getLevelableSkillEntries(
    inspectedHero.hero,
    vocab.abilities.by_id,
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

  return (
    <section className="builds-section" aria-labelledby="builds-heading">
      <div className="builds-header">
        <div>
          <h2 id="builds-heading">Hero forecast</h2>
          <p className="muted-note">
            Item timings, skill priority 1 to 18, talents, farm, and team impact. Select any hero to inspect full details.
          </p>
        </div>

        <div className="builds-header__controls">
          <div
            className="segmented builds-view-segmented"
            role="group"
            aria-label="Comparison view"
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

      <div className="mobile-hero-switcher" role="tablist" aria-label="Select hero build">
        {visibleHeroes.map((hp) => {
          const isSelected = hp.slotIdx === inspectedHero.slotIdx;
          return (
            <button
              key={hp.slotIdx}
              type="button"
              role="tab"
              aria-selected={isSelected}
              className={`hero-switch-chip hero-switch-chip--${hp.side} ${isSelected ? "is-selected" : ""}`}
              onClick={() => setSelectedSlotIdx(hp.slotIdx)}
            >
              <span className="hero-switch-chip__thumb">
                <HeroPortraitImg
                  src={hp.hero.img}
                  alt={hp.hero.display_name}
                  fallbackText={hp.hero.display_name.slice(0, 2)}
                  width={36}
                  height={20}
                  decorative
                />
              </span>
              <span className="hero-switch-chip__name">{hp.hero.display_name}</span>
            </button>
          );
        })}
      </div>

      <div className="builds-table-wrap">
        {inspectorTab === "builds" ? (
          <table className="builds-table">
            <thead>
              <tr>
                <th scope="col">Hero</th>
                <th scope="col">Key items</th>
                <th scope="col">Skill build (1 to 18)</th>
                <th scope="col">Talents</th>
              </tr>
            </thead>
            <tbody>
              {visibleHeroes.map((hp, rowIdx) => {
                const isSelected = hp.slotIdx === inspectedHero.slotIdx;
                const sideClass =
                  hp.side === "radiant" ? "row-side--radiant" : "row-side--dire";
                const isTeamDivider =
                  teamFilter === "all" &&
                  rowIdx > 0 &&
                  visibleHeroes[rowIdx - 1]?.side === "radiant" &&
                  hp.side === "dire";

                return (
                  <tr
                    key={hp.slotIdx}
                    className={`${sideClass} ${isSelected ? "is-selected-row" : ""} ${isTeamDivider ? "row-team-divider" : ""}`}
                  >
                    <th scope="row" className="hero-cell">
                      <button
                        type="button"
                        className="hero-row-trigger"
                        aria-pressed={isSelected}
                        onClick={() => setSelectedSlotIdx(hp.slotIdx)}
                      >
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
                          <span className="hero-cell__role">
                            {ROLE_NAMES[hp.position].full}
                          </span>
                        </span>
                      </button>
                    </th>

                    <td className="items-cell">
                      {hp.items.length === 0 ? (
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
                                <span className="d2pt-item__min measured">
                                  {Math.round(item.expectedMinute)}m
                                </span>
                              </span>
                            </li>
                          ))}
                        </ol>
                      )}
                    </td>

                    <td className="skills-cell">
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
                          const isCheckpoint = ULT_LEVEL_CHECKPOINTS.has(step.level);
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
                                        ? `Slot ${step.shortTag} · Ultimate`
                                        : `Slot ${step.shortTag}`
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
                                  </span>
                                ) : (
                                  <span className="skill-cell__badge">{step.shortTag}</span>
                                )}
                              </span>
                            </li>
                          );
                        })}
                      </ol>
                    </td>

                    <td className="talents-cell">
                      <ul className="talent-list">
                        {hp.talents.map((tal) => (
                          <li key={tal.tier} className="talent-row">
                            <span className="talent-row__tier measured">{tal.tier}</span>
                            <span className="talent-row__text" title={tal.displayName}>
                              {tal.displayName}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </td>
                  </tr>
                );
              })}
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
              </tr>
            </thead>
            <tbody>
              {visibleHeroes.map((hp, rowIdx) => {
                const isSelected = hp.slotIdx === inspectedHero.slotIdx;
                const sideClass =
                  hp.side === "radiant" ? "row-side--radiant" : "row-side--dire";
                const isTeamDivider =
                  teamFilter === "all" &&
                  rowIdx > 0 &&
                  visibleHeroes[rowIdx - 1]?.side === "radiant" &&
                  hp.side === "dire";

                return (
                  <tr
                    key={hp.slotIdx}
                    className={`${sideClass} ${isSelected ? "is-selected-row" : ""} ${isTeamDivider ? "row-team-divider" : ""}`}
                  >
                    <th scope="row" className="hero-cell">
                      <button
                        type="button"
                        className="hero-row-trigger"
                        aria-pressed={isSelected}
                        onClick={() => setSelectedSlotIdx(hp.slotIdx)}
                      >
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
                          <span className="hero-cell__role">
                            {ROLE_NAMES[hp.position].full}
                          </span>
                        </span>
                      </button>
                    </th>

                    <td className="metric-cell measured">
                      <span className="metric-cell__main">{formatRangeText(hp.economy.gpm)}</span>
                    </td>
                    <td className="metric-cell measured">
                      <span className="metric-cell__main">{formatRangeText(hp.economy.xpm)}</span>
                    </td>
                    <td className="metric-cell measured">
                      <span className="metric-cell__main">{formatRangeText(hp.economy.csTotal)}</span>
                      <span className="metric-cell__sub">
                        {formatRangeText(hp.economy.csPerMin, 1)}/min
                      </span>
                    </td>
                    <td className="metric-cell measured">
                      <span className="metric-cell__main">
                        {Math.round(hp.economy.kills.mid)} / {Math.round(hp.economy.deaths.mid)} /{" "}
                        {Math.round(hp.economy.assists.mid)}
                      </span>
                      <span className="metric-cell__sub">
                        {formatRangeText(hp.economy.kills)} · {formatRangeText(hp.economy.deaths)} ·{" "}
                        {formatRangeText(hp.economy.assists)}
                      </span>
                    </td>
                    <td className="combat-cell measured">
                      <CombatCellContent stat={hp.combat.heroDamage} side={hp.side} showShare />
                    </td>
                    <td className="combat-cell measured">
                      <CombatCellContent stat={hp.combat.towerDamage} side={hp.side} showShare />
                    </td>
                    <td className="combat-cell measured">
                      <CombatCellContent stat={hp.combat.damageTaken} side={hp.side} showShare />
                    </td>
                    <td className="combat-cell measured">
                      <CombatCellContent
                        stat={hp.combat.healing}
                        side={hp.side}
                        showShare={hp.combat.showHealingShare}
                        unit="HPM"
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="hero-detail-panel" aria-live="polite">
        <div className="hero-detail-panel__head">
          <div className="hero-detail-panel__identity">
            <span className="hero-detail-panel__portrait">
              <HeroPortraitImg
                src={inspectedHero.hero.img}
                alt={inspectedHero.hero.display_name}
                fallbackText={inspectedHero.hero.display_name.slice(0, 2)}
                width={80}
                height={45}
                decorative
              />
            </span>
            <div>
              <h3>{inspectedHero.hero.display_name}</h3>
              <p className="muted-note">
                {inspectedHero.side === "radiant" ? "Radiant" : "Dire"} ·{" "}
                {ROLE_NAMES[inspectedHero.position].full}
              </p>
            </div>
          </div>
        </div>

        <div className="hero-detail-panel__grid">
          <div className="detail-block detail-block--items">
            <h4>Key items</h4>
            {inspectedHero.items.length === 0 ? (
              <p className="muted-note">No key items predicted for this slot.</p>
            ) : (
              <ol className="detail-items-list">
                {inspectedHero.items.map((item, idx) => (
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
                        <span>{item.expectedMinute.toFixed(1)} min</span>
                      </span>
                    </span>
                  </li>
                ))}
              </ol>
            )}
          </div>

          <div className="detail-block detail-block--talents">
            <h4>Talents</h4>
            <div className="talent-comparison">
              {inspectedHero.talents.map((tal) => (
                <div key={tal.tier} className="talent-tier-row">
                  <span className="talent-tier-row__lvl measured">Lvl {tal.tier}</span>
                  <div className="talent-tier-row__opts">
                    {tal.options.length === 0 ? (
                      <span className="muted-note">{tal.displayName}</span>
                    ) : (
                      tal.options.map((opt) => (
                        <span
                          key={opt.id}
                          className={`talent-opt ${opt.chosen ? "is-chosen" : ""}`}
                        >
                          {opt.displayName}
                        </span>
                      ))
                    )}
                  </div>
                </div>
              ))}
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
                  {formatRangeText(inspectedHero.economy.gpm)}
                </span>
              </div>
              <div className="econ-stat">
                <span className="econ-stat__label">XPM</span>
                <span className="econ-stat__val measured">
                  {formatRangeText(inspectedHero.economy.xpm)}
                </span>
              </div>
              <div className="econ-stat">
                <span className="econ-stat__label">CS</span>
                <span className="econ-stat__val measured">
                  {formatRangeText(inspectedHero.economy.csTotal)}
                </span>
                <span className="econ-stat__sub measured">
                  {formatRangeText(inspectedHero.economy.csPerMin, 1)}/min
                </span>
              </div>
              <div className="econ-stat">
                <span className="econ-stat__label">Kills</span>
                <span className="econ-stat__val measured">
                  {formatRangeText(inspectedHero.economy.kills)}
                </span>
              </div>
              <div className="econ-stat">
                <span className="econ-stat__label">Deaths</span>
                <span className="econ-stat__val measured">
                  {formatRangeText(inspectedHero.economy.deaths)}
                </span>
              </div>
              <div className="econ-stat">
                <span className="econ-stat__label">Assists</span>
                <span className="econ-stat__val measured">
                  {formatRangeText(inspectedHero.economy.assists)}
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
                stat={inspectedHero.combat.heroDamage}
                side={inspectedHero.side}
                unit="DPM"
                showShare
              />
              <CombatMeterRow
                label="Tower damage"
                stat={inspectedHero.combat.towerDamage}
                side={inspectedHero.side}
                unit="DPM"
                showShare
              />
              <CombatMeterRow
                label="Damage taken"
                stat={inspectedHero.combat.damageTaken}
                side={inspectedHero.side}
                unit="DPM"
                showShare
              />
              <CombatMeterRow
                label="Healing"
                stat={inspectedHero.combat.healing}
                side={inspectedHero.side}
                unit="HPM"
                showShare={inspectedHero.combat.showHealingShare}
              />
            </div>
          </div>

          <div className="detail-block detail-block--matrix">
            <h4>Skill matrix (levels 1 to 18)</h4>
            <div className="ability-matrix-wrap">
              <table className="ability-matrix">
                <thead>
                  <tr>
                    <th scope="col">Ability</th>
                    {inspectedHero.skills.map((st) => {
                      const isUltCol = ULT_LEVEL_CHECKPOINTS.has(st.level);
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
                  {inspectedSkillEntries.map((entry) => {
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
                        {inspectedHero.skills.map((st) => {
                          const hit =
                            st.kind === "ability" && st.slotClass === entry.origSlotIdx;
                          const isUltCol = ULT_LEVEL_CHECKPOINTS.has(st.level);
                          return (
                            <td
                              key={st.level}
                              className={`${hit ? "is-picked" : ""} ${isUltCol ? "is-ult-col" : ""}`}
                              aria-label={hit ? `${abName} at level ${st.level}` : undefined}
                            >
                              {hit ? (
                                <>
                                  <span
                                    className={`matrix-mark matrix-mark--slot-${colorKey}`}
                                    aria-hidden="true"
                                  />
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
                    {inspectedHero.skills.map((st) => {
                      const hit = st.kind === "talent";
                      const isUltCol = ULT_LEVEL_CHECKPOINTS.has(st.level);
                      return (
                        <td
                          key={st.level}
                          className={`${hit ? "is-picked-talent" : ""} ${isUltCol ? "is-ult-col" : ""}`}
                        >
                          {hit ? (
                            <>
                              <span
                                className="matrix-mark matrix-mark--talent"
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
                  <tr>
                    <th scope="row" className="ability-matrix__name">
                      <span className="ability-matrix__label-cell">
                        <span className="ability-matrix__key ability-matrix__key--attr measured">
                          +
                        </span>
                        <span className="ability-matrix__title">Stats</span>
                      </span>
                    </th>
                    {inspectedHero.skills.map((st) => {
                      const hit = st.kind === "attribute";
                      const isUltCol = ULT_LEVEL_CHECKPOINTS.has(st.level);
                      return (
                        <td
                          key={st.level}
                          className={`${hit ? "is-picked-attr" : ""} ${isUltCol ? "is-ult-col" : ""}`}
                        >
                          {hit ? (
                            <>
                              <span
                                className="matrix-mark matrix-mark--attr"
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
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {renderFloatingTooltip()}
    </section>
  );
}
