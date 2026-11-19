import { Clock } from "lucide-react";

import { ROLE_NAMES } from "../data/draftPresets";
import { formatLead10m, formatWinOddsPair } from "../ml/decoder";
import type {
  DraftPrediction,
  HeroPrediction,
  LaneKey,
  MatchPaceOption,
} from "../types/oracle";
import { FactionEmblem, GoldCoinIcon, LaneDirectionIcon } from "./DotaIcons";
import { HeroPortraitImg } from "./HeroPortraitImg";

interface MatchSummaryBarProps {
  prediction: DraftPrediction | null;
  loadingBundle: boolean;
  predicting: boolean;
  error: string | null;
  matchPace: MatchPaceOption;
  onSelectMatchPace: (pace: MatchPaceOption) => void;
}

const SCALE_MIN = 20;
const SCALE_MAX = 60;
const SCALE_TICKS = [20, 30, 40, 50, 60] as const;

const PACE_OPTIONS: ReadonlyArray<{ value: MatchPaceOption; label: string }> = [
  { value: 0, label: "Auto" },
  { value: 25, label: "Fast 25m" },
  { value: 35, label: "Mid 35m" },
  { value: 50, label: "Late 50m" },
];

const LANE_POSITIONS: Record<
  LaneKey,
  { radiant: readonly number[]; dire: readonly number[] }
> = {
  top: { radiant: [3, 4], dire: [1, 5] },
  mid: { radiant: [2], dire: [2] },
  bot: { radiant: [1, 5], dire: [3, 4] },
};

function toDurationPct(minutes: number): number {
  const clamped = Math.max(SCALE_MIN, Math.min(SCALE_MAX, minutes));
  return ((clamped - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * 100;
}

function getWinEdgeLabel(radLeadDiff: number): string {
  const absDiff = Math.abs(radLeadDiff);
  if (absDiff <= 1) {
    return "Even";
  }
  if (absDiff < 8) {
    return "Slight edge";
  }
  if (absDiff < 20) {
    return "Favored";
  }
  return "Strong edge";
}

function getLaneHeroes(
  heroes: readonly HeroPrediction[],
  side: "radiant" | "dire",
  positions: readonly number[],
): HeroPrediction[] {
  return positions
    .map((pos) => heroes.find((h) => h.side === side && h.position === pos))
    .filter((h): h is HeroPrediction => h !== undefined);
}

export function MatchSummaryBar({
  prediction,
  loadingBundle,
  predicting,
  error,
  matchPace,
  onSelectMatchPace,
}: MatchSummaryBarProps) {
  if (error) {
    return (
      <section className="summary-state summary-state--error" role="alert">
        <h2>Forecast unavailable</h2>
        <p>{error}</p>
      </section>
    );
  }

  if (loadingBundle) {
    return (
      <section
        className="summary-strip summary-strip--skeleton"
        aria-live="polite"
        aria-busy="true"
        aria-label="Loading active patch data"
      >
        <div className="summary-cell">
          <div className="summary-cell__top">
            <span className="skeleton-box skeleton-box--sm" />
            <span className="skeleton-box skeleton-box--xs" />
          </div>
          <div className="win-readout">
            <span className="skeleton-box skeleton-box--reading" />
            <span className="skeleton-box skeleton-box--reading" />
          </div>
          <div className="skeleton-box skeleton-box--bar" />
        </div>

        <div className="summary-cell">
          <div className="summary-cell__top">
            <span className="skeleton-box skeleton-box--sm" />
            <span className="skeleton-box skeleton-box--xs" />
          </div>
          <span className="skeleton-box skeleton-box--reading" />
          <div className="skeleton-box skeleton-box--bar" />
        </div>
      </section>
    );
  }

  if (!prediction) {
    return null;
  }

  const { radProb, direProb, radPctText, direPctText } = formatWinOddsPair(
    prediction.radiantWinProb,
  );
  const radLeadDiff = Number(radPctText) - Number(direPctText);
  const radiantLeads = radLeadDiff > 1;
  const direLeads = radLeadDiff < -1;
  const bannerTone = radiantLeads ? "radiant" : direLeads ? "dire" : "even";

  const edgeText = getWinEdgeLabel(radLeadDiff);

  const midMin = Math.max(10, Math.round(prediction.durationP50));
  const loMin = Math.max(10, Math.min(midMin, Math.round(prediction.durationP25)));
  const hiMin = Math.max(midMin, Math.round(prediction.durationP75));

  const p25Left = toDurationPct(loMin);
  const p75Left = toDurationPct(hiMin);
  const p50Left = toDurationPct(midMin);
  const windowWidth = Math.max(2, p75Left - p25Left);

  const goldLead = formatLead10m(prediction.goldLead10m);
  const xpLead = formatLead10m(prediction.xpLead10m);

  return (
    <section
      className={`match-banner match-banner--${bannerTone} ${predicting ? "is-updating" : ""}`}
      aria-labelledby="forecast-summary-heading"
    >
      <h2 id="forecast-summary-heading" className="visually-hidden">
        Match summary
      </h2>

      <div className="match-banner__hero">
        <div className="banner-team banner-team--radiant">
          <div className="banner-team__crest banner-team__crest--radiant">
            <FactionEmblem side="radiant" size={28} />
          </div>
          <div className="banner-team__info">
            <div className="banner-team__title-row">
              <span className="banner-team__name">Radiant</span>
              {radiantLeads ? (
                <span className="banner-pill banner-pill--radiant nowrap">{edgeText}</span>
              ) : !direLeads ? (
                <span className="banner-pill banner-pill--even nowrap">Even</span>
              ) : null}
            </div>
            <div className="banner-team__odds reading num">{radPctText}%</div>
          </div>
        </div>

        <div className="banner-center">
          <div className="banner-clock-box">
            <div className="banner-clock-box__top">
              <span className="banner-clock-box__label">
                <Clock size={12} aria-hidden="true" />
                <span>Expected duration</span>
              </span>
              {matchPace !== 0 ? (
                <span className="banner-pill banner-pill--pace measured nowrap">
                  Builds @ {matchPace}m
                </span>
              ) : null}
            </div>

            <div className="banner-clock-box__readout">
              <span className="banner-clock-box__mid reading num">~{midMin}m</span>
              <span className="banner-clock-box__window measured nowrap">
                {loMin}–{hiMin} min window
              </span>
            </div>

            <div
              className="duration-scale"
              role="img"
              aria-label={`Expected match time around ${midMin} minutes, typical window ${loMin} to ${hiMin} minutes`}
            >
              <div className="duration-scale__track">
                {SCALE_TICKS.slice(1, -1).map((tick) => (
                  <span
                    key={tick}
                    className="duration-scale__grid-tick"
                    style={{ left: `${toDurationPct(tick)}%` }}
                    aria-hidden="true"
                  />
                ))}
                <span
                  className="duration-scale__iqr"
                  style={{ left: `${p25Left}%`, width: `${windowWidth}%` }}
                />
                <span
                  className="duration-scale__median"
                  style={{ left: `${p50Left}%` }}
                />
              </div>
              <div className="duration-scale__labels measured" aria-hidden="true">
                {SCALE_TICKS.map((tick, idx) => (
                  <span
                    key={tick}
                    className={`duration-scale__tick-label ${idx === 0 ? "is-start" : idx === SCALE_TICKS.length - 1 ? "is-end" : ""}`}
                    style={{ left: `${toDurationPct(tick)}%` }}
                  >
                    {tick}m
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="banner-pace-row">
            <span className="banner-pace-row__label">Match pace</span>
            <div className="segmented match-pace-segmented" role="group" aria-label="Match pace">
              {PACE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  className={`segmented__btn ${matchPace === opt.value ? "is-active" : ""}`}
                  aria-pressed={matchPace === opt.value}
                  onClick={() => onSelectMatchPace(opt.value)}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="banner-team banner-team--dire">
          <div className="banner-team__info banner-team__info--dire">
            <div className="banner-team__title-row">
              {direLeads ? (
                <span className="banner-pill banner-pill--dire nowrap">{edgeText}</span>
              ) : !radiantLeads ? (
                <span className="banner-pill banner-pill--even nowrap">Even</span>
              ) : null}
              <span className="banner-team__name">Dire</span>
            </div>
            <div className="banner-team__odds reading num">{direPctText}%</div>
          </div>
          <div className="banner-team__crest banner-team__crest--dire">
            <FactionEmblem side="dire" size={28} />
          </div>
        </div>
      </div>

      <div
        className="banner-winbar"
        role="img"
        aria-label={`Radiant ${radPctText} percent versus Dire ${direPctText} percent`}
      >
        <span
          className="banner-winbar__fill banner-winbar__fill--radiant"
          style={{ transform: `scaleX(${radProb})` }}
        />
        <span
          className="banner-winbar__fill banner-winbar__fill--dire"
          style={{ transform: `scaleX(${direProb})` }}
        />
        <span className="banner-winbar__mid" aria-hidden="true" />
      </div>

      <div className="banner-laning">
        <div className="banner-laning__meta">
          <span className="banner-laning__title">Laning (0–10 min)</span>
          <div className="lead10m-group" aria-label="Estimated 10-minute team advantage">
            <span className={`lead10m-pill lead10m-pill--${goldLead.side} measured nowrap`}>
              <GoldCoinIcon size={13} />
              <span className="lead10m-pill__k">Gold @10m</span>
              <strong>{goldLead.label}</strong>
            </span>
            <span className={`lead10m-pill lead10m-pill--${xpLead.side} measured nowrap`}>
              <span className="lead10m-dot lead10m-dot--xp" aria-hidden="true" />
              <span className="lead10m-pill__k">XP @10m</span>
              <strong>{xpLead.label}</strong>
            </span>
          </div>
        </div>

        <div className="laning-grid">
          {prediction.lanes.map((lane) => {
            const posMap = LANE_POSITIONS[lane.lane];
            const radHeroes = getLaneHeroes(prediction.heroes, "radiant", posMap.radiant);
            const direHeroes = getLaneHeroes(prediction.heroes, "dire", posMap.dire);
            const matchupTitle = `${radHeroes.map((h) => `${h.hero.display_name} (${ROLE_NAMES[h.position].short})`).join(" + ")} vs ${direHeroes.map((h) => `${h.hero.display_name} (${ROLE_NAMES[h.position].short})`).join(" + ")}`;

            return (
              <div key={lane.lane} className={`lane-card lane-card--${lane.verdictSide}`}>
                <div className="lane-card__head">
                  <span className="lane-card__title-wrap">
                    <span className={`lane-card__dir lane-card__dir--${lane.verdictSide}`}>
                      <LaneDirectionIcon lane={lane.lane} size={14} />
                    </span>
                    <span className="lane-card__title">{lane.label}</span>
                  </span>
                  <span
                    className={`lane-card__verdict lane-card__verdict--${lane.verdictSide} nowrap`}
                  >
                    {lane.verdict}
                  </span>
                </div>

                {radHeroes.length > 0 && direHeroes.length > 0 ? (
                  <div className="lane-card__matchup" title={matchupTitle}>
                    <span className="lane-card__heroes lane-card__heroes--radiant">
                      {radHeroes.map((hp) => (
                        <span key={hp.slotIdx} className="lane-hero-chip">
                          <HeroPortraitImg
                            src={hp.hero.img}
                            alt={hp.hero.display_name}
                            fallbackText={hp.hero.display_name.slice(0, 2)}
                            width={30}
                            height={17}
                          />
                        </span>
                      ))}
                    </span>
                    <span className="lane-card__vs" aria-hidden="true">
                      vs
                    </span>
                    <span className="lane-card__heroes lane-card__heroes--dire">
                      {direHeroes.map((hp) => (
                        <span key={hp.slotIdx} className="lane-hero-chip">
                          <HeroPortraitImg
                            src={hp.hero.img}
                            alt={hp.hero.display_name}
                            fallbackText={hp.hero.display_name.slice(0, 2)}
                            width={30}
                            height={17}
                          />
                        </span>
                      ))}
                    </span>
                  </div>
                ) : null}

                <div
                  className="lane-bar"
                  role="img"
                  aria-label={`${lane.label} lane: Radiant ${lane.radiantPct} percent, Even ${lane.evenPct} percent, Dire ${lane.direPct} percent`}
                >
                  {lane.radiantPct > 0 ? (
                    <span
                      className="lane-bar__seg lane-bar__seg--radiant"
                      style={{ width: `${lane.radiantPct}%` }}
                    />
                  ) : null}
                  {lane.evenPct > 0 ? (
                    <span
                      className="lane-bar__seg lane-bar__seg--even"
                      style={{ width: `${lane.evenPct}%` }}
                    />
                  ) : null}
                  {lane.direPct > 0 ? (
                    <span
                      className="lane-bar__seg lane-bar__seg--dire"
                      style={{ width: `${lane.direPct}%` }}
                    />
                  ) : null}
                </div>

                <div className="lane-card__nums measured" aria-hidden="true">
                  <span
                    className={`lane-num lane-num--radiant ${lane.verdictSide === "radiant" ? "is-winner" : ""}`}
                  >
                    Rad {lane.radiantPct}%
                  </span>
                  <span className="lane-num lane-num--even">Even {lane.evenPct}%</span>
                  <span
                    className={`lane-num lane-num--dire ${lane.verdictSide === "dire" ? "is-winner" : ""}`}
                  >
                    Dire {lane.direPct}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
