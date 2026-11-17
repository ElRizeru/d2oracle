import { formatLead10m, formatWinOddsPair } from "../ml/decoder";
import type { DraftPrediction, MatchPaceOption } from "../types/oracle";

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

function toDurationPct(minutes: number): number {
  const clamped = Math.max(SCALE_MIN, Math.min(SCALE_MAX, minutes));
  return ((clamped - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * 100;
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
  const absDiff = Math.abs(radLeadDiff);
  const radiantLeads = radLeadDiff > 0.5;
  const direLeads = radLeadDiff < -0.5;

  const favoredBadgeText =
    absDiff <= 0.5
      ? "Even matchup"
      : `${radiantLeads ? "Radiant" : "Dire"} +${absDiff.toFixed(1)}%`;

  const favoredBadgeClass = radiantLeads
    ? "summary-badge--radiant"
    : direLeads
      ? "summary-badge--dire"
      : "";

  const midMin = Math.max(10, Math.round(prediction.durationP50));
  const loMin = Math.max(10, Math.min(midMin, Math.round(prediction.durationP25)));
  const hiMin = Math.max(midMin, Math.round(prediction.durationP75));

  const p25Left = toDurationPct(loMin);
  const p75Left = toDurationPct(hiMin);
  const p50Left = toDurationPct(midMin);
  const windowWidth = Math.max(2, p75Left - p25Left);

  const goldLead = formatLead10m(prediction.goldLead10m);
  const xpLead = formatLead10m(prediction.xpLead10m);
  const effectiveMin = Math.round(prediction.effectiveDurationMinutes);

  return (
    <section
      className={`summary-strip ${predicting ? "is-updating" : ""}`}
      aria-labelledby="forecast-summary-heading"
    >
      <h2 id="forecast-summary-heading" className="visually-hidden">
        Match summary
      </h2>

      <div className="summary-cell summary-cell--win">
        <div className="summary-cell__top">
          <span>Win probability</span>
          <span className={`summary-badge ${favoredBadgeClass} measured nowrap`}>
            {favoredBadgeText}
          </span>
        </div>

        <div className="win-readout">
          <div className="win-side win-side--radiant">
            <span className="win-side__name">Radiant</span>
            <span className="reading num">{radPctText}%</span>
            {radiantLeads ? (
              <span className="win-side__delta win-side__delta--radiant measured">
                +{absDiff.toFixed(1)}%
              </span>
            ) : null}
          </div>
          <div className="win-side win-side--dire">
            {direLeads ? (
              <span className="win-side__delta win-side__delta--dire measured">
                +{absDiff.toFixed(1)}%
              </span>
            ) : null}
            <span className="win-side__name">Dire</span>
            <span className="reading num">{direPctText}%</span>
          </div>
        </div>

        <div
          className="win-bar-wrap"
          role="img"
          aria-label={`Radiant ${radPctText} percent versus Dire ${direPctText} percent`}
        >
          <div className="win-bar">
            <span
              className="win-bar__fill win-bar__fill--radiant"
              style={{ transform: `scaleX(${radProb})` }}
            />
            <span
              className="win-bar__fill win-bar__fill--dire"
              style={{ transform: `scaleX(${direProb})` }}
            />
            <span className="win-bar__mid" aria-hidden="true" />
          </div>
          <div className="win-bar__tick-row" aria-hidden="true">
            <span className="win-bar__tick-label measured">50%</span>
          </div>
        </div>
      </div>

      <div className="summary-cell summary-cell--duration">
        <div className="summary-cell__top">
          <span>Expected match time</span>
          <span className="summary-badge measured nowrap">
            {matchPace === 0 ? `Builds @ ~${effectiveMin}m` : `Builds @ ${matchPace}m`}
          </span>
        </div>

        <div className="duration-headline">
          <div className="duration-readout">
            <span className="reading num">~{midMin}</span>
            <span className="duration-unit nowrap">min</span>
          </div>
          <div className="duration-window measured nowrap">
            <span>
              {loMin}–{hiMin} min window
            </span>
          </div>
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
            {SCALE_TICKS.map((tick) => (
              <span key={tick}>{tick}m</span>
            ))}
          </div>
        </div>

        <div className="match-pace-row">
          <span className="match-pace-row__label">Match pace</span>
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

      <div className="summary-cell summary-cell--laning">
        <div className="summary-cell__top summary-cell__top--laning">
          <span>Laning (0–10 min)</span>
          <div className="lead10m-group" aria-label="Estimated 10-minute team advantage">
            <span className={`lead10m-pill lead10m-pill--${goldLead.side} measured nowrap`}>
              <span className="lead10m-pill__k">Gold @10m</span>
              <strong>{goldLead.label}</strong>
            </span>
            <span className={`lead10m-pill lead10m-pill--${xpLead.side} measured nowrap`}>
              <span className="lead10m-pill__k">XP @10m</span>
              <strong>{xpLead.label}</strong>
            </span>
          </div>
        </div>

        <div className="laning-grid">
          {prediction.lanes.map((lane) => (
            <div key={lane.lane} className="lane-card">
              <div className="lane-card__head">
                <span className="lane-card__title">{lane.label}</span>
                <span
                  className={`lane-card__verdict lane-card__verdict--${lane.verdictSide} measured nowrap`}
                >
                  {lane.verdict}
                </span>
              </div>

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
                <span className="lane-num lane-num--radiant">Rad {lane.radiantPct}%</span>
                <span className="lane-num lane-num--even">Even {lane.evenPct}%</span>
                <span className="lane-num lane-num--dire">Dire {lane.direPct}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
