import { useEffect, useMemo, useRef } from "react";
import { X } from "lucide-react";

import type { ModelConfigJson, VocabJson } from "../types/oracle";

interface ModelInfoModalProps {
  isOpen: boolean;
  vocab: VocabJson | null;
  modelConfig: ModelConfigJson | null;
  onClose: () => void;
}

export interface ModelInfoTile {
  label: string;
  value: string;
  sub?: string;
}

export interface ModelInfoViewData {
  patchName: string;
  overviewTiles: ModelInfoTile[];
  validationCaption: string;
  accuracyTiles: ModelInfoTile[];
}

function formatInt(value: number | null | undefined): string | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return null;
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(Math.round(value));
}

function formatPct(value: number | null | undefined): string | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return `${(value * 100).toFixed(1)}%`;
}

function formatUtcDate(iso: string | null | undefined): string | null {
  if (!iso) {
    return null;
  }
  const ts = Date.parse(iso);
  if (!Number.isFinite(ts)) {
    return null;
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(ts));
}

export function buildModelInfoViewData(
  vocab: VocabJson | null,
  modelConfig: ModelConfigJson | null,
): ModelInfoViewData {
  const summary = modelConfig?.model_summary;
  const patchName =
    modelConfig?.game_version_name || vocab?.game_version_name || "Active patch";

  const heroCount = summary?.num_heroes ?? vocab?.heroes.num_heroes ?? modelConfig?.num_heroes;
  const itemCount = summary?.num_key_items ?? vocab?.items.num_key_items;
  const totalMatches = formatInt(summary?.total_matches);
  const rankedMatches = formatInt(summary?.ranked_matches);
  const proMatches = formatInt(summary?.pro_matches);

  const matchPoolSub =
    rankedMatches && proMatches
      ? `${rankedMatches} Ranked · ${proMatches} Pro`
      : "Active patch matches";

  const heroPoolVal = heroCount != null ? `${heroCount} heroes` : "Active roster";
  const heroPoolSub =
    itemCount != null ? `${itemCount} tracked items` : "Full item catalog";

  const updatedVal =
    formatUtcDate(summary?.trained_at_utc) ?? formatUtcDate(vocab?.generated_at) ?? "Active cycle";
  const snapshotDate = formatUtcDate(summary?.dataset_snapshot_utc ?? vocab?.generated_at);
  const updatedSub = snapshotDate ? `Data through ${snapshotDate}` : "Active patch snapshot";

  const bundleVal =
    typeof summary?.bundle_size_mb === "number" && Number.isFinite(summary.bundle_size_mb)
      ? `${summary.bundle_size_mb.toFixed(1)} MB`
      : "Loaded";

  const overviewTiles: ModelInfoTile[] = [
    {
      label: "Matches Analyzed",
      value: totalMatches ?? "Active pool",
      sub: matchPoolSub,
    },
    {
      label: "Draft Coverage",
      value: heroPoolVal,
      sub: heroPoolSub,
    },
    {
      label: "Model Updated",
      value: updatedVal,
      sub: updatedSub,
    },
    {
      label: "Browser Footprint",
      value: bundleVal,
    },
  ];

  const accuracyTiles: ModelInfoTile[] = [];
  const winPct = formatPct(summary?.win_accuracy);
  if (winPct) {
    accuracyTiles.push({
      label: "Match Winner Pick",
      value: winPct,
    });
  }

  const skillPct = formatPct(summary?.skill_accuracy);
  if (skillPct) {
    accuracyTiles.push({
      label: "Skill Order (Lv 1–18)",
      value: skillPct,
    });
  }

  const talentPct = formatPct(summary?.talent_accuracy);
  if (talentPct) {
    accuracyTiles.push({
      label: "Talent Choices (10–25)",
      value: talentPct,
    });
  }

  const itemPct = formatPct(summary?.item_accuracy);
  if (itemPct) {
    const itemMargin =
      typeof summary?.item_timing_mae_min === "number" && Number.isFinite(summary.item_timing_mae_min)
        ? `±${summary.item_timing_mae_min.toFixed(1)} min timing margin`
        : "Core progression match";
    accuracyTiles.push({
      label: "Next Key Item",
      value: itemPct,
      sub: itemMargin,
    });
  }

  const durCov = formatPct(summary?.duration_window_coverage);
  if (durCov) {
    const durMargin =
      typeof summary?.duration_mae_min === "number" && Number.isFinite(summary.duration_mae_min)
        ? `Inside range · ±${summary.duration_mae_min.toFixed(1)} min mid`
        : "Inside expected match window";
    accuracyTiles.push({
      label: "Duration Window",
      value: durCov,
      sub: durMargin,
    });
  }

  if (typeof summary?.gpm_mae === "number" && Number.isFinite(summary.gpm_mae)) {
    accuracyTiles.push({
      label: "Economy Margin",
      value: `±${Math.round(summary.gpm_mae)} GPM`,
    });
  }

  const valCount = formatInt(summary?.val_matches);
  const validationCaption = valCount
    ? `Measured on ${valCount} holdout matches`
    : "Holdout match evaluation";

  return {
    patchName,
    overviewTiles,
    validationCaption,
    accuracyTiles,
  };
}

export function ModelInfoModal({
  isOpen,
  vocab,
  modelConfig,
  onClose,
}: ModelInfoModalProps) {
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const view = useMemo(
    () => buildModelInfoViewData(vocab, modelConfig),
    [vocab, modelConfig],
  );

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (isOpen && !dialog.open) {
      dialog.showModal();
    } else if (!isOpen && dialog.open) {
      dialog.close();
    }
  }, [isOpen]);

  return (
    <dialog
      ref={dialogRef}
      className="picker-dialog model-info-dialog"
      aria-labelledby="model-info-title"
      onClose={onClose}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === dialogRef.current) {
          onClose();
        }
      }}
    >
      <div className="model-info-dialog__shell">
        <div className="picker-dialog__header">
          <div className="model-info-dialog__title-wrap">
            <span className="model-info__kicker">Active Model</span>
            <h3 id="model-info-title" className="model-info-dialog__title">
              Patch {view.patchName}
            </h3>
          </div>
          <button
            type="button"
            className="btn btn--icon"
            onClick={onClose}
            aria-label="Close model info"
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>

        <div className="model-info-dialog__body">
          <section className="model-info-section" aria-label="Dataset and bundle overview">
            <div className="detail-block__head">
              <h4 className="model-info__section-title">Dataset &amp; Coverage</h4>
            </div>
            <div className="model-info-grid">
              {view.overviewTiles.map((tile) => (
                <div key={tile.label} className="econ-stat">
                  <span className="econ-stat__label">{tile.label}</span>
                  <span className="econ-stat__val measured">{tile.value}</span>
                  {tile.sub && <span className="econ-stat__sub">{tile.sub}</span>}
                </div>
              ))}
            </div>
          </section>

          {view.accuracyTiles.length > 0 && (
            <section className="model-info-section" aria-label="Validation accuracy">
              <div className="detail-block__head">
                <h4 className="model-info__section-title">Holdout Accuracy</h4>
                <span className="detail-block__sub">{view.validationCaption}</span>
              </div>
              <div className="model-info-metrics">
                {view.accuracyTiles.map((tile) => (
                  <div key={tile.label} className="econ-stat">
                    <span className="econ-stat__label">{tile.label}</span>
                    <span className="econ-stat__val measured">{tile.value}</span>
                    {tile.sub && <span className="econ-stat__sub">{tile.sub}</span>}
                  </div>
                ))}
              </div>
            </section>
          )}

          <div className="model-info-notes">
            <p className="model-info-note">
              Forecasts update automatically as you draft heroes, swap slots, or change{" "}
              <strong>Expected Pace</strong>.
            </p>
            <p className="model-info-note">
              Built from Ranked &amp; League Captains Mode matches on the active patch.
            </p>
          </div>
        </div>
      </div>
    </dialog>
  );
}
