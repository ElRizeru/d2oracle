import { useEffect, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronUp,
  GripVertical,
  Link2,
  Loader2,
  Plus,
  RotateCcw,
  Shuffle,
  X,
} from "lucide-react";

import { ROLE_NAMES } from "../data/draftPresets";
import type { DraftSlot, HeroMeta } from "../types/oracle";
import { AttributeIcon, FactionEmblem, RoleIcon } from "./DotaIcons";
import { HeroPortraitImg } from "./HeroPortraitImg";

export { ROLE_NAMES };

interface DraftWorkspaceProps {
  slots: readonly DraftSlot[];
  heroesById: Record<string, HeroMeta>;
  bundleReady: boolean;
  loadingBundle: boolean;
  predicting: boolean;
  filledCount: number;
  draftQuery: string;
  onOpenPicker: (slotIdx: number) => void;
  onDropSwap: (fromSlotIdx: number, toSlotIdx: number) => void;
  onClearSlot: (slotIdx: number) => void;
  onRandomDraft: () => void;
  onClearAll: () => void;
}

export function DraftWorkspace({
  slots,
  heroesById,
  bundleReady,
  loadingBundle,
  predicting,
  filledCount,
  draftQuery,
  onOpenPicker,
  onDropSwap,
  onClearSlot,
  onRandomDraft,
  onClearAll,
}: DraftWorkspaceProps) {
  const radiantSlots = slots.slice(0, 5);
  const direSlots = slots.slice(5, 10);

  const [dragSourceSlotIdx, setDragSourceSlotIdx] = useState<number | null>(null);
  const [dragOverSlotIdx, setDragOverSlotIdx] = useState<number | null>(null);
  const [copiedLink, setCopiedLink] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const copyTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (filledCount < 10 && collapsed) {
      setCollapsed(false);
    }
  }, [filledCount, collapsed]);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current !== null) {
        window.clearTimeout(copyTimerRef.current);
      }
    };
  }, []);

  const handleCopyDraftLink = async () => {
    if (typeof window === "undefined") return;
    const shareUrl = `${window.location.origin}${window.location.pathname}${draftQuery}`;
    let copied = false;
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(shareUrl);
        copied = true;
      } catch {
        copied = false;
      }
    }
    if (!copied) {
      try {
        const ta = document.createElement("textarea");
        ta.value = shareUrl;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        copied = document.execCommand("copy");
        document.body.removeChild(ta);
      } catch {
        copied = false;
      }
    }
    if (copied) {
      setCopiedLink(true);
      if (copyTimerRef.current !== null) {
        window.clearTimeout(copyTimerRef.current);
      }
      copyTimerRef.current = window.setTimeout(() => {
        setCopiedLink(false);
      }, 1800);
    }
  };

  const renderSlotRow = (slot: DraftSlot) => {
    const hero = slot.heroId !== null ? heroesById[String(slot.heroId)] : undefined;
    const isDragging = dragSourceSlotIdx === slot.slotIdx;
    const isDropTarget =
      dragOverSlotIdx === slot.slotIdx && dragSourceSlotIdx !== slot.slotIdx;
    const teamLabel = slot.side === "radiant" ? "Radiant" : "Dire";
    const role = ROLE_NAMES[slot.position];
    const canDrag = bundleReady && Boolean(hero);

    return (
      <div
        key={slot.slotIdx}
        data-slot-idx={slot.slotIdx}
        draggable={canDrag}
        onDragStart={(e) => {
          if (!canDrag) {
            e.preventDefault();
            return;
          }
          setDragSourceSlotIdx(slot.slotIdx);
          e.dataTransfer.effectAllowed = "move";
          e.dataTransfer.setData("text/plain", String(slot.slotIdx));
        }}
        onDragOver={(e) => {
          if (!bundleReady) return;
          e.preventDefault();
          e.dataTransfer.dropEffect = "move";
          if (dragOverSlotIdx !== slot.slotIdx) {
            setDragOverSlotIdx(slot.slotIdx);
          }
        }}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
            if (dragOverSlotIdx === slot.slotIdx) {
              setDragOverSlotIdx(null);
            }
          }
        }}
        onDrop={(e) => {
          e.preventDefault();
          const raw = e.dataTransfer.getData("text/plain");
          const parsed = raw !== "" ? Number.parseInt(raw, 10) : dragSourceSlotIdx;
          setDragOverSlotIdx(null);
          setDragSourceSlotIdx(null);
          if (
            typeof parsed === "number" &&
            Number.isInteger(parsed) &&
            parsed >= 0 &&
            parsed < 10 &&
            parsed !== slot.slotIdx
          ) {
            onDropSwap(parsed, slot.slotIdx);
          }
        }}
        onDragEnd={() => {
          setDragSourceSlotIdx(null);
          setDragOverSlotIdx(null);
        }}
        className={`draft-slot draft-slot--${slot.side} ${hero ? "is-filled" : "is-empty"} ${isDragging ? "is-dragging" : ""} ${isDropTarget ? "is-drop-target" : ""}`}
      >
        <div className="draft-slot__pos">
          {canDrag ? (
            <GripVertical size={13} className="draft-slot__grip" aria-hidden="true" />
          ) : null}
          <span className="draft-slot__role-icon" aria-hidden="true">
            <RoleIcon position={slot.position} size={14} />
          </span>
          <span className="draft-slot__role">
            <span className="role-label--full">{role.full}</span>
            <span className="role-label--short" aria-hidden="true">
              {role.short}
            </span>
          </span>
        </div>

        <button
          type="button"
          className="draft-slot__select"
          disabled={!bundleReady}
          onClick={() => onOpenPicker(slot.slotIdx)}
          aria-label={
            loadingBundle
              ? `Loading ${teamLabel} ${role.full}`
              : hero
                ? `Change hero for ${teamLabel} ${role.full}: ${hero.display_name}`
                : `Pick hero for ${teamLabel} ${role.full}`
          }
        >
          {loadingBundle && !hero ? (
            <span className="draft-slot__skeleton" aria-hidden="true">
              <span className="skeleton-box skeleton-box--thumb" />
              <span className="skeleton-box skeleton-box--name" />
            </span>
          ) : hero ? (
            <>
              <span className="draft-slot__thumb">
                <HeroPortraitImg
                  src={hero.img}
                  alt={hero.display_name}
                  fallbackText={hero.display_name.slice(0, 2)}
                  width={54}
                  height={30}
                  decorative
                  priority={slot.position === 1}
                />
              </span>
              <span className="draft-slot__hero-info">
                <span className="draft-slot__hero-name">{hero.display_name}</span>
              </span>
              <span className="draft-slot__attr">
                <AttributeIcon attr={hero.primary_attr} size={15} />
              </span>
            </>
          ) : (
            <>
              <span className="draft-slot__thumb draft-slot__thumb--empty" aria-hidden="true">
                <Plus size={14} />
              </span>
              <span className="draft-slot__placeholder">Pick hero</span>
            </>
          )}
        </button>

        {hero ? (
          <div className="draft-slot__actions">
            <button
              type="button"
              className="btn btn--icon"
              aria-label={`Clear ${teamLabel} ${role.full}`}
              title="Clear slot"
              disabled={!bundleReady}
              onClick={() => onClearSlot(slot.slotIdx)}
            >
              <X size={14} aria-hidden="true" />
            </button>
          </div>
        ) : null}
      </div>
    );
  };

  const renderCollapsedStrip = () => (
    <div className="draft-mini-bar">
      <div className="draft-mini-team draft-mini-team--radiant">
        <span className="draft-mini-team__label">
          <FactionEmblem side="radiant" size={16} />
          <span>Radiant</span>
        </span>
        <div className="draft-mini-team__portraits">
          {radiantSlots.map((slot) => {
            const hero = slot.heroId !== null ? heroesById[String(slot.heroId)] : undefined;
            return (
              <button
                key={slot.slotIdx}
                type="button"
                className="draft-mini-chip draft-mini-chip--radiant"
                onClick={() => onOpenPicker(slot.slotIdx)}
                title={hero ? `${hero.display_name} (${ROLE_NAMES[slot.position].full})` : "Pick hero"}
                aria-label={hero ? `Change Radiant ${ROLE_NAMES[slot.position].full}: ${hero.display_name}` : "Pick hero"}
              >
                <span className="draft-mini-chip__thumb">
                  <HeroPortraitImg
                    src={hero?.img}
                    alt={hero?.display_name ?? ""}
                    fallbackText={hero?.display_name.slice(0, 2) ?? "?"}
                    width={68}
                    height={32}
                    decorative
                  />
                </span>
                <span className="draft-mini-chip__role">
                  <RoleIcon position={slot.position} size={12} />
                  <span className="draft-mini-chip__role-label">{ROLE_NAMES[slot.position].short}</span>
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <span className="draft-mini-bar__vs" aria-hidden="true">
        vs
      </span>

      <div className="draft-mini-team draft-mini-team--dire">
        <div className="draft-mini-team__portraits">
          {direSlots.map((slot) => {
            const hero = slot.heroId !== null ? heroesById[String(slot.heroId)] : undefined;
            return (
              <button
                key={slot.slotIdx}
                type="button"
                className="draft-mini-chip draft-mini-chip--dire"
                onClick={() => onOpenPicker(slot.slotIdx)}
                title={hero ? `${hero.display_name} (${ROLE_NAMES[slot.position].full})` : "Pick hero"}
                aria-label={hero ? `Change Dire ${ROLE_NAMES[slot.position].full}: ${hero.display_name}` : "Pick hero"}
              >
                <span className="draft-mini-chip__thumb">
                  <HeroPortraitImg
                    src={hero?.img}
                    alt={hero?.display_name ?? ""}
                    fallbackText={hero?.display_name.slice(0, 2) ?? "?"}
                    width={68}
                    height={32}
                    decorative
                  />
                </span>
                <span className="draft-mini-chip__role">
                  <RoleIcon position={slot.position} size={12} />
                  <span className="draft-mini-chip__role-label">{ROLE_NAMES[slot.position].short}</span>
                </span>
              </button>
            );
          })}
        </div>
        <span className="draft-mini-team__label">
          <span>Dire</span>
          <FactionEmblem side="dire" size={16} />
        </span>
      </div>
    </div>
  );

  return (
    <section className="draft-panel" aria-labelledby="draft-heading">
      <div className="draft-toolbar">
        <div className="draft-toolbar__title-group">
          <div className="draft-toolbar__heading-row">
            <h1 id="draft-heading">Draft</h1>
            <span
              className={`draft-count-pill measured ${filledCount === 10 ? "is-complete" : ""}`}
            >
              {filledCount}/10
            </span>
            {predicting ? (
              <span className="draft-predicting" role="status" aria-live="polite">
                <Loader2 size={13} className="spin-icon" aria-hidden="true" />
                <span>Updating…</span>
              </span>
            ) : null}
          </div>
          {!collapsed ? (
            <p className="draft-toolbar__sub">
              Pick or drag 10 unique heroes across roles.
            </p>
          ) : null}
        </div>

        <div className="draft-toolbar__controls">
          <button
            type="button"
            className={`btn ${filledCount === 0 ? "btn--tinted" : ""}`}
            disabled={!bundleReady || predicting}
            onClick={onRandomDraft}
          >
            <Shuffle size={14} aria-hidden="true" />
            <span>Random draft</span>
          </button>

          <button
            type="button"
            className={`btn ${copiedLink ? "is-copied" : ""}`}
            disabled={!bundleReady || filledCount === 0}
            onClick={() => void handleCopyDraftLink()}
          >
            {copiedLink ? (
              <Check size={14} aria-hidden="true" />
            ) : (
              <Link2 size={14} aria-hidden="true" />
            )}
            <span>{copiedLink ? "Copied!" : "Copy draft link"}</span>
          </button>

          <button
            type="button"
            className="btn btn--quiet"
            disabled={!bundleReady || filledCount === 0 || predicting}
            onClick={onClearAll}
          >
            <RotateCcw size={14} aria-hidden="true" />
            <span>Reset</span>
          </button>

          {filledCount === 10 ? (
            <button
              type="button"
              className="btn btn--quiet draft-collapse-btn"
              aria-expanded={!collapsed}
              onClick={() => setCollapsed((prev) => !prev)}
            >
              {collapsed ? (
                <>
                  <ChevronDown size={14} aria-hidden="true" />
                  <span>Expand</span>
                </>
              ) : (
                <>
                  <ChevronUp size={14} aria-hidden="true" />
                  <span>Compact</span>
                </>
              )}
            </button>
          ) : null}
        </div>
      </div>

      {collapsed && filledCount === 10 ? (
        renderCollapsedStrip()
      ) : (
        <div className="teams-grid">
          <div className="team-column" aria-label="Radiant roster">
            <div className="team-column__head team-column__head--radiant">
              <div className="team-column__brand">
                <FactionEmblem side="radiant" size={18} />
                <h2 className="team-column__title">Radiant</h2>
              </div>
              <span className="measured team-column__tag">
                {radiantSlots.filter((s) => s.heroId !== null).length}/5
              </span>
            </div>
            <div className="team-column__rows">{radiantSlots.map(renderSlotRow)}</div>
          </div>

          <div className="team-column" aria-label="Dire roster">
            <div className="team-column__head team-column__head--dire">
              <div className="team-column__brand">
                <FactionEmblem side="dire" size={18} />
                <h2 className="team-column__title">Dire</h2>
              </div>
              <span className="measured team-column__tag">
                {direSlots.filter((s) => s.heroId !== null).length}/5
              </span>
            </div>
            <div className="team-column__rows">{direSlots.map(renderSlotRow)}</div>
          </div>
        </div>
      )}
    </section>
  );
}
