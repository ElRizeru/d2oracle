import { useEffect, useMemo, useRef, useState } from "react";
import { Search, X } from "lucide-react";

import type { DraftSlot, HeroMeta } from "../types/oracle";
import { HeroPortraitImg } from "./HeroPortraitImg";

interface HeroPickerModalProps {
  isOpen: boolean;
  targetSlot: DraftSlot | null;
  heroes: readonly HeroMeta[];
  allSlots: readonly DraftSlot[];
  onSelectHero: (heroId: number) => void;
  onClose: () => void;
}

const ATTR_FILTERS: ReadonlyArray<{ key: string; label: string; dotClass?: string }> = [
  { key: "all_filter", label: "All attributes" },
  { key: "str", label: "Strength", dotClass: "attr-dot--str" },
  { key: "agi", label: "Agility", dotClass: "attr-dot--agi" },
  { key: "int", label: "Intelligence", dotClass: "attr-dot--int" },
  { key: "all", label: "Universal", dotClass: "attr-dot--all" },
];

const ATTR_GROUPS: ReadonlyArray<{
  attr: "str" | "agi" | "int" | "all";
  label: string;
}> = [
  { attr: "str", label: "Strength" },
  { attr: "agi", label: "Agility" },
  { attr: "int", label: "Intelligence" },
  { attr: "all", label: "Universal" },
];

function formatAttrShort(attr: string | null): string {
  if (attr === "str") return "Str";
  if (attr === "agi") return "Agi";
  if (attr === "int") return "Int";
  if (attr === "all") return "Uni";
  return "Hero";
}

export function HeroPickerModal({
  isOpen,
  targetSlot,
  heroes,
  allSlots,
  onSelectHero,
  onClose,
}: HeroPickerModalProps) {
  const dialogRef = useRef<HTMLDialogElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const [query, setQuery] = useState("");
  const [attrFilter, setAttrFilter] = useState("all_filter");

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (isOpen && !dialog.open) {
      setQuery("");
      setAttrFilter("all_filter");
      dialog.showModal();
      requestAnimationFrame(() => {
        searchRef.current?.focus({ preventScroll: true });
      });
    } else if (!isOpen && dialog.open) {
      dialog.close();
    }
  }, [isOpen]);

  const takenHeroMap = useMemo(() => {
    const map = new Map<number, DraftSlot>();
    for (const s of allSlots) {
      if (s.heroId !== null && s.slotIdx !== targetSlot?.slotIdx) {
        map.set(s.heroId, s);
      }
    }
    return map;
  }, [allSlots, targetSlot]);

  const filteredHeroes = useMemo(() => {
    const q = query.trim().toLowerCase();
    return heroes.filter((h) => {
      if (attrFilter !== "all_filter" && h.primary_attr !== attrFilter) {
        return false;
      }
      if (!q) return true;
      return (
        h.display_name.toLowerCase().includes(q) ||
        h.short_name.toLowerCase().includes(q)
      );
    });
  }, [heroes, query, attrFilter]);

  const firstPickableHero = useMemo(
    () => filteredHeroes.find((h) => !takenHeroMap.has(h.id)) ?? null,
    [filteredHeroes, takenHeroMap],
  );

  const groupedHeroes = useMemo(() => {
    if (attrFilter !== "all_filter") {
      return [];
    }
    const buckets: Record<string, HeroMeta[]> = {
      str: [],
      agi: [],
      int: [],
      all: [],
    };
    for (const hero of filteredHeroes) {
      const key =
        hero.primary_attr === "str" ||
        hero.primary_attr === "agi" ||
        hero.primary_attr === "int" ||
        hero.primary_attr === "all"
          ? hero.primary_attr
          : "all";
      buckets[key]!.push(hero);
    }
    return ATTR_GROUPS.map((g) => ({
      ...g,
      heroes: buckets[g.attr]!,
    })).filter((g) => g.heroes.length > 0);
  }, [filteredHeroes, attrFilter]);

  const slotTitle = targetSlot
    ? `Select hero for ${targetSlot.side === "radiant" ? "Radiant" : "Dire"} Pos ${targetSlot.position}`
    : "Select hero";

  const renderHeroOption = (hero: HeroMeta) => {
    const owner = takenHeroMap.get(hero.id);
    const isCurrent = targetSlot?.heroId === hero.id;
    const isEnterTarget =
      query.trim().length > 0 && firstPickableHero?.id === hero.id;
    const ownerTag = owner
      ? `${owner.side === "radiant" ? "R" : "D"} P${owner.position}`
      : null;
    const ownerFullLabel = owner
      ? `${owner.side === "radiant" ? "Radiant" : "Dire"} Pos ${owner.position}`
      : null;

    return (
      <button
        key={hero.id}
        type="button"
        role="option"
        aria-selected={isCurrent}
        disabled={owner !== undefined}
        title={
          ownerFullLabel
            ? `${hero.display_name} is assigned to ${ownerFullLabel}`
            : hero.display_name
        }
        className={`hero-option ${isCurrent ? "is-current" : ""} ${isEnterTarget ? "is-enter-target" : ""} ${owner ? `is-taken-${owner.side}` : ""}`}
        onClick={() => onSelectHero(hero.id)}
      >
        <span className="hero-option__thumb">
          <HeroPortraitImg
            src={hero.img}
            alt={hero.display_name}
            fallbackText={hero.display_name.slice(0, 2)}
            width={64}
            height={36}
            decorative
          />
        </span>
        <span className="hero-option__meta">
          <span className="hero-option__name">{hero.display_name}</span>
          <span className="hero-option__sub">
            {ownerTag && owner ? (
              <span className={`hero-option__taken hero-option__taken--${owner.side} measured`}>
                {ownerTag}
              </span>
            ) : (
              <>
                <span
                  className={`attr-dot attr-dot--${hero.primary_attr ?? "all"}`}
                  aria-hidden="true"
                />
                <span>{formatAttrShort(hero.primary_attr)}</span>
              </>
            )}
          </span>
        </span>
      </button>
    );
  };

  return (
    <dialog
      ref={dialogRef}
      className="picker-dialog"
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
      <div className="picker-dialog__shell">
        <div className="picker-dialog__header">
          <div>
            <h3 className="picker-dialog__title">{slotTitle}</h3>
            <p className="muted-note">
              Type to filter, then press Enter to lock the top match.
            </p>
          </div>
          <button
            type="button"
            className="btn btn--icon"
            onClick={onClose}
            aria-label="Close hero picker"
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>

        <div className="picker-dialog__filters">
          <label htmlFor="hero-search-input" className="visually-hidden">
            Search heroes by name
          </label>
          <div className="search-wrap">
            <Search size={15} className="search-wrap__icon" aria-hidden="true" />
            <input
              ref={searchRef}
              id="hero-search-input"
              type="search"
              className="input search-wrap__input"
              placeholder="Search hero name…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && firstPickableHero) {
                  e.preventDefault();
                  onSelectHero(firstPickableHero.id);
                }
              }}
            />
            {query.trim().length > 0 && firstPickableHero ? (
              <span className="search-wrap__hint measured">
                Enter: {firstPickableHero.display_name}
              </span>
            ) : null}
          </div>

          <div className="attr-tabs" role="group" aria-label="Filter by primary attribute">
            {ATTR_FILTERS.map((f) => (
              <button
                key={f.key}
                type="button"
                className={`chip-btn ${attrFilter === f.key ? "is-selected" : ""}`}
                aria-pressed={attrFilter === f.key}
                onClick={() => setAttrFilter(f.key)}
              >
                {f.dotClass ? (
                  <span className={`attr-dot ${f.dotClass}`} aria-hidden="true" />
                ) : null}
                <span>{f.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="picker-body" role="listbox" aria-label="Available heroes">
          {filteredHeroes.length === 0 ? (
            <div className="empty-inline">
              <p>No heroes match your current filter.</p>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setQuery("");
                  setAttrFilter("all_filter");
                }}
              >
                Reset filters
              </button>
            </div>
          ) : attrFilter === "all_filter" ? (
            <div className="picker-groups">
              {groupedHeroes.map((group) => (
                <section
                  key={group.attr}
                  className="picker-group"
                  aria-label={`${group.label} heroes`}
                >
                  <header className="picker-group__head">
                    <span className={`attr-dot attr-dot--${group.attr}`} aria-hidden="true" />
                    <span className={`picker-group__title picker-group__title--${group.attr}`}>
                      {group.label}
                    </span>
                    <span className="measured picker-group__count">{group.heroes.length}</span>
                  </header>
                  <div className="picker-group__grid">
                    {group.heroes.map(renderHeroOption)}
                  </div>
                </section>
              ))}
            </div>
          ) : (
            <div className="picker-group__grid">
              {filteredHeroes.map(renderHeroOption)}
            </div>
          )}
        </div>
      </div>
    </dialog>
  );
}
