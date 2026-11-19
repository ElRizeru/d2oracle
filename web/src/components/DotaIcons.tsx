import { useId } from "react";
import type { LaneKey, RolePosition, TalentChoice, TeamSide } from "../types/oracle";

export function FactionEmblem({
  side,
  size = 20,
}: {
  side: TeamSide;
  size?: number;
}) {
  return (
    <img
      src={side === "radiant" ? "./assets/factions/radiant.svg" : "./assets/factions/dire.svg"}
      alt=""
      width={size}
      height={size}
      aria-hidden="true"
      className={`faction-emblem faction-emblem--${side}`}
    />
  );
}

export function RoleIcon({
  position,
  size = 16,
}: {
  position: RolePosition;
  size?: number;
}) {
  return (
    <img
      src={`./assets/roles/pos${position}.png`}
      alt=""
      width={size}
      height={size}
      aria-hidden="true"
      className={`role-img role-img--pos${position}`}
    />
  );
}

export function AttributeIcon({
  attr,
  size = 14,
}: {
  attr: string | null | undefined;
  size?: number;
}) {
  const key =
    attr === "str" || attr === "agi" || attr === "int" || attr === "all"
      ? attr
      : "all";
  return (
    <img
      src={`./assets/icons/attr_${key}.png`}
      alt=""
      width={size}
      height={size}
      aria-hidden="true"
      className="attr-icon-img"
    />
  );
}

export function GoldCoinIcon({ size = 14 }: { size?: number }) {
  return (
    <img
      src="./assets/icons/gold.png"
      alt=""
      width={size}
      height={size}
      aria-hidden="true"
      className="gold-icon-img"
    />
  );
}

export function LaneDirectionIcon({
  lane,
  size = 14,
}: {
  lane: LaneKey;
  size?: number;
}) {
  return (
    <img
      src={`./assets/lanes/${lane}.svg`}
      alt=""
      width={size}
      height={size}
      aria-hidden="true"
      className="lane-icon-img"
    />
  );
}

type TalentLeafKey = "25L" | "25R" | "20L" | "20R" | "15L" | "15R" | "10L" | "10R";

const TALENT_LEAF_POLYGONS: Record<TalentLeafKey, string> = {
  "25L": "20,7 36.5,7 36.5,24.5 20,24.5",
  "25R": "36.5,7 53,7 53,24.5 36.5,24.5",
  "20L": "10.5,20 36.5,25.5 36.5,37 10.5,36",
  "20R": "36.5,25.5 62.5,20 62.5,36 36.5,37",
  "15L": "6.5,36.5 36.5,38 36.5,50 6.5,48.5",
  "15R": "36.5,38 66.5,36.5 66.5,48.5 36.5,50",
  "10L": "10.5,52.5 36.5,53 36.5,64 10.5,60.5",
  "10R": "36.5,53 62.5,52.5 62.5,60.5 36.5,64",
};

const TRUNK_MIN_Y_BY_TIER: Record<10 | 15 | 20 | 25, number> = {
  25: 21.5,
  20: 34,
  15: 47,
  10: 60.5,
};

/**
 * Official Valve Dota 2 talent tree icon (`./assets/talents.svg`) rendered as a mask
 * with warm gold leaf-blade fill along the chosen left/right branches (`25 -> 10`).
 */
export function TalentTreeGlyph({
  talents,
  size = 36,
  highlightTier,
}: {
  talents: readonly TalentChoice[];
  size?: number;
  highlightTier?: 10 | 15 | 20 | 25;
}) {
  const reactId = useId();
  const gradId = `talent-gold-${reactId.replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const height = Math.round((size * 77) / 73);

  const activeChoices =
    highlightTier !== undefined
      ? talents.filter((t) => t.tier === highlightTier)
      : talents;

  const trunkTopY =
    highlightTier !== undefined
      ? TRUNK_MIN_Y_BY_TIER[highlightTier]
      : activeChoices.length > 0
        ? Math.min(...activeChoices.map((t) => TRUNK_MIN_Y_BY_TIER[t.tier]))
        : 77;

  return (
    <span
      className="talent-tree-badge"
      style={{ width: `${size}px`, height: `${height}px` }}
      aria-hidden="true"
    >
      <span
        className="talent-tree-badge__mask"
        style={{
          maskImage: 'url("./assets/talents.svg")',
          WebkitMaskImage: 'url("./assets/talents.svg")',
        }}
      >
        <svg
          width={size}
          height={height}
          viewBox="0 0 73 77"
          fill="none"
          className="talent-tree-badge__canvas"
        >
          <defs>
            <linearGradient
              id={gradId}
              x1="36.5"
              y1="9"
              x2="36.5"
              y2="72"
              gradientUnits="userSpaceOnUse"
            >
              <stop offset="0%" stopColor="#ffd778" />
              <stop offset="52%" stopColor="#f29b38" />
              <stop offset="100%" stopColor="#c75f18" />
            </linearGradient>
          </defs>
          {/* Base unchosen leaves */}
          <rect width="73" height="77" fill="#4f5561" />
          {/* Slightly softer outer circular frame inside Valve's SVG mask */}
          <circle
            cx="36.5"
            cy="38.5"
            r="35.6"
            fill="none"
            stroke="#363b45"
            strokeWidth="5.8"
          />
          {activeChoices.length > 0 ? (
            <g fill={`url(#${gradId})`}>
              <rect
                x="34.7"
                y={trunkTopY}
                width="3.6"
                height={Math.max(0, 74 - trunkTopY)}
              />
              {activeChoices.map((choice) => {
                const key = `${choice.tier}${choice.chosenSide === 0 ? "L" : "R"}` as TalentLeafKey;
                return <polygon key={choice.tier} points={TALENT_LEAF_POLYGONS[key]} />;
              })}
            </g>
          ) : null}
        </svg>
      </span>
    </span>
  );
}
