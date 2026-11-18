"""Hugging Face dataset downloader, vocabulary indexer, and PyTorch Dataset."""

from __future__ import annotations

import gc
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi
from torch.utils.data import Dataset

from shared.constants import (
    ATTRIBUTE_BONUS_ABILITY_ID,
    LANE_OUTCOME_5_TO_ID,
    LANE_OUTCOME_TO_ID,
    MISSING_STAT_SENTINEL,
    PARQUET_ARROW_SCHEMA,
    PARQUET_SCHEMA_VERSION,
    VOCAB_SCHEMA_VERSION,
)
from shared.vocab import INVOKER_HERO_ID, MEEPO_HERO_ID
from training.config import (
    HEAL_SHARE_MIN_TEAM_SUM,
    MIN_MATCH_DURATION_MIN,
)

IGNORE_INDEX = -100
PAD_ITEM_ID = 0
BOS_ITEM_ID = 1
EOS_ITEM_ID = 2
TALENT_TIERS: tuple[str, ...] = ("10", "15", "20", "25")
STAGE1_BATCH_KEYS: tuple[str, ...] = (
    "hero_idx",
    "is_radiant",
    "pos_idx",
    "radiant_win",
    "duration_min",
    "lane_targets",
    "lane5_targets",
    "lead10_targets",
    "lead10_mask",
    "sample_weight",
    "is_pro_meta",
)

REQUIRED_PARQUET_COLUMNS: tuple[str, ...] = tuple(PARQUET_ARROW_SCHEMA.names)
TRAINING_PARQUET_COLUMNS: tuple[str, ...] = (
    "match_id",
    "start_datetime",
    "duration_seconds",
    "radiant_win",
    "is_pro",
    "top_lane_outcome",
    "mid_lane_outcome",
    "bottom_lane_outcome",
    "radiant_gold_lead_10m",
    "radiant_xp_lead_10m",
    "is_radiant",
    "hero_ids",
    "positions",
    "kills",
    "deaths",
    "assists",
    "last_hits",
    "gpm",
    "xpm",
    "hero_damage",
    "tower_damage",
    "hero_healing",
    "hero_damage_received",
    "key_item_ids",
    "key_item_times",
    "ability_learn_ids",
    "ability_learn_levels",
    "ability_learn_is_talent",
)


def download_hf_dataset(
    repo_id: str,
    cache_dir: Path,
    *,
    token: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[Path]]:
    """Download latest.json, active vocab.json, and all active-patch Parquet shards from HF."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)

    latest_path = api.hf_hub_download(
        repo_id=repo_id,
        filename="latest.json",
        repo_type="dataset",
        local_dir=str(cache_dir),
        force_download=True,
    )
    manifest = json.loads(Path(latest_path).read_text(encoding="utf-8"))

    parquet_ver = manifest.get("parquet_schema_version")
    if parquet_ver is None:
        parquet_ver = manifest.get("schema_version")
    if parquet_ver != PARQUET_SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible HF manifest parquet_schema_version={parquet_ver!r}; "
            f"expected {PARQUET_SCHEMA_VERSION}."
        )

    vocab_rel = manifest["vocab_path"]
    vocab_local = api.hf_hub_download(
        repo_id=repo_id,
        filename=vocab_rel,
        repo_type="dataset",
        local_dir=str(cache_dir),
        force_download=True,
    )
    vocab = json.loads(Path(vocab_local).read_text(encoding="utf-8"))
    vocab_ver = vocab.get("schema_version")
    if vocab_ver != VOCAB_SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible vocab schema_version={vocab_ver!r}; "
            f"expected {VOCAB_SCHEMA_VERSION}."
        )

    shard_paths: list[Path] = []
    for shard in manifest.get("shards") or []:
        rel = shard["path"]
        local_shard = api.hf_hub_download(
            repo_id=repo_id,
            filename=rel,
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
        shard_paths.append(Path(local_shard))

    return manifest, vocab, shard_paths


class VocabIndex:
    """Fast lookup tables and legal skill/talent masks derived from vocab.json."""

    def __init__(self, vocab: dict[str, Any]) -> None:
        vocab_ver = vocab.get("schema_version")
        if vocab_ver != VOCAB_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible vocab schema_version={vocab_ver!r}; "
                f"expected {VOCAB_SCHEMA_VERSION}."
            )
        self.vocab = vocab
        self.gv_id = int(vocab["game_version_id"])
        self.gv_name = str(vocab["game_version_name"])

        heroes_sec = vocab["heroes"]
        self.num_heroes = int(heroes_sec["num_heroes"])
        self.hero_id_to_idx: dict[int, int] = {
            int(k): int(v) for k, v in heroes_sec["id_to_idx"].items()
        }

        items_sec = vocab["items"]
        self.item_vocab_size = int(items_sec["vocab_size"])
        self.item_id_to_idx: dict[int, int] = {
            int(k): int(v) for k, v in items_sec["id_to_idx"].items()
        }
        self.item_aliases: dict[int, int] = {
            int(k): int(v) for k, v in (items_sec.get("aliases") or {}).items()
        }

        abilities_by_id = vocab["abilities"]["by_id"]
        self.max_hero_skills = max(
            (len(h.get("skill_ability_ids") or []) for h in heroes_sec["by_id"].values()),
            default=4,
        )
        self.slot_talent = self.max_hero_skills
        self.slot_attr = self.max_hero_skills + 1
        self.num_skill_classes = self.max_hero_skills + 2

        self.hero_skill_id_to_slot: dict[int, dict[int, int]] = {}
        self.hero_talent_id_to_choice: dict[int, dict[int, tuple[int, int]]] = {}
        self._hero_num_skills_by_idx: dict[int, int] = {}

        from shared.vocab import (
            INVOKER_HERO_ID,
            LINKED_ABILITY_NAME_ALIASES,
            MISSING_LINKED_ABILITY_ID_FALLBACKS,
            NON_LEVELABLE_ABILITY_NAMES,
        )

        abilities_sec = vocab.get("abilities") or {}
        abilities_by_id = abilities_sec.get("by_id") or {}
        raw_ab_aliases = abilities_sec.get("aliases") or {}

        ab_name_to_id: dict[str, int] = {}
        for aid_str, ameta in abilities_by_id.items():
            if isinstance(ameta, dict) and ameta.get("name"):
                ab_name_to_id[str(ameta["name"])] = int(aid_str)

        ab_alias_map: dict[int, int] = {
            int(src_k): int(dst_v) for src_k, dst_v in raw_ab_aliases.items()
        }
        for src_nm, dst_nm in LINKED_ABILITY_NAME_ALIASES.items():
            src_id = ab_name_to_id.get(src_nm)
            dst_id = ab_name_to_id.get(dst_nm)
            if src_id is not None and dst_id is not None and src_id not in ab_alias_map:
                ab_alias_map[src_id] = dst_id
        for src_id, (dst_id, dst_nm) in MISSING_LINKED_ABILITY_ID_FALLBACKS.items():
            if str(src_id) in abilities_by_id or src_id in ab_alias_map:
                continue
            dst_meta = abilities_by_id.get(str(dst_id))
            if dst_meta is not None and (dst_meta.get("name") or dst_nm) == dst_nm:
                ab_alias_map[src_id] = dst_id

        self.hero_levelable_slots: dict[int, list[int]] = {}
        self.hero_ult_slots: dict[int, set[int]] = {}
        self._hero_id_by_idx: dict[int, int] = {}

        for raw_hid, hmeta in heroes_sec["by_id"].items():
            hid = int(raw_hid)
            hidx = int(hmeta["idx"])
            skills = [int(a) for a in (hmeta.get("skill_ability_ids") or [])]
            self._hero_num_skills_by_idx[hidx] = len(skills)
            self._hero_id_by_idx[hidx] = hid

            lvl_slots: list[int] = []
            ult_slots: set[int] = set()

            for i, aid in enumerate(skills):
                ameta = abilities_by_id.get(str(aid)) or {}
                aname = str(ameta.get("name") or "")
                is_inn = bool(ameta.get("is_innate", False))
                is_lvl = bool(
                    ameta.get(
                        "is_levelable",
                        (not is_inn) and (aname not in NON_LEVELABLE_ABILITY_NAMES),
                    )
                )
                if is_lvl:
                    lvl_slots.append(i)
                    if bool(ameta.get("is_ultimate", False)):
                        ult_slots.add(i)

            if hid != INVOKER_HERO_ID and not ult_slots and len(lvl_slots) >= 4:
                ult_slots.add(lvl_slots[-1])

            # Restrict slot_map strictly to canonical levelable slots so unaliased
            # non-levelable abilities can never emit illegal masked skill targets.
            slot_map: dict[int, int] = {skills[i]: i for i in lvl_slots}

            for src_aid, dst_aid in ab_alias_map.items():
                if dst_aid in slot_map:
                    slot_map[src_aid] = slot_map[dst_aid]

            self.hero_skill_id_to_slot[hid] = slot_map
            self.hero_levelable_slots[hidx] = lvl_slots
            self.hero_ult_slots[hidx] = ult_slots

            tmap: dict[int, tuple[int, int]] = {}
            tiers = hmeta.get("talent_tiers") or {}
            for tier_idx, tier_name in enumerate(TALENT_TIERS):
                seen_names: list[str] = []
                for raw_tid in tiers.get(tier_name) or []:
                    tid = int(raw_tid)
                    tname = (abilities_by_id.get(str(tid)) or {}).get("name") or f"tal_{tid}"
                    if tname not in seen_names:
                        seen_names.append(tname)
                    choice_idx = min(1, seen_names.index(tname))
                    tmap[tid] = (tier_idx, choice_idx)
            self.hero_talent_id_to_choice[hid] = tmap

    def build_skill_mask(self) -> torch.BoolTensor:
        """Build [num_heroes + 1, 18, num_skill_classes] boolean mask of legal skill actions."""
        mask = torch.zeros((self.num_heroes + 1, 18, self.num_skill_classes), dtype=torch.bool)
        # Fallback for hero_idx == 0 (<PAD>) so softmax is never all -inf
        mask[0, :, 0] = True
        for hidx, n_skills in self._hero_num_skills_by_idx.items():
            lvl_slots = self.hero_levelable_slots.get(hidx) or []
            if lvl_slots:
                for slot_i in lvl_slots:
                    mask[hidx, :, slot_i] = True
            elif n_skills > 0:
                mask[hidx, :, :n_skills] = True
            else:
                mask[hidx, :, 0] = True

            hid = self._hero_id_by_idx.get(hidx, 0)
            min_ult_idx = 1 if hid == MEEPO_HERO_ID else 5
            for ult_slot in self.hero_ult_slots.get(hidx, ()):
                mask[hidx, :min_ult_idx, ult_slot] = False

            mask[hidx, 9:, self.slot_talent] = True  # Talents unlock at level 10 (0-based idx >= 9)
            mask[hidx, :, self.slot_attr] = True

            # Zero out levels where a hero gains no skill point (lvl 17 standard, lvl 18 Meepo)
            if hid == MEEPO_HERO_ID:
                mask[hidx, 17, :] = False
            elif hid != INVOKER_HERO_ID:
                mask[hidx, 16, :] = False
        return mask

    def map_item_id(self, raw_item_id: int) -> int | None:
        canonical = self.item_aliases.get(int(raw_item_id), int(raw_item_id))
        return self.item_id_to_idx.get(canonical)


def _accumulate_zero_talent_stats(
    rows: list[dict[str, Any]],
    vocab_index: VocabIndex,
    seen_by_hero_tier: dict[tuple[int, int], Counter[int]],
    aid0_hero_tier: Counter[tuple[int, int]],
) -> None:
    """Accumulate per-(hero, tier) observed talent choices and tier-window aid=0 counts from rows."""
    talent_lookup = vocab_index.hero_talent_id_to_choice
    for row in rows:
        hero_ids = row["hero_ids"]
        ab_ids_all = row["ability_learn_ids"]
        ab_lvls_all = row["ability_learn_levels"]
        for p_i in range(10):
            hid = int(hero_ids[p_i])
            tmap = talent_lookup.get(hid)
            for aid_raw, lvl_raw in zip(ab_ids_all[p_i], ab_lvls_all[p_i], strict=True):
                aid = int(aid_raw)
                lvl = int(lvl_raw)
                if aid == 0:
                    if lvl >= 10:
                        tier_bin = 3 if lvl >= 25 else (lvl - 10) // 5
                        aid0_hero_tier[(hid, tier_bin)] += 1
                elif tmap and aid in tmap:
                    ti, ch = tmap[aid]
                    seen_by_hero_tier[(hid, ti)][ch] += 1


def _finalize_zero_talent_tiers(
    vocab_index: VocabIndex,
    seen_by_hero_tier: dict[tuple[int, int], Counter[int]],
    aid0_hero_tier: Counter[tuple[int, int]],
    *,
    min_seen: int = 15,
    min_hero_aid0: int = 5,
) -> dict[int, dict[int, int | None]]:
    """Convert accumulated counts into {hero_id: {tier_idx: known_choice_or_None}}."""
    zero_tiers: dict[int, dict[int, int | None]] = {}
    for hid, tmap in vocab_index.hero_talent_id_to_choice.items():
        defined_choices: dict[int, set[int]] = defaultdict(set)
        for ti, ch in tmap.values():
            defined_choices[ti].add(ch)

        inferred: dict[int, int | None] = {}
        for ti in range(4):
            if len(defined_choices.get(ti, ())) < 2:
                continue
            if aid0_hero_tier[(hid, ti)] < min_hero_aid0:
                continue
            sc = seen_by_hero_tier.get((hid, ti))
            if not sc:
                inferred[ti] = None
            elif len(sc) == 1 and min(sc.values()) >= min_seen:
                inferred[ti] = next(iter(sc.keys()))
        if inferred:
            zero_tiers[hid] = inferred

    return zero_tiers


def infer_zero_talent_tiers(
    train_rows: list[dict[str, Any]],
    vocab_index: VocabIndex,
    *,
    min_seen: int = 15,
    min_hero_aid0: int = 5,
) -> dict[int, dict[int, int | None]]:
    """Infer per-hero talent tiers emitted as ability_id=0 in STRATZ, using train_rows only."""
    seen_by_hero_tier: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    aid0_hero_tier: Counter[tuple[int, int]] = Counter()
    _accumulate_zero_talent_stats(train_rows, vocab_index, seen_by_hero_tier, aid0_hero_tier)
    return _finalize_zero_talent_tiers(
        vocab_index,
        seen_by_hero_tier,
        aid0_hero_tier,
        min_seen=min_seen,
        min_hero_aid0=min_hero_aid0,
    )


def _validate_parquet_shard_schema(path: Path) -> set[str]:
    schema_names = set(pq.read_schema(path).names)
    missing = sorted(set(REQUIRED_PARQUET_COLUMNS) - schema_names)
    if missing:
        raise ValueError(
            f"Parquet shard {path} is incompatible with schema v{PARQUET_SCHEMA_VERSION}; "
            f"missing columns: {missing}"
        )
    return schema_names


def load_parquet_rows(shard_paths: list[Path]) -> list[dict[str, Any]]:
    """Read and validate v2 Parquet shards, pruning to training-required columns."""
    if not shard_paths:
        raise ValueError("No Parquet shards provided")
    tables: list[pa.Table] = []
    for path in shard_paths:
        schema_names = _validate_parquet_shard_schema(path)
        cols = [c for c in TRAINING_PARQUET_COLUMNS if c in schema_names]
        tables.append(pq.read_table(path, columns=cols))
    combined = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return combined.to_pylist()


def _partition_train_val_indices(
    records: list[tuple[int, int, bool]],
    *,
    val_split: float = 0.1,
    seed: int = 42,
    pro_val_min_count: int = 0,
) -> tuple[list[int], list[int]]:
    """Split (start_datetime, match_id, is_pro) chronologically with optional pro val floor."""
    n = len(records)
    if n < 2 or val_split <= 0.0:
        return list(range(n)), []

    n_val = max(1, round(n * val_split))
    if n - n_val < 1:
        n_val = 1

    # Sort chronologically so validation mirrors future-match generalization without leakage
    sorted_items = sorted(
        ((sdt, mid, is_pro, orig_idx) for orig_idx, (sdt, mid, is_pro) in enumerate(records)),
        key=lambda item: (item[0], item[1]),
    )
    train_items = sorted_items[:-n_val]
    val_items = sorted_items[-n_val:]

    if pro_val_min_count > 0 and n >= 200:
        val_pro = sum(1 for item in val_items if item[2])
        if val_pro < pro_val_min_count:
            train_pro_pos = [pos for pos, item in enumerate(train_items) if item[2]]
            total_pro = val_pro + len(train_pro_pos)
            if total_pro >= 20:
                target_pro = min(
                    pro_val_min_count,
                    max(val_pro, round(total_pro * max(val_split, 0.15))),
                )
                need = max(0, min(target_pro - val_pro, len(train_pro_pos)))
                if need > 0:
                    move_positions = set(train_pro_pos[-need:])
                    moved = [train_items[pos] for pos in sorted(move_positions)]
                    train_items = [
                        item for pos, item in enumerate(train_items) if pos not in move_positions
                    ]
                    val_items = sorted(
                        [*val_items, *moved], key=lambda item: (item[0], item[1])
                    )

    train_ids = {item[1] for item in train_items}
    val_ids = {item[1] for item in val_items}
    if not train_ids.isdisjoint(val_ids):
        overlap = sorted(train_ids & val_ids)
        raise RuntimeError(f"Data leakage detected between train and val match_ids: {overlap[:5]}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_items))
    train_indices = [train_items[int(i)][3] for i in perm]
    val_indices = [item[3] for item in val_items]
    return train_indices, val_indices


def split_rows_train_val(
    rows: list[dict[str, Any]],
    *,
    val_split: float = 0.1,
    seed: int = 42,
    pro_val_min_count: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate by match_id and split chronologically (newest matches to validation)."""
    unique_by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        unique_by_id[int(row["match_id"])] = row
    unique_rows = list(unique_by_id.values())

    records = [
        (int(r["start_datetime"]), int(r["match_id"]), bool(r.get("is_pro", False)))
        for r in unique_rows
    ]
    train_idx, val_idx = _partition_train_val_indices(
        records,
        val_split=val_split,
        seed=seed,
        pro_val_min_count=pro_val_min_count,
    )
    return [unique_rows[i] for i in train_idx], [unique_rows[i] for i in val_idx]


_MISSING_STAT_FLOAT = float(MISSING_STAT_SENTINEL)
_LANE_COLS: tuple[str, str, str] = (
    "top_lane_outcome",
    "mid_lane_outcome",
    "bottom_lane_outcome",
)
_LEAD_COLS: tuple[str, str] = ("radiant_gold_lead_10m", "radiant_xp_lead_10m")


def _allocate_dataset_tensors(n: int, max_item_len: int) -> dict[str, torch.Tensor]:
    """Allocate contiguous zero/sentinel PyTorch tensors for n matches."""
    T = int(max_item_len)
    return {
        "hero_idx": torch.zeros((n, 10), dtype=torch.long),
        "is_radiant": torch.zeros((n, 10), dtype=torch.long),
        "pos_idx": torch.zeros((n, 10), dtype=torch.long),
        "radiant_win": torch.zeros((n,), dtype=torch.float32),
        "duration_min": torch.zeros((n,), dtype=torch.float32),
        "lane_targets": torch.full((n, 3), IGNORE_INDEX, dtype=torch.long),
        "lane5_targets": torch.full((n, 3), IGNORE_INDEX, dtype=torch.long),
        "lead10_targets": torch.zeros((n, 2), dtype=torch.float32),
        "lead10_mask": torch.zeros((n, 2), dtype=torch.float32),
        "combat_rates": torch.zeros((n, 10, 4), dtype=torch.float32),
        "combat_rate_mask": torch.zeros((n, 10, 4), dtype=torch.float32),
        "combat_shares": torch.full((n, 2, 5, 4), 0.2, dtype=torch.float32),
        "combat_share_mask": torch.zeros((n, 2, 4), dtype=torch.float32),
        "econ_kda_targets": torch.zeros((n, 10, 6), dtype=torch.float32),
        "econ_kda_mask": torch.zeros((n, 10, 6), dtype=torch.float32),
        "item_input_ids": torch.full((n, 10, T), PAD_ITEM_ID, dtype=torch.long),
        "item_target_ids": torch.full((n, 10, T), PAD_ITEM_ID, dtype=torch.long),
        "item_target_minutes": torch.zeros((n, 10, T), dtype=torch.float32),
        "item_time_mask": torch.zeros((n, 10, T), dtype=torch.float32),
        "skill_targets": torch.full((n, 10, 18), IGNORE_INDEX, dtype=torch.long),
        "talent_targets": torch.full((n, 10, 4), IGNORE_INDEX, dtype=torch.long),
        # Strictly metadata / optional loss weight — never fed to the neural network
        "sample_weight": torch.ones((n,), dtype=torch.float32),
        "is_pro_meta": torch.zeros((n,), dtype=torch.bool),
    }


def _tensors_to_numpy_views(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Return shared-memory NumPy views of pre-allocated PyTorch dataset tensors."""
    return {k: t.numpy() for k, t in tensors.items()}


def _encode_match_row_into_arrays(
    arrays: dict[str, np.ndarray],
    idx: int,
    row: dict[str, Any],
    v: VocabIndex,
    *,
    max_item_len: int,
    pro_sample_weight: float,
    skill_mask_py: list[list[list[bool]]],
    zero_tier_lookup: Mapping[int, Mapping[int, int | None]],
) -> None:
    """Encode one v2 match dict directly into shared NumPy views with minimal overhead."""
    T = int(max_item_len)
    t_limit = T - 1
    hero_idx = arrays["hero_idx"]
    is_radiant = arrays["is_radiant"]
    pos_idx = arrays["pos_idx"]
    item_input_ids = arrays["item_input_ids"]
    item_target_ids = arrays["item_target_ids"]
    item_target_minutes = arrays["item_target_minutes"]
    item_time_mask = arrays["item_time_mask"]
    skill_targets = arrays["skill_targets"]
    talent_targets = arrays["talent_targets"]

    hero_id_to_idx = v.hero_id_to_idx
    item_aliases_get = v.item_aliases.get
    item_id_to_idx_get = v.item_id_to_idx.get
    skill_map_lookup = v.hero_skill_id_to_slot
    talent_map_lookup = v.hero_talent_id_to_choice
    slot_attr = v.slot_attr
    slot_talent = v.slot_talent

    hero_ids_row = row["hero_ids"]
    is_rad_row = row["is_radiant"]
    pos_row = row["positions"]
    key_item_ids_row = row["key_item_ids"]
    key_item_times_row = row["key_item_times"]
    ab_ids_row = row["ability_learn_ids"]
    ab_lvls_row = row["ability_learn_levels"]
    ab_istal_row = row["ability_learn_is_talent"]

    for p_i in range(10):
        hid = int(hero_ids_row[p_i])
        h_idx = hero_id_to_idx.get(hid, 0)
        hero_idx[idx, p_i] = h_idx
        is_radiant[idx, p_i] = 1 if is_rad_row[p_i] else 0
        pos_val = int(pos_row[p_i]) - 1
        pos_idx[idx, p_i] = 0 if pos_val < 0 else (4 if pos_val > 4 else pos_val)

        # Item build sequence (<BOS>, i_1, ..., i_k) -> (i_1, ..., i_k, <EOS>)
        item_input_ids[idx, p_i, 0] = BOS_ITEM_ID
        step = 0
        for raw_iid, t_sec in zip(
            key_item_ids_row[p_i], key_item_times_row[p_i], strict=True
        ):
            raw_i = int(raw_iid)
            canon_i = item_aliases_get(raw_i, raw_i)
            item_tok = item_id_to_idx_get(canon_i)
            if item_tok is None:
                continue
            minute = float(t_sec) / 60.0
            if minute < 0.0:
                minute = 0.0
            elif minute > 180.0:
                minute = 180.0
            item_target_ids[idx, p_i, step] = item_tok
            item_target_minutes[idx, p_i, step] = minute
            item_time_mask[idx, p_i, step] = 1.0
            step += 1
            if step < T:
                item_input_ids[idx, p_i, step] = item_tok
            if step >= t_limit:
                break
        if step < T:
            item_target_ids[idx, p_i, step] = EOS_ITEM_ID

        # Skill order (levels 1-18) and talent choices (10/15/20/25 across all levels)
        skill_map = skill_map_lookup.get(hid)
        talent_map = talent_map_lookup.get(hid)
        hero_zero_tiers = zero_tier_lookup.get(hid)
        hero_skill_mask = skill_mask_py[h_idx]
        occupied_tiers = [False, False, False, False]
        talent_vals = [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
        skill_vals = [IGNORE_INDEX] * 18

        for aid_raw, lvl_raw, is_tal in zip(
            ab_ids_row[p_i],
            ab_lvls_row[p_i],
            ab_istal_row[p_i],
            strict=True,
        ):
            aid = int(aid_raw)
            lvl = int(lvl_raw)

            skill_target_slot: int | None = None
            if aid == ATTRIBUTE_BONUS_ABILITY_ID:
                skill_target_slot = slot_attr
            elif talent_map is not None and aid in talent_map:
                tier_idx, choice_idx = talent_map[aid]
                if not occupied_tiers[tier_idx] and talent_vals[tier_idx] == IGNORE_INDEX:
                    talent_vals[tier_idx] = choice_idx
                occupied_tiers[tier_idx] = True
                if lvl >= 10:
                    skill_target_slot = slot_talent
            elif aid == 0:
                if lvl >= 10:
                    skill_target_slot = slot_talent
                    if hero_zero_tiers:
                        candidates = [
                            ti
                            for ti in range(4)
                            if ti in hero_zero_tiers
                            and not occupied_tiers[ti]
                            and lvl >= 10 + 5 * ti
                        ]
                        if len(candidates) == 1:
                            ti = candidates[0]
                            occupied_tiers[ti] = True
                            known_c = hero_zero_tiers[ti]
                            if known_c is not None and talent_vals[ti] == IGNORE_INDEX:
                                talent_vals[ti] = 1 - int(known_c)
                        elif len(candidates) > 1:
                            occupied_tiers[candidates[0]] = True
            elif aid > 0 and is_tal:
                if lvl >= 10:
                    skill_target_slot = slot_talent
            elif skill_map is not None and aid in skill_map:
                skill_target_slot = skill_map[aid]

            if (
                skill_target_slot is not None
                and 1 <= lvl <= 18
                and hero_skill_mask[lvl - 1][skill_target_slot]
            ):
                skill_vals[lvl - 1] = skill_target_slot

        talent_targets[idx, p_i] = talent_vals
        skill_targets[idx, p_i] = skill_vals

    dur_min = float(row["duration_seconds"]) / 60.0
    if dur_min < MIN_MATCH_DURATION_MIN:
        dur_min = MIN_MATCH_DURATION_MIN
    arrays["radiant_win"][idx] = 1.0 if row["radiant_win"] else 0.0
    arrays["duration_min"][idx] = dur_min

    lane_targets = arrays["lane_targets"]
    lane5_targets = arrays["lane5_targets"]
    for lane_idx, col in enumerate(_LANE_COLS):
        raw_outcome = row.get(col)
        if raw_outcome is not None:
            raw_s = str(raw_outcome)
            mapped = LANE_OUTCOME_TO_ID.get(raw_s)
            if mapped is not None:
                lane_targets[idx, lane_idx] = mapped
            mapped5 = LANE_OUTCOME_5_TO_ID.get(raw_s)
            if mapped5 is not None:
                lane5_targets[idx, lane_idx] = mapped5

    lead10_targets = arrays["lead10_targets"]
    lead10_mask = arrays["lead10_mask"]
    for lead_idx, col in enumerate(_LEAD_COLS):
        raw_lead = row.get(col)
        if raw_lead is not None:
            lead10_targets[idx, lead_idx] = float(raw_lead)
            lead10_mask[idx, lead_idx] = 1.0

    # Combat DPM/HPM rates & Radiant/Dire team shares
    combat_rates = arrays["combat_rates"]
    combat_rate_mask = arrays["combat_rate_mask"]
    combat_shares = arrays["combat_shares"]
    combat_share_mask = arrays["combat_share_mask"]
    missing = _MISSING_STAT_FLOAT

    raw_combat_cols = (
        row["hero_damage"],
        row["tower_damage"],
        row["hero_damage_received"],
        row["hero_healing"],
    )
    for stat_idx, col_vals in enumerate(raw_combat_cols):
        rad_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
        dire_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
        rad_ok = True
        dire_ok = True
        for p_i in range(10):
            raw_v = col_vals[p_i]
            if raw_v is None:
                if p_i < 5:
                    rad_ok = False
                else:
                    dire_ok = False
                continue
            fv = float(raw_v)
            if fv == missing or fv < 0.0:
                if p_i < 5:
                    rad_ok = False
                else:
                    dire_ok = False
                continue
            combat_rates[idx, p_i, stat_idx] = fv / dur_min
            combat_rate_mask[idx, p_i, stat_idx] = 1.0
            if p_i < 5:
                rad_vals[p_i] = fv
            else:
                dire_vals[p_i - 5] = fv

        is_heal = stat_idx == 3
        if rad_ok:
            r_sum = rad_vals[0] + rad_vals[1] + rad_vals[2] + rad_vals[3] + rad_vals[4]
            if (r_sum >= HEAL_SHARE_MIN_TEAM_SUM) if is_heal else (r_sum > 0.0):
                inv_sum = 1.0 / r_sum
                for slot_i in range(5):
                    combat_shares[idx, 0, slot_i, stat_idx] = rad_vals[slot_i] * inv_sum
                combat_share_mask[idx, 0, stat_idx] = 1.0
        if dire_ok:
            d_sum = dire_vals[0] + dire_vals[1] + dire_vals[2] + dire_vals[3] + dire_vals[4]
            if (d_sum >= HEAL_SHARE_MIN_TEAM_SUM) if is_heal else (d_sum > 0.0):
                inv_sum = 1.0 / d_sum
                for slot_i in range(5):
                    combat_shares[idx, 1, slot_i, stat_idx] = dire_vals[slot_i] * inv_sum
                combat_share_mask[idx, 1, stat_idx] = 1.0

    # Econ/KDA rates: [GPM, XPM, CS/min, Kills/10m, Deaths/10m, Assists/10m]
    econ_kda_targets = arrays["econ_kda_targets"]
    econ_kda_mask = arrays["econ_kda_mask"]
    per_10m_div = dur_min / 10.0
    econ_specs = (
        (row["gpm"], 1.0),
        (row["xpm"], 1.0),
        (row["last_hits"], dur_min),
        (row["kills"], per_10m_div),
        (row["deaths"], per_10m_div),
        (row["assists"], per_10m_div),
    )
    for feat_idx, (vals, divisor) in enumerate(econ_specs):
        inv_div = 1.0 / divisor
        for p_i in range(10):
            raw_v = vals[p_i]
            if raw_v is None:
                continue
            fv = float(raw_v)
            if fv != missing and fv >= 0.0:
                econ_kda_targets[idx, p_i, feat_idx] = fv * inv_div
                econ_kda_mask[idx, p_i, feat_idx] = 1.0

    is_pro_flag = bool(row.get("is_pro", False))
    arrays["sample_weight"][idx] = pro_sample_weight if is_pro_flag else 1.0
    arrays["is_pro_meta"][idx] = is_pro_flag


def _encode_match_row_into_tensors(
    tensors: dict[str, torch.Tensor],
    idx: int,
    row: dict[str, Any],
    v: VocabIndex,
    *,
    max_item_len: int,
    pro_sample_weight: float,
    skill_mask: torch.Tensor,
    zero_tier_lookup: Mapping[int, Mapping[int, int | None]],
) -> None:
    """Encode one v2 match dict directly into row idx of pre-allocated tensor dict."""
    _encode_match_row_into_arrays(
        _tensors_to_numpy_views(tensors),
        idx,
        row,
        v,
        max_item_len=max_item_len,
        pro_sample_weight=pro_sample_weight,
        skill_mask_py=skill_mask.tolist(),
        zero_tier_lookup=zero_tier_lookup,
    )


def _apply_skill_mask_clamp(tensors: dict[str, torch.Tensor], skill_mask: torch.Tensor) -> None:
    """Defense-in-depth clamp setting any illegal skill target vs skill_mask to IGNORE_INDEX."""
    skill_targets = tensors["skill_targets"]
    n = skill_targets.shape[0]
    valid_skills = skill_targets != IGNORE_INDEX
    if n > 0 and valid_skills.any():
        hero_idx = tensors["hero_idx"]
        h_grid = hero_idx.unsqueeze(-1).expand(-1, -1, 18)
        l_grid = torch.arange(18, dtype=torch.long).view(1, 1, 18).expand(n, 10, 18)
        legal_skills = skill_mask[h_grid, l_grid, skill_targets.clamp(min=0)]
        skill_targets[valid_skills & (~legal_skills)] = IGNORE_INDEX


class DotaDraftDataset(Dataset[dict[str, torch.Tensor]]):
    """Pre-tensorized PyTorch Dataset converting v2 Parquet match rows into two-stage tensors."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        vocab_index: VocabIndex,
        *,
        max_item_len: int = 12,
        pro_sample_weight: float = 1.0,
        zero_talent_tiers: Mapping[int, Mapping[int, int | None]] | None = None,
    ) -> None:
        self.vidx = vocab_index
        self.max_item_len = int(max_item_len)
        self.pro_sample_weight = float(pro_sample_weight)
        self._num_samples = len(rows)

        tensors = _allocate_dataset_tensors(self._num_samples, self.max_item_len)
        arrays = _tensors_to_numpy_views(tensors)
        skill_mask = vocab_index.build_skill_mask()
        skill_mask_py: list[list[list[bool]]] = skill_mask.tolist()
        zero_tier_lookup = zero_talent_tiers or {}

        for idx, row in enumerate(rows):
            _encode_match_row_into_arrays(
                arrays,
                idx,
                row,
                vocab_index,
                max_item_len=self.max_item_len,
                pro_sample_weight=self.pro_sample_weight,
                skill_mask_py=skill_mask_py,
                zero_tier_lookup=zero_tier_lookup,
            )

        _apply_skill_mask_clamp(tensors, skill_mask)
        self._tensors = tensors

    @classmethod
    def from_tensors(
        cls,
        tensors: dict[str, torch.Tensor],
        vocab_index: VocabIndex,
        *,
        max_item_len: int = 12,
        pro_sample_weight: float = 1.0,
    ) -> DotaDraftDataset:
        """Construct a DotaDraftDataset directly around pre-allocated tensors."""
        obj = cls.__new__(cls)
        Dataset.__init__(obj)
        obj.vidx = vocab_index
        obj.max_item_len = int(max_item_len)
        obj.pro_sample_weight = float(pro_sample_weight)
        obj._tensors = tensors
        obj._num_samples = int(tensors["hero_idx"].shape[0])
        return obj

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: tensor[idx] for k, tensor in self._tensors.items()}

    def get_batch(
        self,
        indices: torch.Tensor | Sequence[int] | slice,
        *,
        keys: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Gather a mini-batch directly from pre-allocated tensors (optionally pruned to keys)."""
        if keys is None:
            return {k: tensor[indices] for k, tensor in self._tensors.items()}
        return {k: self._tensors[k][indices] for k in keys if k in self._tensors}


class TensorBatchLoader:
    """Fast iterable batch sampler directly slicing DotaDraftDataset pre-allocated tensors.

    Supports stage-scoped key pruning (skipping unused Stage-2 tensors in Phase 1) and
    automatic GPU caching when CUDA free memory comfortably accommodates the sliced keys.
    """

    def __init__(
        self,
        dataset: DotaDraftDataset,
        *,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
        keys: Sequence[str] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))
        self._keys: tuple[str, ...] = (
            tuple(k for k in keys if k in dataset._tensors)
            if keys is not None
            else tuple(dataset._tensors.keys())
        )
        self._source_tensors: dict[str, torch.Tensor] = {
            k: dataset._tensors[k] for k in self._keys
        }
        self._device = torch.device(device) if device is not None else torch.device("cpu")
        if self._device.type == "cuda" and len(dataset) > 0:
            total_bytes = sum(t.numel() * t.element_size() for t in self._source_tensors.values())
            try:
                free_mem, _ = torch.cuda.mem_get_info(self._device)
            except RuntimeError:
                free_mem = 0
            budget = min(4 * 1024**3, int(free_mem * 0.45)) if free_mem > 0 else 0
            if 0 < total_bytes <= budget:
                self._source_tensors = {
                    k: t.to(self._device) for k, t in self._source_tensors.items()
                }
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device("cpu")

    def release(self) -> None:
        """Drop any cached GPU tensor references cleanly."""
        self._source_tensors = {}

    def __len__(self) -> int:
        n = len(self.dataset)
        if n == 0:
            return 0
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        n = len(self.dataset)
        if n == 0:
            return
        bs = self.batch_size
        limit = (n // bs) * bs if self.drop_last else n
        src = self._source_tensors or {k: self.dataset._tensors[k] for k in self._keys}
        if self.shuffle:
            perm = torch.randperm(n, generator=self._rng)
            if self._device.type == "cuda":
                perm = perm.to(self._device)
            for start in range(0, limit, bs):
                idx = perm[start : min(start + bs, limit)]
                yield {k: t[idx] for k, t in src.items()}
        else:
            for start in range(0, limit, bs):
                sl = slice(start, min(start + bs, limit))
                yield {k: t[sl] for k, t in src.items()}


@dataclass
class StreamingDatasetBundle:
    """Result returned by build_streaming_datasets without holding Python dicts in memory."""

    train_ds: DotaDraftDataset
    val_ds: DotaDraftDataset | None
    zero_talent_tiers: dict[int, dict[int, int | None]]
    num_total: int
    num_pro: int
    sample_match: dict[str, Any] | None = None


def build_streaming_datasets(
    shard_paths: list[Path],
    vocab_index: VocabIndex,
    *,
    val_split: float = 0.1,
    seed: int = 42,
    max_item_len: int = 12,
    pro_sample_weight: float = 1.0,
    pro_val_min_count: int = 0,
) -> StreamingDatasetBundle:
    """Two-pass streaming Parquet -> PyTorch Dataset builder bit-equivalent to classic path."""
    if not shard_paths:
        raise ValueError("No Parquet shards provided")

    shard_schema_cols: list[list[str]] = []
    # Pass 1a: lightweight match_id + start_datetime + is_pro index for deduplication and split
    latest_loc: dict[int, tuple[int, bool, int, int]] = {}
    for shard_idx, path in enumerate(shard_paths):
        schema_names = _validate_parquet_shard_schema(path)
        shard_schema_cols.append([c for c in TRAINING_PARQUET_COLUMNS if c in schema_names])
        meta_tbl = pq.read_table(path, columns=["match_id", "start_datetime", "is_pro"])
        mids = meta_tbl["match_id"].to_pylist()
        sdts = meta_tbl["start_datetime"].to_pylist()
        pros = meta_tbl["is_pro"].to_pylist()
        for row_idx, (mid, sdt, is_p) in enumerate(zip(mids, sdts, pros, strict=True)):
            latest_loc[int(mid)] = (int(sdt), bool(is_p), shard_idx, row_idx)
        del meta_tbl, mids, sdts, pros

    unique_records: list[tuple[int, int, bool, int, int]] = [
        (sdt, mid, is_p, sidx, ridx) for mid, (sdt, is_p, sidx, ridx) in latest_loc.items()
    ]
    del latest_loc

    n = len(unique_records)
    split_records = [(sdt, mid, is_p) for sdt, mid, is_p, _, _ in unique_records]
    train_idx, val_idx = _partition_train_val_indices(
        split_records,
        val_split=val_split,
        seed=seed,
        pro_val_min_count=pro_val_min_count,
    )
    train_meta = [unique_records[i] for i in train_idx]
    val_meta = [unique_records[i] for i in val_idx]

    train_by_shard: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for dst_idx, (_, _, _, sidx, ridx) in enumerate(train_meta):
        train_by_shard[sidx].append((ridx, dst_idx))
    val_by_shard: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for dst_idx, (_, _, _, sidx, ridx) in enumerate(val_meta):
        val_by_shard[sidx].append((ridx, dst_idx))

    # Pick single sample match (matches (val_rows or train_rows)[0]) for lightweight diagnostics
    sample_target: tuple[int, int] | None = None
    if val_meta:
        sample_target = (val_meta[0][3], val_meta[0][4])
    elif train_meta:
        sample_target = (train_meta[0][3], train_meta[0][4])
    sample_match: dict[str, Any] | None = None

    # Pass 1b: accumulate infer_zero_talent_tiers strictly from train rows, one shard at a time
    seen_by_hero_tier: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    aid0_hero_tier: Counter[tuple[int, int]] = Counter()
    for shard_idx, path in enumerate(shard_paths):
        assignments = train_by_shard.get(shard_idx)
        if not assignments:
            continue
        ab_tbl = pq.read_table(
            path, columns=["hero_ids", "ability_learn_ids", "ability_learn_levels"]
        )
        row_indices = [ridx for ridx, _ in assignments]
        selected_rows = ab_tbl.take(row_indices).to_pylist()
        _accumulate_zero_talent_stats(
            selected_rows, vocab_index, seen_by_hero_tier, aid0_hero_tier
        )
        del ab_tbl, selected_rows
    zero_talent_tiers = _finalize_zero_talent_tiers(
        vocab_index, seen_by_hero_tier, aid0_hero_tier
    )

    # Pass 2: encode shard-by-shard directly into pre-allocated PyTorch tensors via NumPy views
    n_train = len(train_meta)
    n_val = len(val_meta)
    train_tensors = _allocate_dataset_tensors(n_train, max_item_len)
    train_arrays = _tensors_to_numpy_views(train_tensors)
    val_tensors = _allocate_dataset_tensors(n_val, max_item_len) if n_val > 0 else None
    val_arrays = _tensors_to_numpy_views(val_tensors) if val_tensors is not None else None
    skill_mask = vocab_index.build_skill_mask()
    skill_mask_py: list[list[list[bool]]] = skill_mask.tolist()

    for shard_idx, path in enumerate(shard_paths):
        tr_assign = train_by_shard.get(shard_idx) or []
        va_assign = val_by_shard.get(shard_idx) or []
        if not tr_assign and not va_assign:
            continue
        tbl = pq.read_table(path, columns=shard_schema_cols[shard_idx])
        shard_rows = tbl.to_pylist()
        del tbl

        if sample_target is not None and sample_target[0] == shard_idx:
            sample_match = dict(shard_rows[sample_target[1]])

        for ridx, dst_idx in tr_assign:
            _encode_match_row_into_arrays(
                train_arrays,
                dst_idx,
                shard_rows[ridx],
                vocab_index,
                max_item_len=max_item_len,
                pro_sample_weight=pro_sample_weight,
                skill_mask_py=skill_mask_py,
                zero_tier_lookup=zero_talent_tiers,
            )
        if val_arrays is not None:
            for ridx, dst_idx in va_assign:
                _encode_match_row_into_arrays(
                    val_arrays,
                    dst_idx,
                    shard_rows[ridx],
                    vocab_index,
                    max_item_len=max_item_len,
                    pro_sample_weight=1.0,
                    skill_mask_py=skill_mask_py,
                    zero_tier_lookup=zero_talent_tiers,
                )
        del shard_rows
        gc.collect()

    _apply_skill_mask_clamp(train_tensors, skill_mask)
    train_ds = DotaDraftDataset.from_tensors(
        train_tensors, vocab_index, max_item_len=max_item_len, pro_sample_weight=pro_sample_weight
    )
    num_pro = int(train_tensors["is_pro_meta"].sum().item())
    val_ds: DotaDraftDataset | None = None
    if val_tensors is not None:
        _apply_skill_mask_clamp(val_tensors, skill_mask)
        val_ds = DotaDraftDataset.from_tensors(
            val_tensors, vocab_index, max_item_len=max_item_len, pro_sample_weight=1.0
        )
        num_pro += int(val_tensors["is_pro_meta"].sum().item())

    return StreamingDatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        zero_talent_tiers=zero_talent_tiers,
        num_total=n,
        num_pro=num_pro,
        sample_match=sample_match,
    )
