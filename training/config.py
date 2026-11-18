"""Training and ONNX export configuration for d2oracle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from shared.constants import PARQUET_SCHEMA_VERSION, VOCAB_SCHEMA_VERSION

MODEL_SCHEMA_VERSION = 2
ENCODER_ARCH_VERSION = 4

# Normalization scales used by Stage-1/Stage-2 heads, scenario conditioning, and losses.
DURATION_CENTER_MIN = 35.0
DURATION_SCALE_MIN = 10.0
MIN_MATCH_DURATION_MIN = 15.0
MAX_DURATION_OVERRIDE_MIN = 120.0
GOLD_XP_LEAD_SCALE = 1000.0
DPM_HPM_SCALE = 100.0
COMBAT_RATE_LOSS_SCALES: tuple[float, float, float, float] = (250.0, 120.0, 220.0, 80.0)
GPM_XPM_SCALE = 100.0
CS_PER_MIN_SCALE = 2.0
ITEM_TIME_SCALE_MIN = 10.0
ITEM_TIME_LOSS_SCALE_MIN = 5.0
HEAL_SHARE_MIN_TEAM_SUM = 500.0

COMBAT_RATE_NAMES: tuple[str, ...] = (
    "hero_dpm",
    "tower_dpm",
    "damage_taken_dpm",
    "hero_hpm",
)
COMBAT_SHARE_NAMES: tuple[str, ...] = (
    "hero_damage_share",
    "tower_damage_share",
    "damage_taken_share",
    "hero_healing_share",
)
ECON_KDA_NAMES: tuple[str, ...] = (
    "gpm",
    "xpm",
    "cs_per_min",
    "kills_per_10m",
    "deaths_per_10m",
    "assists_per_10m",
)


def default_target_scales() -> dict[str, float]:
    """Return canonical normalization scales stored inside model_config.json."""
    return {
        "duration_center_min": DURATION_CENTER_MIN,
        "duration_scale_min": DURATION_SCALE_MIN,
        "min_match_duration_min": MIN_MATCH_DURATION_MIN,
        "max_duration_override_min": MAX_DURATION_OVERRIDE_MIN,
        "gold_xp_lead_scale": GOLD_XP_LEAD_SCALE,
        "dpm_hpm_scale": DPM_HPM_SCALE,
        "gpm_xpm_scale": GPM_XPM_SCALE,
        "item_time_scale_min": ITEM_TIME_SCALE_MIN,
        "heal_share_min_team_sum": HEAL_SHARE_MIN_TEAM_SUM,
    }


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for training d2oracle two-stage multi-task model."""

    hf_repo_id: str = "rizeru/d2oracle-data"
    data_dir: Path = Path("data/training")
    checkpoint_dir: Path = Path("checkpoints")
    export_dir: Path = Path("exports")

    # Transformer architecture
    d_model: int = 192
    nhead: int = 6
    num_encoder_layers: int = 2
    stage2_num_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 384
    dropout: float = 0.1
    max_item_len: int = 12  # <BOS> + up to 11 items / <EOS>
    stage1_pair_dim: int = 28
    stage1_pair_dropout: float = 0.25
    fm_l2_weight: float = 1e-4
    stage1_win_trunk_scale: float = 0.45
    use_stage2_block: bool = True
    eos_logit_penalty: float = 0.0
    eos_stop_prob: float = 0.25

    # Optimization & two-phase curriculum
    batch_size: int = 64
    epochs: int = 20
    two_phase: bool = True
    stage1_epochs: int | None = None
    stage1_patience: int = 3
    stage2_patience: int = 4
    fit_win_temperature: bool = True
    lr: float = 3e-4
    stage1_lr: float | None = None
    fm_lr: float | None = None
    stage2_lr: float | None = None
    weight_decay: float = 2e-2
    warmup_ratio: float = 0.05
    val_split: float = 0.1
    pro_val_min_count: int = 100
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    use_amp: bool = True

    # Scheduled scenario teacher forcing (Stage 1 -> Stage 2 bridge during training)
    scenario_teacher_forcing_start: float = 0.5
    scenario_teacher_forcing_end: float = 0.15
    duration_teacher_forcing_prob: float = 0.40
    resume: bool = True

    # Optional soft up-weighting for Captains Mode pro matches inside unified training loss
    pro_sample_weight: float = 1.0
    # Per-player weight on losing builds for discrete build heads (item_ce, skill, talent)
    loser_build_weight: float = 0.35
    # Mild label smoothing on win BCE target to reduce early win_head overfitting
    win_label_smoothing: float = 0.01

    # Multi-task loss weights (normalized scales across Stage 1 and Stage 2)
    w_win: float = 2.0
    w_duration: float = 2.0
    w_lanes: float = 0.8
    w_lead10: float = 0.5
    w_combat_rate: float = 0.8
    w_combat_share: float = 0.6
    w_econ_kda: float = 0.8
    w_item_ce: float = 1.0
    w_item_time: float = 1.5
    w_skill: float = 0.8
    w_skill_rank_penalty: float = 0.08
    w_talent: float = 0.6

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        for key in ("data_dir", "checkpoint_dir", "export_dir"):
            raw[key] = str(raw[key])
        return raw


@dataclass
class ModelSpec:
    """Vocabulary-derived model dimensions saved alongside checkpoints and ONNX exports."""

    game_version_id: int
    game_version_name: str
    num_heroes: int
    item_vocab_size: int
    max_item_len: int
    max_hero_skills: int
    num_skill_classes: int
    slot_talent: int
    slot_attr: int
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    stage2_num_layers: int = 2
    stage1_pair_dim: int = 28
    stage1_pair_dropout: float = 0.25
    stage1_win_trunk_scale: float = 0.45
    use_stage2_block: bool = True
    eos_logit_penalty: float = 0.0
    eos_stop_prob: float = 0.25
    model_schema_version: int = MODEL_SCHEMA_VERSION
    encoder_arch_version: int = ENCODER_ARCH_VERSION
    parquet_schema_version: int = PARQUET_SCHEMA_VERSION
    vocab_schema_version: int = VOCAB_SCHEMA_VERSION
    target_scales: dict[str, float] = field(default_factory=default_target_scales)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSpec:
        ver = data.get("model_schema_version")
        if ver != MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible model_schema_version={ver!r}; expected {MODEL_SCHEMA_VERSION}. "
                "Re-train and re-export the bundle with the v2 two-stage architecture."
            )
        arch_ver = data.get("encoder_arch_version")
        if arch_ver != ENCODER_ARCH_VERSION:
            raise ValueError(
                f"Incompatible encoder_arch_version={arch_ver!r}; expected {ENCODER_ARCH_VERSION}. "
                "Old encoder checkpoints cannot resume against TeamRelative/CausalSkillHead; "
                "start fresh or set resume=False."
            )
        parquet_ver = data.get("parquet_schema_version")
        if parquet_ver != PARQUET_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible parquet_schema_version={parquet_ver!r}; "
                f"expected {PARQUET_SCHEMA_VERSION}."
            )
        vocab_ver = data.get("vocab_schema_version")
        if vocab_ver != VOCAB_SCHEMA_VERSION:
            raise ValueError(
                f"Incompatible vocab_schema_version={vocab_ver!r}; expected {VOCAB_SCHEMA_VERSION}."
            )
        return cls(**data)
