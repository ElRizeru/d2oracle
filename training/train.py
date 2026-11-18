"""Training pipeline for d2oracle two-stage multi-task draft prediction model."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.config import ModelSpec, TrainingConfig
from training.dataset import (
    STAGE1_BATCH_KEYS,
    DotaDraftDataset,
    TensorBatchLoader,
    VocabIndex,
    build_streaming_datasets,
    download_hf_dataset,
    infer_zero_talent_tiers,
)
from training.losses import (
    _OPTIONAL_STRATIFIED_METRICS,
    compute_batch_metric_tensors,
    compute_multitask_loss_tensors,
)
from training.model import D2OracleModel

logger = logging.getLogger("d2oracle.training")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_spec(vidx: VocabIndex, cfg: TrainingConfig) -> ModelSpec:
    return ModelSpec(
        game_version_id=vidx.gv_id,
        game_version_name=vidx.gv_name,
        num_heroes=vidx.num_heroes,
        item_vocab_size=vidx.item_vocab_size,
        max_item_len=cfg.max_item_len,
        max_hero_skills=vidx.max_hero_skills,
        num_skill_classes=vidx.num_skill_classes,
        slot_talent=vidx.slot_talent,
        slot_attr=vidx.slot_attr,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        stage2_num_layers=cfg.stage2_num_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        stage1_pair_dim=cfg.stage1_pair_dim,
        stage1_pair_dropout=cfg.stage1_pair_dropout,
        stage1_win_trunk_scale=cfg.stage1_win_trunk_scale,
        use_stage2_block=cfg.use_stage2_block,
        eos_logit_penalty=cfg.eos_logit_penalty,
        eos_stop_prob=cfg.eos_stop_prob,
    )


def _aggregate_metric_tensors(
    totals: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    if not totals:
        return out
    keys = list(totals.keys())
    nums = torch.stack([totals[k][0] for k in keys]).cpu().tolist()
    dens = torch.stack([totals[k][1] for k in keys]).cpu().tolist()
    for k, num, den in zip(keys, nums, dens, strict=True):
        if den <= 0 and k in _OPTIONAL_STRATIFIED_METRICS:
            continue
        out[k] = float(num / den) if den > 0 else 0.0
    return out


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Compute tie-aware ROC-AUC without external sklearn dependency."""
    pos = y_true > 0.5
    n_pos = int(pos.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    n = int(sorted_scores.size)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    sum_ranks_pos = float(ranks[pos].sum())
    return float((sum_ranks_pos - n_pos * (n_pos + 1) * 0.5) / (n_pos * n_neg))


def _scheduled_scenario_mix(
    cfg: TrainingConfig,
    step_epoch: int,
    total_phase_epochs: int | None = None,
) -> float:
    span = cfg.epochs if total_phase_epochs is None else total_phase_epochs
    if span <= 1:
        return float(cfg.scenario_teacher_forcing_end)
    frac = float(step_epoch - 1) / float(span - 1)
    frac = max(0.0, min(1.0, frac))
    return float(
        cfg.scenario_teacher_forcing_start
        + frac * (cfg.scenario_teacher_forcing_end - cfg.scenario_teacher_forcing_start)
    )


def _compute_val_selection_score(metrics: dict[str, float]) -> float:
    """Weighted validation score across all 11 Stage-1 & Stage-2 losses."""
    return float(
        2.0 * metrics.get("loss_win", 0.0)
        + 1.5 * metrics.get("loss_duration", 0.0)
        + 0.8 * metrics.get("loss_lanes", 0.0)
        + 0.4 * metrics.get("loss_lead10", 0.0)
        + 0.25 * metrics.get("loss_combat_rate", 0.0)
        + 0.35 * metrics.get("loss_combat_share", 0.0)
        + 0.30 * metrics.get("loss_econ_kda", 0.0)
        + 0.20 * metrics.get("loss_item_ce", 0.0)
        + 0.20 * metrics.get("loss_item_time", 0.0)
        + 0.30 * metrics.get("loss_skill", 0.0)
        + 0.25 * metrics.get("loss_talent", 0.0)
    )


def compute_stage1_val_score(metrics: dict[str, float]) -> float:
    """Validation selection score for Phase-1 Stage-1 backbone & outcome heads."""
    return float(
        2.0 * metrics.get("loss_win", 0.0)
        + 1.5 * metrics.get("loss_duration", 0.0)
        + 0.8 * metrics.get("loss_lanes", 0.0)
        + 0.4 * metrics.get("loss_lead10", 0.0)
    )


def compute_stage2_val_score(metrics: dict[str, float]) -> float:
    """Validation selection score for Phase-2 Stage-2 hero projections & builds."""
    return float(
        0.50 * metrics.get("loss_combat_rate", 0.0)
        + 0.40 * metrics.get("loss_combat_share", 0.0)
        + 0.40 * metrics.get("loss_econ_kda", 0.0)
        + 0.35 * metrics.get("loss_item_ce", 0.0)
        + 0.35 * metrics.get("loss_item_time", 0.0)
        + 0.45 * metrics.get("loss_skill", 0.0)
        + 0.35 * metrics.get("loss_talent", 0.0)
    )


def _optimizer_step(
    params_or_model: torch.nn.Module | list[torch.nn.Parameter],
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
    *,
    use_scaler: bool,
) -> bool:
    """Run backward, gradient clipping, optimizer step, and step scheduler only when update applied."""
    params = (
        list(params_or_model.parameters())
        if isinstance(params_or_model, torch.nn.Module)
        else params_or_model
    )
    optimizer_stepped = True
    if scaler is not None and use_scaler:
        prev_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = scaler.get_scale() >= prev_scale
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
    if scheduler is not None and optimizer_stepped:
        scheduler.step()
    return optimizer_stepped


def _run_epoch(
    model: D2OracleModel,
    loader: TensorBatchLoader | DataLoader[dict[str, torch.Tensor]],
    cfg: TrainingConfig,
    device: torch.device,
    *,
    stage_mode: str = "all",
    scenario_mix: float = 0.0,
    duration_mix: float = 0.0,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    amp_enabled = cfg.use_amp and device.type == "cuda"
    trainable_params = [p for p in model.parameters() if p.requires_grad] if is_train else []

    loss_sums: dict[str, torch.Tensor] = {}
    metric_totals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    win_probs_all: list[torch.Tensor] = []
    win_targets_all: list[torch.Tensor] = []
    pro_flags_all: list[torch.Tensor] = []
    steps = 0

    for raw_batch in loader:
        batch = {
            k: (v if v.device == device else v.to(device, non_blocking=True))
            for k, v in raw_batch.items()
        }
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                if stage_mode == "stage1":
                    outputs = model.forward_stage1(
                        batch["hero_idx"],
                        batch["is_radiant"],
                        batch["pos_idx"],
                    )
                else:
                    outputs = model(
                        batch["hero_idx"],
                        batch["is_radiant"],
                        batch["pos_idx"],
                        batch["item_input_ids"],
                        true_win=batch["radiant_win"] if is_train else None,
                        true_duration_min=batch["duration_min"] if is_train else None,
                        true_lane_targets=batch["lane_targets"] if is_train else None,
                        true_lead10=batch["lead10_targets"] if is_train else None,
                        true_lead10_mask=batch["lead10_mask"] if is_train else None,
                        scenario_mix=scenario_mix if is_train else 0.0,
                        duration_mix=duration_mix if is_train else 0.0,
                    )
                loss, scalars = compute_multitask_loss_tensors(
                    outputs, batch, cfg, stage_mode=stage_mode
                )
                if is_train and stage_mode in ("stage1", "all") and cfg.fm_l2_weight > 0.0:
                    loss = loss + float(cfg.fm_l2_weight) * model.encoder.fm_l2()

            if is_train:
                _optimizer_step(
                    trainable_params,
                    loss,
                    optimizer,
                    scheduler,
                    scaler,
                    cfg.grad_clip,
                    use_scaler=amp_enabled,
                )

        steps += 1
        for k, v in scalars.items():
            loss_sums[k] = loss_sums[k] + v if k in loss_sums else v

        win_probs_all.append(outputs["win_prob"].squeeze(-1).detach())
        win_targets_all.append(batch["radiant_win"].detach())
        if "is_pro_meta" in batch:
            pro_flags_all.append(batch["is_pro_meta"].detach().bool())

        batch_metrics = compute_batch_metric_tensors(outputs, batch)
        for k, (num_t, den_t) in batch_metrics.items():
            if k in metric_totals:
                prev_num, prev_den = metric_totals[k]
                metric_totals[k] = (prev_num + num_t, prev_den + den_t)
            else:
                metric_totals[k] = (num_t, den_t)

    if loss_sums:
        l_keys = list(loss_sums.keys())
        l_vals = torch.stack([loss_sums[k] for k in l_keys]).cpu().tolist()
        summary = {k: float(v) / max(1, steps) for k, v in zip(l_keys, l_vals, strict=True)}
    else:
        summary = {}
    summary.update(_aggregate_metric_tensors(metric_totals))
    if win_probs_all:
        yp = torch.cat(win_probs_all).cpu().numpy()
        yt = torch.cat(win_targets_all).cpu().numpy()
        auc = _binary_roc_auc(yt, yp)
        if auc is not None:
            summary["win_auc"] = auc
        if pro_flags_all:
            pm = torch.cat(pro_flags_all).cpu().numpy().astype(bool)
            if pm.any():
                p_auc = _binary_roc_auc(yt[pm], yp[pm])
                if p_auc is not None:
                    summary["pro_win_auc"] = p_auc
    return summary


def save_checkpoint(
    path: Path,
    model: D2OracleModel,
    spec: ModelSpec,
    epoch: int,
    metrics: dict[str, float],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    best_val_score: float | None = None,
    best_val_loss: float | None = None,
    best_stage1_score: float | None = None,
    phase: int | None = None,
    stage2_start_epoch: int | None = None,
    history: list[dict[str, Any]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "spec": spec.to_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if best_val_score is not None:
        payload["best_val_score"] = float(best_val_score)
    if best_val_loss is not None:
        payload["best_val_loss"] = float(best_val_loss)
    if best_stage1_score is not None:
        payload["best_stage1_score"] = float(best_stage1_score)
    if phase is not None:
        payload["phase"] = int(phase)
    if stage2_start_epoch is not None:
        payload["stage2_start_epoch"] = int(stage2_start_epoch)
    if history is not None:
        payload["history"] = history
    torch.save(payload, path)


def _build_batch_loader(
    ds: DotaDraftDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
    keys: Sequence[str] | None = None,
    device: torch.device | None = None,
) -> TensorBatchLoader | DataLoader[dict[str, torch.Tensor]]:
    bs = min(int(batch_size), max(1, len(ds)))
    if num_workers == 0:
        return TensorBatchLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            seed=seed,
            keys=keys,
            device=device,
        )
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _resolve_stage1_epochs(cfg: TrainingConfig) -> int:
    if not cfg.two_phase or cfg.epochs < 2:
        return 0
    if cfg.stage1_epochs is not None:
        s1 = int(cfg.stage1_epochs)
        return max(0, min(s1, cfg.epochs))
    if cfg.epochs < 3:
        return 0
    return min(cfg.epochs - 1, max(2, round(cfg.epochs * 0.35)))


def _build_adamw_cosine(
    params_or_groups: list[torch.nn.Parameter] | list[dict[str, Any]],
    *,
    lr: float,
    weight_decay: float,
    total_steps: int,
    warmup_ratio: float,
    use_amp_cuda: bool,
) -> tuple[
    torch.optim.AdamW,
    torch.optim.lr_scheduler.LambdaLR,
    torch.amp.GradScaler,
]:
    optimizer = torch.optim.AdamW(params_or_groups, lr=lr, weight_decay=weight_decay)
    steps = max(1, int(total_steps))
    warmup_steps = int(steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_cuda)
    return optimizer, scheduler, scaler


def _build_stage1_param_groups(
    model: D2OracleModel,
    *,
    base_lr: float,
    fm_lr: float,
    weight_decay: float,
    include_stage2: bool = False,
) -> list[dict[str, Any]]:
    fm_params = [p for p in model.encoder.fm_parameters() if p.requires_grad]
    fm_ids = {id(p) for p in fm_params}
    pool = model.parameters() if include_stage2 else model.encoder.stage1_parameters()
    other_params = [p for p in pool if p.requires_grad and id(p) not in fm_ids]
    groups: list[dict[str, Any]] = []
    if fm_params:
        groups.append({"params": fm_params, "lr": fm_lr, "weight_decay": 0.0})
    if other_params:
        groups.append({"params": other_params, "lr": base_lr, "weight_decay": weight_decay})
    return groups


@torch.no_grad()
def fit_win_temperature(
    model: D2OracleModel,
    val_loader: TensorBatchLoader | DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    min_samples: int = 64,
    t_min: float = 0.5,
    t_max: float = 3.0,
) -> float:
    """Calibrate scalar win_temperature on validation logits minimizing exact NLL."""
    model.eval()
    logits_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    for raw_batch in val_loader:
        batch = {
            k: (v if v.device == device else v.to(device, non_blocking=True))
            for k, v in raw_batch.items()
        }
        s1 = model.forward_stage1(batch["hero_idx"], batch["is_radiant"], batch["pos_idx"])
        logits_all.append(s1["win_logit"].squeeze(-1).float())
        targets_all.append(batch["radiant_win"].float())

    if not logits_all:
        return float(model.encoder.win_temperature.item())
    logits = torch.cat(logits_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    if logits.numel() < min_samples or not (float(targets.min()) < 0.5 < float(targets.max())):
        return float(model.encoder.win_temperature.item())

    nll_before = float(F.binary_cross_entropy_with_logits(logits, targets).item())
    grid = torch.linspace(t_min, t_max, 251, device=logits.device, dtype=torch.float32)
    nlls = torch.stack(
        [F.binary_cross_entropy_with_logits(logits / t, targets) for t in grid]
    )
    best_idx = int(torch.argmin(nlls).item())
    best_t = float(grid[best_idx].item())
    lo = max(t_min, best_t - 0.05)
    hi = min(t_max, best_t + 0.05)
    fine_grid = torch.linspace(lo, hi, 101, device=logits.device, dtype=torch.float32)
    fine_nlls = torch.stack(
        [F.binary_cross_entropy_with_logits(logits / t, targets) for t in fine_grid]
    )
    fine_idx = int(torch.argmin(fine_nlls).item())
    best_t = float(fine_grid[fine_idx].item())
    nll_after = float(fine_nlls[fine_idx].item())

    if nll_after < nll_before - 1e-6:
        model.encoder.win_temperature.fill_(best_t)
        probs_before = torch.sigmoid(logits)
        probs_after = torch.sigmoid(logits / best_t)
        brier_before = float(((probs_before - targets) ** 2).mean().item())
        brier_after = float(((probs_after - targets) ** 2).mean().item())
        logger.info(
            "Calibrated Stage-1 win_temperature=%.3f | val_nll %.4f -> %.4f | val_brier %.4f -> %.4f",
            best_t,
            nll_before,
            nll_after,
            brier_before,
            brier_after,
        )
        return best_t
    return float(model.encoder.win_temperature.item())


def _load_resume_checkpoint(
    last_path: Path,
    model: D2OracleModel,
    spec: ModelSpec,
    cfg: TrainingConfig,
) -> dict[str, Any]:
    ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
    saved_spec = ckpt.get("spec") or {}
    compat_keys = (
        "model_schema_version",
        "encoder_arch_version",
        "game_version_id",
        "num_heroes",
        "item_vocab_size",
        "max_hero_skills",
        "d_model",
        "nhead",
        "num_encoder_layers",
        "stage2_num_layers",
        "num_decoder_layers",
        "dim_feedforward",
        "max_item_len",
        "stage1_pair_dim",
        "stage1_win_trunk_scale",
        "use_stage2_block",
    )
    cur_spec = spec.to_dict()
    mismatches = [k for k in compat_keys if saved_spec.get(k) != cur_spec.get(k)]
    if mismatches:
        raise ValueError(
            f"Cannot resume from incompatible checkpoint {last_path}; mismatched fields: {mismatches}. "
            "Delete the stale checkpoint or set resume=False."
        )
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Cannot resume from checkpoint {last_path}: model weights are incompatible with current "
            f"architecture ({exc}). Delete the stale checkpoint or set resume=False."
        ) from exc
    return ckpt


def _log_epoch_summary(
    epoch: int,
    total_epochs: int,
    *,
    phase_label: str,
    mix_p: float | None,
    train_stats: dict[str, float],
    val_stats: dict[str, float],
    val_score: float,
    elapsed_sec: float | None = None,
) -> None:
    auc_str = f" auc={val_stats['win_auc']:.4f}" if "win_auc" in val_stats else ""
    pro_str = ""
    if "pro_win_acc" in val_stats:
        pro_auc = f", p_auc={val_stats['pro_win_auc']:.3f}" if "pro_win_auc" in val_stats else ""
        pro_str = f" | pro_win={val_stats['pro_win_acc']:.3f}{pro_auc}"
    time_str = f" | {elapsed_sec:.1f}s" if elapsed_sec is not None else ""

    if phase_label == "Phase 1":
        logger.info(
            "Epoch %02d/%02d [Phase 1] | train=%.4f val=%.4f s1_score=%.4f |%s "
            "win=%.3f dur=%.2fm lane=%.3f lead10_g=%.0f%s%s",
            epoch,
            total_epochs,
            train_stats["loss_total"],
            val_stats["loss_total"],
            val_score,
            auc_str,
            val_stats["win_acc"],
            val_stats["dur_p50_mae"],
            val_stats["lane_acc"],
            val_stats["lead10_gold_mae"],
            pro_str,
            time_str,
        )
        return

    tf_part = f", tf={mix_p:.2f}" if mix_p is not None else ""
    logger.info(
        "Epoch %02d/%02d [%s%s] | train=%.4f val=%.4f score=%.4f |%s "
        "win=%.3f dur=%.2fm lane=%.3f gpm_mae=%.1f hdpm_mae=%.1f hhpm_mae=%.1f "
        "imin=%.2fm item=%.3f skill=%.3f tal=%.3f%s%s",
        epoch,
        total_epochs,
        phase_label,
        tf_part,
        train_stats["loss_total"],
        val_stats["loss_total"],
        val_score,
        auc_str,
        val_stats["win_acc"],
        val_stats["dur_p50_mae"],
        val_stats["lane_acc"],
        val_stats["gpm_p50_mae"],
        val_stats["hero_dpm_mae"],
        val_stats["hero_hpm_mae"],
        val_stats["item_min_mae"],
        val_stats["item_acc"],
        val_stats["skill_acc"],
        val_stats["talent_acc"],
        pro_str,
        time_str,
    )


def train_from_datasets(
    train_ds: DotaDraftDataset,
    val_ds: DotaDraftDataset | None,
    vocab: dict[str, Any],
    cfg: TrainingConfig,
    *,
    vidx: VocabIndex | None = None,
) -> dict[str, Any]:
    """Train D2OracleModel from pre-built PyTorch Datasets and save checkpoints + metadata."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vidx = vidx or VocabIndex(vocab)
    spec = build_model_spec(vidx, cfg)
    skill_mask = vidx.build_skill_mask()
    model = D2OracleModel(spec, skill_mask).to(device)
    amp_cuda = cfg.use_amp and device.type == "cuda"

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (cfg.checkpoint_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (cfg.checkpoint_dir / "model_config.json").write_text(
        json.dumps(spec.to_dict(), indent=2), encoding="utf-8"
    )

    best_loss = float("inf")
    best_score = float("inf")
    best_stage1_score = float("inf")
    best_epoch: int | None = None
    best_val_metrics: dict[str, float] | None = None
    best_stage1_epoch: int | None = None
    best_stage1_metrics: dict[str, float] | None = None
    best_stage1_path = cfg.checkpoint_dir / "best_stage1.pt"
    best_path = cfg.checkpoint_dir / "best.pt"
    last_path = cfg.checkpoint_dir / "last.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    resume_phase: int | None = None
    ckpt: dict[str, Any] | None = None
    train_start = time.monotonic()

    s1_epochs = _resolve_stage1_epochs(cfg)
    has_val = val_ds is not None and len(val_ds) > 0

    if cfg.resume and last_path.exists():
        ckpt = _load_resume_checkpoint(last_path, model, spec, cfg)
        best_score = float(ckpt.get("best_val_score", best_score))
        best_loss = float(ckpt.get("best_val_loss", best_loss))
        best_stage1_score = float(ckpt.get("best_stage1_score", best_stage1_score))
        history = list(ckpt.get("history") or [])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        raw_phase = ckpt.get("phase")
        resume_phase = int(raw_phase) if raw_phase is not None else None
        logger.info("Resumed training from %s at epoch %d/%d", last_path, start_epoch, cfg.epochs)

    effective_bs = min(int(cfg.batch_size), max(1, len(train_ds)))
    steps_per_epoch = max(1, (len(train_ds) + effective_bs - 1) // effective_bs)
    base_s1_lr = float(cfg.stage1_lr if cfg.stage1_lr is not None else cfg.lr)
    base_fm_lr = float(cfg.fm_lr if cfg.fm_lr is not None else max(base_s1_lr, base_s1_lr * 1.5))

    if s1_epochs == 0:
        # ---- Single-phase joint training ----
        train_loader = _build_batch_loader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            pin_memory=device.type == "cuda",
            device=device,
        )
        val_loader = (
            _build_batch_loader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                seed=cfg.seed,
                pin_memory=device.type == "cuda",
                device=device,
            )
            if has_val
            else None
        )
        try:
            opt_groups = _build_stage1_param_groups(
                model,
                base_lr=cfg.lr,
                fm_lr=float(cfg.fm_lr if cfg.fm_lr is not None else max(cfg.lr, cfg.lr * 1.5)),
                weight_decay=cfg.weight_decay,
                include_stage2=True,
            )
            optimizer, scheduler, scaler = _build_adamw_cosine(
                opt_groups,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                total_steps=max(1, cfg.epochs * steps_per_epoch),
                warmup_ratio=cfg.warmup_ratio,
                use_amp_cuda=amp_cuda,
            )
            if ckpt is not None:
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if "scaler_state_dict" in ckpt and amp_cuda:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])

            for epoch in range(start_epoch, cfg.epochs + 1):
                t_ep = time.monotonic()
                mix_p = _scheduled_scenario_mix(cfg, epoch)
                train_stats = _run_epoch(
                    model,
                    train_loader,
                    cfg,
                    device,
                    stage_mode="all",
                    scenario_mix=mix_p,
                    duration_mix=cfg.duration_teacher_forcing_prob,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                )
                val_stats = (
                    _run_epoch(
                        model, val_loader, cfg, device, stage_mode="all", scenario_mix=0.0, duration_mix=0.0
                    )
                    if val_loader is not None
                    else dict(train_stats)
                )
                val_score = _compute_val_selection_score(val_stats)
                val_stats["val_selection_score"] = val_score
                ep_sec = time.monotonic() - t_ep
                history.append(
                    {
                        "epoch": epoch,
                        "phase": 0,
                        "scenario_mix": mix_p,
                        "elapsed_sec": round(ep_sec, 2),
                        "train": train_stats,
                        "val": val_stats,
                    }
                )
                _log_epoch_summary(
                    epoch,
                    cfg.epochs,
                    phase_label="Joint",
                    mix_p=mix_p,
                    train_stats=train_stats,
                    val_stats=val_stats,
                    val_score=val_score,
                    elapsed_sec=ep_sec,
                )
                if val_score <= best_score:
                    best_score = val_score
                    best_loss = val_stats["loss_total"]
                    best_epoch = epoch
                    best_val_metrics = dict(val_stats)
                    save_checkpoint(best_path, model, spec, epoch, val_stats, phase=0)

                save_checkpoint(
                    last_path,
                    model,
                    spec,
                    epoch,
                    val_stats,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if amp_cuda else None,
                    best_val_score=best_score,
                    best_val_loss=best_loss,
                    phase=0,
                    history=history,
                )
                (cfg.checkpoint_dir / "history.json").write_text(
                    json.dumps(history, indent=2), encoding="utf-8"
                )
        finally:
            if isinstance(train_loader, TensorBatchLoader):
                train_loader.release()
            if isinstance(val_loader, TensorBatchLoader):
                val_loader.release()
    else:
        # ---- Two-phase training: Phase 1 (Stage-1) -> Calibrate & Freeze -> Phase 2 (Stage-2) ----
        phase1_last_epoch = s1_epochs
        if start_epoch <= s1_epochs and resume_phase != 2:
            s1_train_loader = _build_batch_loader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                seed=cfg.seed,
                pin_memory=device.type == "cuda",
                keys=STAGE1_BATCH_KEYS,
                device=device,
            )
            s1_val_loader = (
                _build_batch_loader(
                    val_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    seed=cfg.seed,
                    pin_memory=device.type == "cuda",
                    keys=STAGE1_BATCH_KEYS,
                    device=device,
                )
                if has_val
                else None
            )
            try:
                model.set_stage1_frozen(False)
                s1_groups = _build_stage1_param_groups(
                    model,
                    base_lr=base_s1_lr,
                    fm_lr=base_fm_lr,
                    weight_decay=cfg.weight_decay,
                    include_stage2=False,
                )
                opt1, sched1, scaler1 = _build_adamw_cosine(
                    s1_groups,
                    lr=base_s1_lr,
                    weight_decay=cfg.weight_decay,
                    total_steps=max(1, s1_epochs * steps_per_epoch),
                    warmup_ratio=cfg.warmup_ratio,
                    use_amp_cuda=amp_cuda,
                )
                if ckpt is not None and resume_phase == 1:
                    if "optimizer_state_dict" in ckpt:
                        opt1.load_state_dict(ckpt["optimizer_state_dict"])
                    if "scheduler_state_dict" in ckpt:
                        sched1.load_state_dict(ckpt["scheduler_state_dict"])
                    if "scaler_state_dict" in ckpt and amp_cuda:
                        scaler1.load_state_dict(ckpt["scaler_state_dict"])

                no_improve = 0
                min_s1_epochs = min(s1_epochs, max(3, int(cfg.stage1_patience)))
                for epoch in range(start_epoch, s1_epochs + 1):
                    t_ep = time.monotonic()
                    train_stats = _run_epoch(
                        model,
                        s1_train_loader,
                        cfg,
                        device,
                        stage_mode="stage1",
                        optimizer=opt1,
                        scheduler=sched1,
                        scaler=scaler1,
                    )
                    val_stats = (
                        _run_epoch(model, s1_val_loader, cfg, device, stage_mode="stage1")
                        if s1_val_loader is not None
                        else dict(train_stats)
                    )
                    s1_score = compute_stage1_val_score(val_stats)
                    val_stats["val_selection_score"] = s1_score
                    ep_sec = time.monotonic() - t_ep
                    history.append(
                        {
                            "epoch": epoch,
                            "phase": 1,
                            "scenario_mix": 0.0,
                            "elapsed_sec": round(ep_sec, 2),
                            "train": train_stats,
                            "val": val_stats,
                        }
                    )
                    _log_epoch_summary(
                        epoch,
                        cfg.epochs,
                        phase_label="Phase 1",
                        mix_p=None,
                        train_stats=train_stats,
                        val_stats=val_stats,
                        val_score=s1_score,
                        elapsed_sec=ep_sec,
                    )
                    if s1_score <= best_stage1_score:
                        best_stage1_score = s1_score
                        best_stage1_epoch = epoch
                        best_stage1_metrics = dict(val_stats)
                        no_improve = 0
                        save_checkpoint(
                            best_stage1_path,
                            model,
                            spec,
                            epoch,
                            val_stats,
                            best_stage1_score=best_stage1_score,
                            phase=1,
                        )
                    else:
                        no_improve += 1

                    save_checkpoint(
                        last_path,
                        model,
                        spec,
                        epoch,
                        val_stats,
                        optimizer=opt1,
                        scheduler=sched1,
                        scaler=scaler1 if amp_cuda else None,
                        best_val_score=best_score,
                        best_val_loss=best_loss,
                        best_stage1_score=best_stage1_score,
                        phase=1,
                        history=history,
                    )
                    (cfg.checkpoint_dir / "history.json").write_text(
                        json.dumps(history, indent=2), encoding="utf-8"
                    )
                    phase1_last_epoch = epoch

                    if (
                        cfg.stage1_patience > 0
                        and epoch >= min_s1_epochs
                        and epoch < s1_epochs
                        and no_improve >= cfg.stage1_patience
                    ):
                        logger.info(
                            "Phase 1 early stopping at epoch %d (best=%s, s1_score=%.4f)",
                            epoch,
                            best_stage1_epoch,
                            best_stage1_score,
                        )
                        break
            finally:
                if isinstance(s1_train_loader, TensorBatchLoader):
                    s1_train_loader.release()
                if isinstance(s1_val_loader, TensorBatchLoader):
                    s1_val_loader.release()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if resume_phase == 2 and ckpt is not None:
            s2_start = start_epoch
            s2_origin_epoch = int(ckpt.get("stage2_start_epoch") or (s1_epochs + 1))
        elif start_epoch <= s1_epochs:
            s2_start = phase1_last_epoch + 1
            s2_origin_epoch = s2_start
        else:
            s2_start = start_epoch
            s2_origin_epoch = s1_epochs + 1

        if s2_start <= cfg.epochs:
            entering_phase2_fresh = resume_phase != 2
            if entering_phase2_fresh and best_stage1_path.exists():
                s1_ckpt = torch.load(best_stage1_path, map_location="cpu", weights_only=False)
                model.load_state_dict(s1_ckpt["model_state_dict"], strict=True)
                best_stage1_epoch = int(s1_ckpt.get("epoch", best_stage1_epoch or 0)) or None
                best_stage1_metrics = dict(s1_ckpt.get("metrics") or {}) or best_stage1_metrics
                logger.info(
                    "Loaded best Phase-1 Stage-1 checkpoint (epoch=%s, s1_score=%.4f) before Phase 2 freeze",
                    s1_ckpt.get("epoch"),
                    float(s1_ckpt.get("best_stage1_score", best_stage1_score)),
                )
                if cfg.fit_win_temperature and has_val and val_ds is not None:
                    cal_loader = _build_batch_loader(
                        val_ds,
                        batch_size=cfg.batch_size,
                        shuffle=False,
                        num_workers=cfg.num_workers,
                        seed=cfg.seed,
                        pin_memory=device.type == "cuda",
                        keys=STAGE1_BATCH_KEYS,
                        device=device,
                    )
                    try:
                        fit_win_temperature(model, cal_loader, device)
                        save_checkpoint(
                            best_stage1_path,
                            model,
                            spec,
                            best_stage1_epoch or phase1_last_epoch,
                            best_stage1_metrics or {},
                            best_stage1_score=best_stage1_score,
                            phase=1,
                        )
                    finally:
                        if isinstance(cal_loader, TensorBatchLoader):
                            cal_loader.release()

            model.set_stage1_frozen(True)
            s2_params = [p for p in model.parameters() if p.requires_grad]
            s2_total_epochs = max(1, cfg.epochs - s2_origin_epoch + 1)

            s2_train_loader = _build_batch_loader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                seed=cfg.seed,
                pin_memory=device.type == "cuda",
                device=device,
            )
            s2_val_loader = (
                _build_batch_loader(
                    val_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    seed=cfg.seed,
                    pin_memory=device.type == "cuda",
                    device=device,
                )
                if has_val
                else None
            )
            try:
                opt2, sched2, scaler2 = _build_adamw_cosine(
                    s2_params,
                    lr=float(cfg.stage2_lr if cfg.stage2_lr is not None else cfg.lr),
                    weight_decay=cfg.weight_decay,
                    total_steps=max(1, s2_total_epochs * steps_per_epoch),
                    warmup_ratio=cfg.warmup_ratio,
                    use_amp_cuda=amp_cuda,
                )
                if ckpt is not None and resume_phase == 2:
                    if "optimizer_state_dict" in ckpt:
                        opt2.load_state_dict(ckpt["optimizer_state_dict"])
                    if "scheduler_state_dict" in ckpt:
                        sched2.load_state_dict(ckpt["scheduler_state_dict"])
                    if "scaler_state_dict" in ckpt and amp_cuda:
                        scaler2.load_state_dict(ckpt["scaler_state_dict"])

                for epoch in range(s2_start, cfg.epochs + 1):
                    t_ep = time.monotonic()
                    phase2_ep = epoch - s2_origin_epoch + 1
                    mix_p = _scheduled_scenario_mix(cfg, phase2_ep, total_phase_epochs=s2_total_epochs)
                    train_stats = _run_epoch(
                        model,
                        s2_train_loader,
                        cfg,
                        device,
                        stage_mode="stage2",
                        scenario_mix=mix_p,
                        duration_mix=cfg.duration_teacher_forcing_prob,
                        optimizer=opt2,
                        scheduler=sched2,
                        scaler=scaler2,
                    )
                    val_stats = (
                        _run_epoch(
                            model,
                            s2_val_loader,
                            cfg,
                            device,
                            stage_mode="stage2",
                            scenario_mix=0.0,
                            duration_mix=0.0,
                        )
                        if s2_val_loader is not None
                        else dict(train_stats)
                    )
                    val_score = compute_stage2_val_score(val_stats)
                    val_stats["val_selection_score"] = val_score
                    ep_sec = time.monotonic() - t_ep
                    history.append(
                        {
                            "epoch": epoch,
                            "phase": 2,
                            "scenario_mix": mix_p,
                            "elapsed_sec": round(ep_sec, 2),
                            "train": train_stats,
                            "val": val_stats,
                        }
                    )
                    _log_epoch_summary(
                        epoch,
                        cfg.epochs,
                        phase_label="Phase 2",
                        mix_p=mix_p,
                        train_stats=train_stats,
                        val_stats=val_stats,
                        val_score=val_score,
                        elapsed_sec=ep_sec,
                    )
                    if val_score <= best_score:
                        best_score = val_score
                        best_loss = val_stats["loss_total"]
                        best_epoch = epoch
                        best_val_metrics = dict(val_stats)
                        save_checkpoint(
                            best_path,
                            model,
                            spec,
                            epoch,
                            val_stats,
                            best_stage1_score=best_stage1_score,
                            phase=2,
                        )

                    save_checkpoint(
                        last_path,
                        model,
                        spec,
                        epoch,
                        val_stats,
                        optimizer=opt2,
                        scheduler=sched2,
                        scaler=scaler2 if amp_cuda else None,
                        best_val_score=best_score,
                        best_val_loss=best_loss,
                        best_stage1_score=best_stage1_score,
                        phase=2,
                        stage2_start_epoch=s2_origin_epoch,
                        history=history,
                    )
                    (cfg.checkpoint_dir / "history.json").write_text(
                        json.dumps(history, indent=2), encoding="utf-8"
                    )
            finally:
                if isinstance(s2_train_loader, TensorBatchLoader):
                    s2_train_loader.release()
                if isinstance(s2_val_loader, TensorBatchLoader):
                    s2_val_loader.release()

    model.set_stage1_frozen(False)
    if not best_path.exists() and history:
        best_epoch = int(history[-1]["epoch"])
        best_val_metrics = dict(history[-1]["val"])
        save_checkpoint(best_path, model, spec, best_epoch, best_val_metrics)
    elif best_path.exists() and best_epoch is None:
        b_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        best_epoch = int(b_ckpt.get("epoch", 0)) or None
        best_val_metrics = dict(b_ckpt.get("metrics") or {}) or None

    if best_stage1_path.exists() and best_stage1_epoch is None:
        s1_ckpt = torch.load(best_stage1_path, map_location="cpu", weights_only=True)
        best_stage1_epoch = int(s1_ckpt.get("epoch", 0)) or None
        best_stage1_metrics = dict(s1_ckpt.get("metrics") or {}) or None

    result: dict[str, Any] = {
        "best_checkpoint": best_path,
        "last_checkpoint": last_path,
        "vocab_path": cfg.checkpoint_dir / "vocab.json",
        "model_config_path": cfg.checkpoint_dir / "model_config.json",
        "history": history,
        "spec": spec,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_stage1_epoch": best_stage1_epoch,
        "best_stage1_metrics": best_stage1_metrics,
        "train_elapsed_sec": round(time.monotonic() - train_start, 2),
    }
    if best_stage1_path.exists():
        result["best_stage1_checkpoint"] = best_stage1_path
    return result


def train_from_rows(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    vocab: dict[str, Any],
    cfg: TrainingConfig,
) -> dict[str, Any]:
    """Train D2OracleModel on in-memory v2 Parquet rows and save checkpoints + metadata."""
    vidx = VocabIndex(vocab)
    zero_talent_tiers = infer_zero_talent_tiers(train_rows, vidx)
    if zero_talent_tiers:
        logger.info(
            "Inferred aid=0 talent recovery rules from train_rows for %d heroes (%d tiers)",
            len(zero_talent_tiers),
            sum(len(t) for t in zero_talent_tiers.values()),
        )
    train_ds = DotaDraftDataset(
        train_rows,
        vidx,
        max_item_len=cfg.max_item_len,
        pro_sample_weight=cfg.pro_sample_weight,
        zero_talent_tiers=zero_talent_tiers,
    )
    val_ds = (
        DotaDraftDataset(
            val_rows,
            vidx,
            max_item_len=cfg.max_item_len,
            pro_sample_weight=1.0,
            zero_talent_tiers=zero_talent_tiers,
        )
        if val_rows
        else None
    )
    return train_from_datasets(train_ds, val_ds, vocab, cfg, vidx=vidx)


def train_from_shards(
    shard_paths: list[Path],
    vocab: dict[str, Any],
    cfg: TrainingConfig,
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stream v2 Parquet shards directly into compact tensors and train D2OracleModel."""
    if not shard_paths:
        raise RuntimeError("No Parquet shards provided for training.")
    t_ds = time.monotonic()
    vidx = VocabIndex(vocab)
    bundle = build_streaming_datasets(
        shard_paths,
        vidx,
        val_split=cfg.val_split,
        seed=cfg.seed,
        max_item_len=cfg.max_item_len,
        pro_sample_weight=cfg.pro_sample_weight,
        pro_val_min_count=cfg.pro_val_min_count,
    )
    ds_sec = round(time.monotonic() - t_ds, 2)
    if bundle.zero_talent_tiers:
        logger.info(
            "Inferred aid=0 talent recovery rules (train-only) for %d heroes (%d tiers)",
            len(bundle.zero_talent_tiers),
            sum(len(t) for t in bundle.zero_talent_tiers.values()),
        )
    patch_name = str((manifest or {}).get("game_version_name", vidx.gv_name))
    gv_id = int((manifest or {}).get("game_version_id", vidx.gv_id))
    n_val = len(bundle.val_ds) if bundle.val_ds is not None else 0
    logger.info(
        "Streamed %d unique matches on patch %s (gv=%s, pro=%d, ranked=%d): train=%d val=%d in %.1fs",
        bundle.num_total,
        patch_name,
        gv_id,
        bundle.num_pro,
        bundle.num_total - bundle.num_pro,
        len(bundle.train_ds),
        n_val,
        ds_sec,
    )
    out = train_from_datasets(bundle.train_ds, bundle.val_ds, vocab, cfg, vidx=vidx)
    out["sample_match"] = bundle.sample_match
    out["val_ds"] = bundle.val_ds
    out["dataset_summary"] = {
        "patch": patch_name,
        "game_version_id": gv_id,
        "num_shards": len(shard_paths),
        "num_total": bundle.num_total,
        "num_pro": bundle.num_pro,
        "num_ranked": bundle.num_total - bundle.num_pro,
        "num_train": len(bundle.train_ds),
        "num_val": n_val,
        "zero_talent_tiers_heroes": len(bundle.zero_talent_tiers),
        "zero_talent_tiers_count": sum(len(t) for t in bundle.zero_talent_tiers.values()),
        "dataset_build_sec": ds_sec,
    }
    return out


def train_from_hf(cfg: TrainingConfig, *, token: str | None = None) -> dict[str, Any]:
    """Download active-patch dataset from Hugging Face Datasets and train via streaming shards."""
    manifest, vocab, shard_paths = download_hf_dataset(
        cfg.hf_repo_id, cfg.data_dir, token=token
    )
    if not shard_paths:
        raise RuntimeError(
            f"No Parquet shards listed in {cfg.hf_repo_id}/latest.json "
            f"for patch {manifest.get('game_version_name')}"
        )
    return train_from_shards(shard_paths, vocab, cfg, manifest=manifest)


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="d2oracle-train")
    parser.add_argument("--repo-id", default="rizeru/d2oracle-data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    cfg = TrainingConfig(
        hf_repo_id=args.repo_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
    )
    train_from_hf(cfg, token=os.environ.get("HF_TOKEN"))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
