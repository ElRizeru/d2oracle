"""Multi-task losses and validation metrics for d2oracle two-stage prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from training.config import (
    COMBAT_RATE_LOSS_SCALES,
    CS_PER_MIN_SCALE,
    DURATION_SCALE_MIN,
    GOLD_XP_LEAD_SCALE,
    GPM_XPM_SCALE,
    ITEM_TIME_LOSS_SCALE_MIN,
    TrainingConfig,
)
from training.dataset import IGNORE_INDEX, PAD_ITEM_ID


def _weighted_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute sample-weighted mean over masked elements, returning zero with grad on empty masks."""
    while sample_weight.ndim < values.ndim:
        sample_weight = sample_weight.unsqueeze(-1)
    weights = mask.to(values.dtype) * sample_weight.to(values.dtype)
    denom = weights.sum()
    if denom > 0:
        return (values * weights).sum() / denom
    return values.sum() * 0.0


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: TrainingConfig,
    *,
    stage_mode: str = "all",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute normalized weighted multi-task training loss across Stage-1 and Stage-2 heads."""
    if stage_mode not in ("all", "stage1", "stage2"):
        raise ValueError(f"Invalid stage_mode={stage_mode!r}; expected 'all', 'stage1', or 'stage2'")

    win_logit = outputs["win_logit"].squeeze(-1).float()
    device = win_logit.device
    bsz = win_logit.shape[0]
    sample_weight = batch.get("sample_weight")
    if sample_weight is None:
        sample_weight = torch.ones((bsz,), device=device, dtype=torch.float32)
    else:
        sample_weight = sample_weight.float()

    with torch.amp.autocast(device_type=device.type, enabled=False):
        # ---- Stage 1 losses (computed in float32 for AMP stability) ----
        win_target = batch["radiant_win"].float()
        win_smooth = float(getattr(cfg, "win_label_smoothing", 0.0))
        if win_smooth > 0.0:
            win_target = win_target * (1.0 - win_smooth) + 0.5 * win_smooth
        win_loss_vec = F.binary_cross_entropy_with_logits(
            win_logit, win_target, reduction="none"
        )
        loss_win = _weighted_masked_mean(win_loss_vec, torch.ones_like(win_loss_vec), sample_weight)

        quantiles = torch.tensor([0.25, 0.50, 0.75], device=device, dtype=torch.float32)
        dur_err_norm = (
            batch["duration_min"].float().unsqueeze(-1) - outputs["duration_quantiles"].float()
        ) / DURATION_SCALE_MIN
        dur_pinball = torch.maximum(quantiles * dur_err_norm, (quantiles - 1.0) * dur_err_norm)
        loss_duration = _weighted_masked_mean(
            dur_pinball, torch.ones_like(dur_pinball), sample_weight
        )

        lane_logits = outputs["lane_logits"].float()
        lane_targets = batch["lane_targets"]
        lane_mask = (lane_targets != IGNORE_INDEX).to(torch.float32)
        if (lane_targets != IGNORE_INDEX).any():
            safe_lanes = torch.clamp(lane_targets, min=0, max=2)
            lane_ce = F.cross_entropy(
                lane_logits.reshape(-1, 3), safe_lanes.reshape(-1), reduction="none"
            ).view(bsz, 3)
            loss_lanes = _weighted_masked_mean(lane_ce, lane_mask, sample_weight)
        else:
            loss_lanes = lane_logits.sum() * 0.0

        lead10_diff = (
            outputs["lead10"].float() - batch["lead10_targets"].float()
        ) / GOLD_XP_LEAD_SCALE
        lead10_err = F.smooth_l1_loss(
            lead10_diff, torch.zeros_like(lead10_diff), reduction="none", beta=0.5
        )
        loss_lead10 = _weighted_masked_mean(
            lead10_err, batch["lead10_mask"].float(), sample_weight
        )

        stage1_total = (
            cfg.w_win * loss_win
            + cfg.w_duration * loss_duration
            + cfg.w_lanes * loss_lanes
            + cfg.w_lead10 * loss_lead10
        )

        if stage_mode == "stage1":
            scalars = {
                "loss_total": float(stage1_total.detach().item()),
                "loss_win": float(loss_win.detach().item()),
                "loss_duration": float(loss_duration.detach().item()),
                "loss_lanes": float(loss_lanes.detach().item()),
                "loss_lead10": float(loss_lead10.detach().item()),
                "loss_combat_rate": 0.0,
                "loss_combat_share": 0.0,
                "loss_econ_kda": 0.0,
                "loss_item_ce": 0.0,
                "loss_item_time": 0.0,
                "loss_skill": 0.0,
                "loss_talent": 0.0,
            }
            return stage1_total, scalars

        # ---- Stage 2 losses ----
        combat_scale = torch.tensor(
            COMBAT_RATE_LOSS_SCALES, device=device, dtype=torch.float32
        ).view(1, 1, 4)
        # Wider quadratic zone (beta=2.0) on right-skewed / zero-inflated tower_dpm & hero_hpm
        # avoids collapsing conditional healing/building rates toward 0 median.
        combat_beta = torch.tensor(
            (0.8, 2.0, 0.8, 2.0), device=device, dtype=torch.float32
        ).view(1, 1, 4)
        combat_diff = (
            outputs["combat_rates"].float() - batch["combat_rates"].float()
        ) / combat_scale
        abs_c = torch.abs(combat_diff)
        combat_rate_err = torch.where(
            abs_c < combat_beta,
            0.5 * (combat_diff * combat_diff) / combat_beta,
            abs_c - 0.5 * combat_beta,
        )
        loss_combat_rate = _weighted_masked_mean(
            combat_rate_err, batch["combat_rate_mask"].float(), sample_weight
        )

        log_pred_shares = F.log_softmax(outputs["combat_share_logits"].float(), dim=2)  # [B, 2, 5, 4]
        target_shares = batch["combat_shares"].float()
        kl_elem = torch.where(
            target_shares > 1e-8,
            target_shares * (torch.log(torch.clamp(target_shares, min=1e-8)) - log_pred_shares),
            torch.zeros_like(log_pred_shares),
        )
        kl_team = kl_elem.sum(dim=2)  # [B, 2, 4]
        loss_combat_share = _weighted_masked_mean(
            kl_team, batch["combat_share_mask"].float(), sample_weight
        )

        econ_scale = torch.tensor(
            [GPM_XPM_SCALE, GPM_XPM_SCALE, CS_PER_MIN_SCALE, 1.0, 1.0, 1.0],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 6, 1)
        econ_err_norm = (
            batch["econ_kda_targets"].float().unsqueeze(-1)
            - outputs["econ_kda_quantiles"].float()
        ) / econ_scale
        econ_pinball = torch.maximum(quantiles * econ_err_norm, (quantiles - 1.0) * econ_err_norm)
        loss_econ_kda = _weighted_masked_mean(
            econ_pinball,
            batch["econ_kda_mask"].float().unsqueeze(-1).expand_as(econ_pinball),
            sample_weight,
        )

        loser_build_w = float(getattr(cfg, "loser_build_weight", 1.0))
        is_winner = (batch["is_radiant"] > 0) == (batch["radiant_win"].view(bsz, 1) > 0.5)
        build_weight = torch.where(
            is_winner,
            torch.ones((bsz, 10), device=device, dtype=torch.float32),
            torch.full((bsz, 10), loser_build_w, device=device, dtype=torch.float32),
        ) * sample_weight.view(bsz, 1)  # [B, 10]

        item_logits = outputs["item_logits"].float()
        item_targets = batch["item_target_ids"]
        valid_items = item_targets != PAD_ITEM_ID
        if valid_items.any():
            item_ce = F.cross_entropy(
                item_logits.reshape(-1, item_logits.shape[-1]),
                item_targets.reshape(-1),
                ignore_index=PAD_ITEM_ID,
                reduction="none",
            ).view(bsz, 10, item_targets.shape[2])
            loss_item_ce = _weighted_masked_mean(
                item_ce, valid_items.to(torch.float32), build_weight
            )
        else:
            loss_item_ce = item_logits.sum() * 0.0

        time_mask = batch["item_time_mask"].float()
        pred_item_minutes = torch.gather(
            outputs["item_minutes"].float(),
            dim=-1,
            index=batch["item_target_ids"].clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)
        item_time_diff = (
            pred_item_minutes - batch["item_target_minutes"].float()
        ) / ITEM_TIME_LOSS_SCALE_MIN
        item_time_err = F.smooth_l1_loss(
            item_time_diff, torch.zeros_like(item_time_diff), reduction="none", beta=0.4
        )
        loss_item_time = _weighted_masked_mean(item_time_err, time_mask, sample_weight)

        skill_logits = outputs["skill_logits"].float()
        skill_targets = batch["skill_targets"]
        valid_skills = skill_targets != IGNORE_INDEX
        if valid_skills.any():
            skill_ce = F.cross_entropy(
                skill_logits.reshape(-1, skill_logits.shape[-1]),
                skill_targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(bsz, 10, 18)
            loss_skill = _weighted_masked_mean(
                skill_ce, valid_skills.to(torch.float32), build_weight
            )
        else:
            loss_skill = skill_logits.sum() * 0.0

        talent_logits = outputs["talent_logits"].float()
        talent_targets = batch["talent_targets"]
        valid_talents = talent_targets != IGNORE_INDEX
        if valid_talents.any():
            talent_ce = F.cross_entropy(
                talent_logits.reshape(-1, 2),
                talent_targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(bsz, 10, 4)
            loss_talent = _weighted_masked_mean(
                talent_ce, valid_talents.to(torch.float32), build_weight
            )
        else:
            loss_talent = talent_logits.sum() * 0.0

        stage2_total = (
            cfg.w_combat_rate * loss_combat_rate
            + cfg.w_combat_share * loss_combat_share
            + cfg.w_econ_kda * loss_econ_kda
            + cfg.w_item_ce * loss_item_ce
            + cfg.w_item_time * loss_item_time
            + cfg.w_skill * loss_skill
            + cfg.w_talent * loss_talent
        )
        total = stage2_total if stage_mode == "stage2" else (stage1_total + stage2_total)

    scalars = {
        "loss_total": float(total.detach().item()),
        "loss_win": float(loss_win.detach().item()),
        "loss_duration": float(loss_duration.detach().item()),
        "loss_lanes": float(loss_lanes.detach().item()),
        "loss_lead10": float(loss_lead10.detach().item()),
        "loss_combat_rate": float(loss_combat_rate.detach().item()),
        "loss_combat_share": float(loss_combat_share.detach().item()),
        "loss_econ_kda": float(loss_econ_kda.detach().item()),
        "loss_item_ce": float(loss_item_ce.detach().item()),
        "loss_item_time": float(loss_item_time.detach().item()),
        "loss_skill": float(loss_skill.detach().item()),
        "loss_talent": float(loss_talent.detach().item()),
    }
    return total, scalars


@torch.no_grad()
def compute_batch_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, tuple[float, int]]:
    """Compute numerator/count pairs for exact epoch-level metric aggregation."""
    metrics: dict[str, tuple[float, int]] = {}

    win_prob = outputs["win_prob"].squeeze(-1)
    win_pred = (win_prob >= 0.5).to(torch.float32)
    win_correct = float((win_pred == batch["radiant_win"]).sum().item())
    n_matches = int(batch["radiant_win"].numel())
    metrics["win_acc"] = (win_correct, n_matches)
    brier = float(((win_prob - batch["radiant_win"]) ** 2).sum().item())
    metrics["win_brier"] = (brier, n_matches)

    dur_abs = torch.abs(outputs["duration_quantiles"][:, 1] - batch["duration_min"])
    metrics["dur_p50_mae"] = (float(dur_abs.sum().item()), n_matches)

    # Optional stratified Pro vs Pub metrics from read-only metadata
    is_pro_meta = batch.get("is_pro_meta")
    if is_pro_meta is not None:
        pro_mask = is_pro_meta.bool()
        pub_mask = ~pro_mask
        if pro_mask.any():
            metrics["pro_win_acc"] = (
                float((win_pred[pro_mask] == batch["radiant_win"][pro_mask]).sum().item()),
                int(pro_mask.sum().item()),
            )
            metrics["pro_dur_mae"] = (
                float(dur_abs[pro_mask].sum().item()),
                int(pro_mask.sum().item()),
            )
        if pub_mask.any():
            metrics["pub_win_acc"] = (
                float((win_pred[pub_mask] == batch["radiant_win"][pub_mask]).sum().item()),
                int(pub_mask.sum().item()),
            )
            metrics["pub_dur_mae"] = (
                float(dur_abs[pub_mask].sum().item()),
                int(pub_mask.sum().item()),
            )

    # Lane outcomes & @10m leads
    lane_targets = batch["lane_targets"]
    lane_mask = lane_targets != IGNORE_INDEX
    if lane_mask.any():
        lane_pred = outputs["lane_probs"].argmax(dim=-1)
        lane_corr = float(((lane_pred == lane_targets) & lane_mask).sum().item())
        metrics["lane_acc"] = (lane_corr, int(lane_mask.sum().item()))
    else:
        metrics["lane_acc"] = (0.0, 0)

    lead_mask = batch["lead10_mask"] > 0
    for idx_l, name_l in enumerate(("lead10_gold_mae", "lead10_xp_mae")):
        m = lead_mask[:, idx_l]
        if m.any():
            err = float(
                torch.abs(outputs["lead10"][m, idx_l] - batch["lead10_targets"][m, idx_l]).sum().item()
            )
            metrics[name_l] = (err, int(m.sum().item()))
        else:
            metrics[name_l] = (0.0, 0)

    if "combat_rates" not in outputs:
        return metrics

    # Combat DPM/HPM & team shares
    crate_mask = batch["combat_rate_mask"] > 0
    for idx_c, name_c in enumerate(
        ("hero_dpm_mae", "tower_dpm_mae", "dmg_taken_dpm_mae", "hero_hpm_mae")
    ):
        m = crate_mask[:, :, idx_c]
        if m.any():
            err = float(
                torch.abs(outputs["combat_rates"][:, :, idx_c] - batch["combat_rates"][:, :, idx_c])[m]
                .sum()
                .item()
            )
            metrics[name_c] = (err, int(m.sum().item()))
        else:
            metrics[name_c] = (0.0, 0)

    cshare_mask = batch["combat_share_mask"] > 0  # [B, 2, 4]
    if cshare_mask.any():
        expanded_m = cshare_mask.unsqueeze(2).expand(-1, -1, 5, -1)
        share_err = float(
            torch.abs(outputs["combat_shares"] - batch["combat_shares"])[expanded_m].sum().item()
        )
        metrics["combat_share_mae"] = (share_err, int(expanded_m.sum().item()))
    else:
        metrics["combat_share_mae"] = (0.0, 0)

    # Economy & KDA p50 MAE
    econ_mask = batch["econ_kda_mask"] > 0
    econ_p50 = outputs["econ_kda_quantiles"][:, :, :, 1]
    for idx_e, name_e in enumerate(
        ("gpm_p50_mae", "xpm_p50_mae", "cs_min_p50_mae")
    ):
        m = econ_mask[:, :, idx_e]
        if m.any():
            err = float(
                torch.abs(econ_p50[:, :, idx_e] - batch["econ_kda_targets"][:, :, idx_e])[m]
                .sum()
                .item()
            )
            metrics[name_e] = (err, int(m.sum().item()))
        else:
            metrics[name_e] = (0.0, 0)

    kda_m = econ_mask[:, :, 3:6]
    if kda_m.any():
        kda_err = float(
            torch.abs(econ_p50[:, :, 3:6] - batch["econ_kda_targets"][:, :, 3:6])[kda_m]
            .sum()
            .item()
        )
        metrics["kda_10m_p50_mae"] = (kda_err, int(kda_m.sum().item()))
    else:
        metrics["kda_10m_p50_mae"] = (0.0, 0)

    # Items, skills, talents
    item_mask = batch["item_target_ids"] != PAD_ITEM_ID
    if item_mask.any():
        item_pred = outputs["item_logits"].argmax(dim=-1)
        item_corr = float(((item_pred == batch["item_target_ids"]) & item_mask).sum().item())
        metrics["item_acc"] = (item_corr, int(item_mask.sum().item()))
    else:
        metrics["item_acc"] = (0.0, 0)

    time_mask = batch["item_time_mask"] > 0
    if time_mask.any():
        pred_item_minutes_eval = torch.gather(
            outputs["item_minutes"].float(),
            dim=-1,
            index=batch["item_target_ids"].clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)
        time_err = float(
            (
                torch.abs(pred_item_minutes_eval - batch["item_target_minutes"].float())
                * batch["item_time_mask"].float()
            )
            .sum()
            .item()
        )
        metrics["item_min_mae"] = (time_err, int(time_mask.sum().item()))
    else:
        metrics["item_min_mae"] = (0.0, 0)

    skill_mask = batch["skill_targets"] != IGNORE_INDEX
    if skill_mask.any():
        skill_pred = outputs["skill_logits"].argmax(dim=-1)
        skill_corr = float(((skill_pred == batch["skill_targets"]) & skill_mask).sum().item())
        metrics["skill_acc"] = (skill_corr, int(skill_mask.sum().item()))
    else:
        metrics["skill_acc"] = (0.0, 0)

    talent_mask = batch["talent_targets"] != IGNORE_INDEX
    if talent_mask.any():
        talent_pred = outputs["talent_logits"].argmax(dim=-1)
        talent_corr = float(((talent_pred == batch["talent_targets"]) & talent_mask).sum().item())
        metrics["talent_acc"] = (talent_corr, int(talent_mask.sum().item()))
    else:
        metrics["talent_acc"] = (0.0, 0)

    return metrics
