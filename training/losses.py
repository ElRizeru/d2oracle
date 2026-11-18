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
    zero = values.sum() * 0.0
    return torch.where(denom > 0, (values * weights).sum() / denom.clamp(min=1e-12), zero)


def compute_multitask_loss_tensors(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: TrainingConfig,
    *,
    stage_mode: str = "all",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted multi-task loss and detached tensor scalars without CPU-GPU sync."""
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
        safe_lanes = torch.where(
            lane_targets != IGNORE_INDEX,
            torch.clamp(lane_targets, min=0, max=2),
            IGNORE_INDEX,
        )
        lane_logits5 = outputs.get("lane_logits5")
        if lane_logits5 is not None:
            lane_ce3 = F.nll_loss(
                lane_logits.reshape(-1, 3),
                safe_lanes.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(bsz, 3)
        else:
            lane_ce3 = F.cross_entropy(
                lane_logits.reshape(-1, 3),
                safe_lanes.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(bsz, 3)
        loss_lanes = _weighted_masked_mean(lane_ce3, lane_mask, sample_weight)

        lane5_targets = batch.get("lane5_targets")
        if lane_logits5 is not None and lane5_targets is not None:
            valid5 = (lane_targets != IGNORE_INDEX) & (lane5_targets != IGNORE_INDEX)
            safe_lanes5 = torch.where(
                valid5,
                torch.clamp(lane5_targets, min=0, max=4),
                IGNORE_INDEX,
            )
            lane_ce5 = F.cross_entropy(
                lane_logits5.float().reshape(-1, 5),
                safe_lanes5.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="none",
            ).view(bsz, 3)
            loss_lane5 = _weighted_masked_mean(lane_ce5, valid5.to(torch.float32), sample_weight)
            loss_lanes = torch.where(
                valid5.any(),
                0.70 * loss_lanes + 0.30 * loss_lane5,
                loss_lanes,
            )

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
            zero = torch.zeros((), device=device, dtype=torch.float32)
            scalars = {
                "loss_total": stage1_total.detach(),
                "loss_win": loss_win.detach(),
                "loss_duration": loss_duration.detach(),
                "loss_lanes": loss_lanes.detach(),
                "loss_lead10": loss_lead10.detach(),
                "loss_combat_rate": zero,
                "loss_combat_share": zero,
                "loss_econ_kda": zero,
                "loss_item_ce": zero,
                "loss_item_time": zero,
                "loss_skill": zero,
                "loss_talent": zero,
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
        item_ce = F.cross_entropy(
            item_logits.reshape(-1, item_logits.shape[-1]),
            item_targets.reshape(-1),
            ignore_index=PAD_ITEM_ID,
            reduction="none",
        ).view(bsz, 10, item_targets.shape[2])
        loss_item_ce = _weighted_masked_mean(item_ce, valid_items.to(torch.float32), build_weight)

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
        skill_ce = F.cross_entropy(
            skill_logits.reshape(-1, skill_logits.shape[-1]),
            skill_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(bsz, 10, 18)
        loss_skill = _weighted_masked_mean(skill_ce, valid_skills.to(torch.float32), build_weight)

        w_rank_pen = float(getattr(cfg, "w_skill_rank_penalty", 0.0))
        if w_rank_pen > 0.0:
            num_cls = skill_logits.shape[-1]
            act_f = valid_skills.to(torch.float32).unsqueeze(-1)
            skill_probs = F.softmax(skill_logits, dim=-1) * act_f
            pred_cum = skill_probs.cumsum(dim=2)

            true_safe = torch.where(valid_skills, skill_targets.clamp(min=0, max=num_cls - 1), 0)
            true_oh = F.one_hot(true_safe, num_classes=num_cls).to(torch.float32) * act_f
            true_cum = true_oh.cumsum(dim=2)

            # Level-aware cap [1, 1, 18, 1] maxed with true_cum so hero/facet outliers (Invoker/Wyvern/etc.)
            # never fight real targets while excess mass on >4-rank basic skills is cleanly penalized.
            lvl_caps = torch.tensor(
                [min(4, (lvl + 1) // 2) for lvl in range(1, 19)],
                device=device,
                dtype=torch.float32,
            ).view(1, 1, 18, 1).expand(1, 1, 18, num_cls).clone()
            if num_cls >= 2:
                # Attribute Bonus (last class) can reach up to 7 ranks in levels 6..18
                lvl_caps[..., -1] = 7.0
            eff_cap = torch.maximum(lvl_caps, true_cum)
            rank_excess = F.relu(pred_cum - eff_cap).pow(2).sum(dim=-1)  # [B, 10, 18]
            loss_skill_rank = _weighted_masked_mean(
                rank_excess, valid_skills.to(torch.float32), build_weight
            )
            loss_skill = loss_skill + w_rank_pen * loss_skill_rank

        talent_logits = outputs["talent_logits"].float()
        talent_targets = batch["talent_targets"]
        valid_talents = talent_targets != IGNORE_INDEX
        talent_ce = F.cross_entropy(
            talent_logits.reshape(-1, 2),
            talent_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(bsz, 10, 4)
        loss_talent = _weighted_masked_mean(
            talent_ce, valid_talents.to(torch.float32), build_weight
        )

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
        "loss_total": total.detach(),
        "loss_win": loss_win.detach(),
        "loss_duration": loss_duration.detach(),
        "loss_lanes": loss_lanes.detach(),
        "loss_lead10": loss_lead10.detach(),
        "loss_combat_rate": loss_combat_rate.detach(),
        "loss_combat_share": loss_combat_share.detach(),
        "loss_econ_kda": loss_econ_kda.detach(),
        "loss_item_ce": loss_item_ce.detach(),
        "loss_item_time": loss_item_time.detach(),
        "loss_skill": loss_skill.detach(),
        "loss_talent": loss_talent.detach(),
    }
    return total, scalars


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: TrainingConfig,
    *,
    stage_mode: str = "all",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute normalized weighted multi-task training loss across Stage-1 and Stage-2 heads."""
    total, t_scalars = compute_multitask_loss_tensors(outputs, batch, cfg, stage_mode=stage_mode)
    stacked = torch.stack(list(t_scalars.values())).cpu().tolist()
    scalars = dict(zip(t_scalars.keys(), (float(x) for x in stacked), strict=True))
    return total, scalars


def _masked_sum_count(
    values: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m_f = mask.to(torch.float32)
    return (values.to(torch.float32) * m_f).sum(), m_f.sum()


@torch.no_grad()
def compute_batch_metric_tensors(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Compute numerator/count scalar tensors without CPU-GPU syncs inside batch loops."""
    metrics: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    win_prob = outputs["win_prob"].squeeze(-1)
    win_pred = (win_prob >= 0.5).to(torch.float32)
    rw = batch["radiant_win"].to(torch.float32)
    ones_m = torch.ones_like(rw)
    metrics["win_acc"] = ((win_pred == rw).to(torch.float32).sum(), ones_m.sum())
    metrics["win_brier"] = (((win_prob - rw) ** 2).sum(), ones_m.sum())

    dur_abs = torch.abs(outputs["duration_quantiles"][:, 1] - batch["duration_min"].to(torch.float32))
    metrics["dur_p50_mae"] = (dur_abs.sum(), ones_m.sum())

    is_pro_meta = batch.get("is_pro_meta")
    if is_pro_meta is not None:
        pro_mask = is_pro_meta.bool()
        pub_mask = ~pro_mask
        metrics["pro_win_acc"] = _masked_sum_count(win_pred == rw, pro_mask)
        metrics["pro_dur_mae"] = _masked_sum_count(dur_abs, pro_mask)
        metrics["pub_win_acc"] = _masked_sum_count(win_pred == rw, pub_mask)
        metrics["pub_dur_mae"] = _masked_sum_count(dur_abs, pub_mask)

    lane_targets = batch["lane_targets"]
    lane_mask = lane_targets != IGNORE_INDEX
    lane_pred = outputs["lane_probs"].argmax(dim=-1)
    metrics["lane_acc"] = _masked_sum_count(lane_pred == lane_targets, lane_mask)

    lead_mask = batch["lead10_mask"] > 0
    lead_abs = torch.abs(outputs["lead10"] - batch["lead10_targets"])
    for idx_l, name_l in enumerate(("lead10_gold_mae", "lead10_xp_mae")):
        metrics[name_l] = _masked_sum_count(lead_abs[:, idx_l], lead_mask[:, idx_l])

    if "combat_rates" not in outputs:
        return metrics

    crate_mask = batch["combat_rate_mask"] > 0
    crate_abs = torch.abs(outputs["combat_rates"] - batch["combat_rates"])
    for idx_c, name_c in enumerate(
        ("hero_dpm_mae", "tower_dpm_mae", "dmg_taken_dpm_mae", "hero_hpm_mae")
    ):
        metrics[name_c] = _masked_sum_count(crate_abs[:, :, idx_c], crate_mask[:, :, idx_c])

    cshare_mask = (batch["combat_share_mask"] > 0).unsqueeze(2).expand(-1, -1, 5, -1)
    share_abs = torch.abs(outputs["combat_shares"] - batch["combat_shares"])
    metrics["combat_share_mae"] = _masked_sum_count(share_abs, cshare_mask)

    econ_mask = batch["econ_kda_mask"] > 0
    econ_p50_abs = torch.abs(outputs["econ_kda_quantiles"][:, :, :, 1] - batch["econ_kda_targets"])
    for idx_e, name_e in enumerate(("gpm_p50_mae", "xpm_p50_mae", "cs_min_p50_mae")):
        metrics[name_e] = _masked_sum_count(econ_p50_abs[:, :, idx_e], econ_mask[:, :, idx_e])
    metrics["kda_10m_p50_mae"] = _masked_sum_count(econ_p50_abs[:, :, 3:6], econ_mask[:, :, 3:6])

    item_mask = batch["item_target_ids"] != PAD_ITEM_ID
    item_pred = outputs["item_logits"].argmax(dim=-1)
    metrics["item_acc"] = _masked_sum_count(item_pred == batch["item_target_ids"], item_mask)

    time_mask = batch["item_time_mask"] > 0
    pred_item_minutes_eval = torch.gather(
        outputs["item_minutes"].float(),
        dim=-1,
        index=batch["item_target_ids"].clamp(min=0).unsqueeze(-1),
    ).squeeze(-1)
    time_abs = torch.abs(pred_item_minutes_eval - batch["item_target_minutes"].float())
    metrics["item_min_mae"] = _masked_sum_count(time_abs, time_mask)

    skill_mask = batch["skill_targets"] != IGNORE_INDEX
    skill_pred = outputs["skill_logits"].argmax(dim=-1)
    metrics["skill_acc"] = _masked_sum_count(skill_pred == batch["skill_targets"], skill_mask)

    talent_mask = batch["talent_targets"] != IGNORE_INDEX
    talent_pred = outputs["talent_logits"].argmax(dim=-1)
    metrics["talent_acc"] = _masked_sum_count(talent_pred == batch["talent_targets"], talent_mask)
    return metrics


_OPTIONAL_STRATIFIED_METRICS = frozenset(
    {"pro_win_acc", "pro_dur_mae", "pub_win_acc", "pub_dur_mae"}
)


@torch.no_grad()
def compute_batch_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, tuple[float, int]]:
    """Compute numerator/count pairs for exact epoch-level metric aggregation."""
    tensor_metrics = compute_batch_metric_tensors(outputs, batch)
    keys = list(tensor_metrics.keys())
    nums = torch.stack([tensor_metrics[k][0] for k in keys]).cpu().tolist()
    dens = torch.stack([tensor_metrics[k][1] for k in keys]).cpu().tolist()
    out: dict[str, tuple[float, int]] = {}
    for k, num, den in zip(keys, nums, dens, strict=True):
        den_int = round(den)
        if den_int == 0 and k in _OPTIONAL_STRATIFIED_METRICS:
            continue
        out[k] = (float(num), den_int)
    return out
