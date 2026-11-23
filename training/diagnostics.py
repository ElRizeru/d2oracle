"""Post-training validation diagnostics, training curve plotting, and export reporting."""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from shared.constants import (
    INVOKER_HERO_ID,
    LANE_OUTCOME_LABELS,
    LANE_OUTCOME_TO_ID,
    MEEPO_HERO_ID,
)
from training.config import TrainingConfig
from training.dataset import (
    BOS_ITEM_ID,
    EOS_ITEM_ID,
    IGNORE_INDEX,
    PAD_ITEM_ID,
    DotaDraftDataset,
    VocabIndex,
)
from training.export_onnx import load_model_from_checkpoint
from training.model import D2OracleModel
from training.train import _binary_roc_auc
from training.verify_onnx import decode_items_onnx_greedy

_CALIBRATION_BINS: tuple[tuple[str, float, float], ...] = (
    ("<0.42", 0.0, 0.42),
    ("0.42-0.46", 0.42, 0.46),
    ("0.46-0.50", 0.46, 0.50),
    ("0.50-0.54", 0.50, 0.54),
    ("0.54-0.58", 0.54, 0.58),
    (">=0.58", 0.58, 1.01),
)


def _resolve_item_name(item_tok: int, vocab: dict[str, Any]) -> tuple[int | None, str]:
    idx_to_id = vocab["items"]["idx_to_id"]
    if item_tok < 0 or item_tok >= len(idx_to_id):
        return None, str(item_tok)
    raw_id = idx_to_id[item_tok]
    if raw_id is None:
        return None, str(item_tok)
    iid = int(raw_id)
    meta = vocab["items"]["by_id"].get(str(iid), {})
    name = str(meta.get("display_name") or meta.get("name") or iid)
    return iid, name


def _win_calibration_summary(probs: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    n = int(probs.size)
    if n == 0:
        return {"count": 0}
    clipped = np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    y = targets.astype(np.float64)
    acc = float(((clipped >= 0.5) == (y >= 0.5)).mean())
    brier = float(((clipped - y) ** 2).mean())
    logloss = float(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean())
    auc = _binary_roc_auc(y, clipped)

    bins: list[dict[str, Any]] = []
    ece = 0.0
    for label, lo, hi in _CALIBRATION_BINS:
        mask = (clipped >= lo) & (clipped < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            bins.append({"bin": label, "count": 0, "avg_pred": None, "actual_winrate": None})
            continue
        avg_p = float(clipped[mask].mean())
        act_w = float(y[mask].mean())
        ece += (cnt / n) * abs(avg_p - act_w)
        bins.append(
            {
                "bin": label,
                "count": cnt,
                "avg_pred": round(avg_p, 4),
                "actual_winrate": round(act_w, 4),
            }
        )

    return {
        "count": n,
        "base_radiant_winrate": round(float(y.mean()), 4),
        "win_auc": round(auc, 4) if auc is not None else None,
        "win_acc": round(acc, 4),
        "win_brier": round(brier, 4),
        "win_logloss": round(logloss, 4),
        "win_ece": round(float(ece), 4),
        "bins": bins,
    }


def _skill_legality_stats(
    skill_logits: torch.Tensor,
    skill_targets: torch.Tensor,
    hero_idx: torch.Tensor,
    vidx: VocabIndex,
    skill_mask: torch.Tensor,
) -> dict[str, Any]:
    """Measure masked/rank-rule violations in raw argmax skill logits before runtime decoding."""
    n, _, num_lvls, num_classes = skill_logits.shape
    pred = skill_logits.argmax(dim=-1)  # [N, 10, 18]
    active = skill_targets != IGNORE_INDEX
    total_active = int(active.sum().item())
    if total_active == 0:
        return {"total_checked_steps": 0, "masked_violation_rate": 0.0, "rank_rule_violation_rate": 0.0}

    legal_step = skill_mask[hero_idx.unsqueeze(-1).expand(-1, -1, num_lvls), torch.arange(num_lvls).view(1, 1, num_lvls).expand(n, 10, num_lvls), pred]
    masked_violations = int((active & (~legal_step)).sum().item())

    # Cumulative rank violations over active skill steps
    pred_oh = F.one_hot(pred, num_classes=num_classes) * active.unsqueeze(-1).to(torch.long)
    cum_counts = pred_oh.cumsum(dim=2)  # [N, 10, 18, C]
    rank_at_step = torch.gather(cum_counts, dim=-1, index=pred.unsqueeze(-1)).squeeze(-1)

    rank_violations = 0
    for s_i in range(n):
        for slot_i in range(10):
            h_i = int(hero_idx[s_i, slot_i].item())
            hid = vidx._hero_id_by_idx.get(h_i, 0)
            ult_slots = vidx.hero_ult_slots.get(h_i, ())
            is_invoker = hid == INVOKER_HERO_ID
            is_meepo = hid == MEEPO_HERO_ID
            for lvl_idx in range(num_lvls):
                if not bool(active[s_i, slot_i, lvl_idx].item()):
                    continue
                lvl = lvl_idx + 1
                cls = int(pred[s_i, slot_i, lvl_idx].item())
                rk = int(rank_at_step[s_i, slot_i, lvl_idx].item())
                if cls == vidx.slot_talent:
                    cap = 4
                elif cls == vidx.slot_attr:
                    cap = 7
                elif cls in ult_slots:
                    if is_meepo:
                        cap = 3 if lvl >= 17 else (2 if lvl >= 10 else (1 if lvl >= 2 else 0))
                    else:
                        cap = 3 if lvl >= 18 else (2 if lvl >= 12 else (1 if lvl >= 6 else 0))
                elif is_invoker:
                    cap = 8
                else:
                    cap = min(4, (lvl + 1) // 2)
                if rk > cap:
                    rank_violations += 1

    return {
        "total_checked_steps": total_active,
        "masked_violation_rate": round(masked_violations / total_active, 6),
        "rank_rule_violation_rate": round(rank_violations / total_active, 6),
    }


def _collect_dataset_slice(
    ds: DotaDraftDataset,
    indices: np.ndarray,
    batch_size: int = 256,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(indices), batch_size):
        idx_t = torch.from_numpy(indices[start : start + batch_size]).long()
        batches.append(ds.get_batch(idx_t))
    return batches


@torch.no_grad()
def collect_validation_diagnostics(
    ckpt_or_model: Path | D2OracleModel,
    val_ds: DotaDraftDataset,
    vidx: VocabIndex,
    cfg: TrainingConfig,
    *,
    batch_size: int = 256,
    max_eval_matches: int = 4096,
    max_decode_matches: int = 256,
) -> dict[str, Any]:
    """Evaluate best checkpoint on validation data for calibration, role breakdown, and item stats."""
    n_val = len(val_ds)
    if n_val == 0:
        return {"num_evaluated": 0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(ckpt_or_model, Path):
        model, _, _ = load_model_from_checkpoint(ckpt_or_model, vocab=vidx.vocab)
    else:
        model = ckpt_or_model
    model = model.to(device)
    model.eval()
    skill_mask = vidx.build_skill_mask()

    if n_val > max_eval_matches:
        eval_indices = np.linspace(0, n_val - 1, max_eval_matches, dtype=np.int64)
    else:
        eval_indices = np.arange(n_val, dtype=np.int64)

    amp_enabled = cfg.use_amp and device.type == "cuda"
    win_probs_l: list[torch.Tensor] = []
    win_targets_l: list[torch.Tensor] = []
    is_pro_l: list[torch.Tensor] = []
    dur_pred_l: list[torch.Tensor] = []
    dur_true_l: list[torch.Tensor] = []
    lane_pred_l: list[torch.Tensor] = []
    lane_true_l: list[torch.Tensor] = []
    lead_pred_l: list[torch.Tensor] = []
    lead_true_l: list[torch.Tensor] = []
    lead_mask_l: list[torch.Tensor] = []

    hero_idx_l: list[torch.Tensor] = []
    is_rad_l: list[torch.Tensor] = []
    pos_idx_l: list[torch.Tensor] = []
    gpm_pred_l: list[torch.Tensor] = []
    gpm_true_l: list[torch.Tensor] = []
    gpm_mask_l: list[torch.Tensor] = []
    hdpm_pred_l: list[torch.Tensor] = []
    hdpm_true_l: list[torch.Tensor] = []
    hdpm_mask_l: list[torch.Tensor] = []
    hhpm_pred_l: list[torch.Tensor] = []
    hhpm_true_l: list[torch.Tensor] = []
    hhpm_mask_l: list[torch.Tensor] = []
    cshare_abs_l: list[torch.Tensor] = []
    cshare_mask_l: list[torch.Tensor] = []

    item_pred_l: list[torch.Tensor] = []
    item_true_l: list[torch.Tensor] = []
    item_min_pred_l: list[torch.Tensor] = []
    item_min_true_l: list[torch.Tensor] = []
    item_min_mask_l: list[torch.Tensor] = []
    skill_logits_l: list[torch.Tensor] = []
    skill_true_l: list[torch.Tensor] = []
    talent_pred_l: list[torch.Tensor] = []
    talent_true_l: list[torch.Tensor] = []

    for raw_b in _collect_dataset_slice(val_ds, eval_indices, batch_size=batch_size):
        b = {k: v.to(device) for k, v in raw_b.items()}
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(
                b["hero_idx"],
                b["is_radiant"],
                b["pos_idx"],
                b["item_input_ids"],
                scenario_mix=0.0,
                duration_mix=0.0,
            )
        win_probs_l.append(out["win_prob"].squeeze(-1).float().cpu())
        win_targets_l.append(b["radiant_win"].float().cpu())
        is_pro_l.append(b["is_pro_meta"].bool().cpu())
        dur_pred_l.append(out["duration_quantiles"].float().cpu())
        dur_true_l.append(b["duration_min"].float().cpu())
        lane_pred_l.append(out["lane_probs"].argmax(dim=-1).cpu())
        lane_true_l.append(b["lane_targets"].cpu())
        lead_pred_l.append(out["lead10"].float().cpu())
        lead_true_l.append(b["lead10_targets"].float().cpu())
        lead_mask_l.append((b["lead10_mask"] > 0).cpu())

        hero_idx_l.append(b["hero_idx"].cpu())
        is_rad_l.append(b["is_radiant"].cpu())
        pos_idx_l.append(b["pos_idx"].cpu())
        gpm_pred_l.append(out["econ_kda_quantiles"][:, :, 0, 1].float().cpu())
        gpm_true_l.append(b["econ_kda_targets"][:, :, 0].float().cpu())
        gpm_mask_l.append((b["econ_kda_mask"][:, :, 0] > 0).cpu())
        hdpm_pred_l.append(out["combat_rates"][:, :, 0].float().cpu())
        hdpm_true_l.append(b["combat_rates"][:, :, 0].float().cpu())
        hdpm_mask_l.append((b["combat_rate_mask"][:, :, 0] > 0).cpu())
        hhpm_pred_l.append(out["combat_rates"][:, :, 3].float().cpu())
        hhpm_true_l.append(b["combat_rates"][:, :, 3].float().cpu())
        hhpm_mask_l.append((b["combat_rate_mask"][:, :, 3] > 0).cpu())
        cshare_abs_l.append(torch.abs(out["combat_shares"].float() - b["combat_shares"].float()).cpu())
        cshare_mask_l.append((b["combat_share_mask"] > 0).cpu())

        item_pred_l.append(out["item_logits"].argmax(dim=-1).cpu())
        item_true_l.append(b["item_target_ids"].cpu())
        pred_imin = torch.gather(
            out["item_minutes"].float(),
            dim=-1,
            index=b["item_target_ids"].clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)
        item_min_pred_l.append(pred_imin.cpu())
        item_min_true_l.append(b["item_target_minutes"].float().cpu())
        item_min_mask_l.append((b["item_time_mask"] > 0).cpu())
        skill_logits_l.append(out["skill_logits"].float().cpu())
        skill_true_l.append(b["skill_targets"].cpu())
        talent_pred_l.append(out["talent_logits"].argmax(dim=-1).cpu())
        talent_true_l.append(b["talent_targets"].cpu())

    win_probs = torch.cat(win_probs_l).numpy()
    win_targets = torch.cat(win_targets_l).numpy()
    is_pro_arr = torch.cat(is_pro_l).numpy()

    win_cal = _win_calibration_summary(win_probs, win_targets)
    win_cal["temperature"] = round(float(model.encoder.win_temperature.item()), 4)
    # Also evaluate all pro matches in val_ds separately when pro sample in slice is small
    pro_all_mask = val_ds._tensors["is_pro_meta"].bool().numpy()
    if pro_all_mask.any():
        pro_indices = np.nonzero(pro_all_mask)[0]
        p_probs_l: list[torch.Tensor] = []
        p_targs_l: list[torch.Tensor] = []
        for raw_b in _collect_dataset_slice(val_ds, pro_indices, batch_size=batch_size):
            b = {k: v.to(device) for k, v in raw_b.items()}
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                p_out = model.forward_stage1(b["hero_idx"], b["is_radiant"], b["pos_idx"])
            p_probs_l.append(p_out["win_prob"].squeeze(-1).float().cpu())
            p_targs_l.append(b["radiant_win"].float().cpu())
        win_cal["pro"] = _win_calibration_summary(
            torch.cat(p_probs_l).numpy(), torch.cat(p_targs_l).numpy()
        )
    if (~is_pro_arr).any():
        win_cal["pub"] = _win_calibration_summary(win_probs[~is_pro_arr], win_targets[~is_pro_arr])

    dur_pred = torch.cat(dur_pred_l).numpy()
    dur_true = torch.cat(dur_true_l).numpy()
    lane_pred = torch.cat(lane_pred_l).numpy()
    lane_true = torch.cat(lane_true_l).numpy()
    lead_pred = torch.cat(lead_pred_l).numpy()
    lead_true = torch.cat(lead_true_l).numpy()
    lead_mask = torch.cat(lead_mask_l).numpy()

    lane_valid = lane_true != IGNORE_INDEX
    lane_accs: dict[str, float | None] = {}
    for idx_ln, ln_name in enumerate(("top", "mid", "bot")):
        m_ln = lane_valid[:, idx_ln]
        lane_accs[f"{ln_name}_acc"] = (
            round(float((lane_pred[m_ln, idx_ln] == lane_true[m_ln, idx_ln]).mean()), 4)
            if m_ln.any()
            else None
        )
    valid_preds = lane_pred[lane_valid]
    valid_trues = lane_true[lane_valid]
    n_lv = max(1, int(valid_trues.size))
    lane_pred_share = {
        lbl: round(float((valid_preds == cls_i).sum() / n_lv), 4)
        for cls_i, lbl in enumerate(LANE_OUTCOME_LABELS)
    }
    lane_true_share = {
        lbl: round(float((valid_trues == cls_i).sum() / n_lv), 4)
        for cls_i, lbl in enumerate(LANE_OUTCOME_LABELS)
    }

    stage1_breakdown = {
        "duration": {
            "p50_mae_min": round(float(np.abs(dur_pred[:, 1] - dur_true).mean()), 3),
            "p50_bias_min": round(float((dur_pred[:, 1] - dur_true).mean()), 3),
            "below_p25_rate": round(float((dur_true < dur_pred[:, 0]).mean()), 4),
            "within_p25_p75_rate": round(
                float(((dur_true >= dur_pred[:, 0]) & (dur_true <= dur_pred[:, 2])).mean()), 4
            ),
            "above_p75_rate": round(float((dur_true > dur_pred[:, 2]).mean()), 4),
            "avg_iqr_min": round(float((dur_pred[:, 2] - dur_pred[:, 0]).mean()), 3),
        },
        "lanes": {
            **lane_accs,
            "pred_class_share": lane_pred_share,
            "true_class_share": lane_true_share,
        },
        "lead10": {
            "gold_mae": round(float(np.abs(lead_pred[lead_mask[:, 0], 0] - lead_true[lead_mask[:, 0], 0]).mean()), 1)
            if lead_mask[:, 0].any()
            else None,
            "gold_bias": round(float((lead_pred[lead_mask[:, 0], 0] - lead_true[lead_mask[:, 0], 0]).mean()), 1)
            if lead_mask[:, 0].any()
            else None,
            "xp_mae": round(float(np.abs(lead_pred[lead_mask[:, 1], 1] - lead_true[lead_mask[:, 1], 1]).mean()), 1)
            if lead_mask[:, 1].any()
            else None,
            "xp_bias": round(float((lead_pred[lead_mask[:, 1], 1] - lead_true[lead_mask[:, 1], 1]).mean()), 1)
            if lead_mask[:, 1].any()
            else None,
        },
    }

    hero_idx_all = torch.cat(hero_idx_l)
    is_rad_all = torch.cat(is_rad_l)
    pos_idx_all = torch.cat(pos_idx_l)
    gpm_pred_all = torch.cat(gpm_pred_l)
    gpm_true_all = torch.cat(gpm_true_l)
    gpm_mask_all = torch.cat(gpm_mask_l)
    hdpm_pred_all = torch.cat(hdpm_pred_l)
    hdpm_true_all = torch.cat(hdpm_true_l)
    hdpm_mask_all = torch.cat(hdpm_mask_l)
    hhpm_pred_all = torch.cat(hhpm_pred_l)
    hhpm_true_all = torch.cat(hhpm_true_l)
    hhpm_mask_all = torch.cat(hhpm_mask_l)
    cshare_abs_all = torch.cat(cshare_abs_l)
    cshare_mask_all = torch.cat(cshare_mask_l)
    item_pred_all = torch.cat(item_pred_l)
    item_true_all = torch.cat(item_true_l)
    item_min_pred_all = torch.cat(item_min_pred_l)
    item_min_true_all = torch.cat(item_min_true_l)
    item_min_mask_all = torch.cat(item_min_mask_l)
    skill_logits_all = torch.cat(skill_logits_l)
    skill_true_all = torch.cat(skill_true_l)
    skill_pred_all = skill_logits_all.argmax(dim=-1)
    talent_pred_all = torch.cat(talent_pred_l)
    talent_true_all = torch.cat(talent_true_l)

    per_position: dict[str, dict[str, Any]] = {}
    for p_idx in range(5):
        p_mask = pos_idx_all == p_idx
        gm = p_mask & gpm_mask_all
        hm = p_mask & hdpm_mask_all
        hlm = p_mask & hhpm_mask_all
        csm = cshare_mask_all
        cs_vals = cshare_abs_all[:, :, p_idx, :]
        im = p_mask.unsqueeze(-1) & (item_true_all != PAD_ITEM_ID)
        imm = p_mask.unsqueeze(-1) & item_min_mask_all
        sm = p_mask.unsqueeze(-1) & (skill_true_all != IGNORE_INDEX)
        tm = p_mask.unsqueeze(-1) & (talent_true_all != IGNORE_INDEX)
        per_position[f"P{p_idx + 1}"] = {
            "gpm_p50_mae": round(float(torch.abs(gpm_pred_all[gm] - gpm_true_all[gm]).mean().item()), 1)
            if gm.any()
            else None,
            "gpm_p50_bias": round(float((gpm_pred_all[gm] - gpm_true_all[gm]).mean().item()), 1)
            if gm.any()
            else None,
            "hero_dpm_mae": round(float(torch.abs(hdpm_pred_all[hm] - hdpm_true_all[hm]).mean().item()), 1)
            if hm.any()
            else None,
            "hero_hpm_mae": round(float(torch.abs(hhpm_pred_all[hlm] - hhpm_true_all[hlm]).mean().item()), 1)
            if hlm.any()
            else None,
            "combat_share_mae": round(float(cs_vals[csm].mean().item()), 4) if csm.any() else None,
            "item_acc": round(float((item_pred_all[im] == item_true_all[im]).float().mean().item()), 4)
            if im.any()
            else None,
            "item_min_mae": round(
                float(torch.abs(item_min_pred_all[imm] - item_min_true_all[imm]).mean().item()), 3
            )
            if imm.any()
            else None,
            "skill_acc": round(float((skill_pred_all[sm] == skill_true_all[sm]).float().mean().item()), 4)
            if sm.any()
            else None,
            "talent_acc": round(
                float((talent_pred_all[tm] == talent_true_all[tm]).float().mean().item()), 4
            )
            if tm.any()
            else None,
        }

    # Winner vs loser build accuracy
    rw_eval = torch.from_numpy(win_targets).view(-1, 1) > 0.5
    is_win_slot = (is_rad_all > 0) == rw_eval
    winner_vs_loser: dict[str, Any] = {}
    for lbl, slot_m in (("winners", is_win_slot), ("losers", ~is_win_slot)):
        im = slot_m.unsqueeze(-1) & (item_true_all != PAD_ITEM_ID)
        sm = slot_m.unsqueeze(-1) & (skill_true_all != IGNORE_INDEX)
        tm = slot_m.unsqueeze(-1) & (talent_true_all != IGNORE_INDEX)
        winner_vs_loser[lbl] = {
            "item_acc": round(float((item_pred_all[im] == item_true_all[im]).float().mean().item()), 4)
            if im.any()
            else None,
            "skill_acc": round(float((skill_pred_all[sm] == skill_true_all[sm]).float().mean().item()), 4)
            if sm.any()
            else None,
            "talent_acc": round(
                float((talent_pred_all[tm] == talent_true_all[tm]).float().mean().item()), 4
            )
            if tm.any()
            else None,
        }

    skill_legality = _skill_legality_stats(
        skill_logits_all, skill_true_all, hero_idx_all, vidx, skill_mask
    )

    # Greedy decode + duration_override sensitivity on compact slice
    n_dec = min(len(eval_indices), max_decode_matches)
    dec_slice = eval_indices[:n_dec] if n_dec == len(eval_indices) else eval_indices[
        np.linspace(0, len(eval_indices) - 1, n_dec, dtype=np.int64)
    ]
    item_decode_stats: dict[str, Any] = {}
    duration_override_check: dict[str, Any] = {}

    if n_dec > 0:
        pred_counts: Counter[int] = Counter()
        true_counts: Counter[int] = Counter()
        pred_lens_by_pos: list[list[int]] = [[] for _ in range(5)]
        true_lens_by_pos: list[list[int]] = [[] for _ in range(5)]
        empty_builds = 0
        max_len_hits = 0
        total_slots = 0

        max_steps = cfg.max_item_len - 1
        max_win_diff_25 = 0.0
        max_win_diff_50 = 0.0
        max_dur_diff_50 = 0.0
        gpm_means: dict[str, list[float]] = {"auto": [], "25m": [], "50m": []}
        hdpm_means: dict[str, list[float]] = {"auto": [], "25m": [], "50m": []}
        item_lens: dict[str, list[float]] = {"auto": [], "25m": [], "50m": []}
        raw_over_25_num = 0
        raw_over_25_den = 0

        for raw_b in _collect_dataset_slice(val_ds, dec_slice, batch_size=min(batch_size, 128)):
            b = {k: v.to(device) for k, v in raw_b.items()}
            bsz = b["hero_idx"].shape[0]
            dur_0 = torch.zeros((bsz,), device=device, dtype=torch.float32)
            dur_25 = torch.full((bsz,), 25.0, device=device, dtype=torch.float32)
            dur_50 = torch.full((bsz,), 50.0, device=device, dtype=torch.float32)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                enc_0 = model.encoder.forward_with_logits(
                    b["hero_idx"], b["is_radiant"], b["pos_idx"], dur_0
                )
                enc_25 = model.encoder.forward_with_logits(
                    b["hero_idx"], b["is_radiant"], b["pos_idx"], dur_25
                )
                enc_50 = model.encoder.forward_with_logits(
                    b["hero_idx"], b["is_radiant"], b["pos_idx"], dur_50
                )

            dec_0, _ = model.decode_items_greedy(
                enc_0["slot_memory"],
                max_steps=max_steps,
                bos_id=BOS_ITEM_ID,
                eos_id=EOS_ITEM_ID,
                eos_penalty=cfg.eos_logit_penalty,
                eos_stop_prob=cfg.eos_stop_prob,
            )
            dec_25, min_25 = model.decode_items_greedy(
                enc_25["slot_memory"],
                max_steps=max_steps,
                bos_id=BOS_ITEM_ID,
                eos_id=EOS_ITEM_ID,
                eos_penalty=cfg.eos_logit_penalty,
                eos_stop_prob=cfg.eos_stop_prob,
                max_minute=None,
            )
            dec_50, _ = model.decode_items_greedy(
                enc_50["slot_memory"],
                max_steps=max_steps,
                bos_id=BOS_ITEM_ID,
                eos_id=EOS_ITEM_ID,
                eos_penalty=cfg.eos_logit_penalty,
                eos_stop_prob=cfg.eos_stop_prob,
                max_minute=50.0,
            )

            max_win_diff_25 = max(
                max_win_diff_25,
                float((enc_25["win_prob"] - enc_0["win_prob"]).abs().max().item()),
            )
            max_win_diff_50 = max(
                max_win_diff_50,
                float((enc_50["win_prob"] - enc_0["win_prob"]).abs().max().item()),
            )
            max_dur_diff_50 = max(
                max_dur_diff_50,
                float((enc_50["duration_quantiles"] - enc_0["duration_quantiles"]).abs().max().item()),
            )

            gpm_means["auto"].append(float(enc_0["econ_kda_quantiles"][:, :, 0, 1].mean().item()))
            gpm_means["25m"].append(float(enc_25["econ_kda_quantiles"][:, :, 0, 1].mean().item()))
            gpm_means["50m"].append(float(enc_50["econ_kda_quantiles"][:, :, 0, 1].mean().item()))
            hdpm_means["auto"].append(float(enc_0["combat_rates"][:, :, 0].mean().item()))
            hdpm_means["25m"].append(float(enc_25["combat_rates"][:, :, 0].mean().item()))
            hdpm_means["50m"].append(float(enc_50["combat_rates"][:, :, 0].mean().item()))

            dec_0_cpu = dec_0.cpu().numpy()
            dec_25_cpu = dec_25.cpu().numpy()
            min_25_cpu = min_25.cpu().numpy()
            dec_50_cpu = dec_50.cpu().numpy()
            t_items = b["item_target_ids"].cpu().numpy()
            pos_cpu = b["pos_idx"].cpu().numpy()

            for i in range(bsz):
                for s in range(10):
                    p = int(pos_cpu[i, s])
                    p_toks: list[int] = []
                    for t in dec_0_cpu[i, s]:
                        ti = int(t)
                        if ti == EOS_ITEM_ID:
                            break
                        if ti >= 3:
                            p_toks.append(ti)
                    t_toks = [int(t) for t in t_items[i, s] if t >= 3]
                    pred_lens_by_pos[p].append(len(p_toks))
                    true_lens_by_pos[p].append(len(t_toks))
                    if len(p_toks) == 0:
                        empty_builds += 1
                    if len(p_toks) >= max_steps:
                        max_len_hits += 1
                    total_slots += 1
                    pred_counts.update(p_toks)
                    true_counts.update(t_toks)
                    item_lens["auto"].append(float(len(p_toks)))

                    l25 = 0
                    for step_k, t25 in enumerate(dec_25_cpu[i, s]):
                        if int(t25) == EOS_ITEM_ID:
                            break
                        if int(t25) >= 3:
                            l25 += 1
                            raw_over_25_den += 1
                            if float(min_25_cpu[i, s, step_k]) > 25.5:
                                raw_over_25_num += 1
                    item_lens["25m"].append(float(l25))

                    l50 = 0
                    for t50 in dec_50_cpu[i, s]:
                        if int(t50) == EOS_ITEM_ID:
                            break
                        if int(t50) >= 3:
                            l50 += 1
                    item_lens["50m"].append(float(l50))

        total_pred_items = max(1, sum(pred_counts.values()))
        total_true_items = max(1, sum(true_counts.values()))
        item_decode_stats = {
            "num_decoded_matches": int(n_dec),
            "avg_pred_len_by_pos": {
                f"P{p + 1}": round(float(np.mean(pred_lens_by_pos[p])), 2)
                if pred_lens_by_pos[p]
                else 0.0
                for p in range(5)
            },
            "avg_true_len_by_pos": {
                f"P{p + 1}": round(float(np.mean(true_lens_by_pos[p])), 2)
                if true_lens_by_pos[p]
                else 0.0
                for p in range(5)
            },
            "empty_build_rate": round(empty_builds / max(1, total_slots), 4),
            "max_len_hit_rate": round(max_len_hits / max(1, total_slots), 4),
            "top_predicted_items": [
                {
                    "token": tok,
                    "item_id": _resolve_item_name(tok, vidx.vocab)[0],
                    "name": _resolve_item_name(tok, vidx.vocab)[1],
                    "count": cnt,
                    "share": round(cnt / total_pred_items, 4),
                }
                for tok, cnt in pred_counts.most_common(15)
            ],
            "top_true_items": [
                {
                    "token": tok,
                    "item_id": _resolve_item_name(tok, vidx.vocab)[0],
                    "name": _resolve_item_name(tok, vidx.vocab)[1],
                    "count": cnt,
                    "share": round(cnt / total_true_items, 4),
                }
                for tok, cnt in true_counts.most_common(15)
            ],
        }
        duration_override_check = {
            "stage1_win_max_diff": float(max(max_win_diff_25, max_win_diff_50)),
            "stage1_dur_max_diff": float(max_dur_diff_50),
            "gpm_p50_mean": {k: round(float(np.mean(v)), 1) for k, v in gpm_means.items()},
            "hdpm_mean": {k: round(float(np.mean(v)), 1) for k, v in hdpm_means.items()},
            "avg_items_unclamped": {k: round(float(np.mean(v)), 2) for k, v in item_lens.items()},
            "raw_items_over_25_5m_at_25m_override": round(
                raw_over_25_num / max(1, raw_over_25_den), 4
            ),
        }

    return {
        "num_evaluated": len(eval_indices),
        "win_calibration": win_cal,
        "stage1_breakdown": stage1_breakdown,
        "per_position": per_position,
        "winner_vs_loser_builds": winner_vs_loser,
        "skill_legality": skill_legality,
        "item_decode_stats": item_decode_stats,
        "duration_override_check": duration_override_check,
    }


def format_sample_comparison(
    sample_match: dict[str, Any] | None,
    vidx: VocabIndex,
    vocab: dict[str, Any],
    cfg: TrainingConfig,
    export_dir: Path,
) -> str:
    """Compare ONNX predictions (auto vs 50m) against actual outcomes & builds for one match."""
    if not sample_match:
        return "No sample_match available."

    enc_path = export_dir / "d2oracle_encoder.onnx"
    dec_path = export_dir / "d2oracle_item_decoder.onnx"
    if not enc_path.exists() or not dec_path.exists():
        return "ONNX exports not found for sample comparison."

    hero_by_id = vocab["heroes"]["by_id"]
    item_idx_to_id = vocab["items"]["idx_to_id"]
    item_by_id = vocab["items"]["by_id"]

    hero_idx_np = np.array(
        [[vidx.hero_id_to_idx.get(int(h), 0) for h in sample_match["hero_ids"]]], dtype=np.int64
    )
    is_rad_np = np.array([[1 if f else 0 for f in sample_match["is_radiant"]]], dtype=np.int64)
    pos_idx_np = np.array([[int(p) - 1 for p in sample_match["positions"]]], dtype=np.int64)

    enc_sess = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])

    def _run_mode(dur_override: float, max_min: float | None) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        outs = enc_sess.run(
            None,
            {
                "hero_idx": hero_idx_np,
                "is_radiant": is_rad_np,
                "pos_idx": pos_idx_np,
                "duration_override": np.array([dur_override], dtype=np.float32),
            },
        )
        p_ids, p_mins = decode_items_onnx_greedy(
            dec_sess,
            outs[9],
            max_steps=cfg.max_item_len - 1,
            bos_id=BOS_ITEM_ID,
            eos_id=EOS_ITEM_ID,
            eos_penalty=cfg.eos_logit_penalty,
            eos_stop_prob=cfg.eos_stop_prob,
            max_minute=max_min,
        )
        return outs, p_ids, p_mins

    auto_out, pred_ids, pred_mins = _run_mode(0.0, None)
    long_out, long_ids, long_mins = _run_mode(50.0, 50.0)
    win_prob, dur_q, lane_p, lead10, combat_r, combat_s, econ_q, _, _, _ = auto_out
    long_econ_q = long_out[6]

    dur_min = float(sample_match["duration_seconds"]) / 60.0
    lines = [
        f"Match {sample_match['match_id']} | Actual Radiant Win: {sample_match['radiant_win']} | Actual Duration: {dur_min:.1f} min",
        f"Pred Radiant Win: {win_prob[0, 0]:.1%} | Duration: {dur_q[0, 0]:.1f}-{dur_q[0, 2]:.1f}m (~{dur_q[0, 1]:.1f}m)",
    ]
    lane_pred = ", ".join(
        f"{nm}={LANE_OUTCOME_LABELS[int(np.argmax(lane_p[0, i]))]} ({np.max(lane_p[0, i]):.0%})"
        for i, nm in enumerate(("Top", "Mid", "Bot"))
    )
    act_lanes = []
    for nm, col in (("Top", "top_lane_outcome"), ("Mid", "mid_lane_outcome"), ("Bot", "bottom_lane_outcome")):
        raw_v = sample_match.get(col)
        if raw_v is None:
            continue
        lane_id = LANE_OUTCOME_TO_ID.get(str(raw_v))
        if lane_id is None and isinstance(raw_v, (int, np.integer)) and 0 <= int(raw_v) < len(LANE_OUTCOME_LABELS):
            lane_id = int(raw_v)
        if lane_id is not None:
            act_lanes.append(f"{nm}={LANE_OUTCOME_LABELS[lane_id]}")
    act_gold = sample_match.get("radiant_gold_lead_10m")
    act_xp = sample_match.get("radiant_xp_lead_10m")
    lines.append(
        f"Pred Lanes: {lane_pred} | @10m Leads: Gold={lead10[0, 0]:+.0f}, XP={lead10[0, 1]:+.0f}"
    )
    if act_lanes or act_gold is not None:
        lines.append(
            f"True Lanes: {', '.join(act_lanes) or 'n/a'} | @10m Leads: Gold={act_gold}, XP={act_xp}"
        )
    lines.append("")

    raw_item_lists = sample_match.get("key_item_ids") or [[] for _ in range(10)]
    raw_item_times = sample_match.get("key_item_times") or [[] for _ in range(10)]
    raw_gpm = sample_match.get("gpm") or [None] * 10
    raw_hdmg = sample_match.get("hero_damage") or [None] * 10

    for s in range(10):
        hid = int(sample_match["hero_ids"][s])
        hmeta = hero_by_id.get(str(hid), {})
        team = "Radiant" if sample_match["is_radiant"][s] else "Dire"
        pos = sample_match["positions"][s]
        t_idx = 0 if s < 5 else 1
        slot_t = s % 5

        def _fmt_seq(tok_arr: np.ndarray, min_arr: np.ndarray) -> str:
            items_fmt = []
            for tok, m in zip(tok_arr, min_arr, strict=True):
                ti = int(tok)
                if ti == EOS_ITEM_ID:
                    break
                if ti >= 3:
                    iid = item_idx_to_id[ti]
                    iname = item_by_id.get(str(iid), {}).get("display_name", str(iid))
                    items_fmt.append(f"{iname} (~{m:.0f}m)")
            return ", ".join(items_fmt) or "none"

        true_items_fmt = []
        for iid_raw, t_sec in zip(raw_item_lists[s], raw_item_times[s], strict=False):
            iid = vidx.item_aliases.get(int(iid_raw), int(iid_raw))
            iname = item_by_id.get(str(iid), {}).get("display_name", str(iid))
            true_items_fmt.append(f"{iname} ({float(t_sec)/60.0:.0f}m)")

        true_gpm_str = f"{raw_gpm[s]}" if raw_gpm[s] is not None and raw_gpm[s] >= 0 else "n/a"
        true_hdpm_str = (
            f"{float(raw_hdmg[s]) / max(1.0, dur_min):.0f}"
            if raw_hdmg[s] is not None and raw_hdmg[s] >= 0
            else "n/a"
        )
        gpm_rng = f"{econ_q[0, s, 0, 0]:.0f}-{econ_q[0, s, 0, 2]:.0f}"
        gpm_50 = f"{long_econ_q[0, s, 0, 1]:.0f}"
        hdpm = f"{combat_r[0, s, 0]:.0f} ({combat_s[0, t_idx, slot_t, 0]:.0%})"
        lines.append(
            f"{team:7s} P{pos} {hmeta.get('display_name', hid):16s} | "
            f"GPM:{gpm_rng} (50m:{gpm_50}, true:{true_gpm_str}) | "
            f"HDPM:{hdpm} (true:{true_hdpm_str})"
        )
        lines.append(f"  Pred(auto): {_fmt_seq(pred_ids[0, s], pred_mins[0, s])}")
        lines.append(f"  Pred(50m) : {_fmt_seq(long_ids[0, s], long_mins[0, s])}")
        lines.append(f"  True      : {', '.join(true_items_fmt) or 'none'}")

    return "\n".join(lines)


def plot_training_curves(
    history: list[dict[str, Any]],
    out_path: Path | None = None,
    *,
    show: bool = True,
) -> Path | None:
    """Plot 12-panel Stage-1/Stage-2 training & validation curves and optionally save PNG."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(17, 15))
    plots = [
        ("loss_total", "Total Normalized Loss"),
        ("win_auc", "Win ROC-AUC (all vs pro val)"),
        ("loss_win", "Win BCE Loss"),
        ("dur_p50_mae", "Duration p50 MAE (min)"),
        ("lane_acc", "Lane Outcome Accuracy"),
        ("lead10_gold_mae", "@10m Gold Lead MAE"),
        ("hero_dpm_mae", "Hero DPM MAE"),
        ("gpm_p50_mae", "GPM p50 MAE"),
        ("item_acc", "Item Next-Token Accuracy"),
        ("item_min_mae", "Item Timing MAE (min)"),
        ("skill_acc", "Skill Build Accuracy (1-18)"),
        ("talent_acc", "Talent Accuracy"),
    ]
    p2_epochs = [h["epoch"] for h in history if h.get("phase") == 2]
    phase_split = min(p2_epochs) - 0.5 if p2_epochs else None

    for ax, (key, title) in zip(axes.flat, plots, strict=True):
        tr_pts = [(h["epoch"], h["train"][key]) for h in history if key in h.get("train", {})]
        va_pts = [(h["epoch"], h["val"][key]) for h in history if key in h.get("val", {})]
        if tr_pts:
            ax.plot([p[0] for p in tr_pts], [p[1] for p in tr_pts], label="train")
        if va_pts:
            ax.plot([p[0] for p in va_pts], [p[1] for p in va_pts], label="val")
        if key == "win_auc":
            pro_pts = [
                (h["epoch"], h["val"]["pro_win_auc"])
                for h in history
                if "pro_win_auc" in h.get("val", {})
            ]
            if pro_pts:
                ax.plot(
                    [p[0] for p in pro_pts],
                    [p[1] for p in pro_pts],
                    linestyle="--",
                    label="val pro",
                )
        if phase_split is not None:
            ax.axvline(phase_split, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def _format_summary_markdown(summary: dict[str, Any]) -> str:
    ds = summary.get("dataset") or {}
    env = summary.get("environment") or {}
    best_s1 = summary.get("best_stage1") or {}
    best_all = summary.get("best_overall") or {}
    last_ep = summary.get("last_epoch") or {}
    diag = summary.get("validation_diagnostics") or {}

    lines = [
        "# d2oracle training report",
        "",
        f"- Generated UTC: `{summary.get('generated_at_utc')}`",
        f"- Patch: `{ds.get('patch')}` (gv={ds.get('game_version_id')}), shards={ds.get('num_shards')}",
        f"- Matches: total={ds.get('num_total')} (pro={ds.get('num_pro')}, ranked={ds.get('num_ranked')}), train={ds.get('num_train')}, val={ds.get('num_val')}",
        f"- Environment: `{env.get('device')}`, PyTorch `{env.get('torch_version')}`, elapsed `{env.get('train_elapsed_sec')}s`",
        f"- Export bundle size: `{summary.get('total_export_mb')} MB` | ONNX parity passed: `{((summary.get('onnx_parity') or {}).get('passed'))}`",
        "",
        "## Checkpoint Epochs",
        f"- Best Stage 1: epoch `{best_s1.get('epoch')}` | s1_score=`{(best_s1.get('metrics') or {}).get('val_selection_score')}` | win_auc=`{(best_s1.get('metrics') or {}).get('win_auc')}` | pro_win_auc=`{(best_s1.get('metrics') or {}).get('pro_win_auc')}`",
        f"- Best Overall: epoch `{best_all.get('epoch')}` | val_score=`{(best_all.get('metrics') or {}).get('val_selection_score')}` | item_acc=`{(best_all.get('metrics') or {}).get('item_acc')}` | skill_acc=`{(best_all.get('metrics') or {}).get('skill_acc')}` | talent_acc=`{(best_all.get('metrics') or {}).get('talent_acc')}`",
        f"- Last Epoch: epoch `{last_ep.get('epoch')}` | val_score=`{(last_ep.get('metrics') or {}).get('val_selection_score')}`",
    ]

    win_cal = diag.get("win_calibration") or {}
    if win_cal.get("count"):
        lines.extend(
            [
                "",
                "## Win Calibration (best checkpoint on val)",
                f"- All (n={win_cal.get('count')}, T={win_cal.get('temperature', 1.0)}): AUC={win_cal.get('win_auc')}, Acc={win_cal.get('win_acc')}, Brier={win_cal.get('win_brier')}, LogLoss={win_cal.get('win_logloss')}, ECE={win_cal.get('win_ece')}",
            ]
        )
        if "pro" in win_cal:
            pr = win_cal["pro"]
            lines.append(
                f"- Pro (n={pr.get('count')}): AUC={pr.get('win_auc')}, Acc={pr.get('win_acc')}, Brier={pr.get('win_brier')}, LogLoss={pr.get('win_logloss')}"
            )
        lines.extend(["", "| Prob Bin | Count | Avg Pred | Actual Winrate |", "|---|---:|---:|---:|"])
        for b in win_cal.get("bins") or []:
            lines.append(
                f"| {b['bin']} | {b['count']} | {b['avg_pred']} | {b['actual_winrate']} |"
            )

    s1_bd = diag.get("stage1_breakdown") or {}
    if s1_bd:
        dur = s1_bd.get("duration") or {}
        ln = s1_bd.get("lanes") or {}
        l10 = s1_bd.get("lead10") or {}
        lines.extend(
            [
                "",
                "## Duration, Lanes & @10m",
                f"- Duration p50 MAE={dur.get('p50_mae_min')}m, bias={dur.get('p50_bias_min')}m, IQR={dur.get('avg_iqr_min')}m | inside [p25,p75]={dur.get('within_p25_p75_rate')} (<p25={dur.get('below_p25_rate')}, >p75={dur.get('above_p75_rate')})",
                f"- Lanes acc: top={ln.get('top_acc')}, mid={ln.get('mid_acc')}, bot={ln.get('bot_acc')} | pred={ln.get('pred_class_share')} | true={ln.get('true_class_share')}",
                f"- @10m leads: gold_mae={l10.get('gold_mae')} (bias={l10.get('gold_bias')}), xp_mae={l10.get('xp_mae')} (bias={l10.get('xp_bias')})",
            ]
        )

    pos_map = diag.get("per_position") or {}
    if pos_map:
        lines.extend(
            [
                "",
                "## Per-Position Stage 2 (P1-P5)",
                "| Pos | GPM MAE | GPM Bias | HDPM MAE | Item Acc | Item Min MAE | Skill Acc | Talent Acc |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for p in ("P1", "P2", "P3", "P4", "P5"):
            row = pos_map.get(p) or {}
            lines.append(
                f"| {p} | {row.get('gpm_p50_mae')} | {row.get('gpm_p50_bias')} | {row.get('hero_dpm_mae')} | {row.get('item_acc')} | {row.get('item_min_mae')} | {row.get('skill_acc')} | {row.get('talent_acc')} |"
            )

    wvl = diag.get("winner_vs_loser_builds") or {}
    if wvl:
        win_m = wvl.get("winners") or {}
        lose_m = wvl.get("losers") or {}
        lines.extend(
            [
                "",
                "## Winner vs Loser Builds & Legality",
                f"- Winners: item_acc={win_m.get('item_acc')}, skill_acc={win_m.get('skill_acc')}, talent_acc={win_m.get('talent_acc')}",
                f"- Losers: item_acc={lose_m.get('item_acc')}, skill_acc={lose_m.get('skill_acc')}, talent_acc={lose_m.get('talent_acc')}",
                f"- Skill raw legality: {diag.get('skill_legality')}",
            ]
        )

    idec = diag.get("item_decode_stats") or {}
    if idec:
        lines.extend(
            [
                "",
                f"## Item Greedy Decode (n={idec.get('num_decoded_matches')} matches)",
                f"- Avg pred len: {idec.get('avg_pred_len_by_pos')}",
                f"- Avg true len: {idec.get('avg_true_len_by_pos')}",
                f"- Empty builds: {idec.get('empty_build_rate')} | Max-len hits: {idec.get('max_len_hit_rate')}",
                f"- Duration override check: {diag.get('duration_override_check')}",
                "",
                "Top predicted items: "
                + ", ".join(f"{it['name']} ({it['share']:.1%})" for it in idec.get("top_predicted_items", [])[:10]),
                "Top true items: "
                + ", ".join(f"{it['name']} ({it['share']:.1%})" for it in idec.get("top_true_items", [])[:10]),
            ]
        )

    return "\n".join(lines) + "\n"


def _first_non_none(*vals: Any) -> Any:
    for val in vals:
        if val is not None:
            return val
    return None


def _round_or_none(val: Any, ndigits: int) -> float | None:
    if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
        return round(float(val), ndigits)
    return None


def build_model_summary_metadata(
    summary_payload: dict[str, Any],
    vocab: dict[str, Any],
) -> dict[str, Any]:
    """Build compact dataset & holdout accuracy metadata for web model_config.json."""
    ds = dict(summary_payload.get("dataset") or {})
    best_overall = (summary_payload.get("best_overall") or {}).get("metrics") or {}
    best_stage1 = (summary_payload.get("best_stage1") or {}).get("metrics") or {}
    diag = summary_payload.get("validation_diagnostics") or {}
    s1_bd = diag.get("stage1_breakdown") or {}
    dur_bd = s1_bd.get("duration") or {}

    return {
        "trained_at_utc": summary_payload.get("generated_at_utc"),
        "dataset_snapshot_utc": vocab.get("generated_at"),
        "total_matches": int(ds.get("num_total") or 0),
        "ranked_matches": int(ds.get("num_ranked") or 0),
        "pro_matches": int(ds.get("num_pro") or 0),
        "train_matches": int(ds.get("num_train") or 0),
        "val_matches": int(ds.get("num_val") or 0),
        "num_shards": int(ds.get("num_shards") or 0),
        "num_heroes": int((vocab.get("heroes") or {}).get("num_heroes") or 0),
        "num_key_items": int((vocab.get("items") or {}).get("num_key_items") or 0),
        "bundle_size_mb": _round_or_none(summary_payload.get("total_export_mb"), 2),
        "win_accuracy": _round_or_none(
            _first_non_none(best_overall.get("win_acc"), best_stage1.get("win_acc")),
            4,
        ),
        "lane_accuracy": _round_or_none(
            _first_non_none(best_overall.get("lane_acc"), best_stage1.get("lane_acc")),
            4,
        ),
        "duration_mae_min": _round_or_none(
            _first_non_none(
                best_overall.get("dur_p50_mae"),
                best_stage1.get("dur_p50_mae"),
                dur_bd.get("p50_mae_min"),
            ),
            2,
        ),
        "duration_window_coverage": _round_or_none(dur_bd.get("within_p25_p75_rate"), 4),
        "skill_accuracy": _round_or_none(best_overall.get("skill_acc"), 4),
        "talent_accuracy": _round_or_none(best_overall.get("talent_acc"), 4),
        "item_accuracy": _round_or_none(best_overall.get("item_acc"), 4),
        "item_timing_mae_min": _round_or_none(best_overall.get("item_min_mae"), 2),
        "gpm_mae": _round_or_none(best_overall.get("gpm_p50_mae"), 1),
    }


def write_training_report(
    report_dir: Path,
    *,
    train_result: dict[str, Any],
    cfg: TrainingConfig,
    vocab: dict[str, Any],
    model_config: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    exported: dict[str, Path] | None = None,
    parity: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    curves_path: Path | None = None,
    sample_comparison: str | None = None,
) -> dict[str, Path]:
    """Write history, config, summary JSON/Markdown, sample prediction, and plot to report_dir."""
    report_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    history = list(train_result.get("history") or [])
    history_path = report_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    written["history"] = history_path

    effective_cfg = train_result.get("resolved_config") or cfg
    cfg_payload = {
        "training_config": effective_cfg.to_dict(),
        "model_config": model_config
        or (train_result["spec"].to_dict() if "spec" in train_result else None),
        "dataset": train_result.get("dataset_summary"),
        "manifest": manifest,
    }
    cfg_path = report_dir / "training_config.json"
    cfg_path.write_text(json.dumps(cfg_payload, indent=2, default=str), encoding="utf-8")
    written["training_config"] = cfg_path

    export_sizes = {}
    total_mb = 0.0
    if exported:
        for k, p in exported.items():
            if Path(p).exists():
                mb = round(Path(p).stat().st_size / (1024 * 1024), 4)
                export_sizes[k] = mb
                total_mb += mb

    device_desc = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    last_entry = history[-1] if history else None
    summary_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "torch_version": torch.__version__,
            "device": device_desc,
            "train_elapsed_sec": train_result.get("train_elapsed_sec"),
        },
        "dataset": train_result.get("dataset_summary"),
        "best_stage1": {
            "epoch": train_result.get("best_stage1_epoch"),
            "metrics": train_result.get("best_stage1_metrics"),
        },
        "best_overall": {
            "epoch": train_result.get("best_epoch"),
            "metrics": train_result.get("best_val_metrics"),
        },
        "last_epoch": (
            {"epoch": last_entry["epoch"], "metrics": last_entry.get("val")}
            if last_entry is not None
            else None
        ),
        "export_sizes_mb": export_sizes,
        "total_export_mb": round(total_mb, 3),
        "onnx_parity": parity,
        "validation_diagnostics": diagnostics,
    }
    summary_json_path = report_dir / "training_summary.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    written["training_summary"] = summary_json_path

    if exported and "model_config" in exported:
        mc_path = Path(exported["model_config"])
        if mc_path.exists():
            mc_payload = json.loads(mc_path.read_text(encoding="utf-8"))
            if isinstance(mc_payload, dict):
                mc_payload["model_summary"] = build_model_summary_metadata(summary_payload, vocab)
                mc_path.write_text(json.dumps(mc_payload, indent=2), encoding="utf-8")

    summary_md_path = report_dir / "summary.md"
    summary_md_path.write_text(_format_summary_markdown(summary_payload), encoding="utf-8")
    written["summary_md"] = summary_md_path

    if sample_comparison:
        sample_path = report_dir / "sample_predictions.txt"
        sample_path.write_text(sample_comparison + "\n", encoding="utf-8")
        written["sample_predictions"] = sample_path

    if curves_path is not None and curves_path.exists():
        dest_curves = report_dir / "training_curves.png"
        if curves_path.resolve() != dest_curves.resolve():
            shutil.copy2(curves_path, dest_curves)
        written["training_curves"] = dest_curves

    return written


def package_export_bundle(
    archive_path: Path | str,
    export_dir: Path,
    report_dir: Path | None = None,
) -> Path:
    """Create a zip containing ONNX/JSON model files at root and diagnostic artifacts under report/."""
    out_zip = Path(archive_path)
    if out_zip.suffix != ".zip":
        out_zip = out_zip.with_suffix(".zip")
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(export_dir.iterdir()):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
        if report_dir is not None and report_dir.exists():
            for file_path in sorted(report_dir.rglob("*")):
                if file_path.is_file():
                    rel = file_path.relative_to(report_dir)
                    zf.write(file_path, arcname=str(Path("report") / rel))
    return out_zip
