"""PyTorch vs ONNX Runtime numerical and greedy-decoding parity verification."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from training.config import DPM_HPM_SCALE, GOLD_XP_LEAD_SCALE, GPM_XPM_SCALE
from training.dataset import BOS_ITEM_ID, EOS_ITEM_ID
from training.export_onnx import load_model_from_checkpoint

logger = logging.getLogger("d2oracle.verify_onnx")

ENCODER_OUTPUT_NAMES: tuple[str, ...] = (
    "win_prob",
    "duration_quantiles",
    "lane_probs",
    "lead10",
    "combat_rates",
    "combat_shares",
    "econ_kda_quantiles",
    "skill_logits",
    "talent_logits",
    "slot_memory",
)


def _output_scale_array(name: str) -> float | np.ndarray:
    """Return normalization divisor so float32 ULP checks run in O(1) units."""
    if name == "lead10":
        return GOLD_XP_LEAD_SCALE
    if name == "combat_rates":
        return DPM_HPM_SCALE
    if name == "econ_kda_quantiles":
        return np.array(
            [GPM_XPM_SCALE, GPM_XPM_SCALE, 1.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )[None, None, :, None]
    return 1.0


def decode_items_onnx_greedy(
    decoder_sess: ort.InferenceSession,
    slot_memory: np.ndarray,
    *,
    max_steps: int,
    bos_id: int = BOS_ITEM_ID,
    eos_id: int = EOS_ITEM_ID,
    eos_penalty: float = 0.0,
    eos_stop_prob: float = 0.0,
    max_minute: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run autoregressive greedy item decoding using ONNX Runtime."""
    bsz, _, d = slot_memory.shape
    expanded_memory = np.repeat(slot_memory[:, None, :, :], 10, axis=1).reshape(bsz * 10, 10, d)
    target_slots = np.tile(np.arange(10, dtype=np.int64), bsz)
    eff_max_min = float(max_minute) if (max_minute is not None and max_minute > 0.0) else None

    n_streams = bsz * 10
    seq = np.full((n_streams, 1), bos_id, dtype=np.int64)
    finished = np.zeros((n_streams,), dtype=bool)
    last_min = np.zeros((n_streams,), dtype=np.float32)
    stream_rows = np.arange(n_streams, dtype=np.int64)
    pred_ids: list[np.ndarray] = []
    pred_mins: list[np.ndarray] = []

    for step_idx in range(max_steps):
        logits, mins = decoder_sess.run(
            None,
            {
                "slot_memory": expanded_memory.astype(np.float32),
                "target_slot_idx": target_slots,
                "item_seq": seq,
            },
        )
        step_logits = logits[:, -1, :].copy()
        step_logits[:, 0] = -1e9
        step_logits[:, bos_id] = -1e9
        for row_i in range(n_streams):
            for prev_tok in seq[row_i, 1:]:
                if prev_tok >= 3:
                    step_logits[row_i, prev_tok] = -1e9
        if eos_stop_prob > 0.0:
            row_max = np.max(step_logits, axis=-1, keepdims=True)
            exp_s = np.exp(np.clip(step_logits - row_max, -80.0, 0.0))
            p_eos = exp_s[:, eos_id] / np.sum(exp_s, axis=-1)
            has_item = np.any(step_logits[:, 3:] > -1e8, axis=-1)
            block_eos = has_item & (p_eos < eos_stop_prob)
            step_logits[block_eos, eos_id] = -1e9
        if eos_penalty != 0.0:
            step_logits[:, eos_id] -= eos_penalty

        cand_tok = np.argmax(step_logits, axis=-1).astype(np.int64)
        next_tok = np.where(finished, eos_id, cand_tok).astype(np.int64)
        raw_min = mins[stream_rows, -1, next_tok].astype(np.float32)
        mono_min = np.maximum(last_min, raw_min)
        if step_idx > 0 and eff_max_min is not None:
            over_budget = (~finished) & (next_tok != eos_id) & (mono_min > eff_max_min)
            next_tok = np.where(over_budget, eos_id, next_tok).astype(np.int64)
        stop_now = finished | (next_tok == eos_id)
        next_min = np.where(stop_now, np.float32(0.0), mono_min).astype(np.float32)
        finished = stop_now
        last_min = np.where(finished, last_min, next_min).astype(np.float32)

        pred_ids.append(next_tok)
        pred_mins.append(next_min)
        seq = np.concatenate([seq, next_tok[:, None]], axis=1)
        if bool(np.all(finished)):
            break

    out_ids = np.stack(pred_ids, axis=1).reshape(bsz, 10, -1)
    out_mins = np.stack(pred_mins, axis=1).reshape(bsz, 10, -1)
    return out_ids, out_mins


def verify_onnx_parity(
    checkpoint_path: Path,
    export_dir: Path,
    *,
    vocab_path: Path | None = None,
    atol: float = 1e-4,
    seed: int = 123,
) -> dict[str, Any]:
    """Verify ONNX outputs match PyTorch across auto and scenario-overridden duration modes."""
    vocab = (
        json.loads(vocab_path.read_text(encoding="utf-8"))
        if vocab_path is not None
        else None
    )
    model, spec, vocab = load_model_from_checkpoint(checkpoint_path, vocab=vocab)

    enc_path = export_dir / "d2oracle_encoder.onnx"
    dec_path = export_dir / "d2oracle_item_decoder.onnx"
    onnx.checker.check_model(onnx.load(str(enc_path)))
    onnx.checker.check_model(onnx.load(str(dec_path)))

    enc_sess = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(seed)
    valid_hero_indices = sorted(int(v) for v in vocab["heroes"]["id_to_idx"].values())
    max_diffs: dict[str, float] = {}

    for bsz in (1, 3):
        hero_idx_np = rng.choice(valid_hero_indices, size=(bsz, 10), replace=True).astype(np.int64)
        is_rad_np = np.tile(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int64), (bsz, 1))
        pos_idx_np = np.tile(np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64), (bsz, 1))

        mode_outputs: dict[float, dict[str, np.ndarray]] = {}
        for override_val in (0.0, 22.0, 50.0):
            override_np = np.full((bsz,), override_val, dtype=np.float32)

            with torch.no_grad():
                pt_tuple = model.encoder(
                    torch.from_numpy(hero_idx_np),
                    torch.from_numpy(is_rad_np),
                    torch.from_numpy(pos_idx_np),
                    torch.from_numpy(override_np),
                )

            ort_tuple = enc_sess.run(
                None,
                {
                    "hero_idx": hero_idx_np,
                    "is_radiant": is_rad_np,
                    "pos_idx": pos_idx_np,
                    "duration_override": override_np,
                },
            )
            ort_by_name = dict(zip(ENCODER_OUTPUT_NAMES, ort_tuple, strict=True))
            mode_outputs[override_val] = ort_by_name

            for name, pt_tensor, ort_arr in zip(
                ENCODER_OUTPUT_NAMES, pt_tuple, ort_tuple, strict=True
            ):
                scale = _output_scale_array(name)
                norm_diff = float(np.max(np.abs(pt_tensor.numpy() - ort_arr) / scale))
                max_diffs[name] = max(max_diffs.get(name, 0.0), norm_diff)
                if norm_diff > atol:
                    raise AssertionError(
                        f"ONNX encoder mismatch on '{name}' (B={bsz}, override={override_val}): "
                        f"max_norm_diff={norm_diff:.6e} > {atol}"
                    )

            # Structural invariants check
            dur_q = ort_by_name["duration_quantiles"]
            assert np.all(dur_q[:, 0] >= 15.0 - 1e-5)
            assert np.all(dur_q[:, 0] <= dur_q[:, 1] + 1e-5)
            assert np.all(dur_q[:, 1] <= dur_q[:, 2] + 1e-5)

            econ_q = ort_by_name["econ_kda_quantiles"]
            assert np.all(econ_q >= -1e-5)
            assert np.all(econ_q[..., 0] <= econ_q[..., 1] + 1e-4)
            assert np.all(econ_q[..., 1] <= econ_q[..., 2] + 1e-4)

            assert np.all(ort_by_name["combat_rates"] >= -1e-5)
            assert np.allclose(ort_by_name["lane_probs"].sum(axis=-1), 1.0, atol=1e-4)
            assert np.allclose(ort_by_name["combat_shares"].sum(axis=2), 1.0, atol=1e-4)

            ort_skill = ort_by_name["skill_logits"]
            for b in range(bsz):
                for s in range(10):
                    h_idx = int(hero_idx_np[b, s])
                    expected_mask = model.encoder.skill_mask_bias[h_idx].numpy()
                    illegal = expected_mask < -1000.0
                    if np.any(illegal):
                        assert float(np.max(ort_skill[b, s][illegal])) < -5000.0

        # Verify Stage 1 invariance & Stage 2 scenario responsiveness across duration_override
        auto_out = mode_outputs[0.0]
        short_out = mode_outputs[22.0]
        long_out = mode_outputs[50.0]
        for s1_key in ("win_prob", "duration_quantiles", "lane_probs", "lead10"):
            s1_atol = 1e-3 if s1_key == "lead10" else 1e-6
            if not np.allclose(auto_out[s1_key], long_out[s1_key], atol=s1_atol):
                raise AssertionError(f"duration_override mutated Stage-1 output '{s1_key}'")

        slot_delta = float(np.max(np.abs(long_out["slot_memory"] - short_out["slot_memory"])))
        if slot_delta < 1e-5:
            raise AssertionError(
                f"Stage-2 slot_memory did not respond to duration_override (delta={slot_delta})"
            )

        # Verify decoder & greedy decoding across auto (0.0), short (22.0), and long (50.0) modes
        for override_val in (0.0, 22.0, 50.0):
            ort_mem = mode_outputs[override_val]["slot_memory"]
            pt_mem = torch.from_numpy(ort_mem)

            for seq_len in (1, 5, min(10, spec.max_item_len)):
                n_dec = bsz * 2
                mem_np = ort_mem[:1].repeat(n_dec, axis=0)
                slots_np = rng.integers(0, 10, size=(n_dec,), dtype=np.int64)
                seq_np = np.ones((n_dec, seq_len), dtype=np.int64)
                if seq_len > 1 and spec.item_vocab_size > 3:
                    seq_np[:, 1:] = rng.integers(3, spec.item_vocab_size, size=(n_dec, seq_len - 1))

                with torch.no_grad():
                    pt_ilog, pt_imin = model.item_decoder(
                        torch.from_numpy(mem_np),
                        torch.from_numpy(slots_np),
                        torch.from_numpy(seq_np),
                    )
                ort_ilog, ort_imin = dec_sess.run(
                    None,
                    {
                        "slot_memory": mem_np,
                        "target_slot_idx": slots_np,
                        "item_seq": seq_np,
                    },
                )
                for name, (pt_arr, ort_arr) in (
                    ("next_item_logits", (pt_ilog.numpy(), ort_ilog)),
                    ("next_item_minutes", (pt_imin.numpy(), ort_imin)),
                ):
                    diff = float(np.max(np.abs(pt_arr - ort_arr)))
                    max_diffs[name] = max(max_diffs.get(name, 0.0), diff)
                    if diff > atol:
                        raise AssertionError(
                            f"ONNX decoder mismatch on '{name}' (T={seq_len}, override={override_val}): "
                            f"max_abs_diff={diff:.6e} > {atol}"
                        )

            max_min_arg = override_val if override_val > 0.0 else None
            pt_ids, pt_mins = model.decode_items_greedy(
                pt_mem,
                max_steps=spec.max_item_len - 1,
                eos_penalty=spec.eos_logit_penalty,
                eos_stop_prob=spec.eos_stop_prob,
                max_minute=max_min_arg,
            )
            ort_ids, ort_mins = decode_items_onnx_greedy(
                dec_sess,
                ort_mem,
                max_steps=spec.max_item_len - 1,
                eos_penalty=spec.eos_logit_penalty,
                eos_stop_prob=spec.eos_stop_prob,
                max_minute=max_min_arg,
            )
            if not np.array_equal(pt_ids.numpy(), ort_ids):
                raise AssertionError(
                    f"Greedy item sequence mismatch (B={bsz}, override={override_val}):\n"
                    f"PyTorch={pt_ids.numpy()}\nONNX={ort_ids}"
                )
            if max_min_arg is not None and ort_mins.shape[2] > 1:
                assert float(np.max(ort_mins[:, :, 1:])) <= max_min_arg + 1e-4
            greedy_min_diff = float(np.max(np.abs(pt_mins.numpy() - ort_mins)))
            max_diffs["greedy_item_minutes"] = max(
                max_diffs.get("greedy_item_minutes", 0.0), greedy_min_diff
            )
            if greedy_min_diff > atol:
                raise AssertionError(
                    f"Greedy item minutes mismatch (B={bsz}, override={override_val}): "
                    f"max_abs_diff={greedy_min_diff:.6e} > {atol}"
                )

    logger.info("ONNX parity PASSED (max_norm_diffs=%s)", max_diffs)
    return {"passed": True, "max_diffs": max_diffs}


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="d2oracle-verify-onnx")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--export-dir", type=Path, default=Path("exports"))
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    result = verify_onnx_parity(args.checkpoint, args.export_dir, atol=args.atol)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
