"""Export trained d2oracle two-stage encoder and autoregressive item decoder to ONNX."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import torch

from training.config import ModelSpec
from training.dataset import VocabIndex
from training.model import D2OracleModel

logger = logging.getLogger("d2oracle.export")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    vocab: dict[str, Any] | None = None,
) -> tuple[D2OracleModel, ModelSpec, dict[str, Any]]:
    """Load trained D2OracleModel, ModelSpec, and vocab from checkpoint directory."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    spec = ModelSpec.from_dict(ckpt["spec"])

    if vocab is None:
        vocab_file = checkpoint_path.parent / "vocab.json"
        if not vocab_file.exists():
            raise FileNotFoundError(
                f"vocab.json not found next to checkpoint at {vocab_file}"
            )
        vocab = json.loads(vocab_file.read_text(encoding="utf-8"))

    vidx = VocabIndex(vocab)
    skill_mask = vidx.build_skill_mask()
    model = D2OracleModel(spec, skill_mask)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, spec, vocab


def _onnx_export_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "opset_version": 17,
        "do_constant_folding": True,
    }
    sig = inspect.signature(torch.onnx.export)
    if "dynamo" in sig.parameters:
        kwargs["dynamo"] = False
    return kwargs


def export_to_onnx(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    vocab_path: Path | None = None,
) -> dict[str, Path]:
    """Export d2oracle_encoder.onnx, d2oracle_item_decoder.onnx, model_config.json, and vocab.json."""
    vocab = (
        json.loads(vocab_path.read_text(encoding="utf-8"))
        if vocab_path is not None
        else None
    )
    model, spec, vocab = load_model_from_checkpoint(checkpoint_path, vocab=vocab)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_path = output_dir / "d2oracle_encoder.onnx"
    decoder_path = output_dir / "d2oracle_item_decoder.onnx"
    config_path = output_dir / "model_config.json"
    out_vocab_path = output_dir / "vocab.json"

    export_kwargs = _onnx_export_kwargs()

    # 1. Export two-stage draft encoder (include both auto 0.0 and override > 0 in dummy batch)
    dummy_hero = torch.ones((2, 10), dtype=torch.long)
    dummy_rad = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * 2, dtype=torch.long)
    dummy_pos = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]] * 2, dtype=torch.long)
    dummy_dur_override = torch.tensor([0.0, 45.0], dtype=torch.float32)

    encoder_outputs = [
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
    ]
    dynamic_axes = {
        name: {0: "batch"}
        for name in (
            "hero_idx",
            "is_radiant",
            "pos_idx",
            "duration_override",
            *encoder_outputs,
        )
    }

    torch.onnx.export(
        model.encoder,
        (dummy_hero, dummy_rad, dummy_pos, dummy_dur_override),
        str(encoder_path),
        input_names=["hero_idx", "is_radiant", "pos_idx", "duration_override"],
        output_names=encoder_outputs,
        dynamic_axes=dynamic_axes,
        **export_kwargs,
    )

    # 2. Export autoregressive item decoder
    dummy_mem = torch.zeros((2, 10, spec.d_model), dtype=torch.float32)
    dummy_slot = torch.tensor([0, 5], dtype=torch.long)
    dummy_seq = torch.tensor([[1, 3, 4], [1, 3, 0]], dtype=torch.long)

    torch.onnx.export(
        model.item_decoder,
        (dummy_mem, dummy_slot, dummy_seq),
        str(decoder_path),
        input_names=["slot_memory", "target_slot_idx", "item_seq"],
        output_names=["next_item_logits", "next_item_minutes"],
        dynamic_axes={
            "slot_memory": {0: "batch"},
            "target_slot_idx": {0: "batch"},
            "item_seq": {0: "batch", 1: "seq_len"},
            "next_item_logits": {0: "batch", 1: "seq_len"},
            "next_item_minutes": {0: "batch", 1: "seq_len"},
        },
        **export_kwargs,
    )

    config_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
    out_vocab_path.write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )

    logger.info("Exported ONNX bundle to %s", output_dir)
    return {
        "encoder": encoder_path,
        "decoder": decoder_path,
        "model_config": config_path,
        "vocab": out_vocab_path,
    }


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="d2oracle-export-onnx")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("exports"))
    parser.add_argument("--vocab", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    export_to_onnx(args.checkpoint, args.output_dir, vocab_path=args.vocab)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
