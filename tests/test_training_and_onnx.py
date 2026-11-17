"""Training and ONNX parity tests."""

from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from collector.validator import validate_and_transform_match
from shared.constants import (
    ATTRIBUTE_BONUS_ABILITY_ID,
    MISSING_STAT_SENTINEL,
    PARQUET_ARROW_SCHEMA,
)
from tests.conftest import make_raw_stratz_match
from training.config import (
    ENCODER_ARCH_VERSION,
    MODEL_SCHEMA_VERSION,
    ModelSpec,
    TrainingConfig,
)
from training.dataset import (
    BOS_ITEM_ID,
    EOS_ITEM_ID,
    IGNORE_INDEX,
    PAD_ITEM_ID,
    DotaDraftDataset,
    TensorBatchLoader,
    VocabIndex,
    build_streaming_datasets,
    download_hf_dataset,
    infer_zero_talent_tiers,
    load_parquet_rows,
    split_rows_train_val,
)
from training.diagnostics import (
    collect_validation_diagnostics,
    format_sample_comparison,
    package_export_bundle,
    write_training_report,
)
from training.export_onnx import export_to_onnx
from training.losses import (
    compute_batch_metrics,
    compute_multitask_loss,
    compute_multitask_loss_tensors,
)
from training.model import CausalSkillHead, D2OracleEncoder, D2OracleModel
from training.train import (
    _compute_val_selection_score,
    _optimizer_step,
    build_model_spec,
    compute_stage1_val_score,
    compute_stage2_val_score,
    train_from_datasets,
    train_from_rows,
    train_from_shards,
)
from training.verify_onnx import verify_onnx_parity


def _make_training_row(
    *,
    match_id: int,
    start_datetime: int,
    radiant_win: bool = True,
    duration_seconds: int = 2100,
    is_pro: bool = False,
    top_lane_outcome: str | None = "RADIANT_VICTORY",
    mid_lane_outcome: str | None = "TIE",
    bottom_lane_outcome: str | None = "DIRE_STOMP",
    gold_lead_10m: int | None = 1400,
    xp_lead_10m: int | None = -600,
    dmg_taken_override: list[int] | None = None,
    healing_override: list[int] | None = None,
) -> dict[str, Any]:
    """Create a synthetic v2 Parquet-compatible row valid for sample_vocab."""
    hero_ids = [1, 155, 1, 155, 1, 155, 1, 155, 1, 155]
    dmg_taken = (
        dmg_taken_override
        if dmg_taken_override is not None
        else [18000, 14000, 26000, 12000, 10000, 19000, 15000, 24000, 13000, 11000]
    )
    healing = (
        healing_override
        if healing_override is not None
        else [0, 3200, 400, 1800, 4200, 0, 100, 50, 100, 50]
    )
    return {
        "match_id": match_id,
        "start_datetime": start_datetime,
        "duration_seconds": duration_seconds,
        "radiant_win": radiant_win,
        "game_version_id": 190,
        "game_version_name": "7.41f",
        "game_mode": "CAPTAINS_MODE" if is_pro else "ALL_PICK_RANKED",
        "lobby_type": "RANKED",
        "bracket": 8,
        "rank": 80,
        "average_rank": 80,
        "actual_rank": 80,
        "league_id": 19222 if is_pro else 0,
        "is_pro": is_pro,
        "source": "stratz_pro" if is_pro else "opendota_public",
        "top_lane_outcome": top_lane_outcome,
        "mid_lane_outcome": mid_lane_outcome,
        "bottom_lane_outcome": bottom_lane_outcome,
        "radiant_gold_lead_10m": gold_lead_10m,
        "radiant_xp_lead_10m": xp_lead_10m,
        "player_slots": [0, 1, 2, 3, 4, 128, 129, 130, 131, 132],
        "is_radiant": [True] * 5 + [False] * 5,
        "hero_ids": hero_ids,
        "positions": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "variant": [1] * 10,
        "kills": [8, 9, 5, 3, 2, 6, 7, 4, 2, 1],
        "deaths": [3, 4, 6, 7, 8, 5, 5, 7, 8, 9],
        "assists": [9, 10, 14, 18, 19, 8, 9, 11, 15, 17],
        "last_hits": [310, 260, 195, 75, 45, 290, 240, 180, 65, 35],
        "denies": [18, 15, 11, 3, 2, 16, 12, 9, 2, 1],
        "gpm": [680, 610, 490, 340, 290, 640, 580, 460, 310, 270],
        "xpm": [760, 740, 580, 420, 360, 720, 700, 540, 390, 330],
        "networth": [23000, 20500, 16500, 11000, 9200, 21500, 19200, 15200, 10200, 8600],
        "levels": [24, 24, 21, 18, 17, 23, 23, 20, 17, 16],
        "hero_damage": [26000, 29000, 19000, 12000, 9000, 24000, 25000, 17000, 11000, 8000],
        "tower_damage": [7500, 3200, 2100, 400, 200, 4100, 2000, 1500, 300, 100],
        "hero_healing": healing,
        "hero_damage_received": dmg_taken,
        "final_items": [[63, 1, 116, 204, 0, 0] for _ in range(10)],
        "purchase_item_ids": [[63, 1, 116, 204] for _ in range(10)],
        "purchase_times": [[420, 780, 1200, 1600] for _ in range(10)],
        "key_item_ids": [[63, 1, 116, 204] for _ in range(10)],
        "key_item_times": [[420, 780, 1200, 1600] for _ in range(10)],
        "ability_learn_ids": [
            [
                5003,
                5004,
                7314,
                5003,
                5004,
                5006,
                1495,
                6800,
                6606,
                6012,
                0,
                ATTRIBUTE_BONUS_ABILITY_ID,
            ]
            if hid == 1
            else [1659, 1660, 1661, 1659, 1660, 1662, 8001]
            for hid in hero_ids
        ],
        "ability_learn_times": [
            list(range(60, 12 * 60 + 1, 60)) if hid == 1 else list(range(60, 7 * 60 + 1, 60))
            for hid in hero_ids
        ],
        "ability_learn_levels": [
            [1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 16, 18]
            if hid == 1
            else [1, 2, 3, 4, 5, 6, 10]
            for hid in hero_ids
        ],
        "ability_learn_ranks": [
            [1] * 12 if hid == 1 else [1] * 7 for hid in hero_ids
        ],
        "ability_learn_is_talent": [
            [False, False, False, False, False, False, True, True, True, True, False, False]
            if hid == 1
            else [False, False, False, False, False, False, True]
            for hid in hero_ids
        ],
    }


def test_vocab_index_masks_and_duplicate_talent_names(sample_vocab: dict[str, Any]) -> None:
    vocab = copy.deepcopy(sample_vocab)
    vocab["heroes"]["by_id"]["1"]["skill_ability_ids"] = [5003, 5004, 7314]
    vocab["heroes"]["by_id"]["1"]["talent_tiers"]["15"] = [6800, 999901, 665]
    vocab["abilities"]["by_id"]["999901"] = {
        "id": 999901,
        "name": "special_bonus_dup_test",
        "is_talent": True,
        "display_name": "+10 Test",
    }
    vocab["abilities"]["by_id"]["6800"] = {
        "id": 6800,
        "name": "special_bonus_dup_test",
        "is_talent": True,
        "display_name": "+10 Test",
    }

    vidx = VocabIndex(vocab)
    mask = vidx.build_skill_mask()
    assert mask.shape == (vidx.num_heroes + 1, 18, vidx.num_skill_classes)

    am_idx = vidx.hero_id_to_idx[1]
    assert torch.all(mask[am_idx, :9, :3])
    assert not torch.any(mask[am_idx, :, 3])
    assert not torch.any(mask[am_idx, :9, vidx.slot_talent])
    assert not torch.any(mask[am_idx, 16, :])  # Standard heroes gain no point at level 17 (idx 16)
    active_talent_lvls = [i for i in range(9, 18) if i != 16]
    active_all_lvls = [i for i in range(18) if i != 16]
    assert torch.all(mask[am_idx, active_talent_lvls, vidx.slot_talent])
    assert torch.all(mask[am_idx, active_all_lvls, vidx.slot_attr])

    choices = {vidx.hero_talent_id_to_choice[1][tid] for tid in (6800, 999901, 665)}
    assert choices == {(1, 0), (1, 1)}

    tok_204 = vidx.map_item_id(204)
    assert tok_204 is not None and tok_204 >= 3
    assert tok_204 == vidx.map_item_id(104)
    assert vidx.map_item_id(999999) is None


def test_schema_version_rejection_and_leak_free_split(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    vidx = VocabIndex(sample_vocab)
    cfg = TrainingConfig(d_model=32, nhead=4)
    spec = build_model_spec(vidx, cfg)
    assert spec.model_schema_version == MODEL_SCHEMA_VERSION

    bad_spec = spec.to_dict()
    bad_spec["model_schema_version"] = 1
    with pytest.raises(ValueError, match="model_schema_version"):
        ModelSpec.from_dict(bad_spec)

    bad_vocab = copy.deepcopy(sample_vocab)
    bad_vocab["schema_version"] = 999
    with pytest.raises(ValueError, match="vocab schema_version"):
        VocabIndex(bad_vocab)

    # Reject legacy Parquet missing v2 columns
    old_shard = tmp_path / "legacy_v1.parquet"
    pq.write_table(pa.Table.from_pylist([{"match_id": 1}]), old_shard)
    with pytest.raises(ValueError, match="missing columns"):
        load_parquet_rows([old_shard])

    # Deduplicate duplicate match_id across shards and enforce zero leakage
    row1 = _make_training_row(match_id=101, start_datetime=1000)
    row1_dup = _make_training_row(match_id=101, start_datetime=1000)
    row2 = _make_training_row(match_id=102, start_datetime=2000)
    row3 = _make_training_row(match_id=103, start_datetime=3000)
    train_rows, val_rows = split_rows_train_val([row1, row1_dup, row2, row3], val_split=0.33)
    assert len(train_rows) == 2
    assert len(val_rows) == 1
    assert val_rows[0]["match_id"] == 103
    assert {r["match_id"] for r in train_rows}.isdisjoint({r["match_id"] for r in val_rows})


def test_dataset_encoding_and_stage2_target_masks(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    vidx = VocabIndex(sample_vocab)

    raw1 = make_raw_stratz_match(match_id=101, start_datetime=1794580000, did_radiant_win=True)
    raw2 = make_raw_stratz_match(match_id=102, start_datetime=1794590000, did_radiant_win=False)
    rec1 = validate_and_transform_match(
        raw1, active_gv_id=190, active_gv_name="7.41f", source="opendota_public", vocab=sample_vocab
    ).record
    rec2 = validate_and_transform_match(
        raw2, active_gv_id=190, active_gv_name="7.41f", source="opendota_public", vocab=sample_vocab
    ).record
    assert rec1 is not None and rec2 is not None

    shard_path = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist([rec1, rec2], schema=PARQUET_ARROW_SCHEMA), shard_path)
    rows = load_parquet_rows([shard_path])
    train_rows, val_rows = split_rows_train_val(rows, val_split=0.5, seed=42)
    assert len(train_rows) == 1 and len(val_rows) == 1
    assert val_rows[0]["match_id"] == 102

    # Build row with:
    # - slot 2 hero_damage_received = -1 (Radiant damaged_taken share masked, Dire share still valid)
    # - Radiant heal sum = 9600 >= 500 (share valid), Dire heal sum = 300 < 500 (share masked, HPM still valid)
    # - gold lead None -> masked; xp lead present -> unmasked
    rich_row = _make_training_row(
        match_id=200,
        start_datetime=1794600000,
        duration_seconds=2100,  # 35.0 min
        top_lane_outcome="RADIANT_STOMP",
        mid_lane_outcome="TIE",
        bottom_lane_outcome=None,
        gold_lead_10m=None,
        xp_lead_10m=-850,
        dmg_taken_override=[
            18000,
            14000,
            MISSING_STAT_SENTINEL,
            12000,
            10000,
            19000,
            15000,
            24000,
            13000,
            11000,
        ],
        healing_override=[0, 3200, 400, 1800, 4200, 0, 100, 50, 100, 50],
    )
    ds = DotaDraftDataset([rich_row], vidx, max_item_len=8)
    sample = ds[0]

    assert "is_pro" not in sample
    assert sample["hero_idx"].shape == (10,)
    assert sample["item_input_ids"][0, 0].item() == BOS_ITEM_ID
    assert sample["item_target_ids"][0, 4].item() == EOS_ITEM_ID
    assert torch.all(sample["item_target_ids"][0, 5:] == PAD_ITEM_ID)
    assert torch.all(sample["item_time_mask"][0, :4] == 1.0)
    assert torch.all(sample["item_time_mask"][0, 4:] == 0.0)
    assert pytest.approx(sample["item_target_minutes"][0, 0].item(), rel=1e-4) == 7.0

    # Lane & @10m targets
    assert sample["lane_targets"].tolist() == [0, 1, IGNORE_INDEX]
    assert sample["lead10_mask"].tolist() == [0.0, 1.0]
    assert pytest.approx(sample["lead10_targets"][1].item(), rel=1e-4) == -850.0

    # Combat DPM / HPM & team shares
    # Slot 2 had hero_damage_received == -1 -> rate masked (0.0) & Radiant damage_taken share masked
    assert sample["combat_rate_mask"][2, 2].item() == 0.0
    assert sample["combat_rate_mask"][0, 2].item() == 1.0
    assert sample["combat_share_mask"][0, 2].item() == 0.0
    assert sample["combat_share_mask"][1, 2].item() == 1.0
    # Dire heal sum = 300 < 500 -> Dire healing share masked, Radiant valid, Dire HPM still unmasked
    assert sample["combat_share_mask"][0, 3].item() == 1.0
    assert sample["combat_share_mask"][1, 3].item() == 0.0
    assert torch.all(sample["combat_rate_mask"][:, 3] == 1.0)
    assert pytest.approx(sample["combat_shares"][0, :, 0].sum().item(), abs=1e-5) == 1.0

    # Econ/KDA rates at 35 min (3.5 x 10m): slot 0 kills=8 -> 8/3.5, last_hits=310 -> 310/35
    assert pytest.approx(sample["econ_kda_targets"][0, 0].item(), rel=1e-4) == 680.0
    assert pytest.approx(sample["econ_kda_targets"][0, 2].item(), rel=1e-4) == 310.0 / 35.0
    assert pytest.approx(sample["econ_kda_targets"][0, 3].item(), rel=1e-4) == 8.0 / 3.5

    # Skills & talents (aid=0 at 10<=lvl<=18 on active levels maps to slot_talent; lvl 17 skipped)
    assert sample["skill_targets"][0, 9].item() == vidx.slot_talent
    assert sample["skill_targets"][0, 15].item() == vidx.slot_talent
    assert sample["skill_targets"][0, 16].item() == IGNORE_INDEX
    assert sample["skill_targets"][0, 17].item() == vidx.slot_attr
    assert sample["talent_targets"][0].tolist() == [0, 0, 0, 0]


def test_encoder_twostage_invariants_and_gradient_isolation(
    sample_vocab: dict[str, Any],
) -> None:
    vidx = VocabIndex(sample_vocab)
    cfg = TrainingConfig(d_model=32, nhead=4, num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=64, dropout=0.0)
    spec = build_model_spec(vidx, cfg)
    enc = D2OracleEncoder(spec, vidx.build_skill_mask())
    enc.eval()
    assert not hasattr(enc, "pro_emb")

    hero_idx = torch.tensor([[vidx.hero_id_to_idx[1]] * 5 + [vidx.hero_id_to_idx[155]] * 5])
    is_rad = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    pos_idx = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])

    out_auto = enc(hero_idx, is_rad, pos_idx, torch.tensor([0.0]))
    out_long = enc(hero_idx, is_rad, pos_idx, torch.tensor([50.0]))
    (
        win_prob,
        dur_q,
        lane_probs,
        lead10,
        combat_rates,
        combat_shares,
        econ_q,
        skill_logits,
        talent_logits,
        slot_mem,
    ) = out_auto

    assert win_prob.shape == (1, 1) and 0.0 <= win_prob.item() <= 1.0
    assert dur_q.shape == (1, 3) and dur_q[0, 0] >= 15.0 and dur_q[0, 0] <= dur_q[0, 1] <= dur_q[0, 2]
    assert lane_probs.shape == (1, 3, 3) and torch.allclose(lane_probs.sum(dim=-1), torch.ones(1, 3), atol=1e-5)
    assert lead10.shape == (1, 2)
    assert combat_rates.shape == (1, 10, 4) and torch.all(combat_rates >= 0.0)
    assert combat_shares.shape == (1, 2, 5, 4) and torch.allclose(combat_shares.sum(dim=2), torch.ones(1, 2, 4), atol=1e-5)
    assert econ_q.shape == (1, 10, 6, 3) and torch.all(econ_q >= 0.0)
    assert torch.all(econ_q[..., 0] <= econ_q[..., 1]) and torch.all(econ_q[..., 1] <= econ_q[..., 2])
    assert talent_logits.shape == (1, 10, 4, 2)
    assert slot_mem.shape == (1, 10, 32)
    assert torch.all(skill_logits[0, :, :9, vidx.slot_talent] < -5000.0)

    # duration_override=50.0 must preserve Stage-1 outputs identically while modifying Stage-2 slot_memory
    for idx_s1 in range(4):
        assert torch.allclose(out_auto[idx_s1], out_long[idx_s1], atol=1e-6)
    assert float(torch.max(torch.abs(out_auto[-1] - out_long[-1])).item()) > 1e-5

    # Stage-2 backward pass must never propagate gradients into Stage-1 heads (detached scenario)
    enc.train()
    enc.zero_grad(set_to_none=True)
    full = enc.forward_with_logits(hero_idx, is_rad, pos_idx)
    stage2_only_loss = (
        full["combat_rates"].sum()
        + full["econ_kda_quantiles"].sum()
        + full["combat_share_logits"].sum()
        + full["skill_logits"].sum()
        + full["talent_logits"].sum()
    )
    stage2_only_loss.backward()
    for s1_mod in (enc.win_head, enc.duration_head, enc.lane_head, enc.lead10_head):
        for p in s1_mod.parameters():
            assert p.grad is None or torch.all(p.grad == 0)


def test_multitask_loss_all_masked_targets_no_nan(sample_vocab: dict[str, Any]) -> None:
    vidx = VocabIndex(sample_vocab)
    cfg = TrainingConfig(
        d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64, max_item_len=6
    )
    spec = build_model_spec(vidx, cfg)
    model = D2OracleModel(spec, vidx.build_skill_mask())

    bsz, T = 2, 6
    batch = {
        "hero_idx": torch.ones((bsz, 10), dtype=torch.long),
        "is_radiant": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * bsz, dtype=torch.long),
        "pos_idx": torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]] * bsz, dtype=torch.long),
        "radiant_win": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "duration_min": torch.tensor([35.0, 42.0], dtype=torch.float32),
        "lane_targets": torch.full((bsz, 3), IGNORE_INDEX, dtype=torch.long),
        "lead10_targets": torch.zeros((bsz, 2), dtype=torch.float32),
        "lead10_mask": torch.zeros((bsz, 2), dtype=torch.float32),
        "combat_rates": torch.zeros((bsz, 10, 4), dtype=torch.float32),
        "combat_rate_mask": torch.zeros((bsz, 10, 4), dtype=torch.float32),
        "combat_shares": torch.full((bsz, 2, 5, 4), 0.2, dtype=torch.float32),
        "combat_share_mask": torch.zeros((bsz, 2, 4), dtype=torch.float32),
        "econ_kda_targets": torch.zeros((bsz, 10, 6), dtype=torch.float32),
        "econ_kda_mask": torch.zeros((bsz, 10, 6), dtype=torch.float32),
        "item_input_ids": torch.full((bsz, 10, T), BOS_ITEM_ID, dtype=torch.long),
        "item_target_ids": torch.full((bsz, 10, T), PAD_ITEM_ID, dtype=torch.long),
        "item_target_minutes": torch.zeros((bsz, 10, T), dtype=torch.float32),
        "item_time_mask": torch.zeros((bsz, 10, T), dtype=torch.float32),
        "skill_targets": torch.full((bsz, 10, 18), IGNORE_INDEX, dtype=torch.long),
        "talent_targets": torch.full((bsz, 10, 4), IGNORE_INDEX, dtype=torch.long),
        "sample_weight": torch.ones((bsz,), dtype=torch.float32),
    }
    outputs = model(
        batch["hero_idx"],
        batch["is_radiant"],
        batch["pos_idx"],
        batch["item_input_ids"],
    )
    loss, scalars = compute_multitask_loss(outputs, batch, cfg)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(v)) for v in scalars.values())
    loss.backward()


def test_end_to_end_train_export_and_onnx_parity(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    rows = [
        _make_training_row(
            match_id=1000 + i,
            start_datetime=1794580000 + i * 600,
            radiant_win=(i % 2 == 0),
            duration_seconds=1800 + i * 90,
            is_pro=(i % 3 == 0),
        )
        for i in range(12)
    ]
    ckpt_dir = tmp_path / "ckpt"
    export_dir = tmp_path / "exports"
    cfg = TrainingConfig(
        checkpoint_dir=ckpt_dir,
        export_dir=export_dir,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_item_len=8,
        batch_size=4,
        epochs=2,
        lr=1e-3,
        use_amp=False,
    )
    train_res = train_from_rows(rows[:10], rows[10:], sample_vocab, cfg)
    assert train_res["best_checkpoint"].exists()
    assert train_res["vocab_path"].exists()
    assert train_res["model_config_path"].exists()

    exported = export_to_onnx(train_res["best_checkpoint"], export_dir)
    assert exported["encoder"].exists()
    assert exported["decoder"].exists()
    assert exported["model_config"].exists()
    assert exported["vocab"].exists()

    parity = verify_onnx_parity(train_res["best_checkpoint"], export_dir, atol=1e-4)
    assert parity["passed"] is True
    for name, diff in parity["max_diffs"].items():
        assert diff < 1e-4, f"{name} exceeded tolerance: {diff}"


def test_real_patch_vocab_export_parity_and_payload_budget_if_present(tmp_path: Path) -> None:
    vocab_file = Path("data/constants/vocab_190.json")
    if not vocab_file.exists():
        pytest.skip("Local data/constants/vocab_190.json not present")

    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    vidx = VocabIndex(vocab)
    cfg = TrainingConfig(
        d_model=192,
        nhead=6,
        num_encoder_layers=4,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.0,
        max_item_len=12,
    )
    spec = build_model_spec(vidx, cfg)
    model = D2OracleModel(spec, vidx.build_skill_mask())
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"
    (ckpt_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    torch.save({"epoch": 1, "spec": spec.to_dict(), "model_state_dict": model.state_dict()}, ckpt_path)

    export_dir = tmp_path / "exports"
    exported = export_to_onnx(ckpt_path, export_dir)
    parity = verify_onnx_parity(ckpt_path, export_dir, atol=1e-4)
    assert parity["passed"] is True

    total_bytes = sum(p.stat().st_size for p in exported.values())
    total_mb = total_bytes / (1024 * 1024)
    # Keep browser model payload within the 25 MB budget
    assert total_mb <= 25.0, f"Browser bundle size {total_mb:.2f} MB exceeded 25 MB budget"


def test_v2_vocab_aliases_preult_mask_and_ptensor_dataset(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    vocab = copy.deepcopy(sample_vocab)
    # Add a paired non-levelable alias slot (9992 -> 5003) inside hero 1 skill list, plus Meepo (82)
    am = vocab["heroes"]["by_id"]["1"]
    am["skill_ability_ids"] = [5003, 9992, 5004, 7314, 5006]
    vocab["abilities"]["by_id"]["9992"] = {
        "id": 9992,
        "name": "antimage_paired_alias",
        "display_name": "Paired Alias",
        "is_talent": False,
        "is_innate": False,
        "is_levelable": False,
        "is_ultimate": False,
        "img": None,
    }
    vocab["abilities"]["by_id"]["5006"]["is_ultimate"] = True
    vocab["abilities"]["aliases"]["9992"] = 5003

    meepo_idx = int(vocab["heroes"]["num_heroes"]) + 1
    vocab["heroes"]["num_heroes"] = meepo_idx
    vocab["heroes"]["id_to_idx"]["82"] = meepo_idx
    vocab["heroes"]["by_id"]["82"] = {
        "idx": meepo_idx,
        "id": 82,
        "short_name": "meepo",
        "display_name": "Meepo",
        "primary_attr": "agi",
        "attack_type": "Melee",
        "roles": ["Carry"],
        "skill_ability_ids": [5426, 5427, 5428, 5429],
        "talent_tiers": {"10": [], "15": [], "20": [], "25": []},
        "img": None,
        "icon": None,
    }
    for sid, nm, ult in (
        (5426, "meepo_earthbind", False),
        (5427, "meepo_poof", False),
        (5428, "meepo_ransack", False),
        (5429, "meepo_divided_we_stand", True),
    ):
        vocab["abilities"]["by_id"][str(sid)] = {
            "id": sid,
            "name": nm,
            "display_name": nm,
            "is_talent": False,
            "is_innate": False,
            "is_levelable": True,
            "is_ultimate": ult,
            "img": None,
        }

    vidx = VocabIndex(vocab)
    # Paired alias 9992 in skill_ability_ids[1] must remap cleanly to canonical slot 0 (5003)
    assert vidx.hero_skill_id_to_slot[1][9992] == 0

    mask = vidx.build_skill_mask()
    am_hidx = vidx.hero_id_to_idx[1]
    assert not bool(mask[am_hidx, :, 1].any())  # non-levelable alias slot masked at all levels
    assert not bool(mask[am_hidx, :5, 4].any())  # standard ult blocked at levels 1..5
    assert bool(mask[am_hidx, 5, 4].item())  # unlocked at level 6

    m_hidx = vidx.hero_id_to_idx[82]
    assert not bool(mask[m_hidx, :1, 3].any())  # Meepo Divided We Stand blocked at level 1
    assert bool(mask[m_hidx, 1, 3].item())  # unlocked at level 2 in 7.41f

    # Verify column pruning + pre-tensorized dataset
    shard = tmp_path / "matches_test.parquet"
    rows = [_make_training_row(match_id=2000 + i, start_datetime=1794580000 + i * 60) for i in range(4)]
    table = pa.Table.from_pylist(rows, schema=PARQUET_ARROW_SCHEMA)
    pq.write_table(table, shard, compression="zstd")

    loaded = load_parquet_rows([shard])
    assert len(loaded) == 4
    assert "player_slots" not in loaded[0]
    assert "networth" not in loaded[0]

    ds = DotaDraftDataset(loaded, vidx, max_item_len=8)
    assert hasattr(ds, "_tensors")
    sample = ds[0]
    assert sample["hero_idx"].shape == (10,)
    assert sample["skill_targets"].dtype == torch.long


def test_per_token_item_minutes_monotonic_and_resume_last_checkpoint(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    rows = [
        _make_training_row(
            match_id=3000 + i,
            start_datetime=1794580000 + i * 600,
            radiant_win=(i % 2 == 0),
            duration_seconds=1800 + i * 120,
        )
        for i in range(10)
    ]
    ckpt_dir = tmp_path / "ckpt_resume"
    export_dir = tmp_path / "exp_resume"

    cfg_1 = TrainingConfig(
        checkpoint_dir=ckpt_dir,
        export_dir=export_dir,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_item_len=6,
        batch_size=4,
        epochs=1,
        lr=1e-3,
        use_amp=False,
        resume=False,
    )
    res_1 = train_from_rows(rows[:8], rows[8:], sample_vocab, cfg_1)
    assert len(res_1["history"]) == 1
    assert (ckpt_dir / "last.pt").exists()

    cfg_2 = TrainingConfig(
        checkpoint_dir=ckpt_dir,
        export_dir=export_dir,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_item_len=6,
        batch_size=4,
        epochs=2,
        lr=1e-3,
        use_amp=False,
        resume=True,
    )
    res_2 = train_from_rows(rows[:8], rows[8:], sample_vocab, cfg_2)
    assert len(res_2["history"]) == 2
    assert res_2["history"][-1]["epoch"] == 2
    assert "val_selection_score" in res_2["history"][-1]["val"]

    vidx = VocabIndex(sample_vocab)
    model = D2OracleModel(build_model_spec(vidx, cfg_2), vidx.build_skill_mask())
    model.eval()
    with torch.no_grad():
        hero_idx = torch.ones((1, 10), dtype=torch.long)
        is_rad = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.long)
        pos_idx = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], dtype=torch.long)
        out = model.encoder.forward_with_logits(hero_idx, is_rad, pos_idx)
        mem = out["slot_memory"]
        items_in = torch.full((1, 10, 6), BOS_ITEM_ID, dtype=torch.long)
        full_out = model(hero_idx, is_rad, pos_idx, items_in)
        assert full_out["item_minutes"].shape == full_out["item_logits"].shape
        tokens, pred_mins = model.decode_items_greedy(mem, max_steps=6)
        assert tokens.shape[:2] == (1, 10)
        n_steps = tokens.shape[2]
        for s in range(10):
            active = [float(pred_mins[0, s, t]) for t in range(n_steps) if int(tokens[0, s, t]) > EOS_ITEM_ID]
            assert active == sorted(active)
            assert all(
                float(pred_mins[0, s, t]) == 0.0
                for t in range(n_steps)
                if int(tokens[0, s, t]) <= EOS_ITEM_ID
            )


def test_attribute_bonus_with_is_talent_maps_to_slot_attr_and_aid_zero_handling(
    sample_vocab: dict[str, Any],
) -> None:
    vidx = VocabIndex(sample_vocab)
    row = _make_training_row(match_id=9001, start_datetime=1794600000)
    row["ability_learn_ids"][0] = [5003, 0, 999999, ATTRIBUTE_BONUS_ABILITY_ID, 0, 0]
    row["ability_learn_levels"][0] = [1, 8, 12, 14, 17, 18]
    row["ability_learn_times"][0] = [60, 420, 720, 900, 1140, 1200]
    row["ability_learn_ranks"][0] = [1, 1, 1, 1, 1, 1]
    # aid=730 even with is_talent=True encodes as slot_attr;
    # aid=0 at lvl<10 stays IGNORE_INDEX; unmapped positive aid stays IGNORE_INDEX;
    # lvl=17 is skipped on standard heroes -> IGNORE_INDEX; aid=0 at lvl=18 maps to slot_talent.
    row["ability_learn_is_talent"][0] = [False, False, False, True, False, False]

    sample = DotaDraftDataset([row], vidx)[0]
    assert sample["skill_targets"][0, 0].item() == 0
    assert sample["skill_targets"][0, 7].item() == IGNORE_INDEX
    assert sample["skill_targets"][0, 11].item() == IGNORE_INDEX
    assert sample["skill_targets"][0, 13].item() == vidx.slot_attr
    assert sample["skill_targets"][0, 16].item() == IGNORE_INDEX
    assert sample["skill_targets"][0, 17].item() == vidx.slot_talent


def test_meepo_level2_ult_encodes_and_trains_with_finite_skill_loss(
    sample_vocab: dict[str, Any],
) -> None:
    vocab = copy.deepcopy(sample_vocab)
    meepo_idx = int(vocab["heroes"]["num_heroes"]) + 1
    vocab["heroes"]["num_heroes"] = meepo_idx
    vocab["heroes"]["id_to_idx"]["82"] = meepo_idx
    vocab["heroes"]["by_id"]["82"] = {
        "idx": meepo_idx,
        "id": 82,
        "short_name": "meepo",
        "display_name": "Meepo",
        "primary_attr": "agi",
        "attack_type": "Melee",
        "roles": ["Carry"],
        "skill_ability_ids": [5426, 5427, 5428, 5429],
        "talent_tiers": {"10": [], "15": [], "20": [], "25": []},
        "img": None,
        "icon": None,
    }
    for sid, nm, ult in (
        (5426, "meepo_earthbind", False),
        (5427, "meepo_poof", False),
        (5428, "meepo_ransack", False),
        (5429, "meepo_divided_we_stand", True),
    ):
        vocab["abilities"]["by_id"][str(sid)] = {
            "id": sid,
            "name": nm,
            "display_name": nm,
            "is_talent": False,
            "is_innate": False,
            "is_levelable": True,
            "is_ultimate": ult,
            "img": None,
        }

    vidx = VocabIndex(vocab)
    row = _make_training_row(match_id=9002, start_datetime=1794600000)
    row["hero_ids"][0] = 82
    row["ability_learn_ids"][0] = [5426, 5429, 5427, 5428]
    row["ability_learn_levels"][0] = [1, 2, 3, 4]
    row["ability_learn_ranks"][0] = [1, 1, 1, 1]
    row["ability_learn_times"][0] = [30, 58, 120, 180]
    row["ability_learn_is_talent"][0] = [False, False, False, False]

    sample = DotaDraftDataset([row], vidx)[0]
    assert sample["skill_targets"][0, 1].item() == 3  # Divided We Stand learned at level 2

    cfg = TrainingConfig(
        d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64, max_item_len=12
    )
    spec = build_model_spec(vidx, cfg)
    model = D2OracleModel(spec, vidx.build_skill_mask())
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}
    outputs = model(
        batch["hero_idx"],
        batch["is_radiant"],
        batch["pos_idx"],
        batch["item_input_ids"],
    )
    loss, scalars = compute_multitask_loss(outputs, batch, cfg)
    assert torch.isfinite(loss)
    assert scalars["loss_skill"] < 20.0


def test_io_wisp_tether_break_aliases_to_tether_and_never_hits_masked_slot(
    sample_vocab: dict[str, Any],
) -> None:
    vocab = copy.deepcopy(sample_vocab)
    io_idx = int(vocab["heroes"]["num_heroes"]) + 1
    vocab["heroes"]["num_heroes"] = io_idx
    vocab["heroes"]["id_to_idx"]["91"] = io_idx
    vocab["heroes"]["by_id"]["91"] = {
        "idx": io_idx,
        "id": 91,
        "short_name": "wisp",
        "display_name": "Io",
        "primary_attr": "all",
        "attack_type": "Ranged",
        "roles": ["Support"],
        "skill_ability_ids": [5485, 5486, 5487, 5488, 5489, 5490],
        "talent_tiers": {"10": [], "15": [], "20": [], "25": []},
        "img": None,
        "icon": None,
    }
    vocab["abilities"].setdefault("aliases", {})["5489"] = 5485
    for sid, nm, lvlable, ult in (
        (5485, "wisp_tether", True, False),
        (5486, "wisp_spirits", True, False),
        (5487, "wisp_overcharge", True, False),
        (5488, "wisp_relocate", True, True),
        (5489, "wisp_tether_break", False, False),
        (5490, "wisp_unaliased_innate", False, False),
    ):
        vocab["abilities"]["by_id"][str(sid)] = {
            "id": sid,
            "name": nm,
            "display_name": nm,
            "is_talent": False,
            "is_innate": False,
            "is_levelable": lvlable,
            "is_ultimate": ult,
            "img": None,
        }

    vidx = VocabIndex(vocab)
    slot_map = vidx.hero_skill_id_to_slot[91]
    assert slot_map[5485] == 0
    assert slot_map[5489] == 0
    assert 5490 not in slot_map

    row = _make_training_row(match_id=9003, start_datetime=1794600500)
    row["hero_ids"][0] = 91
    row["ability_learn_ids"][0] = [5489, 5486, 5487, 5489, 5490]
    row["ability_learn_levels"][0] = [1, 2, 3, 4, 5]
    row["ability_learn_ranks"][0] = [1, 1, 1, 2, 1]
    row["ability_learn_times"][0] = [30, 60, 120, 180, 240]
    row["ability_learn_is_talent"][0] = [False, False, False, False, False]

    sample = DotaDraftDataset([row], vidx)[0]
    assert sample["skill_targets"][0, 0].item() == 0
    assert sample["skill_targets"][0, 3].item() == 0
    assert sample["skill_targets"][0, 4].item() == -100

    cfg = TrainingConfig(
        d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64, max_item_len=12
    )
    spec = build_model_spec(vidx, cfg)
    model = D2OracleModel(spec, vidx.build_skill_mask())
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}
    outputs = model(
        batch["hero_idx"],
        batch["is_radiant"],
        batch["pos_idx"],
        batch["item_input_ids"],
    )
    loss, scalars = compute_multitask_loss(outputs, batch, cfg)
    assert torch.isfinite(loss)
    assert scalars["loss_skill"] < 20.0


def test_download_hf_dataset_forces_redownload_on_vocab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_vocab: dict[str, Any]
) -> None:
    import training.dataset as dataset_mod
    from shared.constants import PARQUET_SCHEMA_VERSION, VOCAB_SCHEMA_VERSION

    download_calls: list[dict[str, Any]] = []
    manifest_payload = {
        "schema_version": PARQUET_SCHEMA_VERSION,
        "parquet_schema_version": PARQUET_SCHEMA_VERSION,
        "vocab_schema_version": VOCAB_SCHEMA_VERSION,
        "game_version_id": 190,
        "game_version_name": "7.41f",
        "vocab_path": "data/gv_190_7.41f/vocab.json",
        "shards": [{"path": "data/gv_190_7.41f/shard_0000.parquet", "num_rows": 1}],
    }

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def hf_hub_download(self, **kwargs: Any) -> str:
            download_calls.append(kwargs)
            target = Path(kwargs["local_dir"]) / kwargs["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if kwargs["filename"] == "latest.json":
                target.write_text(json.dumps(manifest_payload), encoding="utf-8")
            elif kwargs["filename"] == "data/gv_190_7.41f/vocab.json":
                target.write_text(json.dumps(sample_vocab), encoding="utf-8")
            else:
                target.write_bytes(b"")
            return str(target)

    monkeypatch.setattr(dataset_mod, "HfApi", FakeHfApi)
    manifest, vocab, shard_paths = download_hf_dataset("rizeru/d2oracle-data", cache_dir=tmp_path)
    assert manifest["vocab_path"] == "data/gv_190_7.41f/vocab.json"
    assert vocab["schema_version"] == VOCAB_SCHEMA_VERSION
    assert len(shard_paths) == 1

    vocab_calls = [c for c in download_calls if c["filename"] == "data/gv_190_7.41f/vocab.json"]
    assert len(vocab_calls) == 1
    assert vocab_calls[0].get("force_download") is True


def test_grad_scaler_overflow_skips_scheduler_step() -> None:
    lin = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    optimizer = torch.optim.SGD(lin.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / float(step + 1))
    scaler = torch.amp.GradScaler("cpu", enabled=True)

    # Step 1: overflowing gradient -> scaler skips optimizer step, reduces scale, scheduler must NOT step.
    optimizer.zero_grad(set_to_none=True)
    overflow_loss = (lin(torch.tensor([[1e38]], dtype=torch.float32))).sum()
    stepped_overflow = _optimizer_step(
        lin, overflow_loss, optimizer, scheduler, scaler, grad_clip=1.0, use_scaler=True
    )
    assert stepped_overflow is False
    assert scaler.get_scale() < 65536.0
    assert scheduler._step_count == 1
    assert scheduler.get_last_lr()[0] == pytest.approx(0.1)
    assert lin.weight.item() == pytest.approx(1.0)

    # Step 2: normal gradient -> optimizer steps and scheduler advances cleanly.
    optimizer.zero_grad(set_to_none=True)
    normal_loss = (lin(torch.tensor([[1.0]], dtype=torch.float32))).sum()
    stepped_normal = _optimizer_step(
        lin, normal_loss, optimizer, scheduler, scaler, grad_clip=1.0, use_scaler=True
    )
    assert stepped_normal is True
    assert scheduler._step_count == 2
    assert scheduler.get_last_lr()[0] == pytest.approx(0.05)
    assert lin.weight.item() == pytest.approx(0.9)


def test_talent_overwrite_protection_and_skill_mask_clamping(
    sample_vocab: dict[str, Any],
) -> None:
    vidx = VocabIndex(sample_vocab)
    row = _make_training_row(match_id=9101, start_datetime=1794601000)
    # Hero 1 (Anti-Mage):
    # - holds lvl 5 point, levels Mana Void (5006) at index 5 (forbidden at lvl<=5 by skill_mask -> IGNORE_INDEX)
    # - picks T10=0 (1495) at lvl 10, then leftover T10=1 (966) at lvl 27 (must NOT overwrite 0)
    # - picks T15=1 (665) at lvl 15, then leftover T15=0 (6800) at lvl 28 (must NOT overwrite 1)
    row["levels"][0] = 30
    row["ability_learn_ids"][0] = [5003, 5004, 7314, 5003, 5006, 5004, 1495, 665, 6606, 6012, 966, 6800]
    row["ability_learn_levels"][0] = [1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 27, 28]
    row["ability_learn_times"][0] = list(range(60, 12 * 60 + 1, 60))
    row["ability_learn_ranks"][0] = [1] * 12
    row["ability_learn_is_talent"][0] = [
        False, False, False, False, False, False, True, True, True, True, True, True
    ]

    ds = DotaDraftDataset([row], vidx)
    sample = ds[0]
    assert sample["talent_targets"][0, 0].item() == 0
    assert sample["talent_targets"][0, 1].item() == 1
    assert sample["talent_targets"][0, 2].item() == 0
    assert sample["talent_targets"][0, 3].item() == 0
    # Illegal held-point lvl-5 ultimate target is clamped to IGNORE_INDEX so CrossEntropy never spikes +1e4
    assert sample["skill_targets"][0, 4].item() == IGNORE_INDEX
    assert sample["skill_targets"][0, 5].item() == 1

    cfg = TrainingConfig(
        d_model=32, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64, max_item_len=12
    )
    model = D2OracleModel(build_model_spec(vidx, cfg), vidx.build_skill_mask())
    batch = {k: v.unsqueeze(0) for k, v in sample.items()}
    outputs = model(batch["hero_idx"], batch["is_radiant"], batch["pos_idx"], batch["item_input_ids"])
    loss, scalars = compute_multitask_loss(outputs, batch, cfg)
    assert torch.isfinite(loss)
    assert scalars["loss_skill"] < 20.0
    assert scalars["loss_talent"] < 10.0


def test_infer_zero_talent_tiers_and_recovery_without_val_leakage(
    sample_vocab: dict[str, Any],
) -> None:
    vocab = copy.deepcopy(sample_vocab)
    h12_idx = int(vocab["heroes"]["num_heroes"]) + 1
    vocab["heroes"]["num_heroes"] = h12_idx
    vocab["heroes"]["id_to_idx"]["12"] = h12_idx
    vocab["heroes"]["idx_to_id"].append(12)
    vocab["heroes"]["by_id"]["12"] = {
        "idx": h12_idx,
        "id": 12,
        "short_name": "phantom_lancer",
        "display_name": "Phantom Lancer",
        "primary_attr": "agi",
        "attack_type": "Melee",
        "roles": ["Carry"],
        "skill_ability_ids": [5003, 5004, 7314, 5006],
        "talent_tiers": {
            "10": [9101, 9102],
            "15": [9103, 9104],
            "20": [9105, 9106],
            "25": [9107, 9108],
        },
        "img": None,
        "icon": None,
    }
    for tid in range(9101, 9109):
        vocab["abilities"]["by_id"][str(tid)] = {
            "id": tid,
            "name": f"special_bonus_pl_{tid}",
            "display_name": f"PL Talent {tid}",
            "is_talent": True,
            "is_innate": False,
            "is_levelable": False,
            "is_ultimate": False,
            "img": None,
        }

    # Add full T15/T20/T25 on Largo (155) only to verify val_rows never leak into zero_tiers
    vocab["heroes"]["by_id"]["155"]["talent_tiers"] = {
        "10": [8001, 8002],
        "15": [8003, 8004],
        "20": [8005, 8006],
        "25": [8007, 8008],
    }
    for tid in range(8003, 8009):
        vocab["abilities"]["by_id"][str(tid)] = {
            "id": tid,
            "name": f"special_bonus_largo_{tid}",
            "display_name": f"Largo {tid}",
            "is_talent": True,
            "is_innate": False,
            "is_levelable": False,
            "is_ultimate": False,
            "img": None,
        }

    vidx = VocabIndex(vocab)
    train_rows = []
    for i in range(16):
        r = _make_training_row(match_id=9200 + i, start_datetime=1794602000 + i * 60)
        # Keep only one AM slot (0) and one Hero-12 slot (2); remaining slots are Largo (155) with full choices
        r["hero_ids"] = [1, 155, 12, 155, 155, 155, 155, 155, 155, 155]
        # Slot 0: AM with T10=[0: 16], T15=both, T20=[0: 2] (<15 -> must NOT be marked!), T25=both, aid=0 at lvl>=10
        am_ab = [5003, 1495, 6800 if i % 2 == 0 else 665, 6012 if i % 2 == 0 else 6353, 0]
        am_lvl = [1, 10, 15, 25, 11]
        if i < 2:
            am_ab.append(6606)
            am_lvl.append(20)
        r["ability_learn_ids"][0] = am_ab
        r["ability_learn_levels"][0] = am_lvl
        r["ability_learn_times"][0] = [60 * j for j in range(1, len(am_ab) + 1)]
        r["ability_learn_ranks"][0] = [1] * len(am_ab)
        r["ability_learn_is_talent"][0] = [False] * len(am_ab)

        # Slot 2: Hero 12 (PL-like): aid=0 occurs across Tiers 0/1/2 (lvl 11, 15, 21), T25 observed both
        r["ability_learn_ids"][2] = [5003, 9101, 0, 0, 9105, 0, 9107 if i % 2 == 0 else 9108]
        r["ability_learn_levels"][2] = [1, 10, 11, 15, 20, 21, 25]
        r["ability_learn_times"][2] = [60, 600, 660, 900, 1200, 1260, 1500]
        r["ability_learn_ranks"][2] = [1] * 7
        r["ability_learn_is_talent"][2] = [False, True, False, False, True, False, True]

        # All Largo slots observe both choices cleanly across all 4 tiers in train (no aid=0)
        for s_l in (1, 3, 4, 5, 6, 7, 8, 9):
            c = i % 2
            r["ability_learn_ids"][s_l] = [1659, 8001 + c, 8003 + c, 8005 + c, 8007 + c]
            r["ability_learn_levels"][s_l] = [1, 10, 15, 20, 25]
            r["ability_learn_times"][s_l] = [60, 600, 900, 1200, 1500]
            r["ability_learn_ranks"][s_l] = [1] * 5
            r["ability_learn_is_talent"][s_l] = [False, True, True, True, True]
        train_rows.append(r)

    # Validation row tries to inject aid=0 on Largo (155) ONLY in val -> must NOT enter infer_zero_talent_tiers(train_rows)
    val_row = _make_training_row(match_id=9900, start_datetime=1794609000)
    val_row["ability_learn_ids"][1] = [1659, 0, 0, 0, 0, 0]
    val_row["ability_learn_levels"][1] = [1, 10, 15, 20, 25, 26]
    val_row["ability_learn_times"][1] = [60, 600, 900, 1200, 1500, 1600]
    val_row["ability_learn_ranks"][1] = [1] * 6
    val_row["ability_learn_is_talent"][1] = [False] * 6

    zero_tiers = infer_zero_talent_tiers(train_rows, vidx)
    assert zero_tiers[1] == {0: 0}
    assert zero_tiers[12] == {0: 0, 1: None, 2: 0}
    assert 155 not in zero_tiers

    # Test recovery on sequential hero 12 + lvl 27/28 leftover protection
    seq_row = _make_training_row(match_id=9950, start_datetime=1794610000)
    seq_row["hero_ids"][0] = 12
    seq_row["levels"][0] = 30
    seq_row["ability_learn_ids"][0] = [5003, 0, 0, 0, 9108, 9101, 9105]
    seq_row["ability_learn_levels"][0] = [1, 10, 15, 20, 25, 27, 28]
    seq_row["ability_learn_times"][0] = [60 * j for j in range(1, 8)]
    seq_row["ability_learn_ranks"][0] = [1] * 7
    seq_row["ability_learn_is_talent"][0] = [False] * 7

    # Test ambiguous multi-candidate aid=0 when lvl 15 picks aid=0 before lvl 10 was picked
    amb_row = _make_training_row(match_id=9951, start_datetime=1794610100)
    amb_row["hero_ids"][0] = 12
    amb_row["ability_learn_ids"][0] = [5003, 0, 0, 0]
    amb_row["ability_learn_levels"][0] = [1, 15, 16, 20]
    amb_row["ability_learn_times"][0] = [60, 900, 960, 1200]
    amb_row["ability_learn_ranks"][0] = [1] * 4
    amb_row["ability_learn_is_talent"][0] = [False] * 4

    ds = DotaDraftDataset([seq_row, amb_row], vidx, zero_talent_tiers=zero_tiers)
    s0 = ds[0]
    # T10 -> 1 (from aid=0, protected against lvl 27 leftover 9101), T15 -> IGNORE (both missing), T20 -> 1, T25 -> 1
    assert s0["talent_targets"][0].tolist() == [1, IGNORE_INDEX, 1, 1]
    assert s0["skill_targets"][0, 9].item() == vidx.slot_talent
    assert s0["skill_targets"][0, 14].item() == vidx.slot_talent

    s1 = ds[1]
    # Ambiguous lvl 15 consumes slot 0 (stays IGNORE), lvl 16 resolves [1] (both missing -> IGNORE), lvl 20 resolves T20=1
    assert s1["talent_targets"][0].tolist() == [IGNORE_INDEX, IGNORE_INDEX, 1, IGNORE_INDEX]


def test_io_wisp_spirits_subabilities_fallback_when_missing_from_vocab_by_id(
    sample_vocab: dict[str, Any],
) -> None:
    vocab = copy.deepcopy(sample_vocab)
    io_idx = int(vocab["heroes"]["num_heroes"]) + 1
    vocab["heroes"]["num_heroes"] = io_idx
    vocab["heroes"]["id_to_idx"]["91"] = io_idx
    vocab["heroes"]["idx_to_id"].append(91)
    vocab["heroes"]["by_id"]["91"] = {
        "idx": io_idx,
        "id": 91,
        "short_name": "wisp",
        "display_name": "Io",
        "primary_attr": "all",
        "attack_type": "Ranged",
        "roles": ["Support"],
        "skill_ability_ids": [5485, 5486, 5487, 5488],
        "talent_tiers": {"10": [], "15": [], "20": [], "25": []},
        "img": None,
        "icon": None,
    }
    # Deliberately omit 5490 (wisp_spirits_in) and 5493 (wisp_spirits_out) from abilities.by_id & aliases
    for sid, nm, lvlable, ult in (
        (5485, "wisp_tether", True, False),
        (5486, "wisp_spirits", True, False),
        (5487, "wisp_overcharge", True, False),
        (5488, "wisp_relocate", True, True),
    ):
        vocab["abilities"]["by_id"][str(sid)] = {
            "id": sid,
            "name": nm,
            "display_name": nm,
            "is_talent": False,
            "is_innate": False,
            "is_levelable": lvlable,
            "is_ultimate": ult,
            "img": None,
        }

    vidx = VocabIndex(vocab)
    slot_map = vidx.hero_skill_id_to_slot[91]
    assert slot_map[5486] == 1
    assert slot_map[5490] == 1
    assert slot_map[5493] == 1

    row = _make_training_row(match_id=9301, start_datetime=1794603000)
    row["hero_ids"][0] = 91
    row["ability_learn_ids"][0] = [5485, 5490, 5493, 5487, 5488]
    row["ability_learn_levels"][0] = [1, 2, 3, 4, 6]
    row["ability_learn_times"][0] = [30, 60, 120, 180, 360]
    row["ability_learn_ranks"][0] = [1] * 5
    row["ability_learn_is_talent"][0] = [False] * 5

    sample = DotaDraftDataset([row], vidx)[0]
    assert sample["skill_targets"][0, 1].item() == 1
    assert sample["skill_targets"][0, 2].item() == 1
    assert sample["skill_targets"][0, 5].item() == 3


def test_train_from_rows_uses_train_only_zero_talent_tiers_finite_losses(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    rows = [_make_training_row(match_id=9400 + i, start_datetime=1794604000 + i * 60) for i in range(18)]
    # Mark held lvl-5 ult + aid=0 talent in training rows to ensure finite losses end-to-end
    rows[0]["ability_learn_ids"][0][4] = 5006
    train_rows, val_rows = split_rows_train_val(rows, val_split=0.1, seed=42)

    cfg = TrainingConfig(
        checkpoint_dir=tmp_path / "ckpt",
        data_dir=tmp_path / "data",
        export_dir=tmp_path / "export",
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        max_item_len=8,
        batch_size=8,
        epochs=1,
        use_amp=False,
    )
    res = train_from_rows(train_rows, val_rows, sample_vocab, cfg)
    val_stats = res["history"][-1]["val"]
    assert all(torch.isfinite(torch.tensor(v)) for v in val_stats.values())
    assert val_stats["loss_skill"] < 20.0
    assert val_stats["loss_talent"] < 10.0


def test_loser_build_weight_and_win_label_smoothing(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    rows = [
        _make_training_row(match_id=9500, start_datetime=1794605000, radiant_win=True),
        _make_training_row(match_id=9501, start_datetime=1794605100, radiant_win=False),
    ]
    vidx = VocabIndex(sample_vocab)
    ds = DotaDraftDataset(rows, vidx, max_item_len=8)
    batch = {k: torch.stack([ds[0][k], ds[1][k]], dim=0) for k in ds[0]}

    cfg_base = TrainingConfig(
        checkpoint_dir=tmp_path / "ckpt",
        data_dir=tmp_path / "data",
        export_dir=tmp_path / "export",
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        max_item_len=8,
        use_amp=False,
        loser_build_weight=0.0,
        win_label_smoothing=0.0,
    )
    spec = build_model_spec(vidx, cfg_base)
    skill_mask = vidx.build_skill_mask()
    model = D2OracleModel(spec, skill_mask)
    model.eval()
    with torch.no_grad():
        outputs = model(
            batch["hero_idx"],
            batch["is_radiant"],
            batch["pos_idx"],
            batch["item_input_ids"],
        )

    perturbed = {k: v.clone() for k, v in outputs.items()}
    # Match 0 winner=Radiant(0..4), loser=Dire(5..9); Match 1 winner=Dire(5..9), loser=Radiant(0..4)
    loser_slots = [slice(5, 10), slice(0, 5)]
    loser_teams = [1, 0]
    for b_idx, (sl, tm) in enumerate(zip(loser_slots, loser_teams, strict=True)):
        perturbed["item_logits"][b_idx, sl, :, 0] += 5.0
        perturbed["skill_logits"][b_idx, sl, :, 0] += 5.0
        perturbed["talent_logits"][b_idx, sl, :, 0] += 5.0
        perturbed["econ_kda_quantiles"][b_idx, sl] += 500.0
        perturbed["item_minutes"][b_idx, sl] += 20.0
        perturbed["combat_rates"][b_idx, sl] += 400.0
        perturbed["combat_share_logits"][b_idx, tm, 0] += 5.0

    _, sc_zero_clean = compute_multitask_loss(outputs, batch, cfg_base)
    _, sc_zero_pert = compute_multitask_loss(perturbed, batch, cfg_base)
    # Discrete build heads must ignore losing slots when loser_build_weight=0.0
    for key in ("loss_item_ce", "loss_skill", "loss_talent"):
        assert sc_zero_pert[key] == pytest.approx(sc_zero_clean[key], abs=1e-6)
    # Continuous match-context heads still see losing slots identically across loser_build_weight
    for key in ("loss_econ_kda", "loss_item_time", "loss_combat_rate", "loss_combat_share"):
        assert sc_zero_pert[key] != pytest.approx(sc_zero_clean[key], abs=1e-3)

    cfg_35 = TrainingConfig(
        checkpoint_dir=cfg_base.checkpoint_dir,
        data_dir=cfg_base.data_dir,
        export_dir=cfg_base.export_dir,
        max_item_len=8,
        loser_build_weight=0.35,
        win_label_smoothing=0.0,
    )
    cfg_100 = TrainingConfig(
        checkpoint_dir=cfg_base.checkpoint_dir,
        data_dir=cfg_base.data_dir,
        export_dir=cfg_base.export_dir,
        max_item_len=8,
        loser_build_weight=1.0,
        win_label_smoothing=0.0,
    )
    _, sc_35_clean = compute_multitask_loss(outputs, batch, cfg_35)
    _, sc_35_pert = compute_multitask_loss(perturbed, batch, cfg_35)
    _, sc_100_pert = compute_multitask_loss(perturbed, batch, cfg_100)
    for key in ("loss_econ_kda", "loss_item_time", "loss_combat_rate", "loss_combat_share"):
        assert sc_35_pert[key] == pytest.approx(sc_zero_pert[key], abs=1e-6)
        assert sc_100_pert[key] == pytest.approx(sc_zero_pert[key], abs=1e-6)
    for key in ("loss_item_ce", "loss_skill", "loss_talent"):
        assert sc_zero_pert[key] != pytest.approx(sc_35_pert[key], abs=1e-4)
        assert sc_35_pert[key] != pytest.approx(sc_100_pert[key], abs=1e-4)
        assert sc_35_pert[key] != pytest.approx(sc_35_clean[key], abs=1e-4)

    cfg_smooth = TrainingConfig(
        checkpoint_dir=cfg_base.checkpoint_dir,
        data_dir=cfg_base.data_dir,
        export_dir=cfg_base.export_dir,
        max_item_len=8,
        loser_build_weight=0.35,
        win_label_smoothing=0.02,
    )
    outputs_win = {k: v.clone() for k, v in outputs.items()}
    outputs_win["win_logit"] = torch.tensor([[3.0], [-3.0]], dtype=torch.float32)
    _, sc_win_unsmoothed = compute_multitask_loss(outputs_win, batch, cfg_base)
    _, sc_smooth = compute_multitask_loss(outputs_win, batch, cfg_smooth)
    target_smooth = batch["radiant_win"].float() * 0.98 + 0.5 * 0.02
    expected_win = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs_win["win_logit"].squeeze(-1).float(), target_smooth
    ).item()
    assert sc_smooth["loss_win"] == pytest.approx(expected_win, abs=1e-6)
    assert sc_smooth["loss_win"] > sc_win_unsmoothed["loss_win"] + 1e-2


def test_team_relative_scenario_flips_dire_and_lane_mapping(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    vidx = VocabIndex(sample_vocab)
    cfg = TrainingConfig(
        checkpoint_dir=tmp_path / "ckpt",
        data_dir=tmp_path / "data",
        export_dir=tmp_path / "export",
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        max_item_len=8,
    )
    spec = build_model_spec(vidx, cfg)
    encoder = D2OracleEncoder(spec, vidx.build_skill_mask())
    assert encoder.SCENARIO_DIM == 16
    assert encoder.rad_pos_to_lane.tolist() == [2, 1, 0, 0, 2]
    assert encoder.dire_pos_to_lane.tolist() == [0, 1, 2, 2, 0]

    scen_win = torch.tensor([[0.72]], dtype=torch.float32)
    dur_norm = torch.tensor([[0.25]], dtype=torch.float32)
    scen_lanes = torch.tensor(
        [
            [
                [0.80, 0.15, 0.05],  # Top: [Rad, Even, Dire]
                [0.20, 0.50, 0.30],  # Mid
                [0.10, 0.20, 0.70],  # Bot
            ]
        ],
        dtype=torch.float32,
    )
    lead10_norm = torch.tensor([[1.4, -0.6]], dtype=torch.float32)
    is_radiant = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.long)
    pos_idx = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], dtype=torch.long)

    vec = encoder._build_team_relative_scenario(
        scen_win, dur_norm, scen_lanes, lead10_norm, is_radiant, pos_idx
    )
    assert vec.shape == (1, 10, 16)
    # own_win
    assert torch.allclose(vec[0, :5, 0], torch.full((5,), 0.72), atol=1e-6)
    assert torch.allclose(vec[0, 5:, 0], torch.full((5,), 0.28), atol=1e-6)
    # dur_norm
    assert torch.allclose(vec[0, :, 1], torch.full((10,), 0.25), atol=1e-6)
    # own_lanes flattened [9]
    rad_lanes_exp = torch.tensor([0.80, 0.15, 0.05, 0.20, 0.50, 0.30, 0.10, 0.20, 0.70])
    dire_lanes_exp = torch.tensor([0.05, 0.15, 0.80, 0.30, 0.50, 0.20, 0.70, 0.20, 0.10])
    for s in range(5):
        assert torch.allclose(vec[0, s, 2:11], rad_lanes_exp, atol=1e-6)
    for s in range(5, 10):
        assert torch.allclose(vec[0, s, 2:11], dire_lanes_exp, atol=1e-6)
    # own_lead10
    assert torch.allclose(vec[0, :5, 11:13], torch.tensor([[1.4, -0.6]]).expand(5, 2), atol=1e-6)
    assert torch.allclose(vec[0, 5:, 11:13], torch.tensor([[-1.4, 0.6]]).expand(5, 2), atol=1e-6)
    # own_slot_lane: Rad [Bot, Mid, Top, Top, Bot], Dire [Top, Mid, Bot, Bot, Top] flipped for Dire
    expected_slot_lanes = torch.tensor(
        [
            [0.10, 0.20, 0.70],  # Rad P1 Safe -> Bot
            [0.20, 0.50, 0.30],  # Rad P2 Mid -> Mid
            [0.80, 0.15, 0.05],  # Rad P3 Off -> Top
            [0.80, 0.15, 0.05],  # Rad P4 Soft -> Top
            [0.10, 0.20, 0.70],  # Rad P5 Hard -> Bot
            [0.05, 0.15, 0.80],  # Dire P1 Safe -> Top (Dire view)
            [0.30, 0.50, 0.20],  # Dire P2 Mid -> Mid (Dire view)
            [0.70, 0.20, 0.10],  # Dire P3 Off -> Bot (Dire view)
            [0.70, 0.20, 0.10],  # Dire P4 Soft -> Bot (Dire view)
            [0.05, 0.15, 0.80],  # Dire P5 Hard -> Top (Dire view)
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(vec[0, :, 13:16], expected_slot_lanes, atol=1e-6)


def test_causal_skill_head_masking_and_causality(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    torch.manual_seed(7)
    vidx = VocabIndex(sample_vocab)
    cfg = TrainingConfig(
        checkpoint_dir=tmp_path / "ckpt",
        data_dir=tmp_path / "data",
        export_dir=tmp_path / "export",
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        max_item_len=8,
        use_amp=False,
    )
    spec = build_model_spec(vidx, cfg)
    skill_mask = vidx.build_skill_mask()
    model = D2OracleModel(spec, skill_mask)
    assert isinstance(model.encoder.skill_head, CausalSkillHead)

    # Activate delta_head from zero-init so causal cross-level coupling is exercised
    torch.nn.init.normal_(model.encoder.skill_head.delta_head.weight, std=0.25)
    torch.nn.init.normal_(model.encoder.skill_head.delta_head.bias, std=0.05)
    model.eval()

    bsz = 2
    hero_idx = torch.tensor([[1, 2, 1, 2, 1, 2, 1, 2, 1, 2]] * bsz, dtype=torch.long)
    is_radiant = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * bsz, dtype=torch.long)
    pos_idx = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]] * bsz, dtype=torch.long)

    with torch.no_grad():
        cond_mem = torch.randn(bsz, 10, spec.d_model)
        valid_a = model.encoder.skill_mask_valid[hero_idx].clone()
        bias_a = model.encoder.skill_mask_bias[hero_idx].clone()

        valid_future = valid_a.clone()
        bias_future = bias_a.clone()
        # Perturb only level 16 (index 15)
        valid_future[:, :, 15, :] = 0.0
        valid_future[:, :, 15, 0] = 1.0
        bias_future[:, :, 15, :] = -1e4
        bias_future[:, :, 15, 0] = 0.0

        out_a = model.encoder.skill_head(cond_mem, valid_a, bias_a)
        out_future = model.encoder.skill_head(cond_mem, valid_future, bias_future)
        assert out_a.shape == (bsz, 10, 18, spec.num_skill_classes)
        assert torch.allclose(out_a[:, :, :15, :], out_future[:, :, :15, :], atol=1e-6)

        valid_past = valid_a.clone()
        bias_past = bias_a.clone()
        # Perturb level 1 (index 0): force different skill availability at level 1
        valid_past[:, :, 0, :] = 0.0
        valid_past[:, :, 0, 1] = 1.0
        bias_past[:, :, 0, :] = -1e4
        bias_past[:, :, 0, 1] = 0.0
        out_past = model.encoder.skill_head(cond_mem, valid_past, bias_past)
        assert (out_a[:, :, 1:, :] - out_past[:, :, 1:, :]).abs().max().item() > 1e-4

        enc_out = model.encoder.forward_with_logits(hero_idx, is_radiant, pos_idx)
        illegal_mask = ~(skill_mask[hero_idx])
        assert illegal_mask.any()
        assert enc_out["skill_logits"].masked_select(illegal_mask).max().item() < -1e3
        dur_zero = torch.zeros(bsz, dtype=torch.float32)
        enc_tuple = model.encoder(hero_idx, is_radiant, pos_idx, dur_zero)
        assert torch.allclose(enc_tuple[7], enc_out["skill_logits"], atol=1e-6)


def test_streaming_datasets_bit_equivalent_to_classic_and_train_from_shards(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    rows_raw: list[dict[str, Any]] = []
    for i in range(25):
        row = _make_training_row(
            match_id=9600 + i,
            start_datetime=1794620000 - i * 120,  # Intentionally descending to test chronological sort
            is_pro=(i % 4 == 0),
        )
        # Replace tier-0 talent (index 6 at level 10) with aid=0 for hero 1 on every row so
        # train-only infer_zero_talent_tiers satisfies min_hero_aid0 & min_seen across shards
        for s_idx, hid in enumerate(row["hero_ids"]):
            if hid == 1:
                row["ability_learn_ids"][s_idx][6] = 0
        rows_raw.append(row)

    # Inject duplicate match_id across shards with different duration to test identical dedup behavior
    dup_row = copy.deepcopy(rows_raw[0])
    dup_row["duration_seconds"] = 3333
    shard0_rows = rows_raw[:8]
    shard1_rows = rows_raw[8:17]
    shard2_rows = [*rows_raw[17:], dup_row]

    shard_paths: list[Path] = []
    for idx, srows in enumerate([shard0_rows, shard1_rows, shard2_rows]):
        spath = tmp_path / f"shard_{idx:04d}.parquet"
        table = pa.Table.from_pylist(srows, schema=PARQUET_ARROW_SCHEMA)
        pq.write_table(table, spath)
        shard_paths.append(spath)

    vidx = VocabIndex(sample_vocab)
    classic_rows = load_parquet_rows(shard_paths)
    train_rows, val_rows = split_rows_train_val(classic_rows, val_split=0.2, seed=123)
    unique_classic = train_rows + val_rows
    classic_tiers = infer_zero_talent_tiers(train_rows, vidx)
    assert classic_tiers
    classic_train = DotaDraftDataset(
        train_rows, vidx, max_item_len=8, pro_sample_weight=1.5, zero_talent_tiers=classic_tiers
    )
    classic_val = DotaDraftDataset(
        val_rows, vidx, max_item_len=8, pro_sample_weight=1.0, zero_talent_tiers=classic_tiers
    )

    bundle = build_streaming_datasets(
        shard_paths, vidx, val_split=0.2, seed=123, max_item_len=8, pro_sample_weight=1.5
    )
    assert bundle.zero_talent_tiers == classic_tiers
    assert bundle.num_total == len(unique_classic) == 25
    assert bundle.num_pro == sum(1 for r in unique_classic if r["is_pro"])
    assert bundle.val_ds is not None
    assert len(bundle.train_ds) == len(classic_train)
    assert len(bundle.val_ds) == len(classic_val)

    for key in classic_train._tensors:
        assert torch.equal(bundle.train_ds._tensors[key], classic_train._tensors[key]), f"train mismatch {key}"
        assert torch.equal(bundle.val_ds._tensors[key], classic_val._tensors[key]), f"val mismatch {key}"

    cfg = TrainingConfig(
        checkpoint_dir=tmp_path / "ckpt",
        data_dir=tmp_path / "data",
        export_dir=tmp_path / "export",
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        max_item_len=8,
        batch_size=8,
        epochs=1,
        val_split=0.2,
        seed=123,
        use_amp=False,
    )
    res = train_from_shards(
        shard_paths,
        sample_vocab,
        cfg,
        manifest={"game_version_id": 190, "game_version_name": "7.41f"},
    )
    assert res["sample_match"] is not None
    assert int(res["sample_match"]["match_id"]) >= 9600
    val_stats = res["history"][-1]["val"]
    assert all(torch.isfinite(torch.tensor(v)) for v in val_stats.values())


def test_resume_rejects_stale_encoder_arch_version_and_selection_score(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    vidx = VocabIndex(sample_vocab)
    cfg = TrainingConfig(
        checkpoint_dir=tmp_path / "ckpt",
        data_dir=tmp_path / "data",
        export_dir=tmp_path / "export",
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        max_item_len=8,
        batch_size=8,
        epochs=1,
        use_amp=False,
        resume=False,
    )
    spec_dict = build_model_spec(vidx, cfg).to_dict()
    assert spec_dict["encoder_arch_version"] == ENCODER_ARCH_VERSION

    missing_arch = dict(spec_dict)
    missing_arch.pop("encoder_arch_version")
    with pytest.raises(ValueError, match="encoder_arch_version"):
        ModelSpec.from_dict(missing_arch)

    stale_arch = dict(spec_dict)
    stale_arch["encoder_arch_version"] = ENCODER_ARCH_VERSION - 1
    with pytest.raises(ValueError, match="encoder_arch_version"):
        ModelSpec.from_dict(stale_arch)

    rows = [_make_training_row(match_id=9700 + i, start_datetime=1794630000 + i * 60) for i in range(12)]
    train_rows, val_rows = split_rows_train_val(rows, val_split=0.2, seed=42)
    train_from_rows(train_rows, val_rows, sample_vocab, cfg)

    last_pt = cfg.checkpoint_dir / "last.pt"
    orig_ckpt = torch.load(last_pt, map_location="cpu", weights_only=True)

    # Case A: stale encoder_arch_version in saved spec
    bad_spec_ckpt = copy.deepcopy(orig_ckpt)
    bad_spec_ckpt["spec"]["encoder_arch_version"] = ENCODER_ARCH_VERSION - 1
    torch.save(bad_spec_ckpt, last_pt)
    cfg_resume = TrainingConfig(
        checkpoint_dir=cfg.checkpoint_dir,
        data_dir=cfg.data_dir,
        export_dir=cfg.export_dir,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        max_item_len=8,
        batch_size=8,
        epochs=2,
        use_amp=False,
        resume=True,
    )
    with pytest.raises(ValueError, match="encoder_arch_version"):
        train_from_rows(train_rows, val_rows, sample_vocab, cfg_resume)

    # Case B: matching spec metadata, corrupted state_dict shape -> clean ValueError instead of RuntimeError
    bad_weights_ckpt = copy.deepcopy(orig_ckpt)
    bad_weights_ckpt["model_state_dict"]["encoder.win_head.0.weight"] = torch.zeros(1, 1)
    torch.save(bad_weights_ckpt, last_pt)
    with pytest.raises(ValueError, match="resume=False"):
        train_from_rows(train_rows, val_rows, sample_vocab, cfg_resume)

    # Verify _compute_val_selection_score tracks all 11 losses with exact expected weights
    assert _compute_val_selection_score({}) == 0.0
    expected_weights = {
        "loss_win": 2.0,
        "loss_duration": 1.5,
        "loss_lanes": 0.8,
        "loss_lead10": 0.4,
        "loss_combat_rate": 0.25,
        "loss_combat_share": 0.35,
        "loss_econ_kda": 0.30,
        "loss_item_ce": 0.20,
        "loss_item_time": 0.20,
        "loss_skill": 0.30,
        "loss_talent": 0.25,
    }
    for key, w in expected_weights.items():
        assert _compute_val_selection_score({key: 1.0}) == pytest.approx(w, abs=1e-7)


def test_causal_skill_head_zeros_skipped_level_phantom_ranks_and_zero_talent_tier_window(
    sample_vocab: dict[str, Any],
) -> None:
    # 1. CausalSkillHead must produce zero base_probs contribution on fully masked skipped levels (idx 16)
    head = CausalSkillHead(d_model=16, num_skill_classes=6, d_skill=16, nhead=2, dropout=0.0)
    head.eval()
    valid_mask = torch.ones((1, 1, 18, 6), dtype=torch.float32)
    valid_mask[:, :, 16, :] = 0.0  # Skipped level 17
    bias_mask = torch.where(valid_mask > 0.5, 0.0, -1e4)

    captured_prev_ranks: list[torch.Tensor] = []

    def _capture_hook(_mod: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured_prev_ranks.append(inputs[0].detach().clone())

    handle = head.prob_rank_proj.register_forward_pre_hook(_capture_hook)
    try:
        with torch.no_grad():
            head(torch.randn(1, 1, 16), valid_mask, bias_mask)
    finally:
        handle.remove()

    cat_in = captured_prev_ranks[0][0]  # [18, 12] = [base_probs(6), prev_ranks(6)]
    assert torch.allclose(cat_in[16, :6], torch.zeros(6), atol=1e-7)
    # Every class probability sums cleanly: prev_ranks sum across classes equals number of *active* prior levels / 4
    # At idx 17 (level 18), exactly 16 prior levels are active (0..15), NOT 17 -> sum == 16 / 4 == 4.0
    assert float(cat_in[17, 6:].sum().item()) == pytest.approx(4.0, abs=1e-5)

    # 2. aid=0 occurring only at level 25 (Tier 3) must NOT falsely infer Tier 2 (level 20) as single-choice aid=0
    vocab = copy.deepcopy(sample_vocab)
    vocab["heroes"]["by_id"]["12"] = {
        "idx": 3,
        "id": 12,
        "short_name": "phantom_lancer",
        "display_name": "Phantom Lancer",
        "primary_attr": "agi",
        "attack_type": "Melee",
        "roles": ["Carry"],
        "skill_ability_ids": [5003, 5004, 5005, 5006],
        "talent_tiers": {"10": [9101, 9102], "15": [9103, 9104], "20": [9105, 9106], "25": [9107, 9108]},
        "img": None,
        "icon": None,
    }
    vocab["heroes"]["id_to_idx"]["12"] = 3
    vocab["heroes"]["num_heroes"] = 3
    for tid, name in (
        (9101, "t10_l"),
        (9102, "t10_r"),
        (9103, "t15_l"),
        (9104, "t15_r"),
        (9105, "t20_l"),
        (9106, "t20_r"),
        (9107, "t25_l"),
        (9108, "t25_r"),
    ):
        vocab["abilities"]["by_id"][str(tid)] = {
            "id": tid,
            "name": name,
            "display_name": name,
            "is_talent": True,
            "img": None,
        }
    vidx = VocabIndex(vocab)
    rows = []
    for i in range(18):
        r = _make_training_row(match_id=88000 + i, start_datetime=1794600000 + i * 60)
        r["hero_ids"][0] = 12
        # Seen T20 choice 0 >= 15 times, while aid=0 appears ONLY at lvl=25 (Tier 3), never lvl 20..24
        r["ability_learn_ids"][0] = [5003, 9101, 9103, 9105, 0]
        r["ability_learn_levels"][0] = [1, 10, 15, 20, 25]
        r["ability_learn_times"][0] = [60, 600, 900, 1200, 1500]
        r["ability_learn_ranks"][0] = [1] * 5
        r["ability_learn_is_talent"][0] = [False, True, True, True, False]
        rows.append(r)

    tiers = infer_zero_talent_tiers(rows, vidx)
    assert 2 not in (tiers.get(12) or {})
    assert tiers.get(12) == {3: None}


def test_tensor_batch_loader_and_pro_val_min_count_zero_leakage(
    sample_vocab: dict[str, Any],
) -> None:
    vidx = VocabIndex(sample_vocab)
    rows = [
        _make_training_row(
            match_id=98000 + i,
            start_datetime=1794640000 + i * 60,
            is_pro=(i < 24),
        )
        for i in range(250)
    ]
    train_rows, val_rows = split_rows_train_val(
        rows, val_split=0.1, seed=42, pro_val_min_count=10
    )
    train_ids = {int(r["match_id"]) for r in train_rows}
    val_ids = {int(r["match_id"]) for r in val_rows}
    assert train_ids.isdisjoint(val_ids)
    assert sum(1 for r in val_rows if r["is_pro"]) >= 4

    ds = DotaDraftDataset(train_rows[:17], vidx, max_item_len=8)
    seq_loader = TensorBatchLoader(ds, batch_size=5, shuffle=False)
    batches = list(seq_loader)
    assert [int(b["hero_idx"].shape[0]) for b in batches] == [5, 5, 5, 2]
    rebuilt = torch.cat([b["hero_idx"] for b in batches], dim=0)
    assert torch.equal(rebuilt, torch.stack([ds[i]["hero_idx"] for i in range(17)], dim=0))

    shuf_loader = TensorBatchLoader(ds, batch_size=5, shuffle=True, seed=9)
    shuf_wins = torch.cat([b["radiant_win"] for b in shuf_loader], dim=0)
    assert shuf_wins.numel() == 17


def test_stage1_pairwise_symmetry_freeze_isolation_and_two_phase_resume(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    vidx = VocabIndex(sample_vocab)
    cfg = TrainingConfig(
        checkpoint_dir=tmp_path / "ckpt_two_phase",
        data_dir=tmp_path / "data",
        export_dir=tmp_path / "export",
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        max_item_len=8,
        stage1_pair_dim=8,
        stage1_pair_dropout=0.2,
        use_stage2_block=True,
        eos_logit_penalty=0.75,
        batch_size=8,
        epochs=4,
        two_phase=True,
        stage1_epochs=2,
        lr=1e-3,
        use_amp=False,
        resume=True,
    )
    spec = build_model_spec(vidx, cfg)
    model = D2OracleModel(spec, vidx.build_skill_mask())
    model.eval()

    # Check pairwise prior team antisymmetry and <PAD> masking (hero_idx == 0)
    hero_a = torch.tensor([[1, 2, 1, 2, 0, 2, 1, 2, 0, 1]], dtype=torch.long)
    rad_a = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.long)
    pos_a = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]], dtype=torch.long)

    hero_b = torch.cat([hero_a[:, 5:], hero_a[:, :5]], dim=1)
    rad_b = rad_a.clone()
    pos_b = torch.cat([pos_a[:, 5:], pos_a[:, :5]], dim=1)

    with torch.no_grad():
        torch.nn.init.normal_(model.encoder.hero_pos_win_bias.weight, std=0.5)
        hp_a = model.encoder.hero_emb(hero_a) + model.encoder.pos_emb(pos_a)
        hp_b = model.encoder.hero_emb(hero_b) + model.encoder.pos_emb(pos_b)
        p_a = model.encoder._compute_pairwise_win_prior(hero_a, rad_a, pos_a, hp_a)
        p_b = model.encoder._compute_pairwise_win_prior(hero_b, rad_b, pos_b, hp_b)
    assert torch.allclose(p_a, -p_b, atol=1e-5)

    # Check Stage-1 freeze isolation during a Phase-2 optimizer step
    model.set_stage1_frozen(True)
    for mod in model.encoder._stage1_submodules():
        assert mod.training is False
        for param in mod.parameters():
            assert param.requires_grad is False
    assert model.encoder.cls_token.requires_grad is False

    with torch.no_grad():
        s1_before = {
            k: v.clone()
            for k, v in model.forward_stage1(hero_a[:, :10], rad_a, pos_a).items()
        }

    model.train(True)
    for mod in model.encoder._stage1_submodules():
        assert mod.training is False

    rows = [
        _make_training_row(
            match_id=99000 + i,
            start_datetime=1794650000 + i * 60,
            radiant_win=(i % 2 == 0),
        )
        for i in range(16)
    ]
    ds = DotaDraftDataset(rows, vidx, max_item_len=8)
    batch = next(iter(TensorBatchLoader(ds, batch_size=8, shuffle=False)))
    opt2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    opt2.zero_grad()
    out2 = model(
        batch["hero_idx"],
        batch["is_radiant"],
        batch["pos_idx"],
        batch["item_input_ids"],
    )
    loss2, _ = compute_multitask_loss(out2, batch, cfg, stage_mode="stage2")
    loss2.backward()
    assert model.encoder.hero_emb.weight.grad is None
    assert model.encoder.combat_rate_head[-1].bias.grad is not None
    assert float(model.encoder.combat_rate_head[-1].bias.grad[3].abs().item()) > 0.0
    assert model.item_decoder.minute_head[-1].bias.grad is not None
    opt2.step()

    model.eval()
    with torch.no_grad():
        s1_after = model.forward_stage1(hero_a[:, :10], rad_a, pos_a)
    for k in ("win_logit", "duration_quantiles", "lane_logits", "lead10", "slot_memory"):
        assert torch.allclose(s1_before[k], s1_after[k], atol=1e-6)

    # Full two-phase train + interrupt at epoch 2 + resume to epoch 4 + ONNX export/parity
    train_rows, val_rows = split_rows_train_val(rows, val_split=0.25, seed=42)
    cfg_interrupt = copy.deepcopy(cfg)
    cfg_interrupt.epochs = 2
    cfg_interrupt.stage1_epochs = 2
    res_p1 = train_from_rows(train_rows, val_rows, sample_vocab, cfg_interrupt)
    assert res_p1["history"][-1]["phase"] == 1
    assert (cfg.checkpoint_dir / "best_stage1.pt").exists()

    # Phase 2 continuation via full cfg (epochs=4, stage1_epochs=2)
    train_ds = DotaDraftDataset(train_rows, vidx, max_item_len=8)
    val_ds = DotaDraftDataset(val_rows, vidx, max_item_len=8)
    res_full = train_from_datasets(train_ds, val_ds, sample_vocab, cfg, vidx=vidx)
    assert len(res_full["history"]) == 4
    assert [h["phase"] for h in res_full["history"]] == [1, 1, 2, 2]
    assert compute_stage1_val_score(res_full["history"][1]["val"]) > 0.0
    assert compute_stage2_val_score(res_full["history"][-1]["val"]) > 0.0

    exported = export_to_onnx(res_full["best_checkpoint"], cfg.export_dir)
    assert exported["encoder"].exists()
    parity = verify_onnx_parity(res_full["best_checkpoint"], cfg.export_dir, atol=1e-4)
    assert parity["passed"] is True


def test_fully_masked_branchless_loss_and_diagnostics_bundle(
    tmp_path: Path, sample_vocab: dict[str, Any]
) -> None:
    vidx = VocabIndex(sample_vocab)
    shard_path = tmp_path / "shard_diag.parquet"
    rows = [
        _make_training_row(
            match_id=9700 + i,
            start_datetime=1763000000 + i * 600,
            radiant_win=(i % 2 == 0),
            is_pro=(i % 4 == 0),
            duration_seconds=1800 + i * 90,
        )
        for i in range(12)
    ]
    tbl = pa.Table.from_pylist(rows, schema=PARQUET_ARROW_SCHEMA)
    pq.write_table(tbl, shard_path, compression="zstd")

    cfg = TrainingConfig(
        checkpoint_dir=tmp_path / "ckpt",
        data_dir=tmp_path / "data",
        export_dir=tmp_path / "export",
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        stage1_pair_dim=8,
        max_item_len=8,
        batch_size=6,
        epochs=2,
        stage1_epochs=1,
        val_split=0.25,
        pro_val_min_count=0,
        use_amp=False,
    )
    res = train_from_shards(
        [shard_path],
        sample_vocab,
        cfg,
        manifest={"game_version_id": 190, "game_version_name": "7.41f"},
    )
    assert res["best_epoch"] in (1, 2)
    assert res["best_stage1_epoch"] == 1
    assert res["dataset_summary"]["num_total"] == 12
    assert res["dataset_summary"]["num_val"] == 3
    assert res["val_ds"] is not None

    exported = export_to_onnx(res["best_checkpoint"], cfg.export_dir)
    parity = verify_onnx_parity(res["best_checkpoint"], cfg.export_dir, atol=1e-4)
    assert parity["passed"] is True

    # Branchless loss & metric tensors must match float outputs and keep grads finite on all-masked batch
    model = D2OracleModel(res["spec"], vidx.build_skill_mask())
    raw_b = next(iter(TensorBatchLoader(res["val_ds"], batch_size=3, shuffle=False)))
    masked_b = {k: v.clone() for k, v in raw_b.items()}
    masked_b["lane_targets"].fill_(IGNORE_INDEX)
    masked_b["lead10_mask"].zero_()
    masked_b["combat_rate_mask"].zero_()
    masked_b["combat_share_mask"].zero_()
    masked_b["econ_kda_mask"].zero_()
    masked_b["item_target_ids"].fill_(PAD_ITEM_ID)
    masked_b["item_time_mask"].zero_()
    masked_b["skill_targets"].fill_(IGNORE_INDEX)
    masked_b["talent_targets"].fill_(IGNORE_INDEX)
    masked_b["is_pro_meta"].zero_()

    out = model(
        masked_b["hero_idx"],
        masked_b["is_radiant"],
        masked_b["pos_idx"],
        masked_b["item_input_ids"],
    )
    loss_f, sc_f = compute_multitask_loss(out, masked_b, cfg, stage_mode="all")
    loss_t, sc_t = compute_multitask_loss_tensors(out, masked_b, cfg, stage_mode="all")
    assert float(loss_f.item()) == pytest.approx(float(loss_t.item()), abs=1e-6)
    for k in sc_f:
        assert sc_f[k] == pytest.approx(float(sc_t[k].item()), abs=1e-6)
    loss_t.backward()
    assert torch.isfinite(loss_t)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    bm = compute_batch_metrics(out, masked_b)
    assert "pro_win_acc" not in bm
    assert bm["lane_acc"] == (0.0, 0)

    # Full validation diagnostics + sample comparison + report packaging
    diag = collect_validation_diagnostics(
        res["best_checkpoint"],
        res["val_ds"],
        vidx,
        cfg,
        max_eval_matches=10,
        max_decode_matches=3,
    )
    assert diag["num_evaluated"] == 3
    assert diag["win_calibration"]["count"] == 3
    assert "P1" in diag["per_position"] and "combat_share_mae" in diag["per_position"]["P1"]
    assert diag["duration_override_check"]["stage1_win_max_diff"] < 1e-5
    assert diag["item_decode_stats"]["num_decoded_matches"] == 3

    sample_txt = format_sample_comparison(
        res["sample_match"], vidx, sample_vocab, cfg, cfg.export_dir
    )
    assert "True Lanes:" in sample_txt
    assert "Pred(auto):" in sample_txt
    assert "True      :" in sample_txt

    report_dir = tmp_path / "report"
    written = write_training_report(
        report_dir,
        train_result=res,
        cfg=cfg,
        vocab=sample_vocab,
        exported=exported,
        parity=parity,
        diagnostics=diag,
        sample_comparison=sample_txt,
    )
    assert written["summary_md"].exists()
    assert written["training_summary"].exists()
    assert written["history"].exists()

    archive_path = package_export_bundle(tmp_path / "d2oracle_exports", cfg.export_dir, report_dir)
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
        assert "d2oracle_encoder.onnx" in names
        assert "d2oracle_item_decoder.onnx" in names
        assert "model_config.json" in names
        assert "vocab.json" in names
        assert "report/history.json" in names
        assert "report/training_config.json" in names
        assert "report/training_summary.json" in names
        assert "report/summary.md" in names
        assert "report/sample_predictions.txt" in names
    assert sorted(p.name for p in cfg.export_dir.iterdir()) == [
        "d2oracle_encoder.onnx",
        "d2oracle_item_decoder.onnx",
        "model_config.json",
        "vocab.json",
    ]
