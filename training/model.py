"""ONNX-compatible Two-Stage Multi-Task Transformer model for Dota 2 draft prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import (
    CS_PER_MIN_SCALE,
    DPM_HPM_SCALE,
    DURATION_CENTER_MIN,
    DURATION_SCALE_MIN,
    GOLD_XP_LEAD_SCALE,
    GPM_XPM_SCALE,
    MAX_DURATION_OVERRIDE_MIN,
    MIN_MATCH_DURATION_MIN,
    ModelSpec,
)


class ExplicitMultiheadAttention(nn.Module):
    """Standard QKV multi-head attention using explicit matmul/softmax (100% ONNX-safe)."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = query.shape
        k_len = key.shape[1]

        q = self.q_proj(query).view(bsz, q_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, k_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, k_len, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            scores = scores + attn_bias
        probs = F.softmax(scores, dim=-1)
        probs = self.attn_drop(probs)

        ctx = torch.matmul(probs, v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        return self.out_proj(ctx)


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer encoder layer."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = ExplicitMultiheadAttention(d_model, nhead, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop1(self.self_attn(h, h, h))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """Pre-LN Transformer decoder layer with causal self-attention and draft cross-attention."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = ExplicitMultiheadAttention(d_model, nhead, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = ExplicitMultiheadAttention(d_model, nhead, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_bias: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop1(self.self_attn(h, h, h, attn_bias=causal_bias))
        h = self.norm2(x)
        x = x + self.drop2(self.cross_attn(h, memory, memory))
        x = x + self.ffn(self.norm3(x))
        return x


class CausalSkillHead(nn.Module):
    """Per-level base skill predictor with causal ExplicitMultiheadAttention rank refinement."""

    def __init__(
        self,
        d_model: int,
        num_skill_classes: int,
        *,
        d_skill: int = 64,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_skill = d_skill
        self.num_skill_classes = num_skill_classes

        self.base_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.base_head = nn.Linear(d_model, 18 * num_skill_classes)

        self.ctx_proj = nn.Linear(d_model, d_skill)
        self.prob_rank_proj = nn.Linear(num_skill_classes * 2, d_skill)
        self.level_emb = nn.Embedding(18, d_skill)

        self.norm1 = nn.LayerNorm(d_skill)
        self.attn = ExplicitMultiheadAttention(d_skill, nhead, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_skill)
        self.ff = nn.Sequential(
            nn.Linear(d_skill, d_skill * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_skill * 2, d_skill),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_skill)
        self.delta_head = nn.Linear(d_skill, num_skill_classes)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

        causal = torch.triu(torch.ones(18, 18, dtype=torch.float32), diagonal=1) * -1e4
        self.register_buffer("causal_bias", causal.view(1, 1, 18, 18))
        self.register_buffer(
            "strict_lower", torch.tril(torch.ones(18, 18, dtype=torch.float32), diagonal=-1)
        )

    def forward(
        self,
        cond_slot_memory: torch.Tensor,
        valid_mask: torch.Tensor,
        bias_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_slots, _ = cond_slot_memory.shape
        n = bsz * n_slots

        base_h = self.base_mlp(cond_slot_memory)
        base_raw = self.base_head(base_h).view(bsz, n_slots, 18, self.num_skill_classes)
        base_masked = base_raw * valid_mask + bias_mask
        base_probs = F.softmax(base_masked.reshape(n, 18, self.num_skill_classes), dim=-1)
        # Zero completely masked (skipped) hero levels before cumulative rank aggregation
        active_levels = torch.clamp(
            valid_mask.reshape(n, 18, self.num_skill_classes).sum(dim=-1, keepdim=True),
            max=1.0,
        )
        base_probs = base_probs * active_levels
        # Expected points already spent per class strictly before level l via static ONNX MatMul
        prev_ranks = torch.matmul(self.strict_lower, base_probs) / 4.0

        h = (
            self.ctx_proj(base_h).reshape(n, 1, self.d_skill)
            + self.level_emb.weight.unsqueeze(0)
            + self.prob_rank_proj(torch.cat([base_probs, prev_ranks], dim=-1))
        )
        h_norm = self.norm1(h)
        h = h + self.drop1(self.attn(h_norm, h_norm, h_norm, attn_bias=self.causal_bias))
        h = h + self.ff(self.norm2(h))
        delta = self.delta_head(self.out_norm(h)).view(bsz, n_slots, 18, self.num_skill_classes)
        return base_raw + delta


class D2OracleEncoder(nn.Module):
    """Two-stage draft encoder: Stage 1 predicts win/duration/lanes/@10m, Stage 2 predicts hero stats/builds."""

    SCENARIO_DIM = 16  # own_win(1) + dur_norm(1) + own_lanes(9) + own_lead10_norm(2) + own_slot_lane(3)

    def __init__(self, spec: ModelSpec, skill_mask: torch.BoolTensor) -> None:
        super().__init__()
        self.spec = spec
        self._stage1_frozen = False
        d = spec.d_model
        stage1_drop = min(0.35, spec.dropout * 1.5) if spec.dropout > 0.0 else 0.0
        pair_drop = float(spec.stage1_pair_dropout) if spec.dropout > 0.0 else 0.0

        self.hero_emb = nn.Embedding(spec.num_heroes + 1, d)
        self.team_emb = nn.Embedding(2, d)
        self.pos_emb = nn.Embedding(5, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)

        self.emb_drop = nn.Dropout(spec.dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d, spec.nhead, spec.dim_feedforward, dropout=spec.dropout
                )
                for _ in range(spec.num_encoder_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d)

        # ONNX-safe role lookup matrix for pooling per-team position embeddings
        self.register_buffer("pos_eye", torch.eye(5, dtype=torch.float32))
        self.register_buffer(
            "pair_off_diag", (1.0 - torch.eye(10, dtype=torch.float32)).unsqueeze(0)
        )
        # ONNX-safe position -> lane (0=Top, 1=Mid, 2=Bot) per team side
        self.register_buffer("rad_pos_to_lane", torch.tensor([2, 1, 0, 0, 2], dtype=torch.long))
        self.register_buffer("dire_pos_to_lane", torch.tensor([0, 1, 2, 2, 0], dtype=torch.long))
        self.register_buffer(
            "econ_scale",
            torch.tensor(
                [[[[GPM_XPM_SCALE], [GPM_XPM_SCALE], [CS_PER_MIN_SCALE], [1.0], [1.0], [1.0]]]],
                dtype=torch.float32,
            ),
        )

        # ---- Stage 1 heads (unconditioned match-level outcomes) ----
        self.win_head = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Dropout(stage1_drop),
            nn.Linear(d, 1),
        )
        self.pair_dim = int(spec.stage1_pair_dim)
        if self.pair_dim > 0:
            self.hero_pos_win_bias = nn.Embedding((spec.num_heroes + 1) * 5, 1)
            nn.init.zeros_(self.hero_pos_win_bias.weight)
            self.win_pair_proj = nn.Linear(d, 3 * self.pair_dim)
            self.pair_drop = nn.Dropout(pair_drop)
            self.win_pair_head = nn.Linear(3, 1, bias=False)
            with torch.no_grad():
                self.win_pair_head.weight.copy_(torch.tensor([[1.0, 0.25, 0.25]]))
        else:
            self.hero_pos_win_bias = None
            self.win_pair_proj = None
            self.pair_drop = None
            self.win_pair_head = None

        self.duration_head = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Dropout(stage1_drop),
            nn.Linear(d, 3),
        )
        nn.init.constant_(self.duration_head[-1].bias, 0.0)
        with torch.no_grad():
            self.duration_head[-1].bias[0] = 23.0  # ~38 min initial median (15 + softplus(23))

        self.match_context_proj = nn.Linear(3 * d, d)
        self.lane_emb = nn.Embedding(3, d)
        self.lane_pair_mlp = nn.Sequential(
            nn.Linear(6 * d, d),
            nn.GELU(),
            nn.Dropout(stage1_drop),
        )
        self.lane_head = nn.Linear(d, 3)

        self.lead10_head = nn.Sequential(
            nn.Linear(4 * d, d),
            nn.GELU(),
            nn.Dropout(stage1_drop),
            nn.Linear(d, 2),
        )

        # ---- Stage 1 -> Stage 2 Scenario Conditioning Bridge ----
        self.scenario_mlp = nn.Sequential(
            nn.Linear(self.SCENARIO_DIM, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.scenario_film = nn.Linear(d, 2 * d)
        self.stage2_fuse = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(d, d),
        )
        self.stage2_norm = nn.LayerNorm(d)
        self.stage2_block = (
            TransformerEncoderBlock(d, spec.nhead, spec.dim_feedforward, dropout=spec.dropout)
            if spec.use_stage2_block
            else None
        )

        # ---- Stage 2 per-hero heads (conditioned on detached/scheduled Stage-1 scenario) ----
        self.combat_rate_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(d, 4),
        )
        nn.init.constant_(self.combat_rate_head[-1].bias, 1.0)
        self.combat_share_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(d, 4),
        )
        self.econ_kda_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(d, 6 * 3),
        )
        nn.init.constant_(self.econ_kda_head[-1].bias, 0.0)
        with torch.no_grad():
            # Initialize median offsets for realistic initial GPM/XPM/CS/K/D/A magnitudes
            init_q50 = (5.0, 5.5, 4.0, 1.5, 1.2, 2.5)
            for idx_m, val in enumerate(init_q50):
                self.econ_kda_head[-1].bias[idx_m * 3] = val

        self.skill_head = CausalSkillHead(
            d, spec.num_skill_classes, d_skill=64, nhead=4, dropout=spec.dropout
        )
        self.talent_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(d, 4 * 2),
        )

        # Float mask buffers (1.0/0.0 multiply + 0.0/-1e4 bias avoids bool ops and float32 ULP drift)
        valid = skill_mask.to(torch.float32)
        bias = torch.where(skill_mask, 0.0, -1e4).to(torch.float32)
        self.register_buffer("skill_mask_valid", valid)
        self.register_buffer("skill_mask_bias", bias)

    def _stage1_submodules(self) -> tuple[nn.Module, ...]:
        mods: list[nn.Module] = [
            self.hero_emb,
            self.team_emb,
            self.pos_emb,
            self.emb_drop,
            self.layers,
            self.final_norm,
            self.win_head,
            self.duration_head,
            self.match_context_proj,
            self.lane_emb,
            self.lane_pair_mlp,
            self.lane_head,
            self.lead10_head,
        ]
        if self.pair_dim > 0:
            assert self.hero_pos_win_bias is not None
            assert self.win_pair_proj is not None
            assert self.pair_drop is not None
            assert self.win_pair_head is not None
            mods.extend(
                [
                    self.hero_pos_win_bias,
                    self.win_pair_proj,
                    self.pair_drop,
                    self.win_pair_head,
                ]
            )
        return tuple(mods)

    def set_stage1_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze backbone + Stage-1 heads, keeping frozen Stage-1 in eval mode."""
        self._stage1_frozen = bool(frozen)
        self.cls_token.requires_grad_(not frozen)
        for mod in self._stage1_submodules():
            for param in mod.parameters():
                param.requires_grad_(not frozen)
            if frozen:
                mod.eval()

    def train(self, mode: bool = True) -> D2OracleEncoder:
        super().train(mode)
        if mode and self._stage1_frozen:
            for mod in self._stage1_submodules():
                mod.eval()
        return self

    def _compute_lane_features(
        self,
        slot_memory: torch.Tensor,
        is_radiant: torch.Tensor,
        pos_idx: torch.Tensor,
        match_repr: torch.Tensor,
    ) -> torch.Tensor:
        """Build [B, 3, d] Top/Mid/Bot matchup features combining pair embeddings & global context."""
        bsz = slot_memory.shape[0]
        pos_onehot = self.pos_eye[pos_idx]  # [B, 10, 5]
        rad_mask = is_radiant.to(slot_memory.dtype).unsqueeze(-1)  # [B, 10, 1]
        dire_mask = 1.0 - rad_mask

        rad_pos_w = pos_onehot * rad_mask
        dire_pos_w = pos_onehot * dire_mask
        rad_by_pos = torch.bmm(rad_pos_w.transpose(1, 2), slot_memory) / rad_pos_w.sum(
            dim=1
        ).clamp(min=1.0).unsqueeze(-1)  # [B, 5, d]
        dire_by_pos = torch.bmm(dire_pos_w.transpose(1, 2), slot_memory) / dire_pos_w.sum(
            dim=1
        ).clamp(min=1.0).unsqueeze(-1)  # [B, 5, d]

        # Top: Rad pos 3+4 vs Dire pos 1+5
        top_rad = 0.5 * (rad_by_pos[:, 2, :] + rad_by_pos[:, 3, :])
        top_dire = 0.5 * (dire_by_pos[:, 0, :] + dire_by_pos[:, 4, :])
        top_core_diff = rad_by_pos[:, 2, :] - dire_by_pos[:, 0, :]
        top_supp_diff = rad_by_pos[:, 3, :] - dire_by_pos[:, 4, :]

        # Mid: Rad pos 2 vs Dire pos 2
        mid_rad = rad_by_pos[:, 1, :]
        mid_dire = dire_by_pos[:, 1, :]
        mid_core_diff = mid_rad - mid_dire
        mid_supp_diff = torch.zeros_like(mid_core_diff)

        # Bot: Rad pos 1+5 vs Dire pos 3+4
        bot_rad = 0.5 * (rad_by_pos[:, 0, :] + rad_by_pos[:, 4, :])
        bot_dire = 0.5 * (dire_by_pos[:, 2, :] + dire_by_pos[:, 3, :])
        bot_core_diff = rad_by_pos[:, 0, :] - dire_by_pos[:, 2, :]
        bot_supp_diff = rad_by_pos[:, 4, :] - dire_by_pos[:, 3, :]

        match_ctx = F.gelu(self.match_context_proj(match_repr)).unsqueeze(1).expand(bsz, 3, -1)
        rad_lanes = torch.stack([top_rad, mid_rad, bot_rad], dim=1)
        dire_lanes = torch.stack([top_dire, mid_dire, bot_dire], dim=1)
        core_diffs = torch.stack([top_core_diff, mid_core_diff, bot_core_diff], dim=1)
        supp_diffs = torch.stack([top_supp_diff, mid_supp_diff, bot_supp_diff], dim=1)

        lane_inputs = torch.cat(
            [match_ctx, rad_lanes, dire_lanes, rad_lanes - dire_lanes, core_diffs, supp_diffs],
            dim=-1,
        )  # [B, 3, 6*d]
        lane_ids = self.lane_emb.weight.unsqueeze(0).expand(bsz, -1, -1)  # [B, 3, d]
        return self.lane_pair_mlp(lane_inputs) + lane_ids  # [B, 3, d]

    def _compute_pairwise_win_prior(
        self,
        hero_idx: torch.Tensor,
        is_radiant: torch.Tensor,
        pos_idx: torch.Tensor,
        hp_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute antisymmetric (hero, pos) linear prior + low-rank synergy/counter win logit."""
        if (
            self.pair_dim <= 0
            or self.hero_pos_win_bias is None
            or self.win_pair_proj is None
            or self.pair_drop is None
            or self.win_pair_head is None
        ):
            return torch.zeros((hero_idx.shape[0], 1), device=hero_idx.device, dtype=torch.float32)

        dtype = hp_emb.dtype
        active = (hero_idx > 0).to(dtype)  # [B, 10], masks empty <PAD> slots on partial drafts
        rad_w = is_radiant.to(dtype) * active
        dire_w = (1.0 - is_radiant.to(dtype)) * active

        flat_hp = hero_idx * 5 + pos_idx
        hp_bias = self.hero_pos_win_bias(flat_hp).squeeze(-1)  # [B, 10]
        prior_diff = (hp_bias * (rad_w - dire_w)).sum(dim=1, keepdim=True) * 0.2

        pair_feats = self.pair_drop(self.win_pair_proj(hp_emb)) * active.unsqueeze(-1)
        u_syn, a_cnt, b_cnt = pair_feats.chunk(3, dim=-1)
        scale = 1.0 / (math.sqrt(self.pair_dim) * 5.0)

        syn_mat = torch.bmm(u_syn, u_syn.transpose(1, 2))
        same_team_signed = (
            torch.bmm(rad_w.unsqueeze(-1), rad_w.unsqueeze(1))
            - torch.bmm(dire_w.unsqueeze(-1), dire_w.unsqueeze(1))
        ) * self.pair_off_diag
        syn_diff = 0.5 * (syn_mat * same_team_signed).sum(dim=(1, 2)).unsqueeze(-1) * scale

        cnt_mat = torch.bmm(a_cnt, b_cnt.transpose(1, 2)) - torch.bmm(
            b_cnt, a_cnt.transpose(1, 2)
        )
        cross_mask = torch.bmm(rad_w.unsqueeze(-1), dire_w.unsqueeze(1))
        cnt_diff = (cnt_mat * cross_mask).sum(dim=(1, 2)).unsqueeze(-1) * scale

        return self.win_pair_head(torch.cat([prior_diff, syn_diff, cnt_diff], dim=-1))

    def _build_team_relative_scenario(
        self,
        scen_win: torch.Tensor,
        dur_norm: torch.Tensor,
        scen_lanes: torch.Tensor,
        lead10_norm: torch.Tensor,
        is_radiant: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Build [B, 10, 16] per-slot team-relative Stage-1 scenario conditioning vector."""
        bsz = is_radiant.shape[0]
        rad_f = is_radiant.to(scen_win.dtype).unsqueeze(-1)  # [B, 10, 1]
        team_sign = 2.0 * rad_f - 1.0  # +1 Radiant, -1 Dire

        rad_win_exp = scen_win.unsqueeze(1).expand(bsz, 10, 1)
        own_win = torch.where(rad_f > 0.5, rad_win_exp, 1.0 - rad_win_exp)
        dur_exp = dur_norm.unsqueeze(1).expand(bsz, 10, 1)
        own_lead10 = lead10_norm.unsqueeze(1).expand(bsz, 10, 2) * team_sign

        lanes_rad_view = scen_lanes.unsqueeze(1).expand(bsz, 10, 3, 3)  # [Rad, Even, Dire]
        lanes_dire_view = torch.cat(
            [lanes_rad_view[..., 2:3], lanes_rad_view[..., 1:2], lanes_rad_view[..., 0:1]],
            dim=-1,
        )  # [Dire, Even, Rad]
        own_lanes = torch.where(rad_f.unsqueeze(-1) > 0.5, lanes_rad_view, lanes_dire_view)

        slot_lane_idx = torch.where(
            is_radiant > 0, self.rad_pos_to_lane[pos_idx], self.dire_pos_to_lane[pos_idx]
        )
        own_slot_lane = torch.gather(
            own_lanes,
            dim=2,
            index=slot_lane_idx.unsqueeze(-1).unsqueeze(-1).expand(bsz, 10, 1, 3),
        ).squeeze(2)

        return torch.cat(
            [own_win, dur_exp, own_lanes.reshape(bsz, 10, 9), own_lead10, own_slot_lane],
            dim=-1,
        )

    def _condition_slot_memory(
        self,
        slot_memory: torch.Tensor,
        scenario_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Modulate per-slot draft representations with per-slot team-relative Stage-1 scenario."""
        scen_feat = self.scenario_mlp(scenario_vec)  # [B, 10, d]
        gamma, beta = self.scenario_film(scen_feat).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)

        modulated = self.stage2_norm(slot_memory * (1.0 + gamma) + beta)
        delta = self.stage2_fuse(torch.cat([modulated, scen_feat], dim=-1))
        out = self.stage2_norm(modulated + delta)
        if self.stage2_block is not None:
            out = self.stage2_block(out)
        return out

    def forward_stage1(
        self,
        hero_idx: torch.Tensor,
        is_radiant: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run shared draft backbone and Stage-1 match-level heads only."""
        bsz = hero_idx.shape[0]
        hp_emb = self.hero_emb(hero_idx) + self.pos_emb(pos_idx)
        slot_emb = hp_emb + self.team_emb(is_radiant)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = self.emb_drop(torch.cat([cls, slot_emb], dim=1))
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)

        cls_out = x[:, 0, :]
        slot_memory = x[:, 1:, :]  # [B, 10, d]

        rad_w = is_radiant.to(slot_memory.dtype).unsqueeze(-1)
        dire_w = 1.0 - rad_w
        rad_pool = (slot_memory * rad_w).sum(dim=1) / rad_w.sum(dim=1).clamp(min=1.0)
        dire_pool = (slot_memory * dire_w).sum(dim=1) / dire_w.sum(dim=1).clamp(min=1.0)
        match_repr = torch.cat([cls_out, rad_pool - dire_pool, rad_pool + dire_pool], dim=-1)

        pair_win_logit = self._compute_pairwise_win_prior(hero_idx, is_radiant, pos_idx, hp_emb)
        win_logit = self.win_head(match_repr) + pair_win_logit  # [B, 1]
        win_prob = torch.sigmoid(win_logit)

        dur_raw = self.duration_head(match_repr.float())
        p50 = MIN_MATCH_DURATION_MIN + F.softplus(dur_raw[:, 0:1])
        delta_lo = F.softplus(dur_raw[:, 1:2])
        delta_hi = F.softplus(dur_raw[:, 2:3])
        p25 = torch.clamp(p50.detach() - delta_lo, min=MIN_MATCH_DURATION_MIN)
        p75 = p50.detach() + delta_hi
        duration_quantiles = torch.cat([p25, p50, p75], dim=-1)  # [B, 3]

        lane_feats = self._compute_lane_features(slot_memory, is_radiant, pos_idx, match_repr)
        lane_logits = self.lane_head(lane_feats)  # [B, 3, 3]
        lane_probs = F.softmax(lane_logits, dim=-1)

        lead10_input = torch.cat([match_repr, lane_feats.mean(dim=1)], dim=-1)
        lead10 = self.lead10_head(lead10_input) * GOLD_XP_LEAD_SCALE  # [B, 2]

        return {
            "win_logit": win_logit,
            "win_prob": win_prob,
            "duration_quantiles": duration_quantiles,
            "lane_logits": lane_logits,
            "lane_probs": lane_probs,
            "lead10": lead10,
            "slot_memory": slot_memory,
        }

    def forward_with_logits(
        self,
        hero_idx: torch.Tensor,
        is_radiant: torch.Tensor,
        pos_idx: torch.Tensor,
        duration_override: torch.Tensor | None = None,
        *,
        true_win: torch.Tensor | None = None,
        true_duration_min: torch.Tensor | None = None,
        true_lane_targets: torch.Tensor | None = None,
        true_lead10: torch.Tensor | None = None,
        true_lead10_mask: torch.Tensor | None = None,
        scenario_mix: float = 0.0,
        duration_mix: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        bsz = hero_idx.shape[0]
        if duration_override is None:
            duration_override = torch.zeros(bsz, device=hero_idx.device, dtype=torch.float32)

        stage1 = self.forward_stage1(hero_idx, is_radiant, pos_idx)
        win_logit = stage1["win_logit"]
        win_prob = stage1["win_prob"]
        duration_quantiles = stage1["duration_quantiles"]
        lane_logits = stage1["lane_logits"]
        lane_probs = stage1["lane_probs"]
        lead10 = stage1["lead10"]
        slot_memory = stage1["slot_memory"]

        # ---- Detached Stage-1 scenario + optional scheduled teacher forcing + duration override ----
        scen_win = win_prob.detach()
        scen_dur = duration_quantiles[:, 1:2].detach()
        scen_lanes = lane_probs.detach()
        scen_lead10 = lead10.detach()

        if self.training and scenario_mix > 0.0 and true_win is not None:
            use_true = torch.rand((bsz, 1), device=hero_idx.device) < scenario_mix
            soft_true_win = 0.85 * true_win.view(bsz, 1) + 0.15 * scen_win
            scen_win = torch.where(use_true, soft_true_win, scen_win)
            if true_lane_targets is not None:
                valid_lane = (true_lane_targets >= 0) & use_true
                safe_idx = torch.clamp(true_lane_targets, min=0, max=2)
                lane_onehot = F.one_hot(safe_idx, num_classes=3).to(scen_lanes.dtype)
                soft_true_lanes = 0.80 * lane_onehot + 0.20 * scen_lanes
                scen_lanes = torch.where(valid_lane.unsqueeze(-1), soft_true_lanes, scen_lanes)
            if true_lead10 is not None and true_lead10_mask is not None:
                valid_lead = (true_lead10_mask > 0.0) & use_true
                scen_lead10 = torch.where(valid_lead, true_lead10, scen_lead10)

        # Keep true-duration conditioning active across epochs with Gaussian jitter so
        # FiLM remains responsive to inference-time duration_override scenarios.
        if self.training and duration_mix > 0.0 and true_duration_min is not None:
            use_true_dur = torch.rand((bsz, 1), device=hero_idx.device) < duration_mix
            dur_jitter = torch.randn((bsz, 1), device=hero_idx.device, dtype=scen_dur.dtype) * 1.5
            noisy_true_dur = torch.clamp(
                true_duration_min.view(bsz, 1).to(scen_dur.dtype) + dur_jitter,
                min=MIN_MATCH_DURATION_MIN,
                max=MAX_DURATION_OVERRIDE_MIN,
            )
            scen_dur = torch.where(use_true_dur, noisy_true_dur, scen_dur)

        # Apply explicit duration_override when > 0 (ONNX-safe torch.where)
        override_clamped = torch.clamp(
            duration_override.view(bsz, 1),
            min=MIN_MATCH_DURATION_MIN,
            max=MAX_DURATION_OVERRIDE_MIN,
        )
        eff_dur = torch.where(duration_override.view(bsz, 1) > 0.0, override_clamped, scen_dur)

        dur_norm = (eff_dur - DURATION_CENTER_MIN) / DURATION_SCALE_MIN
        lead10_norm = torch.clamp(scen_lead10 / GOLD_XP_LEAD_SCALE, min=-15.0, max=15.0)
        scenario_vec = self._build_team_relative_scenario(
            scen_win, dur_norm, scen_lanes, lead10_norm, is_radiant, pos_idx
        )

        # ---- Stage 2 conditioning & heads ----
        cond_slot_memory = self._condition_slot_memory(slot_memory, scenario_vec)

        combat_rates = F.softplus(self.combat_rate_head(cond_slot_memory).float()) * DPM_HPM_SCALE
        combat_share_logits = self.combat_share_head(cond_slot_memory).view(bsz, 2, 5, 4)
        combat_shares = F.softmax(combat_share_logits, dim=2)

        econ_raw = self.econ_kda_head(cond_slot_memory).float().view(bsz, 10, 6, 3)
        eq50 = F.softplus(econ_raw[..., 0])
        elo = F.softplus(econ_raw[..., 1])
        ehi = F.softplus(econ_raw[..., 2])
        eq25 = torch.clamp(eq50.detach() - elo, min=0.0)
        eq75 = eq50.detach() + ehi
        econ_kda_quantiles = torch.stack([eq25, eq50, eq75], dim=-1) * self.econ_scale

        skill_valid = self.skill_mask_valid[hero_idx]
        skill_bias = self.skill_mask_bias[hero_idx]
        raw_skill = self.skill_head(cond_slot_memory, skill_valid, skill_bias)
        skill_logits = raw_skill * skill_valid + skill_bias
        talent_logits = self.talent_head(cond_slot_memory).view(bsz, 10, 4, 2)

        return {
            "win_logit": win_logit,
            "win_prob": win_prob,
            "duration_quantiles": duration_quantiles,
            "lane_logits": lane_logits,
            "lane_probs": lane_probs,
            "lead10": lead10,
            "combat_rates": combat_rates,
            "combat_share_logits": combat_share_logits,
            "combat_shares": combat_shares,
            "econ_kda_quantiles": econ_kda_quantiles,
            "skill_logits": skill_logits,
            "talent_logits": talent_logits,
            "slot_memory": cond_slot_memory,
        }

    def forward(
        self,
        hero_idx: torch.Tensor,
        is_radiant: torch.Tensor,
        pos_idx: torch.Tensor,
        duration_override: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """ONNX-friendly tuple forward across all Stage-1 and Stage-2 heads."""
        out = self.forward_with_logits(hero_idx, is_radiant, pos_idx, duration_override)
        return (
            out["win_prob"],
            out["duration_quantiles"],
            out["lane_probs"],
            out["lead10"],
            out["combat_rates"],
            out["combat_shares"],
            out["econ_kda_quantiles"],
            out["skill_logits"],
            out["talent_logits"],
            out["slot_memory"],
        )


class D2OracleItemDecoder(nn.Module):
    """Autoregressive Transformer decoder predicting key items and purchase minutes."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        d = spec.d_model

        self.item_emb = nn.Embedding(spec.item_vocab_size, d, padding_idx=0)
        self.step_emb = nn.Embedding(spec.max_item_len, d)
        self.target_proj = nn.Linear(d, d)
        self.drop = nn.Dropout(spec.dropout)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d, spec.nhead, spec.dim_feedforward, dropout=spec.dropout
                )
                for _ in range(spec.num_decoder_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d)
        self.item_head = nn.Linear(d, spec.item_vocab_size)
        self.minute_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, spec.item_vocab_size),
        )

        # Precomputed causal additive bias [max_item_len, max_item_len]
        upper = torch.triu(
            torch.ones(spec.max_item_len, spec.max_item_len, dtype=torch.float32), diagonal=1
        )
        self.register_buffer("causal_bias", upper * -1e4)

    def forward(
        self,
        slot_memory: torch.Tensor,
        target_slot_idx: torch.Tensor,
        item_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next_item_logits [N, T, V] and next_item_minutes [N, T]."""
        seq_len = item_seq.shape[1]
        gather_idx = target_slot_idx.view(-1, 1, 1).expand(-1, 1, self.spec.d_model)
        target_ctx = self.target_proj(torch.gather(slot_memory, dim=1, index=gather_idx))

        pos_weights = self.step_emb.weight[:seq_len].unsqueeze(0)
        x = self.drop(self.item_emb(item_seq) + pos_weights + target_ctx)
        causal = self.causal_bias[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, slot_memory, causal)
        x = self.final_norm(x)

        next_item_logits = self.item_head(x)
        next_item_minutes = F.softplus(self.minute_head(x).float())
        return next_item_logits, next_item_minutes


class D2OracleModel(nn.Module):
    """Combined multi-task training module wrapping D2OracleEncoder and D2OracleItemDecoder."""

    def __init__(self, spec: ModelSpec, skill_mask: torch.BoolTensor) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = D2OracleEncoder(spec, skill_mask)
        self.item_decoder = D2OracleItemDecoder(spec)

    def set_stage1_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze Stage-1 backbone & heads while keeping Stage-2/decoder trainable."""
        self.encoder.set_stage1_frozen(frozen)

    def train(self, mode: bool = True) -> D2OracleModel:
        super().train(mode)
        if mode and self.encoder._stage1_frozen:
            for mod in self.encoder._stage1_submodules():
                mod.eval()
        return self

    def forward_stage1(
        self,
        hero_idx: torch.Tensor,
        is_radiant: torch.Tensor,
        pos_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fast Phase-1 forward pass evaluating Stage-1 heads only."""
        return self.encoder.forward_stage1(hero_idx, is_radiant, pos_idx)

    def forward(
        self,
        hero_idx: torch.Tensor,
        is_radiant: torch.Tensor,
        pos_idx: torch.Tensor,
        item_input_ids: torch.Tensor,
        duration_override: torch.Tensor | None = None,
        *,
        true_win: torch.Tensor | None = None,
        true_duration_min: torch.Tensor | None = None,
        true_lane_targets: torch.Tensor | None = None,
        true_lead10: torch.Tensor | None = None,
        true_lead10_mask: torch.Tensor | None = None,
        scenario_mix: float = 0.0,
        duration_mix: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Full training forward pass with scenario conditioning and teacher-forced item decoding."""
        enc = self.encoder.forward_with_logits(
            hero_idx,
            is_radiant,
            pos_idx,
            duration_override,
            true_win=true_win,
            true_duration_min=true_duration_min,
            true_lane_targets=true_lane_targets,
            true_lead10=true_lead10,
            true_lead10_mask=true_lead10_mask,
            scenario_mix=scenario_mix,
            duration_mix=duration_mix,
        )
        slot_memory = enc["slot_memory"]  # [B, 10, d]
        bsz, _, d = slot_memory.shape
        seq_len = item_input_ids.shape[2]

        # Expand scenario-conditioned slot_memory across all 10 slots -> [B * 10, 10, d]
        expanded_memory = (
            slot_memory.unsqueeze(1).expand(bsz, 10, 10, d).reshape(bsz * 10, 10, d)
        )
        target_slots = (
            torch.arange(10, device=slot_memory.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz, 10)
            .reshape(bsz * 10)
        )
        flat_items = item_input_ids.reshape(bsz * 10, seq_len)

        item_logits_flat, item_minutes_flat = self.item_decoder(
            expanded_memory, target_slots, flat_items
        )
        enc["item_logits"] = item_logits_flat.view(bsz, 10, seq_len, self.spec.item_vocab_size)
        enc["item_minutes"] = item_minutes_flat.view(bsz, 10, seq_len, self.spec.item_vocab_size)
        return enc

    @torch.no_grad()
    def decode_items_greedy(
        self,
        slot_memory: torch.Tensor,
        *,
        max_steps: int | None = None,
        bos_id: int = 1,
        eos_id: int = 2,
        eos_penalty: float | None = None,
        eos_stop_prob: float | None = None,
        max_minute: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively decode greedy key item IDs and minutes for all 10 slots."""
        self.eval()
        bsz, _, d = slot_memory.shape
        steps = max_steps or (self.spec.max_item_len - 1)
        eff_eos_penalty = float(self.spec.eos_logit_penalty if eos_penalty is None else eos_penalty)
        eff_stop_prob = float(self.spec.eos_stop_prob if eos_stop_prob is None else eos_stop_prob)
        eff_max_min = float(max_minute) if (max_minute is not None and max_minute > 0.0) else None
        device = slot_memory.device

        expanded_memory = (
            slot_memory.unsqueeze(1).expand(bsz, 10, 10, d).reshape(bsz * 10, 10, d)
        )
        target_slots = (
            torch.arange(10, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz, 10)
            .reshape(bsz * 10)
        )

        n_streams = bsz * 10
        seq = torch.full((n_streams, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros((n_streams,), dtype=torch.bool, device=device)
        last_min = torch.zeros((n_streams,), dtype=torch.float32, device=device)
        pred_ids: list[torch.Tensor] = []
        pred_mins: list[torch.Tensor] = []

        for step_idx in range(steps):
            logits, mins = self.item_decoder(expanded_memory, target_slots, seq)
            step_logits = logits[:, -1, :].clone()
            step_logits[:, 0] = -1e9
            step_logits[:, bos_id] = -1e9
            for prev_col in range(1, seq.shape[1]):
                prev_tok = seq[:, prev_col]
                valid_prev = prev_tok >= 3
                if valid_prev.any():
                    rows = torch.nonzero(valid_prev, as_tuple=False).squeeze(-1)
                    step_logits[rows, prev_tok[rows]] = -1e9
            if eff_stop_prob > 0.0:
                probs = F.softmax(step_logits, dim=-1)
                has_item = (step_logits[:, 3:] > -1e8).any(dim=-1)
                block_eos = has_item & (probs[:, eos_id] < eff_stop_prob)
                if block_eos.any():
                    step_logits[block_eos, eos_id] = -1e9
            if eff_eos_penalty != 0.0:
                step_logits[:, eos_id] = step_logits[:, eos_id] - eff_eos_penalty

            cand_tok = torch.argmax(step_logits, dim=-1)
            next_tok = torch.where(finished, torch.full_like(cand_tok, eos_id), cand_tok)
            raw_min = torch.gather(
                mins[:, -1, :].float(),
                dim=-1,
                index=next_tok.unsqueeze(-1),
            ).squeeze(-1)
            mono_min = torch.maximum(last_min, raw_min)
            if step_idx > 0 and eff_max_min is not None:
                over_budget = (~finished) & (next_tok != eos_id) & (mono_min > eff_max_min)
                next_tok = torch.where(over_budget, torch.full_like(next_tok, eos_id), next_tok)
            stop_now = finished | (next_tok == eos_id)
            next_min = torch.where(stop_now, torch.zeros_like(mono_min), mono_min)
            finished = stop_now
            last_min = torch.where(finished, last_min, next_min)

            pred_ids.append(next_tok)
            pred_mins.append(next_min)
            seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
            if bool(finished.all()):
                break

        out_ids = torch.stack(pred_ids, dim=1).view(bsz, 10, -1)
        out_mins = torch.stack(pred_mins, dim=1).view(bsz, 10, -1)
        return out_ids, out_mins
