from __future__ import annotations
# =============================================================================
# PAMA-SDD++ CORE CONTRIBUTION -- Cross-Scale Agent Mediation (CSAM)
# A small set of agent tokens (a) aggregates evidence from *every* pyramid level,
# (b) mediates that shared cross-scale context back into *every* level (so all
# scales are enhanced, not just the finest), and (c) serves as the compact
# teacher<->student distillation interface consumed by the GAC loss. The
# two-stage agent attention follows Agent Attention (Han et al., ECCV 2024,
# cited); the cross-scale aggregation/mediation and the "distillation mediator"
# role are novel here.
#
# MAIN SETTING (via PAMA.CSAM_AGENT_INIT="routing"): Semantic-Prototype Routing.
# Instead of grid AdaptiveAvgPool (which ties agents to spatial cells), learnable
# prototype queries softly cluster region tokens into content-adaptive agents, so
# each agent is a *semantic* slot and the agent relation graph (the GAC transfer
# target) encodes semantic organization rather than spatial layout. Routing
# clusters region tokens gathered from every pyramid level, so the prototype
# agents stay cross-scale-aware. Routing follows Slot Attention (Locatello et al.,
# NeurIPS 2020) / query attention (cited); the distillation-mediator use is new.
# "pool" keeps the baseline agent initialization for SPR ablation; the main
# paper setting uses routing only on the student-side distillation mediator.
# The every-level cross-scale mediation is always on.
# =============================================================================
import math
import torch
import torch.nn as nn


class CSAM(nn.Module):
    """Cross-Scale Agent Mediation (CSAM).

    Agent tokens mediate between dense region tokens (region -> agent -> region)
    and act as the cross-network transfer interface for the distillation losses.

    Design points:
    1) Cross-scale agents are built from and aggregate region tokens across
       every pyramid level; the shared agent context is then mediated back into
       every level, so all scales are enhanced (not just the finest).
    2) Agent initialization is pluggable: "pool" (grid AdaptiveAvgPool with
       learnable per-scale weights, default) or "routing" (learnable prototype
       queries softly cluster region tokens gathered from all levels).
    3) LayerScale on the global-context branch avoids noisy early updates.
    4) A depth-wise local positional branch preserves the CNN local inductive bias.
    """

    def __init__(
        self,
        dim: int,
        num_agents: int = 16,
        num_heads: int = 4,
        dropout: float = 0.0,
        layer_scale: float = 1e-4,
        num_levels: int = 1,
        agent_init: str = "pool",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        pool_size = int(math.sqrt(num_agents))
        if pool_size * pool_size != num_agents:
            raise ValueError("num_agents must be a perfect square for ordered adaptive pooling")
        self.dim = dim
        self.num_agents = num_agents
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.agent_init = str(agent_init).lower()

        self.agent_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.agent_proj = nn.Linear(dim, dim)
        # Learnable semantic prototype added to the pooled/routed evidence.
        self.agent_pos = nn.Parameter(torch.zeros(1, num_agents, dim))

        # Semantic-Prototype Routing modules (only when enabled).
        if self.agent_init == "routing":
            self.route_q = nn.Parameter(torch.randn(1, num_agents, dim) * 0.02)
            self.route_k = nn.Linear(dim, dim)
            self.route_v = nn.Linear(dim, dim)
            self.route_scale = dim ** -0.5

        # CSAM: one softmax-normalized weight per pyramid level so the agents
        # summarize multi-scale evidence. num_levels=1 reduces to single-scale.
        self.num_levels = max(1, int(num_levels))
        self.scale_logits = nn.Parameter(torch.zeros(self.num_levels))

        self.q_pix = nn.Linear(dim, dim)
        self.k_pix = nn.Linear(dim, dim)
        self.v_pix = nn.Linear(dim, dim)
        self.q_agent = nn.Linear(dim, dim)
        self.k_agent = nn.Linear(dim, dim)
        self.v_agent = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.agent_norm = nn.LayerNorm(dim)
        self.pixel_norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.ones(dim) * float(layer_scale))
        self.local_pos = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)

    def _pool_tokens(self, feature: torch.Tensor) -> torch.Tensor:
        return self.agent_pool(feature).flatten(2).transpose(1, 2)  # [B, A, C]

    def _init_agents(self, feature: torch.Tensor) -> torch.Tensor:
        return self.agent_proj(self._pool_tokens(feature)) + self.agent_pos

    def _init_agents_cross_scale(self, feats) -> torch.Tensor:
        # Aggregate pooled agent evidence across pyramid levels. When the number
        # of levels matches num_levels (>1), use the learnable per-scale weights;
        # otherwise fall back to a plain mean so single-scale inputs still work.
        pooled = torch.stack([self._pool_tokens(f) for f in feats], dim=0)  # [L, B, A, C]
        if pooled.shape[0] == self.num_levels and self.num_levels > 1:
            w = torch.softmax(self.scale_logits, dim=0).view(-1, 1, 1, 1)
            agg = (pooled * w).sum(dim=0)
        else:
            agg = pooled.mean(dim=0)
        return self.agent_proj(agg) + self.agent_pos

    def _init_agents_routing(self, pix: torch.Tensor) -> torch.Tensor:
        # Learnable prototype queries softly cluster the region tokens `pix`
        # ([B, N, C]) into content-adaptive semantic agents. Single-step routing.
        b = pix.shape[0]
        q = self.route_q.expand(b, -1, -1)               # [B, A, C]
        k = self.route_k(pix)                            # [B, N, C]
        v = self.route_v(pix)                            # [B, N, C]
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.route_scale, dim=-1)  # [B, A, N]
        routed = attn @ v                                # [B, A, C]
        return self.agent_proj(routed) + self.agent_pos

    def forward(self, feature: torch.Tensor, context_feats=None):
        """Cross-scale agent mediation over the whole pyramid.

        ``feature`` is the base (finest) level; ``context_feats`` is the list of
        all pyramid levels to mediate. One shared set of agents is built from and
        aggregates evidence across *every* level, then mediates that context back
        into *every* level. Returns ``(enhanced_levels, agents)`` where
        ``enhanced_levels`` is a list aligned with ``context_feats`` (or
        ``[feature]`` when no context is given) and ``agents`` is [B, A, C].
        """
        feats = list(context_feats) if context_feats is not None else [feature]
        # Region tokens from every level, concatenated: [B, sum_l(H_l*W_l), C].
        pix_all = torch.cat([f.flatten(2).transpose(1, 2) for f in feats], dim=1)

        # Agent initialization (cross-scale): routing over all levels' tokens,
        # or pooled per-level evidence (cross-scale in "pool", legacy fallback).
        if self.agent_init == "routing":
            agents = self._init_agents_routing(pix_all)
        elif context_feats is None:
            agents = self._init_agents(feature)
        else:
            agents = self._init_agents_cross_scale(feats)

        # Agents aggregate global evidence from every level's region tokens.
        q_a = self._split(self.q_agent(agents))
        k_p = self._split(self.k_pix(pix_all))
        v_p = self._split(self.v_pix(pix_all))
        attn_a = (q_a @ k_p.transpose(-2, -1)) * self.scale
        attn_a = self.dropout(attn_a.softmax(dim=-1))
        agents_upd = self._merge(attn_a @ v_p)
        agents_upd = self.agent_norm(agents + agents_upd)

        # Mediate the shared agent context back into EVERY level (agent key/value
        # are shared across levels; each level attends with its own queries).
        k_a = self._split(self.k_agent(agents_upd))
        v_a = self._split(self.v_agent(agents_upd))
        enhanced = []
        for feat in feats:
            fb, fc, fh, fw = feat.shape
            pix = feat.flatten(2).transpose(1, 2)
            q_p = self._split(self.q_pix(self.pixel_norm(pix)))
            attn_p = (q_p @ k_a.transpose(-2, -1)) * self.scale
            attn_p = self.dropout(attn_p.softmax(dim=-1))
            pix_ctx = self._merge(attn_p @ v_a)
            pix_out = pix + self.gamma * self.proj(pix_ctx)
            out = pix_out.transpose(1, 2).reshape(fb, fc, fh, fw).contiguous()
            out = out + self.local_pos(feat)
            enhanced.append(out)
        return enhanced, agents_upd


# Backward-compatible alias (legacy name; CSAM is the canonical class now).
CrossScaleAgentMediation = CSAM  # legacy alias
AgentMediatorAttention = CSAM  # legacy alias
