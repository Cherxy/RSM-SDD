from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentMediatorAttention(nn.Module):
    """AMA: Agent-Mediated Attention.

    It implements two phases:
    1) agent aggregation: agents query pixel tokens;
    2) agent broadcasting: pixels query updated agents.

    Agent tokens are initialized from an ordered adaptive pooling grid, matching
    the paper formulation A_0 = eta(AAP(G)) + E_A instead of using free tokens.
    """
    def __init__(self, dim, num_agents=16, num_heads=4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        pool_size = int(math.sqrt(num_agents))
        if pool_size * pool_size != num_agents:
            raise ValueError('num_agents must be a perfect square for ordered adaptive pooling')
        self.dim = dim
        self.num_agents = num_agents
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.agent_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.agent_proj = nn.Linear(dim, dim)
        self.agent_pos = nn.Parameter(torch.zeros(1, num_agents, dim))
        self.q_pix = nn.Linear(dim, dim)
        self.k_pix = nn.Linear(dim, dim)
        self.v_pix = nn.Linear(dim, dim)
        self.q_agent = nn.Linear(dim, dim)
        self.k_agent = nn.Linear(dim, dim)
        self.v_agent = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    def _split(self, x):
        b, n, c = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
    def _merge(self, x):
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)
    def forward(self, feature):
        b, c, h, w = feature.shape
        pix = feature.flatten(2).transpose(1, 2)  # B,HW,C
        pooled = self.agent_pool(feature).flatten(2).transpose(1, 2)
        agents = self.agent_proj(pooled) + self.agent_pos
        # agents aggregate pixels
        q_a = self._split(self.q_agent(agents))
        k_p = self._split(self.k_pix(pix))
        v_p = self._split(self.v_pix(pix))
        attn_a = (q_a @ k_p.transpose(-2, -1)) * self.scale
        attn_a = self.dropout(attn_a.softmax(dim=-1))
        agents_upd = self._merge(attn_a @ v_p)
        agents_upd = self.norm(agents + agents_upd)
        # pixels receive global summary from agents
        q_p = self._split(self.q_pix(pix))
        k_a = self._split(self.k_agent(agents_upd))
        v_a = self._split(self.v_agent(agents_upd))
        attn_p = (q_p @ k_a.transpose(-2, -1)) * self.scale
        attn_p = self.dropout(attn_p.softmax(dim=-1))
        pix_ctx = self._merge(attn_p @ v_a)
        pix_out = pix + self.proj(pix_ctx)
        out = pix_out.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return out, agents_upd
