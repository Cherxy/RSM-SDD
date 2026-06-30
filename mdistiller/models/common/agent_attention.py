from __future__ import annotations

import math

import torch
import torch.nn as nn


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0):
    with torch.no_grad():
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=mean - 2 * std, b=mean + 2 * std)


class AgentAttention(nn.Module):
    """
    Conv-friendly Agent Attention adapted from the supplied AgentAttention code.

    Inputs/outputs use [B, C, H, W] to match CNN feature maps in distillation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        agent_num: int = 49,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        pool_size = int(math.sqrt(agent_num))
        if pool_size * pool_size != agent_num:
            raise ValueError("agent_num must be a perfect square.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.agent_num = agent_num

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, 1))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1, agent_num))

        self._reset_parameters()

    def _reset_parameters(self):
        trunc_normal_(self.an_bias, std=0.02)
        trunc_normal_(self.ah_bias, std=0.02)
        trunc_normal_(self.aw_bias, std=0.02)
        trunc_normal_(self.na_bias, std=0.02)
        trunc_normal_(self.ha_bias, std=0.02)
        trunc_normal_(self.wa_bias, std=0.02)

    def _build_bias(self, base: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        return nn.functional.interpolate(base, size=target_hw, mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        n = h * w

        seq = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(seq).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        agent_tokens = self.pool(x).flatten(2).transpose(1, 2)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        pos1 = self._build_bias(self.an_bias, (h, w)).reshape(1, self.num_heads, self.agent_num, n)
        pos2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, 1)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + pos1 + pos2)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        pos3 = self._build_bias(self.na_bias, (h, w)).reshape(1, self.num_heads, self.agent_num, n).permute(0, 1, 3, 2)
        pos4 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, 1, self.agent_num)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + pos3 + pos4)
        q_attn = self.attn_drop(q_attn)
        out = q_attn @ agent_v

        out = out.transpose(1, 2).reshape(b, n, c)
        out = out.transpose(1, 2).reshape(b, c, h, w)

        v_img = v.transpose(1, 2).reshape(b, n, c).transpose(1, 2).reshape(b, c, h, w)
        out = out + self.dwc(v_img)

        out = out.flatten(2).transpose(1, 2)
        out = self.proj_drop(self.proj(out))
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out, agent_v.transpose(1, 2).reshape(b, self.agent_num, c)

