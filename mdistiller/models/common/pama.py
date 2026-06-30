from __future__ import annotations

import torch
import torch.nn as nn

from .afpn import AFPN
from .agent_attention import AgentAttention


class PAMAEnhancer(nn.Module):
    """
    PAMA feature enhancement stack: APF -> AMA.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_heads: int = 4,
        agent_num: int = 49,
    ):
        super().__init__()
        self.apf = AFPN(in_channels=in_channels, out_channels=out_channels)
        self.ama = AgentAttention(
            dim=out_channels,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
            agent_num=agent_num,
        )

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        fused = self.apf(features)
        # The first APF output is the final top-down calibrated feature, not a
        # raw shallow map; use it as the distillation feature for AMA.
        enhanced, agent_tokens = self.ama(fused[0])
        fused[-1] = enhanced
        return fused, agent_tokens
