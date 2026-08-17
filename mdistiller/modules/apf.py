from __future__ import annotations
# -----------------------------------------------------------------------------
# PAMA-SDD++ CORE CONTRIBUTION -- APF++ (channel-spatial adaptive pyramid fusion)
# Progressive feature-pyramid fusion is prior work (cited); the channel-spatial
# reliability gate + residual scaling are the local refinement. Produces
# calibrated multi-scale features consumed by CSAM (csam.py) and local SDD (spp.py).
#
# OPTIONAL ABLATION (via PAMA.APF_GSMF): Global-Semantic Modulated Fusion (GSMF).
# A global semantic anchor pooled from the coarsest (most semantic) pyramid level
# FiLM-modulates the reliability gate at every level, so fine-scale fusion is
# conditioned on global semantics -- attacking semantic fragmentation at the
# feature-calibration stage (complementing the logit-stage LGC loss). FiLM
# conditioning follows Perez et al., AAAI 2018 (cited); the coarse-anchor->gate
# modulation for distillation calibration is the new part. Default OFF keeps the
# module byte-identical to the baseline APF. The main paper setting keeps GSMF
# disabled and uses SPR + reliability-aware distillation as the primary novelty.
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class APFGate(nn.Module):
    """Channel-spatial adaptive fusion used by APF++.

    Predicts a C x H x W reliability map so each semantic channel decides how
    much shallow detail vs. deep context to keep. A residual scale keeps early
    training stable when the fusion module is randomly initialized.

    When ``gsmf`` is enabled, the gate's hidden activation is FiLM-
    modulated by a global semantic anchor ``g`` (see module banner).
    """

    def __init__(self, channels: int, init_gamma: float = 0.5, gsmf: bool = False):
        super().__init__()
        self.gsmf = bool(gsmf)
        if self.gsmf:
            # Split the gate so FiLM can be injected on the hidden features.
            self.gate_hidden = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.gate_out = nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=True),
                nn.Sigmoid(),
            )
            # Zero-init -> (scale, shift) = (0, 0) at start, i.e. identity FiLM,
            # so training begins exactly at the un-modulated baseline.
            self.film = nn.Linear(channels, channels * 2)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        else:
            # Baseline gate (unchanged): keeps parameter names/behavior identical.
            self.gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 1, bias=True),
                nn.Sigmoid(),
            )
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, low: torch.Tensor, high: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        if high.shape[-2:] != low.shape[-2:]:
            high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([low, high], dim=1)
        if self.gsmf:
            h = self.gate_hidden(x)
            if g is not None:
                scale, shift = self.film(g).chunk(2, dim=1)   # each [B, C]
                h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
            mask = self.gate_out(h)
        else:
            mask = self.gate(x)
        fused = mask * low + (1.0 - mask) * high
        return low + self.gamma * (fused - low)


class APF(nn.Module):
    """APF++: progressive pyramid semantic calibration.

    Each backbone feature is projected to a shared channel dimension. Deep
    semantic context is then propagated from coarse to fine levels with
    channel-spatial adaptive fusion. All calibrated levels are returned so local
    distillation can use scale-aware feature levels.

    ``gsmf`` enables Global-Semantic Modulated Fusion (see module banner):
    a global anchor from the coarsest calibrated level conditions every gate.
    """

    def __init__(self, in_channels, out_channels=None, init_gamma: float = 0.5, gsmf: bool = False):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = int(out_channels or self.in_channels[-1])
        self.gsmf = bool(gsmf)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            ) for c in self.in_channels
        ])
        self.fuse = nn.ModuleList([
            APFGate(self.out_channels, init_gamma=init_gamma, gsmf=self.gsmf)
            for _ in range(len(self.in_channels) - 1)
        ])

    def forward(self, features):
        xs = [proj(feat) for proj, feat in zip(self.proj, features)]
        out = xs[-1]
        # Global semantic anchor from the coarsest (most semantic) level.
        g = F.adaptive_avg_pool2d(xs[-1], 1).flatten(1) if self.gsmf else None
        outs = [None] * len(xs)
        outs[-1] = out
        for i in range(len(xs) - 2, -1, -1):
            out = self.fuse[i](xs[i], out, g)
            outs[i] = out
        return outs

# Backward-compatible alias (legacy name).
AdaptiveSpatialFusion = APFGate
