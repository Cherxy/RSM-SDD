from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSpatialFusion(nn.Module):
    """ASF used inside APF.

    M = sigmoid(conv([X,Y])); F_out = M*X + (1-M)*Y.
    """
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, low, high):
        if high.shape[-2:] != low.shape[-2:]:
            high = F.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)
        m = self.weight(torch.cat([low, high], dim=1))
        return m * low + (1.0 - m) * high


class APF(nn.Module):
    """Progressive Pyramid Fusion.

    The module projects each feature level to a shared channel dimension and then
    propagates deep semantics progressively from high level to shallow level.
    """
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = int(out_channels or self.in_channels[-1])
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU(inplace=True),
            ) for c in self.in_channels
        ])
        self.fuse = nn.ModuleList([AdaptiveSpatialFusion(self.out_channels) for _ in range(len(self.in_channels)-1)])
    def forward(self, features):
        xs = [p(f) for p, f in zip(self.proj, features)]
        out = xs[-1]
        outs = [None] * len(xs)
        outs[-1] = out
        # top-down progressive adjacent fusion
        for i in range(len(xs) - 2, -1, -1):
            out = self.fuse[i](xs[i], out)
            outs[i] = out
        return outs
