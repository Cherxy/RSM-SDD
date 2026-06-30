from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidPooling(nn.Module):
    """
    Region decoupling used by SDD/PAMA-SDD.

    Returns:
    - pooled feature tensor of shape [B, C, K]
    - normalized region masks of shape [B, K, H, W]
    """

    def __init__(self, M: Iterable[int]):
        super().__init__()
        self.levels = [int(level) for level in M]
        if not self.levels:
            raise ValueError("M must contain at least one pyramid level.")

    def _build_masks(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        masks = []
        for level in self.levels:
            for row in range(level):
                h0 = int(row * h / level)
                h1 = int((row + 1) * h / level)
                for col in range(level):
                    w0 = int(col * w / level)
                    w1 = int((col + 1) * w / level)
                    mask = torch.zeros(h, w, device=device, dtype=dtype)
                    area = max((h1 - h0) * (w1 - w0), 1)
                    mask[h0:h1, w0:w1] = 1.0 / area
                    masks.append(mask)
        return torch.stack(masks, dim=0)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        pooled = []
        for level in self.levels:
            level_feat = F.adaptive_avg_pool2d(x, output_size=(level, level))
            pooled.append(level_feat.flatten(2))
        pooled = torch.cat(pooled, dim=2)

        base_masks = self._build_masks(h, w, x.device, x.dtype)
        masks = base_masks.unsqueeze(0).expand(b, -1, -1, -1).contiguous()
        return pooled, masks

