from __future__ import annotations
import torch
import torch.nn.functional as F


def split_feature(feature: torch.Tensor, scales=(1,2,4)):
    """Return pooled local descriptors [B, C, sum(s*s)]."""
    outs = []
    for s in scales:
        pooled = F.adaptive_avg_pool2d(feature, output_size=(s, s))
        outs.append(pooled.flatten(2))
    return torch.cat(outs, dim=2)


def spp_logits(feature: torch.Tensor, classifier, scales=(1,2,4)):
    """Compute local logits with the model classifier.

    Input feature: [B, C, H, W]
    Output: [B, num_classes, N_regions]
    """
    desc = split_feature(feature, scales=scales)          # B,C,N
    b, c, n = desc.shape
    tokens = desc.permute(0, 2, 1).reshape(b * n, c)      # B*N,C
    logits = classifier(tokens)                           # B*N,K
    k = logits.shape[-1]
    return logits.reshape(b, n, k).permute(0, 2, 1).contiguous()
