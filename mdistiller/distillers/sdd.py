from __future__ import annotations
import torch
import torch.nn.functional as F
from .kd import kd_loss
from .dkd import dkd_loss
from .nkd import nkd_loss


def local_kd_loss(local_s, local_t, target, base='dkd', scales=(1,2,4), alpha=1.0, beta=8.0, temperature=4.0):
    """Local SDD loss on local logits [B,K,N].

    It reshapes B,K,N -> B*N,K and repeats target N times, following the
    SDD code comment: convert B x C x N to N*B x C and average only over class.
    """
    b, k, n = local_s.shape
    ls = local_s.permute(0, 2, 1).reshape(b*n, k)
    lt = local_t.permute(0, 2, 1).reshape(b*n, k)
    tgt = target[:, None].expand(b, n).reshape(b*n)
    base = base.lower()
    if base == 'kd':
        return kd_loss(ls, lt, temperature=temperature)
    if base == 'dkd':
        return dkd_loss(ls, lt, tgt, alpha=alpha, beta=beta, temperature=temperature)
    if base == 'nkd':
        return nkd_loss(ls, lt, tgt, gamma=1.0, temperature=temperature)
    raise KeyError(base)
