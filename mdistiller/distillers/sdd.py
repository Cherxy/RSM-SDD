from __future__ import annotations
# =============================================================================
# PAMA-SDD++ local SDD distillation loss (faithful Scale Decoupled Distillation).
# `local_kd_loss` applies the per-sample base objective (KD/DKD/NKD) to every
# SDD region logit ([B, K, N]) and, following SDD (CVPR 2024), reweights each
# region by the consistent/complementary decoupling (`_sdd_decouple_weight`):
# regions whose teacher LOCAL prediction disagrees with the teacher GLOBAL (1x1)
# prediction carry complementary knowledge and get a larger weight. A PAMA++
# per-region reliability weight (see PAMASDD._reliability_weight) can be
# multiplied in on top. The `*_loss_per_sample` helpers mirror kd/dkd/nkd but
# keep the per-region dimension; `legacy_local_kd_loss` and `sdd_decouple=False`
# reproduce the un-decoupled baselines for ablation.
# =============================================================================
import torch
import torch.nn.functional as F
from .kd import kd_loss
from .dkd import dkd_loss
from .nkd import nkd_loss


def _weighted_reduce(loss_vec: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return loss_vec.mean()
    weight = weight.reshape(-1).to(loss_vec.device, dtype=loss_vec.dtype).detach()
    denom = weight.sum().clamp_min(1e-6)
    return (loss_vec * weight).sum() / denom


def kd_loss_per_sample(logits_s, logits_t, temperature=4.0):
    loss = F.kl_div(
        F.log_softmax(logits_s / temperature, dim=1),
        F.softmax(logits_t / temperature, dim=1),
        reduction="none",
    ).sum(dim=1)
    return loss * (temperature ** 2)


def dkd_loss_per_sample(logits_s, logits_t, target, alpha=1.0, beta=8.0, temperature=4.0):
    gt_mask = torch.zeros_like(logits_s).scatter_(1, target.reshape(-1, 1), 1).bool()
    other_mask = torch.ones_like(logits_s).scatter_(1, target.reshape(-1, 1), 0).bool()

    pred_s = F.softmax(logits_s / temperature, dim=1)
    pred_t = F.softmax(logits_t / temperature, dim=1)
    pred_s_cat = torch.cat([
        (pred_s * gt_mask).sum(dim=1, keepdim=True),
        (pred_s * other_mask).sum(dim=1, keepdim=True),
    ], dim=1)
    pred_t_cat = torch.cat([
        (pred_t * gt_mask).sum(dim=1, keepdim=True),
        (pred_t * other_mask).sum(dim=1, keepdim=True),
    ], dim=1)
    tckd = F.kl_div(torch.log(pred_s_cat.clamp_min(1e-8)), pred_t_cat, reduction="none").sum(dim=1)

    pred_t_part2 = F.softmax(logits_t / temperature - 1000.0 * gt_mask.float(), dim=1)
    log_pred_s_part2 = F.log_softmax(logits_s / temperature - 1000.0 * gt_mask.float(), dim=1)
    nckd = F.kl_div(log_pred_s_part2, pred_t_part2, reduction="none").sum(dim=1)
    return (alpha * tckd + beta * nckd) * (temperature ** 2)


def nkd_loss_per_sample(logits_s, logits_t, target, gamma=1.0, temperature=4.0):
    mask = torch.zeros_like(logits_s).scatter_(1, target[:, None], 1).bool()
    s = F.log_softmax(logits_s / temperature - 1000.0 * mask.float(), dim=1)
    t = F.softmax(logits_t / temperature - 1000.0 * mask.float(), dim=1)
    return F.kl_div(s, t, reduction="none").sum(dim=1) * (temperature ** 2) * gamma


def _base_loss_per_sample(out_s, out_t, target, base, alpha, beta, temperature):
    """Per-region-sample base distillation loss selected by ``base``."""
    if base == "kd":
        return kd_loss_per_sample(out_s, out_t, temperature=temperature)
    if base == "dkd":
        return dkd_loss_per_sample(out_s, out_t, target, alpha=alpha, beta=beta, temperature=temperature)
    if base == "nkd":
        return nkd_loss_per_sample(out_s, out_t, target, gamma=1.0, temperature=temperature)
    raise KeyError(base)


def _sdd_decouple_weight(out_t, target_r, num_regions, batch, consistent_w=1.0, complementary_w=2.0):
    """Scale-Decoupled Distillation (SDD, CVPR 2024) per-region weights.

    Each region-sample is split by whether the TEACHER's local prediction agrees
    with its global prediction (region 0 -- the 1x1 pooled region):
      * consistent   (local == global): weight ``consistent_w``   (default 1.0)
      * complementary (local != global): weight ``complementary_w`` (default 2.0)
    Complementary regions carry the extra, harder-to-transfer knowledge, so SDD
    emphasizes them. ``out_t`` is region-major ``[num_regions * batch, K]`` (the
    first ``batch`` rows are region 0 == the global prediction). Returns a
    ``[num_regions * batch]`` weight vector. Region 0 is always consistent
    (it equals the global prediction), so it keeps ``consistent_w``.
    """
    local_pred = out_t.argmax(dim=1)                            # [R*B]
    local_correct = local_pred.eq(target_r)                     # [R*B]
    global_correct = local_correct[:batch].repeat(num_regions)  # region 0 == global
    consistent = local_correct.eq(global_correct)               # both right or both wrong
    weight = out_t.new_full((out_t.shape[0],), float(complementary_w))
    weight[consistent] = float(consistent_w)
    return weight


def local_kd_loss(
    local_s,
    local_t,
    target,
    base="dkd",
    scales=(1, 2, 4),
    alpha=1.0,
    beta=8.0,
    temperature=4.0,
    region_weight=None,
    sdd_decouple=True,
    consistent_weight=1.0,
    complementary_weight=2.0,
):
    """Local SDD distillation loss on logits shaped [B, K, N].

    With ``sdd_decouple=True`` (default) this faithfully reproduces Scale
    Decoupled Distillation (CVPR 2024): the per-region base loss (KD/DKD/NKD) is
    reweighted by the consistent/complementary decoupling and averaged over all
    region-samples (``mean`` over N*B, matching the reference normalization). An
    optional PAMA++ ``region_weight`` ([B, N] reliability) is multiplied onto the
    decoupling weight when provided.

    With ``sdd_decouple=False`` the legacy behavior is used: a plain mean, or a
    reliability-weighted mean when ``region_weight`` is given.
    """
    base = base.lower()
    b, k, n = local_s.shape

    if not sdd_decouple:
        # ---- legacy path (unchanged): plain / reliability-weighted mean ----
        ls = local_s.permute(0, 2, 1).reshape(b * n, k)
        lt = local_t.permute(0, 2, 1).reshape(b * n, k)
        tgt = target[:, None].expand(b, n).reshape(b * n)
        weight = None if region_weight is None else region_weight.reshape(b * n)
        loss_vec = _base_loss_per_sample(ls, lt, tgt, base, alpha, beta, temperature)
        return _weighted_reduce(loss_vec, weight)

    # ---- faithful SDD path: region-major [N*B, K], region 0 = global (1x1) ----
    out_s = local_s.permute(2, 0, 1).reshape(n * b, k)
    out_t = local_t.permute(2, 0, 1).reshape(n * b, k)
    target_r = target.repeat(n)
    loss_vec = _base_loss_per_sample(out_s, out_t, target_r, base, alpha, beta, temperature)

    index = _sdd_decouple_weight(out_t, target_r, n, b, consistent_weight, complementary_weight)
    if region_weight is not None:
        # PAMA++ reliability weight, aligned to the same region-major [N*B] order.
        rw = region_weight.permute(1, 0).reshape(n * b).to(loss_vec.dtype).detach()
        index = index * rw
    # SDD reduction: mean of (base loss x decoupling weight) over all region-samples.
    return (loss_vec * index).mean()


def legacy_local_kd_loss(local_s, local_t, target, base="dkd", scales=(1, 2, 4), alpha=1.0, beta=8.0, temperature=4.0):
    """Compatibility wrapper that reproduces the unweighted baseline behavior."""
    base = base.lower()
    b, k, n = local_s.shape
    ls = local_s.permute(0, 2, 1).reshape(b * n, k)
    lt = local_t.permute(0, 2, 1).reshape(b * n, k)
    tgt = target[:, None].expand(b, n).reshape(b * n)
    if base == "kd":
        return kd_loss(ls, lt, temperature=temperature)
    if base == "dkd":
        return dkd_loss(ls, lt, tgt, alpha=alpha, beta=beta, temperature=temperature)
    if base == "nkd":
        return nkd_loss(ls, lt, tgt, gamma=1.0, temperature=temperature)
    raise KeyError(base)
