from __future__ import annotations

import torch
import torch.nn.functional as F

from ._base import Distiller
from .DKD import dkd_loss


def _multi_dkd(out_s_multi, out_t_multi, target, alpha, beta, temperature):
    patch_count = out_s_multi.shape[-1]
    student = out_s_multi.permute(0, 2, 1).reshape(-1, out_s_multi.shape[1])
    teacher = out_t_multi.permute(0, 2, 1).reshape(-1, out_t_multi.shape[1])
    target_repeat = target.repeat_interleave(patch_count)
    return dkd_loss(student, teacher, target_repeat, alpha, beta, temperature)


class SDD_DKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.SOLVER.WARMUP_EPOCHS
        self.M = cfg.SDD.M

    def forward_train(self, image, target, **kwargs):
        logits_student, patch_s, _, _, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, patch_t, _, _, _ = self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        if self.M == "[1]" or self.M == "{1}":
            loss_kd = dkd_loss(logits_student, logits_teacher, target, self.alpha, self.beta, self.temperature)
        else:
            loss_kd = _multi_dkd(patch_s, patch_t, target, self.alpha, self.beta, self.temperature)
        loss_kd = min(kwargs["epoch"] / max(self.warmup, 1), 1.0) * loss_kd
        return logits_student, {"loss_ce": loss_ce, "loss_kd": loss_kd}

