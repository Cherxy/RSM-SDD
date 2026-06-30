from __future__ import annotations

import torch
import torch.nn.functional as F

from ._base import Distiller


def nkd_loss_origin(logit_s, logit_t, target, temperature, gamma):
    s_i = F.log_softmax(logit_s, dim=1)
    t_i = F.softmax(logit_t, dim=1)
    loss_t = F.nll_loss(s_i, target)

    mask = torch.ones_like(logit_s).scatter_(1, target.unsqueeze(1), 0).bool()
    s_t = F.log_softmax(logit_s[mask].view(logit_s.size(0), -1) / temperature, dim=1)
    t_t = F.softmax(logit_t[mask].view(logit_t.size(0), -1) / temperature, dim=1)
    loss_non = F.kl_div(s_t, t_t, reduction="batchmean") * (temperature ** 2)
    return loss_t + gamma * loss_non


def multi_nkd_loss(out_s_multi, out_t_multi, target, temperature, gamma):
    patch_count = out_s_multi.shape[-1]
    student = out_s_multi.permute(0, 2, 1).reshape(-1, out_s_multi.shape[1])
    teacher = out_t_multi.permute(0, 2, 1).reshape(-1, out_t_multi.shape[1])
    target_repeat = target.repeat_interleave(patch_count)
    return nkd_loss_origin(student, teacher, target_repeat, temperature, gamma)


class SDD_NKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.ce_loss_weight = cfg.NKD.CE_WEIGHT
        self.warmup = cfg.SOLVER.WARMUP_EPOCHS
        self.temperature = cfg.NKD.T
        self.gamma = cfg.NKD.GAMMA
        self.M = cfg.SDD.M

    def forward_train(self, image, target, **kwargs):
        logits_student, patch_s, _, _, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, patch_t, _, _, _ = self.teacher(image)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        if self.M == "[1]" or self.M == "{1}":
            loss_kd = nkd_loss_origin(logits_student, logits_teacher, target, self.temperature, self.gamma)
        else:
            loss_kd = multi_nkd_loss(patch_s, patch_t, target, self.temperature, self.gamma)
        loss_kd = min(kwargs["epoch"] / max(self.warmup, 1), 1.0) * loss_kd
        return logits_student, {"loss_ce": loss_ce, "loss_kd": loss_kd}

