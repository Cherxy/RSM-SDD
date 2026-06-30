from __future__ import annotations

import torch
import torch.nn.functional as F

from ._base import Distiller
from .PAMA_KD import gac_loss
from .SDD_NKD import multi_nkd_loss, nkd_loss_origin


class PAMA_NKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.ce_loss_weight = cfg.NKD.CE_WEIGHT
        self.warmup = cfg.SOLVER.WARMUP_EPOCHS
        self.temperature = cfg.NKD.T
        self.gamma = cfg.NKD.GAMMA
        self.gac_weight = cfg.PAMA.GAC_WEIGHT
        self.M = cfg.SDD.M

    def forward_train(self, image, target, **kwargs):
        logits_student, patch_s, _, _, agent_s = self.student(image)
        with torch.no_grad():
            logits_teacher, patch_t, _, _, agent_t = self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        if self.M == "[1]" or self.M == "{1}":
            loss_kd = nkd_loss_origin(logits_student, logits_teacher, target, self.temperature, self.gamma)
        else:
            loss_kd = multi_nkd_loss(patch_s, patch_t, target, self.temperature, self.gamma)
        loss_kd = min(kwargs["epoch"] / max(self.warmup, 1), 1.0) * loss_kd
        loss_gac = self.gac_weight * gac_loss(agent_s, agent_t)
        return logits_student, {"loss_ce": loss_ce, "loss_kd": loss_kd, "loss_gac": loss_gac}

