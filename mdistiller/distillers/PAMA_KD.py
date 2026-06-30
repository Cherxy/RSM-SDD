from __future__ import annotations

import torch
import torch.nn.functional as F

from ._base import Distiller
from .KD import kd_loss, sdd_kd_loss


def gac_loss(agent_tokens_student: torch.Tensor, agent_tokens_teacher: torch.Tensor):
    return F.mse_loss(agent_tokens_student, agent_tokens_teacher)


class PAMA_KD(Distiller):
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.gac_weight = cfg.PAMA.GAC_WEIGHT
        self.M = cfg.SDD.M

    def forward_train(self, image, target, **kwargs):
        logits_student, patch_s, _, _, agent_s = self.student(image)
        with torch.no_grad():
            logits_teacher, patch_t, _, _, agent_t = self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        if self.M == "[1]" or self.M == "{1}":
            loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.temperature)
        else:
            loss_kd = self.kd_loss_weight * sdd_kd_loss(patch_s, patch_t, self.temperature)
        loss_gac = self.gac_weight * gac_loss(agent_s, agent_t)
        return logits_student, {"loss_ce": loss_ce, "loss_kd": loss_kd, "loss_gac": loss_gac}

