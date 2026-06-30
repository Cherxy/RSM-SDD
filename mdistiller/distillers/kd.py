import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def kd_loss(logits_s, logits_t, temperature=4.0, reduction='batchmean'):
    return F.kl_div(
        F.log_softmax(logits_s / temperature, dim=1),
        F.softmax(logits_t / temperature, dim=1),
        reduction=reduction,
    ) * (temperature ** 2)


def sdd_kd_loss(patch_s, patch_t, temperature=4.0):
    """KD over local logits shaped [B, K, N]."""
    b, k, n = patch_s.shape
    s = patch_s.permute(0, 2, 1).reshape(b * n, k)
    t = patch_t.permute(0, 2, 1).reshape(b * n, k)
    return kd_loss(s, t, temperature=temperature)

class KD(Distiller):
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher, cfg)
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = float(cfg.DISTILLER.CE_WEIGHT)
        self.kd_weight = float(cfg.DISTILLER.KD_WEIGHT)
        self.temperature = float(cfg.DISTILLER.T)
    def forward_train(self, image, target, **kwargs):
        logits_s = self.student(image)
        with torch.no_grad(): logits_t = self.teacher(image)
        loss_ce = self.ce(logits_s, target) * self.ce_weight
        loss_kd = kd_loss(logits_s, logits_t, self.temperature) * self.kd_weight
        return {'loss': loss_ce + loss_kd, 'loss_ce': loss_ce.detach(), 'loss_kd': loss_kd.detach(), 'logits_s': logits_s}
