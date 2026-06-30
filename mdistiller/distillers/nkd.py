import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def nkd_loss(logits_s, logits_t, target, gamma=1.0, temperature=4.0):
    # A stable non-target KD approximation: mask GT and align non-target distribution.
    mask = torch.zeros_like(logits_s).scatter_(1, target[:, None], 1).bool()
    s = F.log_softmax(logits_s / temperature - 1000.0 * mask.float(), dim=1)
    t = F.softmax(logits_t / temperature - 1000.0 * mask.float(), dim=1)
    return F.kl_div(s, t, reduction='batchmean') * (temperature ** 2) * gamma

class NKD(Distiller):
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
        loss_kd = nkd_loss(logits_s, logits_t, target, self.kd_weight, self.temperature)
        return {'loss': loss_ce + loss_kd, 'loss_ce': loss_ce.detach(), 'loss_kd': loss_kd.detach(), 'logits_s': logits_s}
