import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller


def _get_gt_mask(logits, target):
    return torch.zeros_like(logits).scatter_(1, target.reshape(-1,1), 1).bool()

def _get_other_mask(logits, target):
    return torch.ones_like(logits).scatter_(1, target.reshape(-1,1), 0).bool()

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdim=True)
    t2 = (t * mask2).sum(dim=1, keepdim=True)
    return torch.cat([t1, t2], dim=1)

def dkd_loss(logits_s, logits_t, target, alpha=1.0, beta=8.0, temperature=4.0, reduction='batchmean'):
    gt_mask = _get_gt_mask(logits_s, target)
    other_mask = _get_other_mask(logits_s, target)
    pred_s = F.softmax(logits_s / temperature, dim=1)
    pred_t = F.softmax(logits_t / temperature, dim=1)
    pred_s_cat = cat_mask(pred_s, gt_mask, other_mask)
    pred_t_cat = cat_mask(pred_t, gt_mask, other_mask)
    log_pred_s_cat = torch.log(pred_s_cat.clamp_min(1e-8))
    tckd_loss = F.kl_div(log_pred_s_cat, pred_t_cat, reduction=reduction) * (temperature ** 2)
    pred_t_part2 = F.softmax(logits_t / temperature - 1000.0 * gt_mask.float(), dim=1)
    log_pred_s_part2 = F.log_softmax(logits_s / temperature - 1000.0 * gt_mask.float(), dim=1)
    nckd_loss = F.kl_div(log_pred_s_part2, pred_t_part2, reduction=reduction) * (temperature ** 2)
    return alpha * tckd_loss + beta * nckd_loss

class DKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher, cfg)
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = float(cfg.DISTILLER.CE_WEIGHT)
        self.alpha = float(cfg.DISTILLER.ALPHA)
        self.beta = float(cfg.DISTILLER.BETA)
        self.temperature = float(cfg.DISTILLER.T)
        self.warmup = int(getattr(cfg.DISTILLER, 'WARMUP', 0))
    def forward_train(self, image, target, epoch=0, **kwargs):
        logits_s = self.student(image)
        with torch.no_grad(): logits_t = self.teacher(image)
        loss_ce = self.ce(logits_s, target) * self.ce_weight
        warm = min(float(epoch + 1) / max(1, self.warmup), 1.0) if self.warmup > 0 else 1.0
        loss_kd = dkd_loss(logits_s, logits_t, target, self.alpha, self.beta, self.temperature) * warm
        return {'loss': loss_ce + loss_kd, 'loss_ce': loss_ce.detach(), 'loss_kd': loss_kd.detach(), 'logits_s': logits_s}
