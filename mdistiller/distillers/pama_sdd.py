from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller
from .kd import kd_loss
from .dkd import dkd_loss
from .nkd import nkd_loss
from .sdd import local_kd_loss
from ..modules.apf import APF
from ..modules.ama import AgentMediatorAttention
from ..modules.spp import spp_logits


class PAMASDDBase(Distiller):
    """End-to-end PAMA-SDD framework.

    The framework follows thesis Chapter 4:
    APF -> AMA -> local SDD distillation + GAC loss + CE/global logit distillation.
    """
    base_name = 'dkd'
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher, cfg)
        label_smoothing = float(getattr(cfg.DISTILLER, 'LABEL_SMOOTHING', 0.0))
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ce_weight = float(cfg.DISTILLER.CE_WEIGHT)
        self.kd_weight = float(cfg.DISTILLER.KD_WEIGHT)
        self.sdd_weight = float(getattr(cfg.DISTILLER, 'SDD_WEIGHT', 1.0))
        self.gac_weight = float(getattr(cfg.DISTILLER, 'GAC_WEIGHT', 1.0))
        self.alpha = float(getattr(cfg.DISTILLER, 'ALPHA', 1.0))
        self.beta = float(getattr(cfg.DISTILLER, 'BETA', 8.0))
        self.temperature = float(getattr(cfg.DISTILLER, 'T', 4.0))
        self.warmup = int(getattr(cfg.DISTILLER, 'WARMUP', 0))
        self.scales = tuple(int(x) for x in getattr(cfg.PAMA, 'M', [1,2,4]))
        num_agents = int(getattr(cfg.PAMA, 'NUM_AGENTS', 16))
        heads = int(getattr(cfg.PAMA, 'NUM_HEADS', 4))
        embed_dim = int(getattr(cfg.PAMA, 'EMBED_DIM', 256))
        self.max_spatial_size = int(getattr(cfg.PAMA, 'MAX_SPATIAL_SIZE', 0))
        s_ch = student.get_feature_channels()
        t_ch = teacher.get_feature_channels()
        self.apf_s = APF(s_ch, out_channels=s_ch[-1])
        self.apf_t = APF(t_ch, out_channels=t_ch[-1])
        self.ama_s = AgentMediatorAttention(s_ch[-1], num_agents=num_agents, num_heads=max(1, min(heads, s_ch[-1] // 8 if s_ch[-1] >= 8 else 1)))
        self.ama_t = AgentMediatorAttention(t_ch[-1], num_agents=num_agents, num_heads=max(1, min(heads, t_ch[-1] // 8 if t_ch[-1] >= 8 else 1)))
        self.agent_proj_s = nn.Linear(s_ch[-1], embed_dim)
        self.agent_proj_t = nn.Linear(t_ch[-1], embed_dim)

    def train(self, mode: bool = True):
        """Keep teacher frozen in eval mode even when distiller enters train mode.

        This avoids teacher BN/Dropout statistic drift during student training,
        which can destabilize distillation targets and hurt final accuracy.
        """
        super().train(mode)
        self.teacher.eval()
        return self

    def _global_kd(self, logits_s, logits_t, target):
        if self.base_name == 'kd':
            return kd_loss(logits_s, logits_t, self.temperature) * self.kd_weight
        if self.base_name == 'dkd':
            return dkd_loss(logits_s, logits_t, target, self.alpha, self.beta, self.temperature) * self.kd_weight
        if self.base_name == 'nkd':
            return nkd_loss(logits_s, logits_t, target, self.kd_weight, self.temperature)
        raise KeyError(self.base_name)
    def _gac_loss(self, agents_s, agents_t):
        ps = F.normalize(self.agent_proj_s(agents_s), dim=-1)
        pt = F.normalize(self.agent_proj_t(agents_t.detach()), dim=-1)
        return F.mse_loss(ps, pt)

    def _cap_spatial_size(self, features):
        if self.max_spatial_size <= 0:
            return features
        capped = []
        for feat in features:
            h, w = feat.shape[-2:]
            longest = max(h, w)
            if longest > self.max_spatial_size:
                scale = self.max_spatial_size / float(longest)
                size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
                feat = F.interpolate(feat, size=size, mode='bilinear', align_corners=False)
            capped.append(feat)
        return capped

    def forward_train(self, image, target, epoch=0, **kwargs):
        logits_s, feats_s = self.student(image, return_features=True)
        with torch.no_grad():
            logits_t, feats_t = self.teacher(image, return_features=True)
        feats_s = self._cap_spatial_size(feats_s)
        feats_t = self._cap_spatial_size(feats_t)
        # APF teacher branch remains no-grad to save memory.
        pfeats_s = self.apf_s(feats_s)
        with torch.no_grad():
            pfeats_t = self.apf_t(feats_t)
        # Use the final top-down APF output: the highest-resolution feature now
        # contains progressively injected deep semantics from all pyramid levels.
        distill_feat_s = pfeats_s[0]
        distill_feat_t = pfeats_t[0]
        enh_s, agents_s = self.ama_s(distill_feat_s)
        with torch.no_grad():
            enh_t, agents_t = self.ama_t(distill_feat_t)
        local_s = spp_logits(enh_s, self.student.classifier, scales=self.scales)
        local_t = spp_logits(enh_t, self.teacher.classifier, scales=self.scales)
        loss_ce = self.ce(logits_s, target) * self.ce_weight
        warm = min(float(epoch + 1) / max(1, self.warmup), 1.0) if self.warmup > 0 else 1.0
        loss_global = self._global_kd(logits_s, logits_t, target) * warm
        loss_local = local_kd_loss(local_s, local_t, target, base=self.base_name, scales=self.scales, alpha=self.alpha, beta=self.beta, temperature=self.temperature) * self.sdd_weight * warm
        loss_gac = self._gac_loss(agents_s, agents_t) * self.gac_weight * warm
        loss = loss_ce + loss_global + loss_local + loss_gac
        return {'loss': loss, 'loss_ce': loss_ce.detach(), 'loss_global': loss_global.detach(), 'loss_local': loss_local.detach(), 'loss_gac': loss_gac.detach(), 'logits_s': logits_s}

class PAMAKD(PAMASDDBase):
    base_name = 'kd'
class PAMADKD(PAMASDDBase):
    base_name = 'dkd'
class PAMANKD(PAMASDDBase):
    base_name = 'nkd'
