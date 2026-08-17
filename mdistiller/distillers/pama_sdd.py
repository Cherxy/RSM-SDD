"""PAMA-SDD++ -- unified local-global consistency distillation framework.

PAMA-SDD++ transfers knowledge from a frozen teacher to a student through a
single pipeline that reconciles *local* (region-level) and *global*
(image-level) predictions. The macro data flow is::

    student features
        |
        v
    [Stage 1] APF++   : calibrate the feature pyramid          (modules/apf.py)
        |
        v
    [Stage 2] CSAM    : cross-scale agent mediation            (modules/csam.py)
        |
        +--> enhanced base feature --> [Stage 3] SDD local logits  (modules/spp.py)
        |
        +--> agent tokens --------------------------------+
                                                          v
    teacher (frozen) --> stable targets --> losses:  CE + global KD
                                                     + reliability-aware local SDD  (++)
                                                     + GAC relation-graph agents    (++)
                                                     + LGC local-global coherence   (++)

Innovation map (paper contribution -> code)
--------------------------------------------
* APF    channel-spatial adaptive pyramid fusion ....... modules/apf.py: APF / APFGate
    - GSMF global-semantic modulated fusion is kept as an optional ablation
* CSAM   cross-scale agent mediation ................... modules/csam.py: CSAM
    - SPR semantic-prototype routing on the student side (main setting)
* SDD    scale-decoupled local logits (prior work) ..... modules/spp.py: spp_logits
* ++ 1   reliability-aware local SDD ................... _reliability_weight + distillers/sdd.py: local_kd_loss
* ++ 2   GAC relation-graph agent consistency .......... _gac_loss
* ++ 3   LGC local-global coherence .................... _lgc_loss
* stable teacher targets (no trainable teacher-side APF/AMA) ... _teacher_targets

``base_name`` selects the backbone global/local objective (KD / DKD / NKD); the
subclasses ``PAMAKD`` / ``PAMADKD`` / ``PAMANKD`` bind it.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from .kd import kd_loss
from .dkd import dkd_loss
from .nkd import nkd_loss
from .sdd import local_kd_loss
from ..modules.apf import APF
from ..modules.csam import CSAM
from ..modules.spp import spp_logits


def _cfg_get(obj: Any, name: str, default):
    """Read ``obj.name`` with a fallback, tolerating a missing config node."""
    return getattr(obj, name, default) if obj is not None else default


class PAMASDD(Distiller):
    """PAMA-SDD++ local-global consistency distiller.

    See the module docstring for the full pipeline and the innovation map. In
    short: the student pyramid is calibrated (APF++), mediated by cross-scale
    agents (CSAM), and scored into multi-scale local logits (SDD); the student
    is then supervised by CE + global KD plus the three ++ objectives
    (reliability-aware local SDD, GAC agent-relation consistency, LGC
    local-global coherence) against stable frozen-teacher targets.

    Subclasses set :attr:`base_name` to pick the backbone KD objective.
    """

    #: backbone global/local objective: "kd" | "dkd" | "nkd" (set by subclasses)
    base_name = "dkd"

    # =====================================================================
    # Construction & config
    # =====================================================================
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher, cfg)
        pama_cfg = getattr(cfg, "PAMA", None)
        dist_cfg = cfg.DISTILLER

        # --- Loss weights, temperature, warmup -------------------------------
        label_smoothing = float(_cfg_get(dist_cfg, "LABEL_SMOOTHING", 0.0))
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ce_weight = float(dist_cfg.CE_WEIGHT)
        self.kd_weight = float(_cfg_get(dist_cfg, "KD_WEIGHT", 1.0))
        self.sdd_weight = float(_cfg_get(dist_cfg, "SDD_WEIGHT", 1.0))
        self.gac_weight = float(_cfg_get(dist_cfg, "GAC_WEIGHT", 0.5))
        self.lgc_weight = float(_cfg_get(dist_cfg, "LGC_WEIGHT", 0.5))
        self.alpha = float(_cfg_get(dist_cfg, "ALPHA", 1.0))
        self.beta = float(_cfg_get(dist_cfg, "BETA", 8.0))
        self.temperature = float(_cfg_get(dist_cfg, "T", 4.0))
        self.warmup = int(_cfg_get(dist_cfg, "WARMUP", 0))
        # GAC variant: "relation_graph" matches each agent's relational
        # distribution (KL) + a structural MSE; "gram" is the legacy cosine-Gram
        # MSE kept for ablation.
        self.gac_mode = str(_cfg_get(dist_cfg, "GAC_MODE", "relation_graph")).lower()
        self.gac_tau = float(_cfg_get(dist_cfg, "GAC_TAU", 1.0))

        # --- Component on/off switches & hyper-parameters (PAMA.*) -----------
        self.scales = tuple(int(x) for x in _cfg_get(pama_cfg, "M", [1, 2, 4]))
        self.use_apf = bool(_cfg_get(pama_cfg, "USE_APF", True))
        self.use_ama = bool(_cfg_get(pama_cfg, "USE_AMA", True))
        self.use_gac = bool(_cfg_get(pama_cfg, "USE_GAC", True))
        self.use_lgc = bool(_cfg_get(pama_cfg, "USE_LGC", True))
        self.use_reliability = bool(_cfg_get(pama_cfg, "USE_RELIABILITY", True))
        self.pyramid_local = bool(_cfg_get(pama_cfg, "PYRAMID_LOCAL", True))
        # Faithful Scale-Decoupled Distillation (SDD, CVPR 2024): reweight local
        # regions by the consistent vs complementary (teacher local-vs-global) split.
        self.use_sdd_decouple = bool(_cfg_get(pama_cfg, "SDD_DECOUPLE", True))
        self.sdd_consistent_w = float(_cfg_get(pama_cfg, "SDD_CONSISTENT_W", 1.0))
        self.sdd_complementary_w = float(_cfg_get(pama_cfg, "SDD_COMPLEMENTARY_W", 2.0))
        self.teacher_target_mode = str(_cfg_get(pama_cfg, "TEACHER_TARGET_MODE", "stable")).lower()
        self.reliability_floor = float(_cfg_get(pama_cfg, "RELIABILITY_FLOOR", 0.2))
        self.reliability_power = float(_cfg_get(pama_cfg, "RELIABILITY_POWER", 1.0))
        self.max_spatial_size = int(_cfg_get(pama_cfg, "MAX_SPATIAL_SIZE", 0))

        # --- Agent geometry --------------------------------------------------
        num_agents = int(_cfg_get(pama_cfg, "NUM_AGENTS", 16))
        pool_size = int(math.sqrt(num_agents))
        if pool_size * pool_size != num_agents:
            raise ValueError("PAMA.NUM_AGENTS must be a perfect square")
        self.num_agents = num_agents
        self.agent_pool_size = pool_size
        heads = int(_cfg_get(pama_cfg, "NUM_HEADS", 4))
        layer_scale = float(_cfg_get(pama_cfg, "CSAM_LAYER_SCALE", _cfg_get(pama_cfg, "AMA_LAYER_SCALE", 1e-4)))
        apf_gamma = float(_cfg_get(pama_cfg, "APF_INIT_GAMMA", 0.5))
        # Main paper setting:
        #   CSAM_AGENT_INIT="routing" enables student-side SPR while the teacher
        #   remains a frozen stable target.
        # GSMF is kept as an optional APF ablation/extension, not a default
        # main-method component.
        apf_gsmf = bool(_cfg_get(pama_cfg, "APF_GSMF", _cfg_get(pama_cfg, "APF_SEMANTIC_MOD", False)))
        ama_agent_init = str(_cfg_get(pama_cfg, "CSAM_AGENT_INIT", _cfg_get(pama_cfg, "AMA_AGENT_INIT", "pool"))).lower()

        # --- Student-side trainable modules (APF++ then CSAM) ----------------
        s_ch = student.get_feature_channels()
        self.student_channels = list(s_ch)
        self.classifier_channels = int(s_ch[-1])
        self.aux_channels = int(_cfg_get(pama_cfg, "AUX_CHANNELS", _cfg_get(pama_cfg, "EMBED_DIM", self.classifier_channels)))
        self.aux_channels = max(1, self.aux_channels)
        if self.use_apf:
            self.apf_s = APF(s_ch, out_channels=self.aux_channels, init_gamma=apf_gamma, gsmf=apf_gsmf)
            self.aux_proj_s = nn.Identity()
        else:
            self.apf_s = nn.Identity()
            if self.aux_channels == self.classifier_channels:
                self.aux_proj_s = nn.Identity()
            else:
                self.aux_proj_s = nn.Sequential(
                    nn.Conv2d(self.classifier_channels, self.aux_channels, 1, bias=False),
                    nn.BatchNorm2d(self.aux_channels),
                    nn.ReLU(inplace=True),
                )
        if self.aux_channels == self.classifier_channels:
            self.local_proj_s = nn.Identity()
        else:
            self.local_proj_s = nn.Conv2d(self.aux_channels, self.classifier_channels, 1, bias=True)

        safe_heads = max(1, min(heads, self.aux_channels))
        while self.aux_channels % safe_heads != 0 and safe_heads > 1:
            safe_heads -= 1
        # CSAM agents aggregate every calibrated pyramid level when APF is on;
        # otherwise a single level (legacy single-scale).
        ama_levels = len(self.student_channels) if self.use_apf else 1
        self.ama_s = CSAM(
            self.aux_channels, num_agents=num_agents, num_heads=safe_heads,
            layer_scale=layer_scale, num_levels=ama_levels, agent_init=ama_agent_init,
        ) if self.use_ama else None

        # Teacher side is frozen: stable targets come from the teacher backbone
        # + classifier and a non-parametric pooled-agent readout (no trainable
        # teacher-side APF/AMA).
        self.teacher_agent_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

    def train(self, mode: bool = True):
        # Keep the teacher frozen (eval) regardless of the distiller's mode.
        super().train(mode)
        self.teacher.eval()
        return self

    # =====================================================================
    # Stage 1 -- APF++: student feature-pyramid calibration  (modules/apf.py)
    # =====================================================================
    def _cap_spatial_size(self, features):
        """Optionally downsample oversized feature maps (memory guard).

        No-op unless ``PAMA.MAX_SPATIAL_SIZE`` > 0. Only maps whose longest side
        exceeds the cap are bilinearly shrunk, preserving aspect ratio.
        """
        if self.max_spatial_size <= 0:
            return features
        capped = []
        for feat in features:
            h, w = feat.shape[-2:]
            longest = max(h, w)
            if longest > self.max_spatial_size:
                scale = self.max_spatial_size / float(longest)
                size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
                feat = F.interpolate(feat, size=size, mode="bilinear", align_corners=False)
            capped.append(feat)
        return capped

    def _student_pyramid(self, feats_s):
        """Return the APF++-calibrated pyramid (coarse->fine order from APF).

        Without APF this degrades to the single deepest feature, matching the
        legacy single-scale behavior.
        """
        if self.use_apf:
            return self.apf_s(feats_s)
        return [self.aux_proj_s(feats_s[-1])]

    # =====================================================================
    # Stage 2 -- CSAM: cross-scale agent mediation  (modules/csam.py)
    # =====================================================================
    def _enhance_student(self, pyramid):
        """Enhance EVERY pyramid level via cross-scale agents; read out agents.

        Returns ``(enhanced_pyramid, agents)``. CSAM builds one shared set of
        cross-scale agents from all levels and mediates that context back into
        each level, so every scale is both a source and a target of enhancement.
        ``agents`` is the compact [B, A, C] token set later distilled by GAC.
        Without CSAM the pyramid passes through unchanged and agents are pooled
        from the finest level.
        """
        if self.use_ama:
            enh_pyramid, agents = self.ama_s(pyramid[0], context_feats=pyramid)
        else:
            enh_pyramid = list(pyramid)
            agents = self._pool_agents(pyramid[0])
        return enh_pyramid, agents

    def _pool_agents(self, feat):
        """Non-parametric agent readout: grid pool -> [B, A, C] tokens."""
        return self.teacher_agent_pool(feat).flatten(2).transpose(1, 2)

    # =====================================================================
    # Stage 3 -- SDD local logits: multi-scale region scoring  (modules/spp.py)
    # =====================================================================
    def _feature_for_scale(self, pyramid, scale: int):
        """Pick the pyramid level to score at a given SDD grid ``scale``.

        Coarser grids (small ``scale``) read from deeper/semantically stronger
        levels; finer grids read from shallower levels. Falls back to the base
        level when pyramid-aware local scoring is disabled or not applicable.
        """
        if not self.pyramid_local or len(pyramid) == 1 or len(self.scales) == 1:
            return pyramid[0]
        ordered = sorted(set(self.scales))
        rank = ordered.index(scale)
        denom = max(1, len(ordered) - 1)
        idx = int(round((len(pyramid) - 1) * (1.0 - rank / denom)))
        idx = max(0, min(len(pyramid) - 1, idx))
        return pyramid[idx]

    def _pyramid_local_logits(self, pyramid, classifier):
        """Concatenate per-scale SDD local logits -> [B, K, sum_s(s*s)]."""
        logits = []
        for scale in self.scales:
            feat = self._feature_for_scale(pyramid, scale)
            feat = self.local_proj_s(feat)
            logits.append(spp_logits(feat, classifier, scales=(scale,)))
        return torch.cat(logits, dim=2)

    # =====================================================================
    # Stable teacher targets (frozen teacher; no teacher-side APF/AMA)
    # =====================================================================
    def _teacher_targets(self, feats_t):
        """Stable teacher targets: deepest feature, local logits, pooled agents."""
        feat_t = feats_t[-1]
        agents_t = self._pool_agents(feat_t)
        local_t = spp_logits(feat_t, self.teacher.classifier, scales=self.scales)
        return feat_t, local_t, agents_t

    # =====================================================================
    # ++ objective 1 -- reliability-aware local weighting
    # =====================================================================
    def _reliability_weight(self, local_t):
        """Per-region reliability weight in [floor, 1] from teacher confidence.

        Confident teacher regions get weight ~1; near-uniform (background/noisy)
        regions are down-weighted toward ``reliability_floor``. Returns ``None``
        when disabled, which makes the local/LGC losses fall back to plain means.
        Shapes: local_t [B, K, N] -> weight [B, N].
        """
        if not self.use_reliability:
            return None
        with torch.no_grad():
            prob_t = F.softmax(local_t / self.temperature, dim=1)
            conf = prob_t.max(dim=1).values
            num_classes = local_t.shape[1]
            lower = 1.0 / max(1, num_classes)
            rel = ((conf - lower) / (1.0 - lower)).clamp(0.0, 1.0)
            if self.reliability_power != 1.0:
                rel = rel.pow(self.reliability_power)
            if self.reliability_floor > 0:
                rel = self.reliability_floor + (1.0 - self.reliability_floor) * rel
            return rel.detach()

    # =====================================================================
    # Loss terms -- global KD + ++ objectives (GAC, LGC)
    # =====================================================================
    def _global_kd(self, logits_s, logits_t, target):
        """Image-level backbone KD selected by ``base_name`` (KD/DKD/NKD)."""
        if self.base_name == "kd":
            return kd_loss(logits_s, logits_t, self.temperature) * self.kd_weight
        if self.base_name == "dkd":
            return dkd_loss(logits_s, logits_t, target, self.alpha, self.beta, self.temperature) * self.kd_weight
        if self.base_name == "nkd":
            return nkd_loss(logits_s, logits_t, target, self.kd_weight, self.temperature)
        raise KeyError(self.base_name)

    def _gac_loss(self, agents_s, agents_t):
        """++ GAC: transfer the teacher's agent *relation graph* to the student.

        Builds a cosine relation graph over agents for each side, then aligns
        them. "relation_graph" (default) matches each agent's relational
        distribution over the other agents (KL) + a light structural MSE for
        stability; "gram" is the legacy plain-MSE ablation. agents_* : [B, A, C].
        """
        if not self.use_gac or self.gac_weight <= 0:
            return agents_s.new_zeros(())
        s = F.normalize(agents_s, dim=-1)
        t = F.normalize(agents_t.detach(), dim=-1)
        rel_s = s @ s.transpose(-1, -2)  # [B, A, A] cosine agent relation graph
        rel_t = t @ t.transpose(-1, -2)
        if self.gac_mode == "gram":
            return F.mse_loss(rel_s, rel_t)
        a = rel_s.shape[-1]
        tau = self.gac_tau
        log_ps = F.log_softmax(rel_s / tau, dim=-1).reshape(-1, a)
        pt = F.softmax(rel_t / tau, dim=-1).reshape(-1, a)
        kl = F.kl_div(log_ps, pt, reduction="batchmean") * (tau ** 2)
        return kl + F.mse_loss(rel_s, rel_t)

    def _lgc_loss(self, local_s, logits_t, region_weight=None):
        """++ LGC: pull every student local prediction toward the global teacher.

        Reduces fragmented/inconsistent local predictions by KL-aligning each of
        the N region logits to the single global teacher distribution, optionally
        reweighted per region by reliability. Shapes: local_s [B, K, N],
        logits_t [B, K].
        """
        if not self.use_lgc or self.lgc_weight <= 0:
            return local_s.new_zeros(())
        b, k, n = local_s.shape
        ls = local_s.permute(0, 2, 1).reshape(b * n, k)
        gt = logits_t.detach().unsqueeze(1).expand(b, n, k).reshape(b * n, k)
        loss_vec = F.kl_div(
            F.log_softmax(ls / self.temperature, dim=1),
            F.softmax(gt / self.temperature, dim=1),
            reduction="none",
        ).sum(dim=1) * (self.temperature ** 2)
        if region_weight is None:
            return loss_vec.mean()
        weight = region_weight.reshape(b * n).to(loss_vec.device, dtype=loss_vec.dtype).detach()
        return (loss_vec * weight).sum() / weight.sum().clamp_min(1e-6)

    # =====================================================================
    # Forward orchestration
    # =====================================================================
    def _warmup_factor(self, epoch):
        """Linear distillation warmup in (0, 1]; 1.0 when warmup is disabled."""
        if self.warmup > 0:
            return min(float(epoch + 1) / max(1, self.warmup), 1.0)
        return 1.0

    def _collect_outputs(self, image):
        """Run both networks and assemble every tensor the losses/metrics need.

        Student path: features -> APF++ pyramid -> CSAM enhances every level +
        reads out cross-scale agents -> SDD local logits over the fully enhanced
        pyramid. Teacher path: frozen forward -> stable targets.
        """
        logits_s, feats_s = self.student(image, return_features=True)
        with torch.no_grad():
            logits_t, feats_t = self.teacher(image, return_features=True)
        feats_s = self._cap_spatial_size(feats_s)
        feats_t = self._cap_spatial_size(feats_t)
        pyramid_s = self._student_pyramid(feats_s)
        enh_pyramid_s, agents_s = self._enhance_student(pyramid_s)
        # Every level is CSAM-enhanced, so local logits at all scales read from
        # the enhanced pyramid (not just the finest level).
        local_s = self._pyramid_local_logits(enh_pyramid_s, self.student.classifier)
        _, local_t, agents_t = self._teacher_targets(feats_t)
        return {
            "logits_s": logits_s,
            "logits_t": logits_t,
            "feats_s": feats_s,
            "feats_t": feats_t,
            "pyramid_s": pyramid_s,
            "enh_pyramid_s": enh_pyramid_s,
            "enh_s": enh_pyramid_s[0],
            "agents_s": agents_s,
            "agents_t": agents_t,
            "local_s": local_s,
            "local_t": local_t,
        }

    def _compute_losses(self, out, target, epoch):
        """Combine CE + global KD + the three ++ objectives into the loss dict.

        Must be called inside the fp32 island opened by :meth:`forward_train`.
        Every distillation term is warmup-scaled identically (CE is not).
        """
        logits_s = out["logits_s"].float()
        logits_t = out["logits_t"].float()
        local_s = out["local_s"].float()
        local_t = out["local_t"].float()
        agents_s = out["agents_s"].float()
        agents_t = out["agents_t"].float()
        region_weight = self._reliability_weight(local_t)

        warm = self._warmup_factor(epoch)
        loss_ce = self.ce(logits_s, target) * self.ce_weight
        loss_global = self._global_kd(logits_s, logits_t, target) * warm
        loss_local = local_kd_loss(
            local_s,
            local_t,
            target,
            base=self.base_name,
            scales=self.scales,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
            region_weight=region_weight,
            sdd_decouple=self.use_sdd_decouple,
            consistent_weight=self.sdd_consistent_w,
            complementary_weight=self.sdd_complementary_w,
        ) * self.sdd_weight * warm
        loss_gac = self._gac_loss(agents_s, agents_t) * self.gac_weight * warm
        loss_lgc = self._lgc_loss(local_s, logits_t, region_weight) * self.lgc_weight * warm
        loss = loss_ce + loss_global + loss_local + loss_gac + loss_lgc
        return {
            "loss": loss,
            "loss_ce": loss_ce.detach(),
            "loss_global": loss_global.detach(),
            "loss_local": loss_local.detach(),
            "loss_gac": loss_gac.detach(),
            "loss_lgc": loss_lgc.detach(),
            "logits_s": logits_s,
        }

    def forward_train(self, image, target, epoch=0, **kwargs):
        out = self._collect_outputs(image)
        # fp32 island: the distillation math (softmax/log/kl at temperature and
        # the DKD/NKD masked log_softmax) is numerically fragile under AMP -- in
        # fp16 a very confident teacher probability can round to exactly 0, so
        # kl_div evaluates 0*log(0) = NaN (observed as local/gac/lgc going NaN
        # first). The heavy backbone forward above stays fp16 for speed/memory;
        # only this small loss math is forced to fp32.
        with torch.autocast(device_type=image.device.type, enabled=False):
            return self._compute_losses(out, target, epoch)

    @torch.no_grad()
    def forward_analysis(self, image, target=None):
        """Diagnostics for the semantic-fragmentation study (no training effect).

        Exports local-vs-global agreement, per-patch dispersion, and the agent
        relation gap. The metric keys are consumed by
        tools/analyze_semantic_consistency.py -- do not rename them.
        """
        was_training = self.training
        self.eval()
        out = self._collect_outputs(image)
        local_s = out["local_s"]
        logits_s = out["logits_s"]
        logits_t = out["logits_t"]
        prob_local = F.softmax(local_s, dim=1).permute(0, 2, 1)
        prob_global_t = F.softmax(logits_t, dim=1).unsqueeze(1)
        prob_global_s = F.softmax(logits_s, dim=1).unsqueeze(1)
        local_global_teacher_cos = F.cosine_similarity(prob_local, prob_global_t.expand_as(prob_local), dim=-1).mean()
        local_global_student_cos = F.cosine_similarity(prob_local, prob_global_s.expand_as(prob_local), dim=-1).mean()
        patch_variance = prob_local.var(dim=1).mean()
        patch_entropy = -(prob_local * prob_local.clamp_min(1e-8).log()).sum(dim=-1).mean()
        rel_loss = self._gac_loss(out["agents_s"], out["agents_t"])
        metrics = {
            "local_global_teacher_cos": local_global_teacher_cos,
            "local_global_student_cos": local_global_student_cos,
            "patch_variance": patch_variance,
            "patch_entropy": patch_entropy,
            "agent_relation_mse": rel_loss,
        }
        if target is not None:
            metrics["top1"] = (logits_s.argmax(dim=1) == target).float().mean()
        if was_training:
            self.train(True)
        out["metrics"] = metrics
        return out


# ---------------------------------------------------------------------------
# Backbone-objective bindings (registered in distillers/__init__.py)
# ---------------------------------------------------------------------------
class PAMAKD(PAMASDD):
    """PAMA-SDD++ with a vanilla-KD backbone objective."""
    base_name = "kd"


class PAMADKD(PAMASDD):
    """PAMA-SDD++ with a DKD backbone objective."""
    base_name = "dkd"


class PAMANKD(PAMASDD):
    """PAMA-SDD++ with an NKD backbone objective."""
    base_name = "nkd"


# Backward-compatible alias (legacy name).
PAMASDDBase = PAMASDD
