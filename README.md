# PAMA-SDD++

**Pyramid and Agent-Mediated Attention for Scale-Decoupled Knowledge Distillation (enhanced)**

PAMA-SDD++ is a classification-oriented knowledge distillation framework built around a single theme: **local–global semantic-consistency distillation**. A feature pyramid supplies multi-scale *local* evidence, a compact set of *agent tokens* carries *global* semantic organization, and scale-decoupled distillation (SDD) aligns them. The three cooperate toward one objective instead of being three stacked add-ons. All extra modules are training-only; inference keeps the original compact student.

<p align="center">
  <img src="assets/overview1.png" alt="PAMA-SDD++ overview" width="95%">
</p>

## Core Contributions

1. **Unified local–global consistency distillation framework.** Multi-scale local evidence (pyramid / SDD) and global semantic organization (agent tokens) are placed in one distillation framework, transferring both local detail and global semantic structure.
2. **Student-side Semantic-Prototype Routing (SPR).** CSAM uses learnable prototype queries on the student branch to cluster cross-scale region tokens into compact semantic agents. The teacher branch remains a frozen stable target; SPR only builds the student's distillation mediator.
3. **Relation-graph + reliability-aware consistency objectives.** Agent relation-graph consistency (**GAC**, used as an auxiliary relation constraint), reliability-weighted local distillation, and local–global coherence (**LGC**) jointly reduce grid-partition noise and semantic fragmentation without changing the deployed student.

Supporting components (implementation details / ablation knobs, built on cited prior work): APF channel-spatial adaptive pyramid fusion, AMA LayerScale + depth-wise local positional branch, and stable frozen-teacher targets. GSMF is kept only as an optional ablation/extension, not as the main method setting.

> APF (progressive feature pyramid), Agent Attention, and SDD are prior work and are cited in the paper. `GAC_MODE=gram` reverts GAC to the legacy cosine-Gram MSE for ablation.

### Core Contributions ↔ Code

| Contribution | File | Key symbols |
|---|---|---|
| Unified framework + GAC + LGC + reliability-aware local SDD | [`mdistiller/distillers/pama_sdd.py`](mdistiller/distillers/pama_sdd.py) | `PAMASDD`, `_gac_loss` (GAC), `_lgc_loss` (LGC), `_reliability_weight` |
| Student-side SPR / Cross-Scale Agent Mediation (CSAM) | [`mdistiller/modules/csam.py`](mdistiller/modules/csam.py) | `CSAM`, `CSAM_AGENT_INIT: routing` |
| APF++ pyramid calibration (supporting) | [`mdistiller/modules/apf.py`](mdistiller/modules/apf.py) | `APF`, `APFGate` |
| SDD local logits (supporting) | [`mdistiller/modules/spp.py`](mdistiller/modules/spp.py) | `spp_logits` |

Each of these files carries an in-code banner marking whether it is a core contribution or a supporting component.

## Method Overview

Given an input image, a frozen teacher and a trainable student extract hierarchical features. PAMA-SDD++ applies **APF** to produce calibrated pyramid features, then student-side **SPR/CSAM** uses prototype agents as semantic mediators: local region tokens are aggregated into cross-scale agents, and the agent-mediated context is broadcast back to enhanced local features. The teacher branch is not modified by APF/CSAM/SPR; it provides stable logits, local logits, and pooled-agent relation targets.

Enhanced student features are partitioned into multi-scale grids (e.g. `M = {1, 2, 4}`) and distilled against stable teacher targets with a reliability-aware local SDD loss. In parallel, **GAC** acts as an auxiliary relation constraint between student semantic agents and frozen teacher pooled-agent relations, while **LGC** keeps local predictions consistent with the teacher's global semantics. The objective combines classification, global logit distillation, local SDD, GAC and LGC:

```text
L_total = L_CE + L_global + w_sdd * L_local + w_gac * L_GAC + w_lgc * L_LGC
```

Three base logit objectives are supported: `PAMA_KD`, `PAMA_DKD`, `PAMA_NKD` (registered in `mdistiller/distillers/__init__.py`).

## Repository Layout

```text
configs/                 Experiment configs for CIFAR-100, ImageNet-1K, and CUB-200-2011
mdistiller/
  dataset/               Dataset builders (cifar100, imagenet, cub200) + augmentation
  models/                Teacher/student backbones + registry
  modules/               Core contribution modules: apf.py (APF++), csam.py (CSAM), spp.py (SDD)
  distillers/            pama_sdd.py (framework + GAC/LGC/reliability) and base losses (kd/dkd/nkd)
  engine/                cfg, trainer, utils
scripts/                 Example training scripts
tests/                   Smoke tests and dataset-path checks
tools/                   Utility / analysis scripts
train.py                 Main training entry point
README_zh.md             Chinese notes: contributions, config map, full command reference
```

Large local artifacts are intentionally git-ignored: `data/ runs/ weights/ *.log *.pth *.pt *.ckpt *.tar *.zip`.

## Installation

```bash
conda create -n pama-sdd python=3.10 -y
conda activate pama-sdd
# Install a PyTorch build matching your CUDA first, e.g. CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

Quick checks:

```bash
python tools/verify_install.py
pytest
```

## Data Preparation

Place datasets under `data/` or pass a custom path with `--data-root`:

```text
data/cifar100
data/imagenet
data/CUB_200_2011
```

For CUB-200-2011 (CaltechDATA mirror):

```bash
mkdir -p data && cd data
wget -O CUB_200_2011.tgz "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
tar -xzf CUB_200_2011.tgz && cd ..
```

## Teacher Checkpoints

Final accuracy requires pretrained teacher checkpoints; set `MODEL.TEACHER_CKPT` in each YAML. Example paths:

```text
save/models/resnet32x4_vanilla/ckpt_epoch_240.pth
save/cub200/resnet32x4/ckpt.pth
save/imagenet/resnet34/ckpt.pth
```

`--allow-random-teacher` is for debugging only and must not be used for reported accuracy.

## Quick Start

```bash
# CIFAR-100, ResNet32x4 -> MobileNetV2, PAMA-DKD
python train.py --cfg configs/cifar100/pama_dkd/res32x4_mv2.yaml \
  --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_mv2_pama_dkd --gpu 0

# Main paper setting: student-side SPR + reliability + auxiliary GAC, GSMF off
python train.py --cfg configs/cifar100/pama_dkd/res32x4_mv2_spr.yaml \
  --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_mv2_pama_dkd_spr --gpu 0

# CUB-200-2011, ResNet32x4 -> ShuffleNetV1, PAMA-KD
python train.py --cfg configs/cub200/res32x4_shuv1/pama_kd.yaml \
  --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_shuv1_pama_kd --gpu 0

# Debug with fake data and a random teacher (one epoch)
python train.py --cfg configs/cifar100/pama_dkd/res32x4_mv2.yaml \
  --debug-fake-data --allow-random-teacher --epochs 1 --output ./runs/debug_fake
```

## Config Guide

The method folder chooses the base objective; the leaf file chooses the teacher→student pair:

```text
configs/<dataset>/<method>/<pair>.yaml   # method in {pama_kd, pama_dkd, pama_nkd}
```

Key `DISTILLER` / `PAMA` keys:

- `CE_WEIGHT / KD_WEIGHT / SDD_WEIGHT / GAC_WEIGHT / LGC_WEIGHT` — loss weights. Note: DKD/NKD base losses are intrinsically much larger than KD, so `KD_WEIGHT`/`SDD_WEIGHT` usually need smaller values for those variants to keep CE from being drowned out.
- `ALPHA / BETA / T / WARMUP` — DKD α/β, KD temperature, loss-weight warmup epochs.
- `GAC_MODE` (`relation_graph` | `gram`), `GAC_TAU`.
- `PAMA.M` (SDD scales), `NUM_AGENTS` (perfect square), `NUM_HEADS`, `MAX_SPATIAL_SIZE`.
- **Main paper setting:** keep the teacher as a frozen stable target, set `PAMA.CSAM_AGENT_INIT: routing` for student-side SPR, keep `USE_RELIABILITY: true`, `USE_GAC: true`, and `USE_LGC: true`, and keep `PAMA.APF_GSMF: false`.
- `PAMA.CSAM_AGENT_INIT: routing` swaps grid-pooled student agents for Semantic-Prototype Routing agents in CSAM.
- `PAMA.APF_GSMF: true` turns on Global-Semantic Modulated Fusion in APF and is reserved for a separate GSMF ablation/extension, not the default main method.
- Ablation switches: `USE_APF / USE_AMA / USE_GAC / USE_LGC / USE_RELIABILITY`.

## Ablations & Analysis

Component ablation configs (drop one module at a time), e.g.:

```text
configs/cifar100/pama_dkd/res32x4_res8x4_no_apf.yaml
configs/cifar100/pama_dkd/res32x4_res8x4_no_ama.yaml
configs/cifar100/pama_dkd/res32x4_res8x4_no_gac.yaml
configs/cifar100/pama_dkd/res32x4_res8x4_no_lgc.yaml
configs/cifar100/pama_dkd/res32x4_res8x4_no_reliability.yaml
```

`PAMASDD.forward_analysis()` and `tools/analyze_semantic_consistency.py` export semantic-consistency metrics (`local_global_teacher_cos`, `patch_variance`, `patch_entropy`, `agent_relation_mse`, ...) that support the semantic-fragmentation motivation.

## Implementation Notes

- **AMP stability.** Distillation losses (softmax/log/KL at temperature, and the DKD/NKD masked `log_softmax`) are computed in fp32 even under AMP — in fp16 a confident teacher probability can round to exactly 0 and make `kl_div` produce NaN. The heavy backbone forward still runs in fp16.
- **DKD/NKD safety.** Those configs enable `GRAD_CLIP_NORM` so the large per-region DKD/NKD losses cannot push the fp16 attention into overflow.
- **Memory.** On 16 GB GPUs the 224² CUB pipeline can exceed VRAM at batch 64 and spill into shared memory (which is very slow); reduce `SOLVER.BATCH_SIZE` (e.g. 32) if `peak_reserved` approaches the card limit.

## Reported Results

Numbers below are from the paper draft; values in parentheses are improvements over the corresponding scale-decoupled baseline (`SD-KD/DKD/NKD`).

### CIFAR-100 Top-1

| Teacher → Student | Best PAMA variant | Top-1 |
|---|---:|---:|
| ResNet32x4 → MobileNetV2 | PAMA-DKD | 71.10 (+1.02) |
| WRN-40-2 → VGG8 | PAMA-KD | 75.22 (+0.78) |
| ResNet50 → ShuffleNetV1 | PAMA-DKD | 79.70 (+1.59) |
| ResNet32x4 → ShuffleNetV1 | PAMA-DKD | 79.18 (+1.88) |
| VGG13 → MobileNetV2 | PAMA-DKD | 71.22 (+0.97) |

### ImageNet-1K

| Teacher → Student | Variant | Top-1 | Top-5 |
|---|---|---:|---:|
| ResNet34 → ResNet18 | PAMA-NKD | 72.57 (+0.24) | 91.42 (+0.11) |
| ResNet50 → MobileNetV2 | PAMA-DKD | 73.60 (+0.52) | 91.83 (+0.74) |

### CUB-200-2011 Top-1

| Teacher → Student | Best PAMA variant | Top-1 |
|---|---:|---:|
| ResNet32x4 → MobileNetV2 | PAMA-NKD | 66.69 (+4.00) |
| ResNet32x4 → ShuffleNetV1 | PAMA-DKD | 67.05 (+1.47) |
| VGG13 → MobileNetV2 | PAMA-NKD | 69.09 (+4.46) |

(See `README_zh.md` for the full per-config command reference and the mapping to the paper's tables 4.1–4.4.)

## Citation

```bibtex
@misc{long2026pamasddpp,
  title  = {PAMA-SDD++: Pyramid and Agent-Mediated Attention for Scale-Decoupled Knowledge Distillation},
  author = {Long, Shiyu},
  year   = {2026},
  note   = {Preprint}
}
```

## Acknowledgement

MDistiller-compatible organization; builds on scale-decoupled distillation and Agent Attention (Han et al., ECCV 2024).
