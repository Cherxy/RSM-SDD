# PAMA-SDD 复现实验 README（中文）
本文档按照用户提供的论文截图表 4.1、表 4.2、表 4.3、表 4.4，对 `configs/` 下的教师-学生组合进行了重新整理，并给出 CIFAR-100、ImageNet、CUB-200-2011 三个数据集的完整训练命令。
> 说明：当前压缩包只覆盖原仓库已有的 PAMA-KD、PAMA-DKD、PAMA-NKD 三类配置。截图中的 FitNet、SP、CRD、SemCKD、ReviewKD、MGD、KD、DKD、NKD、SD-KD、SD-DKD、SD-NKD 等非 PAMA 基线，只有在仓库中实现对应 distiller 并补充配置后才能训练。
## 0. 方法概览与创新点（PAMA-SDD++）

宏观流程：`骨干特征 -> APF 金字塔校准 -> CSAM 跨尺度智能体中介 -> 局部 SDD 蒸馏 + 智能体关系图蒸馏 + 局部-全局一致性`。

统一主题是**局部-全局语义一致性蒸馏**：金字塔提供多尺度局部证据，一组智能体（agent tokens）承载全局语义组织，SDD 负责解耦对齐。三者服务于同一目标，而非独立堆叠的三个模块。

三个核心创新点：

1. **统一的局部-全局语义一致性蒸馏框架**：把多尺度局部证据（金字塔 / SDD）与全局语义组织（智能体）纳入同一蒸馏框架，兼顾局部细节与全局语义结构的迁移。
2. **跨尺度智能体中介（CSAM）**：一组 agent token 以可学习的逐尺度权重从所有金字塔层聚合多尺度证据，并同时作为**师生跨网络的紧凑语义传递接口**。其两段式 agent 注意力形式借鉴 Agent Attention（Han 等, ECCV 2024），此处将其**重新定位**为蒸馏中介——这一角色是本文的新贡献，原始注意力机制按规范引用。
3. **关系图 + 可靠性感知的一致性目标**：智能体关系图蒸馏（GAC，按关系分布 KL + 结构 MSE 对齐师生 agent 关系图）、可靠性加权局部蒸馏、局部-全局一致性，共同缓解局部分区噪声与语义碎片化。

支撑组件（写入实现细节与消融，不单列为创新点）：APF 通道-空间自适应融合、AMA 的 LayerScale 与深度可分离局部位置分支、稳定教师目标（frozen teacher + 非参数化 agent 目标）。

> 说明：APF（渐进式特征金字塔）、Agent Attention、SDD 均为已有工作，正文需正常引用。本框架的贡献在于统一框架、CSAM 的跨尺度聚合与“蒸馏中介”新角色、以及关系图 / 可靠性一致性目标；`GAC_MODE=gram` 可退化为旧版纯 Gram MSE 作消融对照。

### 0.1 创新点 ↔ 代码位置

| 创新点 | 文件 | 关键符号 |
|---|---|---|
| 统一框架 + GAC + LGC + 可靠性加权局部 SDD | `mdistiller/distillers/pama_sdd.py` | `PAMASDD`、`_gac_loss`(GAC)、`_lgc_loss`(LGC)、`_reliability_weight` |
| 跨尺度智能体中介 CSAM | `mdistiller/modules/csam.py` | `CSAM` |
| APF++ 金字塔校准（支撑组件） | `mdistiller/modules/apf.py` | `APF`、`APFGate` |
| SDD 局部 logits（支撑组件） | `mdistiller/modules/spp.py` | `spp_logits` |

以上文件顶部都加了"核心贡献 / 支撑组件"标注头。实现要点补充：蒸馏损失在 AMP 下用 fp32 计算以防 NaN；DKD/NKD 配置开启 `GRAD_CLIP_NORM`；16GB 显卡上 224² 训练如显存吃紧,把 `SOLVER.BATCH_SIZE` 降到 32。

## 1. 配置与截图表格的对应关系
缩写说明：`RN32x4 = ResNet32x4`，`RN50 = ResNet50`，`WRN40-2 = wrn_40_2`，`MobNetv2 = MobileNetV2`，`ShuNetv1 = ShuffleNetV1`，`ShuNetv2 = ShuffleNetV2`。
### 1.1 CIFAR-100：表 4.1 不同特征层级教师-学生对
| 表格 | 教师 -> 学生 | 配置文件名 |
|---|---|---|
| 表 4.1 | ResNet32x4 -> MobileNetV2 | `configs/cifar100/<method>/res32x4_mv2.yaml` |
| 表 4.1 | WRN40-2 -> VGG8 | `configs/cifar100/<method>/wrn40_2_vgg8.yaml` |
| 表 4.1 | WRN40-2 -> MobileNetV2 | `configs/cifar100/<method>/wrn40_2_mv2.yaml` |
| 表 4.1 | ResNet50 -> ShuffleNetV1 | `configs/cifar100/<method>/r50_shuv1.yaml` |

### 1.2 CIFAR-100：表 4.2 相同特征层级教师-学生对
| 表格 | 教师 -> 学生 | 配置文件名 |
|---|---|---|
| 表 4.2 | ResNet32x4 -> ShuffleNetV1 | `configs/cifar100/<method>/res32x4_shuv1.yaml` |
| 表 4.2 | WRN40-2 -> ShuffleNetV1 | `configs/cifar100/<method>/wrn40_2_shuv1.yaml` |
| 表 4.2 | ResNet50 -> MobileNetV2 | `configs/cifar100/<method>/r50_mv2.yaml` |
| 表 4.2 | VGG13 -> MobileNetV2 | `configs/cifar100/<method>/vgg13_mv2.yaml` |
| 表 4.2 | ResNet32x4 -> ShuffleNetV2 | `configs/cifar100/<method>/res32x4_shuv2.yaml` |
| 表 4.2 | ResNet50 -> VGG8 | `configs/cifar100/<method>/r50_vgg8.yaml` |

### 1.3 ImageNet：表 4.3
| 表格 | 教师 -> 学生 | 配置目录 |
|---|---|---|
| 表 4.3 | ResNet34 -> ResNet18 | `configs/imagenet/r34_r18/pama_*.yaml` |
| 表 4.3 | ResNet50 -> MobileNetV2 | `configs/imagenet/r50_mv2/pama_*.yaml` |

### 1.4 CUB-200-2011：表 4.4
| 表格 | 教师 -> 学生 | 配置目录 |
|---|---|---|
| 表 4.4 | ResNet32x4 -> MobileNetV2 | `configs/cub200/res32x4_mv2/pama_*.yaml` |
| 表 4.4 | ResNet32x4 -> ShuffleNetV1 | `configs/cub200/res32x4_shuv1/pama_*.yaml` |
| 表 4.4 | VGG13 -> MobileNetV2 | `configs/cub200/vgg13_mv2/pama_*.yaml` |
| 表 4.4 | VGG13 -> VGG8 | `configs/cub200/vgg13_vgg8/pama_*.yaml` |
| 表 4.4 | ResNet50 -> ShuffleNetV1 | `configs/cub200/resnet50_shuv1/pama_*.yaml` |

## 2. 数据集路径
默认路径如下。训练时也可以通过 `--data-root` 覆盖。

```text
./data/cifar100
./data/imagenet
./data/CUB_200_2011
```
CUB-200-2011 若官方旧链接 404，推荐使用 CaltechDATA 下载：

```bash
mkdir -p data
cd data
aria2c -c -x 16 -s 16 -k 1M -o CUB_200_2011.tgz \
  "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
tar -xzf CUB_200_2011.tgz
cd ..
```

## 3. 教师权重路径
所有配置文件都已经按截图教师模型写入 `MODEL.TEACHER` 和 `MODEL.TEACHER_CKPT`。如果你的教师权重放在其他位置，请修改对应 yaml 中的 `MODEL.TEACHER_CKPT`。

```text
CIFAR-100:
  save/models/resnet32x4_vanilla/ckpt_epoch_240.pth
  save/models/wrn_40_2_vanilla/ckpt_epoch_240.pth
  save/models/resnet50_vanilla/ckpt_epoch_240.pth
  save/models/vgg13_vanilla/ckpt_epoch_240.pth

ImageNet:
  save/imagenet/resnet34/ckpt.pth
  save/imagenet/resnet50/ckpt.pth

CUB-200-2011:
  save/cub200/resnet32x4/ckpt.pth
  save/cub200/vgg13/ckpt.pth
  save/cub200/resnet50/ckpt.pth
```

> 重要：原始仓库的 CIFAR 模型注册表可能没有注册 `resnet50`。本次配置已严格按截图把 CIFAR 中 `RN50/ResNet50` 写为 `MODEL.TEACHER: resnet50`。如果运行时报 `Unknown model resnet50`，需要先在 `mdistiller/models/registry.py` 和 CIFAR ResNet 实现中加入 CIFAR 版 ResNet50，或者将对应配置临时改回仓库已支持的教师模型。

## 4. CIFAR-100 完整训练命令
以下命令覆盖表 4.1 和表 4.2 中的全部教师-学生组合，每个组合均包含 PAMA-KD、PAMA-DKD、PAMA-NKD。

### 4.1 PAMA-KD

```bash
python train.py --cfg configs/cifar100/pama_kd/res32x4_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_mv2_pama_kd --gpu 0
python train.py --cfg configs/cifar100/pama_kd/wrn40_2_vgg8.yaml --data-root ./data/cifar100 --output ./runs/cifar100_wrn40_2_vgg8_pama_kd --gpu 0
python train.py --cfg configs/cifar100/pama_kd/wrn40_2_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_wrn40_2_mv2_pama_kd --gpu 0
python train.py --cfg configs/cifar100/pama_kd/r50_shuv1.yaml --data-root ./data/cifar100 --output ./runs/cifar100_r50_shuv1_pama_kd --gpu 0
python train.py --cfg configs/cifar100/pama_kd/res32x4_shuv1.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_shuv1_pama_kd --gpu 0
python train.py --cfg configs/cifar100/pama_kd/wrn40_2_shuv1.yaml --data-root ./data/cifar100 --output ./runs/cifar100_wrn40_2_shuv1_pama_kd --gpu 0
python train.py --cfg configs/cifar100/pama_kd/r50_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_r50_mv2_pama_kd --gpu 0
python train.py --cfg configs/cifar100/pama_kd/vgg13_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_vgg13_mv2_pama_kd --gpu 0
python train.py --cfg configs/cifar100/pama_kd/res32x4_shuv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_shuv2_pama_kd --gpu 0
python train.py --cfg configs/cifar100/pama_kd/r50_vgg8.yaml --data-root ./data/cifar100 --output ./runs/cifar100_r50_vgg8_pama_kd --gpu 0
```

### 4.2 PAMA-DKD

```bash
python train.py --cfg configs/cifar100/pama_dkd/res32x4_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_mv2_pama_dkd --gpu 0
python train.py --cfg configs/cifar100/pama_dkd/wrn40_2_vgg8.yaml --data-root ./data/cifar100 --output ./runs/cifar100_wrn40_2_vgg8_pama_dkd --gpu 0
python train.py --cfg configs/cifar100/pama_dkd/wrn40_2_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_wrn40_2_mv2_pama_dkd --gpu 0
python train.py --cfg configs/cifar100/pama_dkd/r50_shuv1.yaml --data-root ./data/cifar100 --output ./runs/cifar100_r50_shuv1_pama_dkd --gpu 0
python train.py --cfg configs/cifar100/pama_dkd/res32x4_shuv1.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_shuv1_pama_dkd --gpu 0
python train.py --cfg configs/cifar100/pama_dkd/wrn40_2_shuv1.yaml --data-root ./data/cifar100 --output ./runs/cifar100_wrn40_2_shuv1_pama_dkd --gpu 0
python train.py --cfg configs/cifar100/pama_dkd/r50_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_r50_mv2_pama_dkd --gpu 0
python train.py --cfg configs/cifar100/pama_dkd/vgg13_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_vgg13_mv2_pama_dkd --gpu 0
python train.py --cfg configs/cifar100/pama_dkd/res32x4_shuv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_shuv2_pama_dkd --gpu 0
python train.py --cfg configs/cifar100/pama_dkd/r50_vgg8.yaml --data-root ./data/cifar100 --output ./runs/cifar100_r50_vgg8_pama_dkd --gpu 0
```

### 4.3 PAMA-NKD

```bash
python train.py --cfg configs/cifar100/pama_nkd/res32x4_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_mv2_pama_nkd --gpu 0
python train.py --cfg configs/cifar100/pama_nkd/wrn40_2_vgg8.yaml --data-root ./data/cifar100 --output ./runs/cifar100_wrn40_2_vgg8_pama_nkd --gpu 0
python train.py --cfg configs/cifar100/pama_nkd/wrn40_2_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_wrn40_2_mv2_pama_nkd --gpu 0
python train.py --cfg configs/cifar100/pama_nkd/r50_shuv1.yaml --data-root ./data/cifar100 --output ./runs/cifar100_r50_shuv1_pama_nkd --gpu 0
python train.py --cfg configs/cifar100/pama_nkd/res32x4_shuv1.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_shuv1_pama_nkd --gpu 0
python train.py --cfg configs/cifar100/pama_nkd/wrn40_2_shuv1.yaml --data-root ./data/cifar100 --output ./runs/cifar100_wrn40_2_shuv1_pama_nkd --gpu 0
python train.py --cfg configs/cifar100/pama_nkd/r50_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_r50_mv2_pama_nkd --gpu 0
python train.py --cfg configs/cifar100/pama_nkd/vgg13_mv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_vgg13_mv2_pama_nkd --gpu 0
python train.py --cfg configs/cifar100/pama_nkd/res32x4_shuv2.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_shuv2_pama_nkd --gpu 0
python train.py --cfg configs/cifar100/pama_nkd/r50_vgg8.yaml --data-root ./data/cifar100 --output ./runs/cifar100_r50_vgg8_pama_nkd --gpu 0
```

## 5. ImageNet 完整训练命令

### 5.1 ResNet34 -> ResNet18

```bash
python train.py --cfg configs/imagenet/r34_r18/pama_kd.yaml --data-root ./data/imagenet --output ./runs/imagenet_r34_r18_pama_kd --gpu 0
python train.py --cfg configs/imagenet/r34_r18/pama_dkd.yaml --data-root ./data/imagenet --output ./runs/imagenet_r34_r18_pama_dkd --gpu 0
python train.py --cfg configs/imagenet/r34_r18/pama_nkd.yaml --data-root ./data/imagenet --output ./runs/imagenet_r34_r18_pama_nkd --gpu 0
```

### 5.2 ResNet50 -> MobileNetV2

```bash
python train.py --cfg configs/imagenet/r50_mv2/pama_kd.yaml --data-root ./data/imagenet --output ./runs/imagenet_r50_mv2_pama_kd --gpu 0
python train.py --cfg configs/imagenet/r50_mv2/pama_dkd.yaml --data-root ./data/imagenet --output ./runs/imagenet_r50_mv2_pama_dkd --gpu 0
python train.py --cfg configs/imagenet/r50_mv2/pama_nkd.yaml --data-root ./data/imagenet --output ./runs/imagenet_r50_mv2_pama_nkd --gpu 0
```

## 6. CUB-200-2011 完整训练命令

### 6.1 ResNet32x4 -> MobileNetV2

```bash
python train.py --cfg configs/cub200/res32x4_mv2/pama_kd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_mv2_pama_kd --gpu 0
python train.py --cfg configs/cub200/res32x4_mv2/pama_dkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_mv2_pama_dkd --gpu 0
python train.py --cfg configs/cub200/res32x4_mv2/pama_nkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_mv2_pama_nkd --gpu 0
```

### 6.2 ResNet32x4 -> ShuffleNetV1

```bash
python train.py --cfg configs/cub200/res32x4_shuv1/pama_kd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_shuv1_pama_kd --gpu 0
python train.py --cfg configs/cub200/res32x4_shuv1/pama_dkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_shuv1_pama_dkd --gpu 0
python train.py --cfg configs/cub200/res32x4_shuv1/pama_nkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_shuv1_pama_nkd --gpu 0
```

### 6.3 VGG13 -> MobileNetV2

```bash
python train.py --cfg configs/cub200/vgg13_mv2/pama_kd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_vgg13_mv2_pama_kd --gpu 0
python train.py --cfg configs/cub200/vgg13_mv2/pama_dkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_vgg13_mv2_pama_dkd --gpu 0
python train.py --cfg configs/cub200/vgg13_mv2/pama_nkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_vgg13_mv2_pama_nkd --gpu 0
```

### 6.4 VGG13 -> VGG8

```bash
python train.py --cfg configs/cub200/vgg13_vgg8/pama_kd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_vgg13_vgg8_pama_kd --gpu 0
python train.py --cfg configs/cub200/vgg13_vgg8/pama_dkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_vgg13_vgg8_pama_dkd --gpu 0
python train.py --cfg configs/cub200/vgg13_vgg8/pama_nkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_vgg13_vgg8_pama_nkd --gpu 0
```

### 6.5 ResNet50 -> ShuffleNetV1

```bash
python train.py --cfg configs/cub200/resnet50_shuv1/pama_kd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_resnet50_shuv1_pama_kd --gpu 0
python train.py --cfg configs/cub200/resnet50_shuv1/pama_dkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_resnet50_shuv1_pama_dkd --gpu 0
python train.py --cfg configs/cub200/resnet50_shuv1/pama_nkd.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_resnet50_shuv1_pama_nkd --gpu 0
```

## 7. 后台训练示例
长时间实验建议用 `nohup` 或 `tmux`。例如：

```bash
mkdir -p logs
nohup python train.py \
  --cfg configs/cifar100/pama_dkd/r50_shuv1.yaml \
  --data-root ./data/cifar100 \
  --output ./runs/cifar100_r50_shuv1_pama_dkd \
  --gpu 0 \
  > logs/cifar100_r50_shuv1_pama_dkd.log 2>&1 &

tail -f logs/cifar100_r50_shuv1_pama_dkd.log
```

## 8. 常见问题

### 8.1 `Teacher checkpoint is not loaded`
说明 `MODEL.TEACHER_CKPT` 指向的教师权重不存在或格式不匹配。正式复现实验必须先准备教师权重，不建议使用 `--allow-random-teacher`。

### 8.2 CIFAR-100 的 ResNet50 配置无法运行
本次为保持与截图一致，CIFAR-100 的 RN50 相关配置使用 `MODEL.TEACHER: resnet50`。如果原仓库没有 CIFAR 版 ResNet50，需要补充模型注册。受影响配置包括：

```text
configs/cifar100/pama_*/r50_shuv1.yaml
configs/cifar100/pama_*/r50_mv2.yaml
configs/cifar100/pama_*/r50_vgg8.yaml
```

### 8.3 CUB-200-2011 旧链接 404
旧的 `vision.caltech.edu/visipedia-data/...` 直链可能已经失效，请使用本文第 2 节的 CaltechDATA 链接。

## 9. 关键配置项（含本次新增）

以下键位于各 yaml 的 `DISTILLER` 或 `PAMA` 段。

`DISTILLER`：

- `GAC_WEIGHT`：智能体关系图蒸馏权重。
- `GAC_MODE`：`relation_graph`（默认，关系分布 KL + 结构 MSE）或 `gram`（旧版纯 Gram MSE，消融对照用）。
- `GAC_TAU`：关系图蒸馏温度，默认 `1.0`。
- `LGC_WEIGHT`：局部-全局一致性权重。
- `SDD_WEIGHT` / `KD_WEIGHT` / `CE_WEIGHT` / `ALPHA` / `BETA` / `T` / `WARMUP`：与原框架一致。

`PAMA`：

- `M`：SDD 局部尺度，默认 `[1, 2, 4]`。
- `NUM_AGENTS`：智能体数量（须为完全平方数），默认 `16`。
- `NUM_HEADS`：AMA 注意力头数，默认 `4`。
- `APF_GSMF`：APF 全局语义调制融合（GSMF，默认 `false`）——最粗层全局锚点经 FiLM 调制每一级融合门，把全局语义注入细尺度融合，在特征阶段缓解语义碎片化（与 logit 阶段的 LGC 互补）；零初始化 FiLM，开启初始等价于基线。（旧键 `APF_SEMANTIC_MOD` 仍作 fallback 兼容。）
- `CSAM_AGENT_INIT`：`pool`（默认，网格池化）或 `routing`（语义原型路由）——可学习原型查询把区域 token 软聚类为内容自适应的语义智能体，使 GAC 迁移语义组织而非空间布局；因基座特征已被 APF 跨尺度融合，路由仍保有跨尺度信息。
- `USE_APF` / `USE_AMA` / `USE_GAC` / `USE_LGC` / `USE_RELIABILITY`：各模块开关（消融用）。

关于 CSAM（跨尺度智能体）：当 `USE_APF` 与 `USE_AMA` 同时开启时自动生效——智能体从全部金字塔层以可学习权重聚合多尺度证据；关闭 APF 时自动退化为单尺度，与旧行为一致，无需额外配置项。

## 10. 创新变体命令与命名约定（aug / GSMF+SPR）

本轮新增两类变体，均按固定命名——只需把后缀加到 `--cfg` 与 `--output`，其余参数不变：

- **`_gsmf_spr`（全仓库都有，共 58 个）**：在基础配置上打开两项模块创新 `PAMA.APF_GSMF: true`（APF 全局语义调制融合）+ `PAMA.CSAM_AGENT_INIT: routing`（CSAM 语义原型路由）。**§4–§6 里每一条命令都有对应的 `_gsmf_spr` 版本**，把 cfg 文件名末尾加 `_gsmf_spr` 即可。
- **`_aug` / `_aug_gsmf_spr`（仅 `cub200/res32x4_shuv1`）**：带强增强（RandomResizedCrop + ColorJitter + RandomErasing）+ 标签平滑 + cosine + `GRAD_CLIP_NORM` + DKD/NKD 重平衡权重；`_aug_gsmf_spr` 再叠加上面两项创新。

通用模式：

```bash
python train.py --cfg configs/<路径>/<名字>_gsmf_spr.yaml --data-root <data> --output ./runs/<名字>_gsmf_spr --gpu 0
```

把 §4 的一条改成创新版（示例）：

```bash
python train.py --cfg configs/cifar100/pama_dkd/res32x4_shuv1_gsmf_spr.yaml --data-root ./data/cifar100 --output ./runs/cifar100_res32x4_shuv1_pama_dkd_gsmf_spr --gpu 0
```

### 10.1 旗舰组 `cub200/res32x4_shuv1` 显式命令

强增强版（batch 32，已修复显存溢出/NaN，DKD/NKD 已重平衡权重）：

```bash
python train.py --cfg configs/cub200/res32x4_shuv1/pama_kd_aug.yaml  --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_shuv1_pama_kd_aug  --gpu 0
python train.py --cfg configs/cub200/res32x4_shuv1/pama_dkd_aug.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_shuv1_pama_dkd_aug --gpu 0
python train.py --cfg configs/cub200/res32x4_shuv1/pama_nkd_aug.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_shuv1_pama_nkd_aug --gpu 0
```

强增强 + GSMF + SPR（本文两项模块创新）：

```bash
python train.py --cfg configs/cub200/res32x4_shuv1/pama_kd_aug_gsmf_spr.yaml  --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_shuv1_pama_kd_aug_gsmf_spr  --gpu 0
python train.py --cfg configs/cub200/res32x4_shuv1/pama_dkd_aug_gsmf_spr.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_shuv1_pama_dkd_aug_gsmf_spr --gpu 0
python train.py --cfg configs/cub200/res32x4_shuv1/pama_nkd_aug_gsmf_spr.yaml --data-root ./data/CUB_200_2011 --output ./runs/cub200_res32x4_shuv1_pama_nkd_aug_gsmf_spr --gpu 0
```

### 10.2 模块创新的有/无消融

同一基座下只切换文件名后缀即可做 GSMF+SPR 的有/无对照（其余完全一致）：

| 对照项 | 配置文件 |
|---|---|
| baseline（aug，无模块创新） | `pama_kd_aug.yaml` |
| + GSMF + SPR | `pama_kd_aug_gsmf_spr.yaml` |

若要单独消融某一项：在对应 `_aug.yaml` 的 `PAMA:` 段里只加 `APF_GSMF: true`（仅 GSMF）或只加 `CSAM_AGENT_INIT: routing`（仅 SPR）。两个开关默认关闭，即为不带创新的基线。
