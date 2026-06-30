# PAMA-SDD 复现实验 README（中文）
本文档按照用户提供的论文截图表 4.1、表 4.2、表 4.3、表 4.4，对 `configs/` 下的教师-学生组合进行了重新整理，并给出 CIFAR-100、ImageNet、CUB-200-2011 三个数据集的完整训练命令。
> 说明：当前压缩包只覆盖原仓库已有的 PAMA-KD、PAMA-DKD、PAMA-NKD 三类配置。截图中的 FitNet、SP、CRD、SemCKD、ReviewKD、MGD、KD、DKD、NKD、SD-KD、SD-DKD、SD-NKD 等非 PAMA 基线，只有在仓库中实现对应 distiller 并补充配置后才能训练。
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
