# PAMA-SDD++ 论文重写思路

## 1. 总体写法

这篇论文最适合采用“TPAMI 的技术证据链 + Nature 的清晰故事线”：

- Nature 式主线：不要从模块出发，而是从一个矛盾出发。SDD 暴露了局部证据，但规则网格也带来了不可靠局部目标和语义割裂。
- TPAMI 式展开：每个模块都要回答一个可验证的问题。PSPC 解决局部 logits 的特征语义基础，CSAM 解决局部区域之间缺少全局交互，RA-PAMA 解决局部教师目标可靠性，LGC/GAC 解决局部预测与全局语义结构的一致性。
- 贡献边界要清楚：SDD、Agent Attention、通用金字塔融合不是本文原始贡献；本文贡献在于把它们组织为“可靠性感知的局部-全局语义中介”框架，并加入稳定 teacher targets、relation-graph GAC、RA-PAMA/LGC 这条训练目标链。

## 2. 建议的一句话核心论点

Scale-decoupled distillation should not be treated as independent grid-level matching; it should be trained as reliability-aware local-global semantic mediation.

中文理解：

尺度解耦蒸馏不应该只是把图像切成网格后逐格模仿教师，而应该先判断局部目标是否可靠，再把局部证据放回教师的全局语义结构中迁移。

## 3. 章节写作策略

### Abstract

用四步写：

1. 现有 KD 丢失教师的空间证据。
2. SDD 找回局部证据，但网格会产生噪声和语义割裂。
3. PAMA-SDD++ 用 PSPC、CSAM、RA-PAMA、LGC、GAC 建立局部-全局中介。
4. 三个数据集验证有效，且推理无额外开销。

### Introduction

推荐结构：

1. 模型压缩背景：大模型强，但部署受限。
2. KD 的价值：教师提供软标签、特征或关系知识。
3. 问题缺口：多数分类蒸馏在全局聚合后监督，缺少“where”。
4. SDD 的推进：用多尺度局部 logits 暴露空间证据。
5. SDD 的新问题：局部网格不是语义对象，可能切开目标、混入背景、产生低置信教师预测。
6. 本文中心命题：局部证据需要 reliability 和 global semantics。
7. 方法与贡献：稳定 teacher targets + PSPC + CSAM + RA-PAMA/LGC/GAC。

### Methodology

不要按“模块 1、模块 2、模块 3”机械堆叠。建议按“问题-设计”写：

- Stable teacher targets：解决 teacher-side auxiliary module 可能导致目标漂移的问题。
- PSPC：解决局部 logits 生成前特征语义不一致的问题。
- CSAM：解决局部区域独立、全局上下文不足的问题。
- RA-PAMA：解决局部教师目标置信度不同的问题。
- LGC/GAC：解决局部预测和全局教师语义结构脱节的问题。

### Experiments

实验叙事要围绕三个问题：

1. 是否比 SDD 系列强：看 CIFAR-100、ImageNet-1K、CUB-200-2011 主表。
2. 是否对 base objective 不敏感：KD/DKD/NKD 都有效。
3. 是否真的解决语义割裂：看消融、t-SNE、attention visualization、semantic consistency metrics。

消融实验建议采用主流 KD 论文更常用的设置：

- 主消融：CIFAR-100，ResNet32x4 -> ResNet8x4。这与大量 KD/SDD 类论文习惯一致，也能避免审稿人质疑消融设置过于少见。
- 可选补充：再加一个异构学生模型，如 ResNet32x4 -> ShuffleNetV1/MobileNetV2，用于证明方法对结构差异鲁棒。
- 不建议把 VGG13 -> MobileNetV2 作为主消融设置。这个组合可作为 fine-grained 或 heterogeneous diagnostic 放入补充材料，但不应支撑主消融结论。

### Discussion / Conclusion

不要只重复“效果好”。应收束到方法原则：

局部蒸馏不是增加监督数量，而是控制局部监督质量，并把局部证据放回全局语义结构中。

## 4. 代码到论文的映射

- `mdistiller/distillers/pama_sdd.py`：主框架、RA-PAMA、LGC、GAC、稳定 teacher targets。
- `mdistiller/modules/apf.py`：PSPC/APF，论文中写作“语义金字塔校准”。
- `mdistiller/modules/csam.py`：CSAM，论文中写作“跨尺度 agent 中介”。
- `mdistiller/modules/spp.py`：SDD 局部 logits，属于支撑模块，不建议作为原创贡献写得过重。
- `tools/analyze_semantic_consistency.py` 和 `forward_analysis()`：可支撑 semantic consistency analysis。

## 5. 投稿前需要特别统一的点

- `APF_GSMF` 和 `CSAM_AGENT_INIT: routing` 目前在代码中存在，但正文主线还没有正式展开。若这些是最终投稿的核心新增点，需要补方法公式、消融表和引用；如果实验还没完整跑完，建议先写入 appendix 或 implementation variant，避免主贡献边界变乱。
- GAC/LGC/RA-PAMA 的权重必须和最终配置、表格、文字保持一致。正文已经避免写死过多配置，但最终定稿前仍建议统一。
- 消融表格不要混用不同 teacher-student 设置的数字。若主消融改成 CIFAR-100 ResNet32x4 -> ResNet8x4，就必须重新跑对应配置，不能把 CUB 或 VGG13 -> MobileNetV2 的旧数字平移过来。
- Semantic consistency analysis 目前更像诊断说明。如果有实际数值，建议加一张小表；如果没有数值，保留为定性/工具性分析即可。
- Complexity/inference cost 可以恢复为一个短小节，因为“推理无额外开销”是压缩论文的强卖点。
