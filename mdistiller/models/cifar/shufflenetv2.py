from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mdistiller.models.common import PAMAEnhancer, SpatialPyramidPooling
from .resnet import _parse_levels


def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.reshape(b, c, h, w)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.stride = stride
        branch_features = oup // 2

        if stride == 1:
            self.branch1 = nn.Identity()
            inp_branch2 = branch_features
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
            inp_branch2 = inp

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp_branch2, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride, 1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2CIFAR(nn.Module):
    def __init__(
        self,
        num_classes=100,
        distill_levels=(1, 2, 4),
        use_pama=False,
        pama_out_channels=256,
        pama_heads=4,
        pama_agent_num=49,
    ):
        super().__init__()
        self.distill_levels = _parse_levels(distill_levels)
        self.use_pama = use_pama
        self.num_classes = num_classes

        stage_repeats = [4, 8, 4]
        stage_out_channels = [24, 116, 232, 464, 1024]

        input_channels = stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )

        self.stage2 = self._make_stage(input_channels, stage_out_channels[1], stage_repeats[0])
        self.stage3 = self._make_stage(stage_out_channels[1], stage_out_channels[2], stage_repeats[1])
        self.stage4 = self._make_stage(stage_out_channels[2], stage_out_channels[3], stage_repeats[2])
        self.conv5 = nn.Sequential(
            nn.Conv2d(stage_out_channels[3], stage_out_channels[4], 1, 1, 0, bias=False),
            nn.BatchNorm2d(stage_out_channels[4]),
            nn.ReLU(inplace=True),
        )

        in_channels = [stage_out_channels[1], stage_out_channels[2], stage_out_channels[4]]
        self.pyramid_pool = SpatialPyramidPooling(self.distill_levels)
        self.feature_dim = pama_out_channels if use_pama else stage_out_channels[4]
        if use_pama:
            self.pama = PAMAEnhancer(in_channels, pama_out_channels, pama_heads, pama_agent_num)
        else:
            self.pama = None
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def _make_stage(self, inp, oup, repeat):
        layers = [InvertedResidual(inp, oup, 2)]
        for _ in range(repeat - 1):
            layers.append(InvertedResidual(oup, oup, 1))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        f1 = self.stage2(x)
        f2 = self.stage3(f1)
        f3 = self.conv5(self.stage4(f2))
        return [f1, f2, f3]

    def forward(self, x):
        features = self.forward_features(x)
        agent_tokens = None
        if self.pama is not None:
            features, agent_tokens = self.pama(features)
        distill_feat = features[-1]
        pooled_feature = F.adaptive_avg_pool2d(distill_feat, 1).flatten(1)
        logits = self.fc(pooled_feature)

        patch_features, masks = self.pyramid_pool(distill_feat)
        region_count = patch_features.size(-1)
        patch_logits = self.fc(patch_features.permute(0, 2, 1).reshape(-1, self.feature_dim))
        patch_logits = patch_logits.view(x.size(0), region_count, self.num_classes).permute(0, 2, 1).contiguous()
        return logits, patch_logits, masks, pooled_feature, agent_tokens


def shufflenetv2(**kwargs):
    return ShuffleNetV2CIFAR(**kwargs)

