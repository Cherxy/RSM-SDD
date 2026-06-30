from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mdistiller.models.common import PAMAEnhancer, SpatialPyramidPooling
from .resnet import _parse_levels


def channel_shuffle(x, groups):
    b, c, h, w = x.size()
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=3):
        super().__init__()
        mid_channels = out_channels // 4
        self.stride = stride
        self.groups = groups
        branch_out = out_channels - in_channels if stride == 2 else out_channels

        self.gconv1 = nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.dwconv = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.gconv2 = nn.Conv2d(mid_channels, branch_out, 1, 1, 0, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(branch_out)
        self.shortcut = nn.AvgPool2d(3, stride=2, padding=1) if stride == 2 else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.gconv1(x)), inplace=True)
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.dwconv(out))
        out = self.bn3(self.gconv2(out))
        if self.stride == 1:
            out = F.relu(out + x, inplace=True)
        else:
            out = F.relu(torch.cat([self.shortcut(x), out], dim=1), inplace=True)
        return out


class ShuffleNetV1CIFAR(nn.Module):
    def __init__(
        self,
        num_classes=100,
        groups=3,
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

        out_channels = [240, 480, 960]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.stage2 = self._make_stage(24, out_channels[0], 4, groups)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 8, groups)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 4, groups)

        in_channels = out_channels
        self.pyramid_pool = SpatialPyramidPooling(self.distill_levels)
        self.feature_dim = pama_out_channels if use_pama else out_channels[-1]
        if use_pama:
            self.pama = PAMAEnhancer(in_channels, pama_out_channels, pama_heads, pama_agent_num)
        else:
            self.pama = None
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def _make_stage(self, in_channels, out_channels, repeat, groups):
        layers = [ShuffleUnit(in_channels, out_channels, 2, groups)]
        for _ in range(repeat - 1):
            layers.append(ShuffleUnit(out_channels, out_channels, 1, groups))
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        f1 = self.stage2(x)
        f2 = self.stage3(f1)
        f3 = self.stage4(f2)
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


def shufflenetv1(**kwargs):
    return ShuffleNetV1CIFAR(**kwargs)

