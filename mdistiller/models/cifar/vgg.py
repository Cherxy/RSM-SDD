from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# MDistiller/CRD-style CIFAR VGG definitions.  Public CIFAR teacher checkpoints
# such as save/models/vgg13_vanilla/ckpt_epoch_240.pth are stored with keys
# block0...block4 and classifier.*.  Keep those names so the official teacher
# checkpoints load directly, while exposing the lightweight API used by this
# repo: forward(..., return_features=True), get_feature_channels(), class_num.
_CFGS = {
    "S": [[64], [128], [256], [512], [512]],
    "A": [[64], [128], [256, 256], [512, 512], [512, 512]],
    "B": [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    "D": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    "E": [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
}


class VGG(nn.Module):
    def __init__(self, cfg, batch_norm: bool = True, num_classes: int = 100):
        super().__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Retained for compatibility with 64x64 inputs.  CIFAR-100 32x32 inputs
        # skip this pool exactly as the public MDistiller implementation does.
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)
        self.class_num = num_classes
        self.stage_channels = [stage[-1] for stage in cfg]
        # The PAMA/APF implementation in this repo consumes a 3-level pyramid.
        # Use the last three VGG stages so the classifier input remains 512-D.
        self.feature_channels = self.stage_channels[-3:]
        self._initialize_weights()

    @staticmethod
    def _make_layers(cfg, batch_norm: bool = True, in_channels: int = 3):
        layers = []
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v
        # Match MDistiller's VGG blocks: ReLU after each block is applied in
        # forward(), so the final ReLU is not part of the saved block state.
        return nn.Sequential(*layers[:-1])

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_feature_channels(self):
        return self.feature_channels

    def get_stage_channels(self):
        return self.stage_channels

    def forward_features(self, x):
        h = x.shape[2]
        x = F.relu(self.block0(x), inplace=True)
        f0 = x
        x = self.pool0(x)

        x = F.relu(self.block1(x), inplace=True)
        f1 = x
        x = self.pool1(x)

        x = F.relu(self.block2(x), inplace=True)
        f2 = x
        x = self.pool2(x)

        x = F.relu(self.block3(x), inplace=True)
        f3 = x
        if h == 64:
            x = self.pool3(x)

        x = F.relu(self.block4(x), inplace=True)
        f4 = x
        return [f0, f1, f2, f3, f4]

    def forward(self, x, return_features: bool = False):
        feats_all = self.forward_features(x)
        pooled = self.pool4(feats_all[-1]).reshape(x.size(0), -1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, feats_all[-3:]
        return logits


def vgg8(num_classes: int = 100):
    return VGG(_CFGS["S"], batch_norm=True, num_classes=num_classes)


def vgg8_bn(num_classes: int = 100):
    return VGG(_CFGS["S"], batch_norm=True, num_classes=num_classes)


def vgg13(num_classes: int = 100):
    # Public `vgg13_vanilla` CIFAR teacher checkpoints are VGG13 with BN.
    return VGG(_CFGS["B"], batch_norm=True, num_classes=num_classes)


def vgg13_bn(num_classes: int = 100):
    return VGG(_CFGS["B"], batch_norm=True, num_classes=num_classes)
