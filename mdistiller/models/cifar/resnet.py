import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        return F.relu(out, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride=stride),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)

class CifarResNet(nn.Module):
    def __init__(self, depth=32, width=1, num_classes=100, stem_width=None, imagenet_stem=False):
        super().__init__()
        assert (depth - 2) % 6 == 0
        n = (depth - 2) // 6
        # stage channels
        c1, c2, c3 = 16*width, 32*width, 64*width
        # some public checkpoints (e.g. resnet32x4_vanilla) use a smaller stem
        # than stage-1 channels: stem=32, stages=64/128/256.
        stem_c = 16 * (stem_width if stem_width is not None else width)
        self.imagenet_stem = bool(imagenet_stem)
        if self.imagenet_stem:
            self.conv1 = nn.Conv2d(3, stem_c, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = conv3x3(3, stem_c)
        self.bn1 = nn.BatchNorm2d(stem_c)
        self.inplanes = stem_c
        self.layer1 = self._make_layer(c1, n, stride=1)
        self.layer2 = self._make_layer(c2, n, stride=2)
        self.layer3 = self._make_layer(c3, n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes)
        self.class_num = num_classes
        self.feature_channels = [c1, c2, c3]
    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1))
        return nn.Sequential(*layers)
    def get_feature_channels(self): return self.feature_channels
    def classifier(self, x): return self.fc(x)
    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]
    def forward(self, x, return_features=False):
        feats = self.forward_features(x)
        out = self.avgpool(feats[-1]).flatten(1)
        logits = self.fc(out)
        if return_features: return logits, feats
        return logits


class CifarResNetBottleneck(nn.Module):
    """CIFAR-style bottleneck ResNet used by public `ResNet50_vanilla` teachers.

    Key compatibility points:
    - CIFAR uses a 3x3 stride-1 stem; CUB checkpoints use 7x7 stride-2.
    - Bottleneck stages [3, 4, 6, 3]
    - Projection path named `shortcut` (matches many KD repos)
    - Classifier named `linear` in checkpoint; `classifier()` abstracts usage
    """

    def __init__(self, num_classes=100, imagenet_stem=False):
        super().__init__()
        self.inplanes = 64
        if imagenet_stem:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self.class_num = num_classes
        # Keep 3-level pyramid interface used by distillers/APF.
        self.feature_channels = [512, 1024, 2048]

    def _make_layer(self, planes, blocks, stride):
        layers = [Bottleneck(self.inplanes, planes, stride=stride)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def get_feature_channels(self):
        return self.feature_channels

    def classifier(self, x):
        return self.linear(x)

    def forward_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        f1 = self.layer2(x)
        f2 = self.layer3(f1)
        f3 = self.layer4(f2)
        return [f1, f2, f3]

    def forward(self, x, return_features=False):
        feats = self.forward_features(x)
        out = self.avgpool(feats[-1]).flatten(1)
        logits = self.linear(out)
        if return_features:
            return logits, feats
        return logits

def resnet20(num_classes=100): return CifarResNet(depth=20, width=1, num_classes=num_classes)
def resnet50(num_classes=100): return CifarResNetBottleneck(num_classes=num_classes)
def resnet110(num_classes=100): return CifarResNet(depth=110, width=1, num_classes=num_classes)
def resnet8x4(num_classes=100): return CifarResNet(depth=8, width=4, num_classes=num_classes)
def resnet32x4(num_classes=100):
    # Compatibility with common CIFAR teacher checkpoint naming in distillation repos:
    # `resnet32x4_vanilla` usually follows stem/stages = 32 / (64, 128, 256).
    return CifarResNet(depth=32, width=4, stem_width=2, num_classes=num_classes)

def resnet32x4_imagenet_stem(num_classes=100):
    # CUB200 SDD teacher checkpoints use a 7x7 stride-2 stem without the
    # ImageNet maxpool, while retaining the lightweight resnet32x4 stage layout.
    return CifarResNet(depth=32, width=4, stem_width=2, num_classes=num_classes, imagenet_stem=True)

def resnet50_imagenet_stem(num_classes=100):
    # CUB200 ResNet50 teacher checkpoints use the CIFAR-style bottleneck body
    # with a 7x7 stride-2 stem and no ImageNet maxpool.
    return CifarResNetBottleneck(num_classes=num_classes, imagenet_stem=True)
