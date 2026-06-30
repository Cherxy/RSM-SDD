import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super().__init__()
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, bias=False), nn.BatchNorm2d(out_planes))
        self.use_shortcut = stride == 1
    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu6(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.use_shortcut:
            out = out + self.shortcut(x)
        return out

class MobileNetV2(nn.Module):
    cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
    def __init__(self, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        in_planes = 32
        layers = []
        self.stage_indices = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for st in strides:
                layers.append(Block(in_planes, out_planes, expansion, st))
                in_planes = out_planes
            self.stage_indices.append(len(layers)-1)
        self.layers = nn.ModuleList(layers)
        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, num_classes)
        self.class_num = num_classes
        self.feature_channels = [24, 64, 1280]
    def get_feature_channels(self): return self.feature_channels
    def classifier(self, x): return self.fc(x)
    def forward_features(self, x):
        out = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        feats = []
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i in (2, 8):
                feats.append(out)
        out = F.relu6(self.bn2(self.conv2(out)), inplace=True)
        feats.append(out)
        return feats[-3:]
    def forward(self, x, return_features=False):
        feats = self.forward_features(x)
        out = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
        logits = self.fc(out)
        if return_features: return logits, feats
        return logits

def mobilenetv2(num_classes=100): return MobileNetV2(num_classes=num_classes)
