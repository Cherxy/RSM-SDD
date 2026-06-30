import torch
import torch.nn as nn
import torch.nn.functional as F

class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)
    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x), inplace=True)))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)

class WideResNet(nn.Module):
    def __init__(self, depth=40, widen_factor=2, dropout_rate=0.0, num_classes=100):
        super().__init__()
        n = (depth - 4) // 6
        k = widen_factor
        stages = [16, 16*k, 32*k, 64*k]
        self.conv1 = nn.Conv2d(3, stages[0], 3, padding=1, bias=False)
        self.in_planes = stages[0]
        self.layer1 = self._wide_layer(stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(stages[3])
        self.fc = nn.Linear(stages[3], num_classes)
        self.class_num = num_classes
        self.feature_channels = stages[1:]
    def _wide_layer(self, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for st in strides:
            layers.append(WideBasic(self.in_planes, planes, dropout_rate, st))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def get_feature_channels(self): return self.feature_channels
    def classifier(self, x): return self.fc(x)
    def forward_features(self, x):
        x = self.conv1(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = F.relu(self.bn1(self.layer3(f2)), inplace=True)
        return [f1, f2, f3]
    def forward(self, x, return_features=False):
        feats = self.forward_features(x)
        out = F.avg_pool2d(feats[-1], feats[-1].shape[-1]).flatten(1)
        logits = self.fc(out)
        if return_features: return logits, feats
        return logits

def wrn_40_2(num_classes=100): return WideResNet(40, 2, 0.0, num_classes)
def wrn_16_2(num_classes=100): return WideResNet(16, 2, 0.0, num_classes)
