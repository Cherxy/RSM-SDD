import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleShuffleNet(nn.Module):
    """A lightweight ShuffleNet-like CNN for MDistiller-compatible experiments.

    This implementation is intentionally compact and robust; use official ShuffleNet
    definitions if you need exact paper-level parameter counts.
    """
    def __init__(self, num_classes=100, channels=(24, 116, 232, 464)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, padding=1, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.stage1 = self._stage(channels[0], channels[1], stride=1)
        self.stage2 = self._stage(channels[1], channels[2], stride=2)
        self.stage3 = self._stage(channels[2], channels[3], stride=2)
        self.fc = nn.Linear(channels[3], num_classes)
        self.class_num = num_classes
        self.feature_channels = [channels[1], channels[2], channels[3]]
    def _stage(self, cin, cout, stride):
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, groups=cout, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def get_feature_channels(self): return self.feature_channels
    def classifier(self, x): return self.fc(x)
    def forward_features(self, x):
        x = self.stem(x)
        f1 = self.stage1(x); f2 = self.stage2(f1); f3 = self.stage3(f2)
        return [f1, f2, f3]
    def forward(self, x, return_features=False):
        feats = self.forward_features(x)
        logits = self.fc(F.adaptive_avg_pool2d(feats[-1], 1).flatten(1))
        if return_features: return logits, feats
        return logits

def shufflenetv1(num_classes=100): return SimpleShuffleNet(num_classes=num_classes, channels=(24, 116, 232, 464))
def shufflenetv2(num_classes=100): return SimpleShuffleNet(num_classes=num_classes, channels=(24, 116, 232, 1024))
