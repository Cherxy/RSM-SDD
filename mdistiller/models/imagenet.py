from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TorchvisionFeatureWrapper(nn.Module):
    def __init__(self, base, model_name, num_classes):
        super().__init__()
        self.base = base
        self.model_name = model_name
        if hasattr(base, 'fc'):
            in_ch = base.fc.in_features
            base.fc = nn.Linear(in_ch, num_classes)
            self.fc = base.fc
        elif hasattr(base, 'classifier') and isinstance(base.classifier, nn.Sequential):
            in_ch = base.classifier[-1].in_features
            base.classifier[-1] = nn.Linear(in_ch, num_classes)
            self.fc = base.classifier[-1]
        else:
            raise ValueError('Unsupported model')
        self.class_num = num_classes
        if 'resnet' in model_name:
            if model_name == 'resnet18' or model_name == 'resnet34':
                self.feature_channels = [128, 256, 512]
            else:
                self.feature_channels = [512, 1024, 2048]
        else:
            # torchvision MobileNetV2 features collected below are blocks
            # 3, 13 and 18, with channel counts 24, 96 and 1280.
            self.feature_channels = [24, 96, 1280]
    def get_feature_channels(self): return self.feature_channels
    def classifier(self, x): return self.fc(x)
    def forward_features(self, x):
        if 'resnet' in self.model_name:
            b = self.base
            x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
            x = b.layer1(x)
            f1 = b.layer2(x)
            f2 = b.layer3(f1)
            f3 = b.layer4(f2)
            return [f1, f2, f3]
        else:
            feats = []
            x = self.base.features[0](x)
            for i, m in enumerate(self.base.features[1:], start=1):
                x = m(x)
                if i in (3, 13, len(self.base.features)-1):
                    feats.append(x)
            return feats[-3:]
    def forward(self, x, return_features=False):
        feats = self.forward_features(x)
        out = F.adaptive_avg_pool2d(feats[-1], 1).flatten(1)
        logits = self.fc(out)
        if return_features: return logits, feats
        return logits

def imagenet_model(name, num_classes=1000, pretrained=False):
    key = name.lower().replace('_', '')
    # Use weights=None to avoid implicit downloads. User can load checkpoint separately.
    if key == 'resnet18': base = models.resnet18(weights=None)
    elif key == 'resnet34': base = models.resnet34(weights=None)
    elif key == 'resnet50': base = models.resnet50(weights=None)
    elif key in ('mobilenetv2', 'mobilenetv2imagenet'):
        base = models.mobilenet_v2(weights=None)
        name = 'mobilenetv2'
    else:
        raise KeyError(name)
    return TorchvisionFeatureWrapper(base, name, num_classes)
