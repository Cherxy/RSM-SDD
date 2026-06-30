from .mobilenetv2 import mobilenetv2
from .resnet import (
    CifarResNet,
    resnet8x4,
    resnet20,
    resnet32x4,
    resnet50,
    resnet110,
)
from .shufflenet import shufflenetv1, shufflenetv2
from .vgg import vgg8, vgg8_bn, vgg13, vgg13_bn
from .wrn import wrn_16_2, wrn_40_2

CIFARResNet = CifarResNet

MODEL_DICT = {
    "resnet8x4": resnet8x4,
    "resnet20": resnet20,
    "resnet32x4": resnet32x4,
    "resnet50": resnet50,
    "resnet110": resnet110,
    "vgg8": vgg8,
    "vgg8_bn": vgg8_bn,
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "wrn_16_2": wrn_16_2,
    "wrn_40_2": wrn_40_2,
    "mobilenetv2": mobilenetv2,
    "shufflenetv1": shufflenetv1,
    "shufflenetv2": shufflenetv2,
}

__all__ = ["MODEL_DICT", "CIFARResNet"]
