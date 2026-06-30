from __future__ import annotations
from .cifar.resnet import (
    resnet8x4,
    resnet32x4,
    resnet32x4_imagenet_stem,
    resnet50,
    resnet50_imagenet_stem,
    resnet110,
    resnet20,
)
from .cifar.vgg import vgg8, vgg13
from .cifar.wrn import wrn_40_2, wrn_16_2
from .cifar.mobilenetv2 import mobilenetv2
from .cifar.shufflenet import shufflenetv1, shufflenetv2


def build_model(name, num_classes=100, dataset='cifar100', pretrained=False):
    key = name.lower().replace('-', '_')
    dataset = dataset.lower()
    if dataset in ('cub200', 'cub_200_2011', 'cub-200') and key == 'resnet32x4':
        return resnet32x4_imagenet_stem(num_classes=num_classes)
    if dataset in ('cub200', 'cub_200_2011', 'cub-200') and key == 'resnet50':
        return resnet50_imagenet_stem(num_classes=num_classes)
    if dataset in ('imagenet', 'cub200', 'cub_200_2011', 'cub-200') and key in ('resnet18', 'resnet34', 'resnet50', 'mobilenetv2', 'mobilenet_v2'):
        from .imagenet import imagenet_model
        return imagenet_model(key, num_classes=num_classes, pretrained=pretrained)
    table = {
        'resnet8x4': resnet8x4,
        'resnet32x4': resnet32x4,
        'resnet50': resnet50,
        'resnet110': resnet110,
        'resnet20': resnet20,
        'vgg8': vgg8,
        'vgg13': vgg13,
        'wrn_40_2': wrn_40_2,
        'wrn40_2': wrn_40_2,
        'wrn_16_2': wrn_16_2,
        'mobilenetv2': mobilenetv2,
        'mobilenet_v2': mobilenetv2,
        'shufflenetv1': shufflenetv1,
        'shufflenet_v1': shufflenetv1,
        'shufflenetv2': shufflenetv2,
        'shufflenet_v2': shufflenetv2,
    }
    if key not in table:
        raise KeyError(f'Unknown model {name}. Available={list(table)}')
    return table[key](num_classes=num_classes)
