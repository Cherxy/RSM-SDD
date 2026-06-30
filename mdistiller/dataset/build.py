from __future__ import annotations

def build_loaders(cfg, data_root=None, debug_fake_data=False):
    if debug_fake_data:
        from .fake import build_fake
        return build_fake(cfg)
    name = cfg.DATASET.NAME.lower()
    if name == 'cifar100':
        from .cifar import build_cifar100
        return build_cifar100(cfg, data_root)
    if name == 'imagenet':
        from .imagenet import build_imagenet
        return build_imagenet(cfg, data_root)
    if name in ('cub200', 'cub-200', 'cub_200_2011'):
        from .cub200 import build_cub200
        return build_cub200(cfg, data_root)
    raise KeyError(f'unknown dataset: {cfg.DATASET.NAME}')
