from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .loader_utils import dataloader_kwargs_from_values


def get_cifar100_dataloaders(data_root: str | Path, batch_size: int, num_workers: int = 4):
    data_root = Path(data_root)
    normalize = transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_set = datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=test_transform)

    kwargs = dataloader_kwargs_from_values(batch_size=batch_size, num_workers=num_workers)
    train_loader = DataLoader(train_set, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **kwargs)
    return train_loader, test_loader
