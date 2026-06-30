from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_cifar100(cfg, data_root=None):
    root = data_root or getattr(cfg.DATASET, 'ROOT', './data/cifar100')
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR100(root=root, train=False, download=True, transform=test_tf)
    return (DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=cfg.SOLVER.NUM_WORKERS, pin_memory=True),
            DataLoader(test_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=cfg.SOLVER.NUM_WORKERS, pin_memory=True))
