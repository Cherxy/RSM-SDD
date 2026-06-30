from pathlib import Path


def resolve_imagenet_root(root):
    root = Path(root).expanduser()
    train_dir = root / 'train'
    val_dir = root / 'val'

    if train_dir.is_dir() and val_dir.is_dir():
        return root, train_dir, val_dir

    if root.name.lower() in ('train', 'val') and (root.parent / 'train').is_dir() and (root.parent / 'val').is_dir():
        raise ValueError(
            "ImageNet DATASET.ROOT must be the directory containing both 'train' and 'val'. "
            f"Got split directory: {root}. Use: {root.parent}"
        )

    missing = [str(path) for path in (train_dir, val_dir) if not path.is_dir()]
    raise FileNotFoundError(
        "Invalid ImageNet DATASET.ROOT. Expected the root directory to contain 'train' and 'val'. "
        f"Configured root: {root}. Missing: {', '.join(missing)}"
    )


def build_imagenet(cfg, data_root=None):
    _, train_dir, val_dir = resolve_imagenet_root(data_root or getattr(cfg.DATASET, 'ROOT', './data/imagenet'))
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    img = int(getattr(cfg.DATASET, 'IMG_SIZE', 224))
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256 if img == 224 else int(img * 256 / 224)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = datasets.ImageFolder(train_dir, train_tf)
    val_set = datasets.ImageFolder(val_dir, val_tf)
    return (DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=cfg.SOLVER.NUM_WORKERS, pin_memory=True),
            DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=cfg.SOLVER.NUM_WORKERS, pin_memory=True))
