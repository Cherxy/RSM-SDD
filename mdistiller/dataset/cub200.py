from __future__ import annotations
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets


class CUB200Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = Path(root)
        self.transform = transform
        images_txt = self.root / 'images.txt'
        labels_txt = self.root / 'image_class_labels.txt'
        split_txt = self.root / 'train_test_split.txt'
        if not images_txt.exists():
            raise FileNotFoundError(f'CUB metadata not found under {self.root}. Expected images.txt')
        id_to_path = {}
        for line in images_txt.read_text().splitlines():
            idx, rel = line.split(maxsplit=1)
            id_to_path[int(idx)] = rel
        id_to_label = {}
        for line in labels_txt.read_text().splitlines():
            idx, lab = line.split(maxsplit=1)
            id_to_label[int(idx)] = int(lab) - 1
        id_to_split = {}
        for line in split_txt.read_text().splitlines():
            idx, flag = line.split(maxsplit=1)
            id_to_split[int(idx)] = int(flag)
        self.samples = []
        for idx, rel in id_to_path.items():
            if (id_to_split[idx] == 1) == train:
                self.samples.append((self.root / 'images' / rel, id_to_label[idx]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        path, y = self.samples[i]
        with Image.open(path) as img:
            img = img.convert('RGB')
        if self.transform: img = self.transform(img)
        return img, y


def build_cub200(cfg, data_root=None):
    root = Path(data_root or getattr(cfg.DATASET, 'ROOT', './data/CUB_200_2011'))
    img = int(getattr(cfg.DATASET, 'IMG_SIZE', 224))
    mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize(256 if img == 224 else int(img * 256 / 224)),
        transforms.RandomCrop(img),
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
    try:
        train_set = CUB200Dataset(root, train=True, transform=train_tf)
        val_set = CUB200Dataset(root, train=False, transform=val_tf)
    except FileNotFoundError:
        # Fallback if user manually arranges folders train/val/class/*.jpg
        train_set = datasets.ImageFolder(root / 'train', train_tf)
        val_set = datasets.ImageFolder(root / 'val', val_tf)
    num_workers = int(cfg.SOLVER.NUM_WORKERS)
    loader_kwargs = {
        'batch_size': cfg.SOLVER.BATCH_SIZE,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        loader_kwargs.update({
            'persistent_workers': True,
            'prefetch_factor': 2,
        })
    return (DataLoader(train_set, shuffle=True, **loader_kwargs),
            DataLoader(val_set, shuffle=False, **loader_kwargs))
