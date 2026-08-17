import torch
from torch.utils.data import DataLoader, TensorDataset

from .loader_utils import solver_value


def build_fake(cfg):
    ncls = int(cfg.DATASET.NUM_CLASSES)
    img_size = int(getattr(cfg.DATASET, 'IMG_SIZE', 32))
    train_n = int(getattr(cfg.DATASET, 'FAKE_TRAIN_SAMPLES', 32))
    val_n = int(getattr(cfg.DATASET, 'FAKE_VAL_SAMPLES', 16))
    bsz = int(solver_value(cfg, 'BATCH_SIZE', 64))
    xtr = torch.randn(train_n, 3, img_size, img_size)
    ytr = torch.randint(0, ncls, (train_n,))
    xva = torch.randn(val_n, 3, img_size, img_size)
    yva = torch.randint(0, ncls, (val_n,))
    return (DataLoader(TensorDataset(xtr, ytr), batch_size=bsz, shuffle=True, num_workers=0),
            DataLoader(TensorDataset(xva, yva), batch_size=bsz, shuffle=False, num_workers=0))
