from __future__ import annotations

from torch.utils.data import DataLoader


def solver_value(cfg, name: str, default=None):
    """Read a dataloader option from SOLVER first, then DATASET for legacy configs."""
    for section_name in ("SOLVER", "DATASET"):
        section = getattr(cfg, section_name, None)
        if section is not None and hasattr(section, name):
            return getattr(section, name)
    return default


def dataloader_kwargs(cfg):
    return dataloader_kwargs_from_values(
        batch_size=solver_value(cfg, "BATCH_SIZE", 64),
        num_workers=solver_value(cfg, "NUM_WORKERS", 0),
        pin_memory=solver_value(cfg, "PIN_MEMORY", True),
        persistent_workers=solver_value(cfg, "PERSISTENT_WORKERS", True),
        prefetch_factor=solver_value(cfg, "PREFETCH_FACTOR", 2),
    )


def dataloader_kwargs_from_values(
    batch_size,
    num_workers=0,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
):
    num_workers = int(num_workers)
    kwargs = {
        "batch_size": int(batch_size),
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        prefetch_factor = int(prefetch_factor)
        if prefetch_factor > 0:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def build_dataloader(dataset, cfg, shuffle: bool):
    return DataLoader(dataset, shuffle=shuffle, **dataloader_kwargs(cfg))
