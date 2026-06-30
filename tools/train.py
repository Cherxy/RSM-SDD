from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mdistiller.dataset import get_cifar100_dataloaders
from mdistiller.distillers import DISTILLER_DICT
from mdistiller.engine import Trainer, load_config
from mdistiller.engine.utils import set_seed
from mdistiller.models import MODEL_DICT


def _build_model(model_cfg, use_pama: bool):
    kwargs = dict(
        num_classes=model_cfg.NUM_CLASSES,
        distill_levels=model_cfg.DISTILL_LEVELS,
        use_pama=use_pama,
        pama_out_channels=model_cfg.PAMA_OUT_CHANNELS,
        pama_heads=model_cfg.PAMA_HEADS,
        pama_agent_num=model_cfg.PAMA_AGENT_NUM,
    )
    return MODEL_DICT[model_cfg.NAME](**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Train SDD/PAMA-SDD on CIFAR-100")
    parser.add_argument("--cfg", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--work-dir", type=str, default="work_dirs/default")
    parser.add_argument("--teacher-ckpt", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    set_seed(cfg.SOLVER.SEED)

    train_loader, test_loader = get_cifar100_dataloaders(
        data_root=cfg.DATASET.ROOT,
        batch_size=cfg.DATASET.BATCH_SIZE,
        num_workers=cfg.DATASET.NUM_WORKERS,
    )

    use_pama = cfg.DISTILLER.NAME.startswith("PAMA")
    student = _build_model(cfg.STUDENT, use_pama=use_pama)
    teacher = _build_model(cfg.TEACHER, use_pama=use_pama)

    if args.teacher_ckpt:
        state = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(state["model"] if "model" in state else state, strict=True)

    distiller = DISTILLER_DICT[cfg.DISTILLER.NAME](student, teacher, cfg)

    optimizer = torch.optim.SGD(
        distiller.get_learnable_parameters(),
        lr=cfg.SOLVER.LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        nesterov=bool(cfg.SOLVER.NESTEROV),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(cfg.SOLVER.LR_DECAY_STAGES),
        gamma=cfg.SOLVER.LR_DECAY_RATE,
    )

    trainer = Trainer(distiller, optimizer, scheduler, cfg, work_dir=Path(args.work_dir))
    trainer.train(train_loader, test_loader)


if __name__ == "__main__":
    main()

