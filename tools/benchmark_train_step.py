from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim import SGD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdistiller.dataset import build_loaders
from mdistiller.dataset.loader_utils import solver_value
from mdistiller.distillers import build_distiller
from mdistiller.engine.cfg import load_config
from mdistiller.engine.utils import load_state_safely
from mdistiller.models import build_model


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def solver_bool(cfg, primary: str, default=False, fallback: str | None = None):
    solver = getattr(cfg, "SOLVER", None)
    if solver is not None and hasattr(solver, primary):
        return bool(getattr(solver, primary))
    if fallback and solver is not None and hasattr(solver, fallback):
        return bool(getattr(solver, fallback))
    return bool(default)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--debug-fake-data", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    torch.backends.cudnn.benchmark = True
    cfg = load_config(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    train_loader, _ = build_loaders(cfg, args.data_root, args.debug_fake_data)
    print(
        f"[DATA] batches={len(train_loader)} batch_size={solver_value(cfg, 'BATCH_SIZE', '?')} "
        f"workers={solver_value(cfg, 'NUM_WORKERS', '?')}"
    )
    channels_last = solver_bool(cfg, "CHANNELS_LAST", default=False) and device.type == "cuda"

    student = build_model(
        cfg.MODEL.STUDENT,
        num_classes=cfg.DATASET.NUM_CLASSES,
        dataset=cfg.DATASET.NAME,
    ).to(device)
    teacher = build_model(
        cfg.MODEL.TEACHER,
        num_classes=cfg.DATASET.NUM_CLASSES,
        dataset=cfg.DATASET.NAME,
    ).to(device)
    if getattr(cfg.MODEL, "TEACHER_CKPT", ""):
        load_state_safely(teacher, cfg.MODEL.TEACHER_CKPT, strict=False)

    model = build_distiller(cfg.DISTILLER.TYPE, student, teacher, cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.SOLVER.LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        nesterov=solver_bool(cfg, "NESTEROV", default=False),
    )
    use_amp = solver_bool(cfg, "USE_AMP", default=False, fallback="AMP")
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and device.type == "cuda"))
    print(f"[SOLVER] amp={use_amp} channels_last={channels_last}")

    model.train()
    iterator = iter(train_loader)
    previous_end = time.perf_counter()
    totals = {"data": 0.0, "h2d": 0.0, "forward": 0.0, "backward": 0.0, "total": 0.0}
    for index in range(max(1, args.batches)):
        try:
            images, target = next(iterator)
        except StopIteration:
            break

        data_done = time.perf_counter()
        images = images.to(device, non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        target = target.to(device, non_blocking=True)
        synchronize(device)
        h2d_done = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=bool(use_amp and device.type in ("cuda", "cpu"))):
            out = model.forward_train(images, target, epoch=0)
            loss = out["loss"]
        synchronize(device)
        forward_done = time.perf_counter()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        synchronize(device)
        step_done = time.perf_counter()

        row = {
            "data": data_done - previous_end,
            "h2d": h2d_done - data_done,
            "forward": forward_done - h2d_done,
            "backward": step_done - forward_done,
            "total": step_done - previous_end,
        }
        for key, value in row.items():
            totals[key] += value
        print(
            f"[BATCH {index + 1}] "
            f"data={row['data']:.3f}s h2d={row['h2d']:.3f}s "
            f"forward={row['forward']:.3f}s backward={row['backward']:.3f}s "
            f"total={row['total']:.3f}s loss={loss.item():.4f}"
        )
        previous_end = step_done

    count = index + 1
    print(
        "[AVG] "
        + " ".join(f"{key}={value / count:.3f}s" for key, value in totals.items())
    )


if __name__ == "__main__":
    main()
