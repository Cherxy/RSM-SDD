from __future__ import annotations
import argparse
import csv
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mdistiller.engine.cfg import load_config
from mdistiller.engine.utils import load_state_safely, seed_everything
from mdistiller.dataset import build_loaders
from mdistiller.models import build_model
from mdistiller.distillers import build_distiller


def parse_args():
    p = argparse.ArgumentParser(description="Analyze local-global semantic consistency for PAMA-SDD++.")
    p.add_argument("--cfg", required=True)
    p.add_argument("--student-ckpt", default="", help="Path to a trained student/distiller checkpoint.")
    p.add_argument("--data-root", default=None)
    p.add_argument("--output", default="analysis/semantic_consistency.csv")
    p.add_argument("--gpu", default=None)
    p.add_argument("--max-batches", type=int, default=20)
    p.add_argument("--debug-fake-data", action="store_true")
    p.add_argument("--allow-random-teacher", action="store_true")
    return p.parse_args()


def load_student_checkpoint(distiller, ckpt_path: str):
    if not ckpt_path:
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "student" in ckpt:
            distiller.student.load_state_dict(ckpt["student"], strict=False)
            return True
        if "distiller" in ckpt:
            distiller.load_state_dict(ckpt["distiller"], strict=False)
            return True
    distiller.student.load_state_dict(ckpt, strict=False)
    return True


def main():
    args = parse_args()
    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = load_config(args.cfg)
    seed_everything(int(getattr(cfg.SOLVER, "SEED", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = build_loaders(cfg, args.data_root, args.debug_fake_data)
    student = build_model(cfg.MODEL.STUDENT, cfg.DATASET.NUM_CLASSES, cfg.DATASET.NAME).to(device)
    teacher = build_model(cfg.MODEL.TEACHER, cfg.DATASET.NUM_CLASSES, cfg.DATASET.NAME).to(device)

    ckpt = getattr(cfg.MODEL, "TEACHER_CKPT", "")
    teacher_loaded = load_state_safely(teacher, ckpt, strict=False) if ckpt else False
    if not teacher_loaded and not args.allow_random_teacher:
        raise RuntimeError(f"Teacher checkpoint is not loaded: {ckpt}. Use --allow-random-teacher only for debug.")

    distiller = build_distiller(cfg.DISTILLER.TYPE, student, teacher, cfg).to(device)
    load_student_checkpoint(distiller, args.student_ckpt)
    distiller.eval()

    sums = {}
    count = 0
    rows = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if i >= args.max_batches:
                break
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            out = distiller.forward_analysis(images, target)
            metrics = {k: float(v.detach().cpu()) for k, v in out["metrics"].items()}
            rows.append({"batch": i, **metrics})
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1

    means = {k: v / max(1, count) for k, v in sums.items()}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["batch"] + sorted(means.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"batch": "mean", **means})

    print(f"[semantic-consistency] wrote {out_path}")
    for k in sorted(means):
        print(f"  {k}: {means[k]:.6f}")


if __name__ == "__main__":
    main()
