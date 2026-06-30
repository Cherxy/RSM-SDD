from __future__ import annotations
import argparse, os
import torch
torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
from mdistiller.engine.cfg import load_config
from mdistiller.engine.utils import load_checkpoint
from mdistiller.engine.trainer import evaluate
from mdistiller.dataset import build_loaders
from mdistiller.models import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', default=None)
    p.add_argument('--gpu', default=None)
    p.add_argument('--debug-fake-data', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.cfg)
    _, val_loader = build_loaders(cfg, args.data_root, args.debug_fake_data)
    student = build_model(cfg.MODEL.STUDENT, num_classes=cfg.DATASET.NUM_CLASSES, dataset=cfg.DATASET.NAME).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location='cpu')
    state = ckpt.get('student', ckpt)
    msg = student.load_state_dict({k.replace('module.',''):v for k,v in state.items()}, strict=False)
    print(f'[INFO] loaded student: {msg}')
    res = evaluate(student, val_loader, device, topk=(1,5))
    print(f'[TEST] loss={res["loss"]:.4f} top1={res["top1"]:.2f} top5={res["top5"]:.2f}')

if __name__ == '__main__':
    main()
