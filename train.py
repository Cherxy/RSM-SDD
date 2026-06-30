from __future__ import annotations
import argparse, os
from pathlib import Path
from datetime import datetime
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from mdistiller.engine.cfg import load_config
from mdistiller.engine.utils import seed_everything, save_checkpoint, load_state_safely
from mdistiller.engine.trainer import train_one_epoch, evaluate
from mdistiller.dataset import build_loaders
from mdistiller.models import build_model
from mdistiller.distillers import build_distiller
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def get_teacher_ckpt_fallback(cfg, requested_ckpt: str) -> tuple[str, str] | tuple[None, None]:
    dataset = str(getattr(cfg.DATASET, 'NAME', '')).lower()
    teacher_name = str(getattr(cfg.MODEL, 'TEACHER', '')).lower().replace('-', '_')
    if dataset not in ('cub200', 'cub_200_2011', 'cub-200'):
        return None, None

    fallback_map = {
        'resnet32x4': 'save/models/resnet32x4_vanilla/ckpt_epoch_240.pth',
        'vgg13': 'save/models/vgg13_vanilla/ckpt_epoch_240.pth',
    }
    fallback_ckpt = fallback_map.get(teacher_name)
    if not fallback_ckpt or not Path(fallback_ckpt).exists():
        return None, None

    msg = (
        f'configured CUB200 teacher checkpoint is unavailable: {requested_ckpt}. '
        f'Falling back to architecture-compatible pretrained weights: {fallback_ckpt}. '
        'Classifier-mismatch keys will be skipped automatically during loading.'
    )
    return fallback_ckpt, msg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--data-root', default=None)
    p.add_argument('--output', default='./runs/exp')
    p.add_argument('--gpu', default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--debug-fake-data', action='store_true')
    p.add_argument('--strict-teacher', action='store_true')
    p.add_argument('--allow-random-teacher', action='store_true', help='Allow training when teacher checkpoint is missing (not recommended for final accuracy).')
    p.add_argument('--eval-interval', type=int, default=2, help='Run validation every N epochs (default: 2).')
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output) / 'train_log.txt'

    def log(msg: str):
        print(msg)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log(f'[START] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    cfg = load_config(args.cfg)
    seed_everything(int(getattr(cfg.SOLVER, 'SEED', 42)))
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        log(
            f'[DEVICE] cuda name={torch.cuda.get_device_name(0)} '
            f'visible={os.environ.get("CUDA_VISIBLE_DEVICES", "") or "all"} '
            f'cuda_version={torch.version.cuda}'
        )
    else:
        log('[DEVICE] cpu (torch.cuda.is_available() is False)')
    train_loader, val_loader = build_loaders(cfg, args.data_root, args.debug_fake_data)
    log(
        f'[DATA] train_batches={len(train_loader)} val_batches={len(val_loader)} '
        f'batch_size={cfg.SOLVER.BATCH_SIZE} num_workers={cfg.SOLVER.NUM_WORKERS}'
    )
    student = build_model(cfg.MODEL.STUDENT, num_classes=cfg.DATASET.NUM_CLASSES, dataset=cfg.DATASET.NAME).to(device)
    teacher = build_model(cfg.MODEL.TEACHER, num_classes=cfg.DATASET.NUM_CLASSES, dataset=cfg.DATASET.NAME).to(device)
    ckpt = getattr(cfg.MODEL, 'TEACHER_CKPT', '')
    teacher_loaded = False
    loaded_ckpt = ckpt
    if ckpt:
        teacher_loaded = load_state_safely(teacher, ckpt, strict=False)
        if not teacher_loaded:
            fallback_ckpt, fallback_msg = get_teacher_ckpt_fallback(cfg, ckpt)
            if fallback_ckpt:
                log(f'[WARN] {fallback_msg}')
                teacher_loaded = load_state_safely(teacher, fallback_ckpt, strict=False)
                if teacher_loaded:
                    loaded_ckpt = fallback_ckpt
        if (not teacher_loaded) and args.strict_teacher:
            raise FileNotFoundError(f'Teacher checkpoint missing: {ckpt}')
    else:
        print('[WARN] MODEL.TEACHER_CKPT is empty. Teacher is randomly initialized unless torchvision weights are loaded manually.')
    if not teacher_loaded:
        msg = ('Teacher checkpoint is not loaded. Distillation with a random teacher usually causes very low accuracy. '
               'Please provide MODEL.TEACHER_CKPT or pass --allow-random-teacher to continue intentionally.')
        if args.allow_random_teacher:
            log(f'[WARN] {msg}')
        else:
            raise RuntimeError(msg)
    else:
        log(f'[INFO] teacher checkpoint loaded: {loaded_ckpt}')
    distiller = build_distiller(cfg.DISTILLER.TYPE, student, teacher, cfg).to(device)
    opt = SGD([p for p in distiller.parameters() if p.requires_grad], lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    epochs = args.epochs or int(cfg.SOLVER.EPOCHS)
    sched_type = str(getattr(cfg.SOLVER, 'SCHEDULER', 'multistep')).lower()
    if sched_type == 'cosine':
        eta_min = float(getattr(cfg.SOLVER, 'MIN_LR', 1e-5))
        sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)
    else:
        sched = MultiStepLR(opt, milestones=list(cfg.SOLVER.LR_DECAY_STAGES), gamma=float(cfg.SOLVER.LR_DECAY_RATE))
    use_amp = bool(getattr(cfg.SOLVER, 'USE_AMP', False))
    grad_clip_norm = float(getattr(cfg.SOLVER, 'GRAD_CLIP_NORM', 0.0))
    log(f'[SOLVER] epochs={epochs} amp={use_amp} grad_clip_norm={grad_clip_norm}')
    eval_interval = max(1, int(args.eval_interval))
    best = 0.0
    best_epoch = -1
    for epoch in range(epochs):
        log(f'\n[Epoch {epoch+1}/{epochs}] lr={opt.param_groups[0]["lr"]:.6f}')
        tr = train_one_epoch(distiller, train_loader, opt, device, epoch=epoch, use_amp=use_amp, grad_clip_norm=grad_clip_norm)
        log(
            f'[TRAIN] loss={tr["loss"]:.4f} ce={tr.get("loss_ce", 0.0):.4f} '
            f'kd={tr.get("loss_global", 0.0):.4f} local={tr.get("loss_local", 0.0):.4f} '
            f'gac={tr.get("loss_gac", 0.0):.4f} top1={tr["top1"]:.2f}'
        )
        do_eval = ((epoch + 1) % eval_interval == 0) or ((epoch + 1) == epochs)
        ev = None
        if do_eval:
            ev = evaluate(student, val_loader, device, topk=(1,5))
            log(f'[EVAL] val_loss={ev["loss"]:.4f} val_top1={ev["top1"]:.2f} val_top5={ev["top5"]:.2f}')
        state = {
            'epoch': epoch+1,
            'student': student.state_dict(),
            'distiller': distiller.state_dict(),
            'cfg': args.cfg,
            'best_top1': best,
            'best_epoch': best_epoch,
        }
        save_checkpoint(state, args.output, 'last.pth')
        if ev is not None and ev['top1'] >= best:
            best = ev['top1']
            best_epoch = epoch + 1
            state['best_top1'] = best
            state['best_epoch'] = best_epoch
            save_checkpoint(state, args.output, 'best.pth')
        sched.step()
    log(f'[DONE] best_top1={best:.2f} at epoch={best_epoch}; outputs saved to {args.output}')
    log(f'[LOG] training log saved to {log_path}')

if __name__ == '__main__':
    main()
