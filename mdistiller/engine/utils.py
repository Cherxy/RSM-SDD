from __future__ import annotations
import os, random, time
from pathlib import Path
import numpy as np
import torch


def _resolve_existing_path_case_insensitive(path_str: str) -> str | None:
    """Resolve path with case-insensitive fallback on case-sensitive filesystems.

    This keeps existing behavior for correct paths, while making checkpoints robust
    to naming differences like `resnet50_vanilla` vs `ResNet50_vanilla`.
    """
    if not path_str:
        return None

    p = Path(path_str)
    if p.exists():
        return str(p)

    # Convert relative path to an absolute traversal base.
    p_abs = p if p.is_absolute() else (Path.cwd() / p)
    anchor = Path(p_abs.anchor) if p_abs.is_absolute() else Path.cwd().anchor
    curr = anchor if str(anchor) else Path('/')

    for part in p_abs.parts:
        # Skip drive/root markers already represented by `curr`
        if part in ('/', curr.anchor, ''):
            continue

        exact = curr / part
        if exact.exists():
            curr = exact
            continue

        # Case-insensitive fallback among siblings.
        try:
            children = list(curr.iterdir())
        except Exception:
            return None
        low = part.lower()
        matched = [c for c in children if c.name.lower() == low]
        if not matched:
            return None
        curr = matched[0]

    return str(curr) if curr.exists() else None

class AverageMeter:
    def __init__(self, name='meter'):
        self.name = name
        self.reset()
    def reset(self):
        self.val = 0.0; self.avg = 0.0; self.sum = 0.0; self.count = 0
    def update(self, val, n=1):
        self.val = float(val); self.sum += float(val) * n; self.count += n; self.avg = self.sum / max(1, self.count)


def seed_everything(seed=42):
    torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            k = min(k, output.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / target.size(0)))
        return res


def save_checkpoint(state, output, name='last.pth'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.save(state, output / name)


def load_checkpoint(path, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    return ckpt


def load_state_safely(model, ckpt_path, strict=False, prefix_candidates=('model', 'state_dict', 'student', 'teacher')):
    if not ckpt_path:
        return False
    resolved_ckpt = _resolve_existing_path_case_insensitive(ckpt_path)
    if not resolved_ckpt:
        print(f'[WARN] checkpoint not found: {ckpt_path}')
        return False
    if resolved_ckpt != ckpt_path:
        print(f'[INFO] checkpoint path auto-resolved: {ckpt_path} -> {resolved_ckpt}')
    # PyTorch 2.6+ changed torch.load default: weights_only=True.
    # Some historical checkpoints contain extra python objects (e.g., numpy scalar)
    # and will fail to load in weights-only mode.
    try:
        ckpt = torch.load(resolved_ckpt, map_location='cpu')
    except Exception as e:
        if 'Weights only load failed' in str(e):
            print('[WARN] torch.load failed in weights_only mode, retrying with weights_only=False (trusted checkpoint only).')
            ckpt = torch.load(resolved_ckpt, map_location='cpu', weights_only=False)
        else:
            raise
    state = None
    if isinstance(ckpt, dict):
        for k in prefix_candidates:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]; break
        if state is None:
            state = ckpt
    else:
        state = ckpt
    # remove common prefixes and add key-compat remap for common CIFAR ResNet variants
    clean = {}
    for k, v in state.items():
        nk = k.replace('module.', '')
        clean[nk] = v

    model_state = model.state_dict()
    model_uses_base_prefix = any(k.startswith('base.') for k in model_state)

    # Some public checkpoints use `downsample` while this repo uses `shortcut`.
    # Add bidirectional aliases to maximize loading compatibility.
    remap = {}
    for k, v in list(clean.items()):
        # Torchvision backbones are wrapped under `base` in ImageNet models,
        # while official checkpoints usually save keys like `conv1.weight`.
        if model_uses_base_prefix and not k.startswith('base.'):
            remap[f'base.{k}'] = v

        if '.downsample.' in k:
            remap[k.replace('.downsample.', '.shortcut.')] = v
        if '.shortcut.' in k:
            remap[k.replace('.shortcut.', '.downsample.')] = v

        # Some checkpoints name classifier as `linear.*` while local models use `fc.*`
        # (or vice versa). Add aliases for both directions.
        if k.startswith('linear.'):
            remap[k.replace('linear.', 'fc.', 1)] = v
            remap[k.replace('linear.', 'classifier.', 1)] = v
        if k.startswith('fc.'):
            remap[k.replace('fc.', 'linear.', 1)] = v
            remap[k.replace('fc.', 'classifier.', 1)] = v
        if k.startswith('classifier.'):
            remap[k.replace('classifier.', 'fc.', 1)] = v
            remap[k.replace('classifier.', 'linear.', 1)] = v
    clean.update(remap)

    try:
        if strict:
            msg = model.load_state_dict(clean, strict=True)
        else:
            filtered = {}
            shape_mismatch = []
            unexpected = []
            for k, v in clean.items():
                if k not in model_state:
                    unexpected.append(k)
                    continue
                if hasattr(v, 'shape') and hasattr(model_state[k], 'shape') and tuple(v.shape) == tuple(model_state[k].shape):
                    filtered[k] = v
                else:
                    ckpt_shape = tuple(v.shape) if hasattr(v, 'shape') else type(v)
                    model_shape = tuple(model_state[k].shape) if hasattr(model_state[k], 'shape') else type(model_state[k])
                    shape_mismatch.append((k, ckpt_shape, model_shape))

            if model_uses_base_prefix:
                unexpected = [
                    k for k in unexpected
                    if k.startswith('base.') or f'base.{k}' not in filtered
                ]

            if len(filtered) == 0:
                print(f'[WARN] no compatible parameters can be loaded from {ckpt_path}.')
                if len(shape_mismatch) > 0:
                    print(f'[WARN] first_shape_mismatches: {shape_mismatch[:8]}')
                return False

            if len(shape_mismatch) > 0:
                print(f'[WARN] skipped {len(shape_mismatch)} keys due to shape mismatch.')
                print(f'[WARN] first_shape_mismatches: {shape_mismatch[:8]}')
            if len(unexpected) > 0:
                print(f'[INFO] skipped {len(unexpected)} unexpected keys not in target model.')

            msg = model.load_state_dict(filtered, strict=False)
    except RuntimeError as e:
        print(f'[WARN] failed to load checkpoint {resolved_ckpt}: {e}')
        return False

    if hasattr(msg, 'missing_keys') and hasattr(msg, 'unexpected_keys'):
        print(f'[INFO] loaded checkpoint {resolved_ckpt}: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')
        if len(msg.missing_keys) > 0:
            print(f'[INFO] first_missing_keys: {msg.missing_keys[:8]}')
        if len(msg.unexpected_keys) > 0:
            print(f'[INFO] first_unexpected_keys: {msg.unexpected_keys[:8]}')
    else:
        print(f'[INFO] loaded checkpoint {resolved_ckpt}: {msg}')
    return True
