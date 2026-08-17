from __future__ import annotations
import os
import time
import torch
from tqdm import tqdm
from .utils import AverageMeter, accuracy


def _amp_enabled(use_amp, device):
    return bool(use_amp and device.type in ('cuda', 'cpu'))


def _move_images(images, device, channels_last=False):
    images = images.to(device, non_blocking=True)
    if channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch=0,
    print_freq=50,
    use_amp=False,
    grad_clip_norm=0.0,
    scaler=None,
    channels_last=False,
):
    model.train()
    loss_meter, top1_meter = AverageMeter('loss'), AverageMeter('top1')
    ce_meter = AverageMeter('loss_ce')
    global_meter = AverageMeter('loss_global')
    local_meter = AverageMeter('loss_local')
    gac_meter = AverageMeter('loss_gac')
    lgc_meter = AverageMeter('loss_lgc')
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=bool(use_amp and device.type == 'cuda'))
    amp_enabled = _amp_enabled(use_amp, device)
    profile = os.environ.get('PROFILE', '') not in ('', '0', 'false', 'False')
    _profile_iters = int(os.environ.get('PROFILE_ITERS', '0') or '0')
    if profile and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    pbar = tqdm(loader, desc=f'train epoch {epoch}', ncols=100)
    _t_prev = time.perf_counter()
    _data_ms = _comp_ms = 0.0
    for it, (images, target) in enumerate(pbar):
        _t_got = time.perf_counter()
        images = _move_images(images, device, channels_last=channels_last)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            out = model.forward_train(images, target, epoch=epoch)
            loss = out['loss']
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
        logits = out['logits_s']
        top1 = accuracy(logits.detach(), target, topk=(1,))[0]
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(top1.item(), images.size(0))
        if 'loss_ce' in out:
            ce_meter.update(out['loss_ce'].item(), images.size(0))
        if 'loss_global' in out:
            global_meter.update(out['loss_global'].item(), images.size(0))
        if 'loss_local' in out:
            local_meter.update(out['loss_local'].item(), images.size(0))
        if 'loss_gac' in out:
            gac_meter.update(out['loss_gac'].item(), images.size(0))
        if 'loss_lgc' in out:
            lgc_meter.update(out['loss_lgc'].item(), images.size(0))
        pbar.set_postfix(loss=f'{loss_meter.avg:.4f}', ce=f'{ce_meter.avg:.4f}', kd=f'{global_meter.avg:.4f}', loc=f'{local_meter.avg:.4f}', gac=f'{gac_meter.avg:.4f}', lgc=f'{lgc_meter.avg:.4f}', top1=f'{top1_meter.avg:.2f}')
        if profile:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            _now = time.perf_counter()
            _data_ms += (_t_got - _t_prev) * 1000.0
            _comp_ms += (_now - _t_got) * 1000.0
            if (it + 1) % 10 == 0:
                msg = f'[PROFILE] iter={it + 1} data={_data_ms / 10:.0f}ms compute={_comp_ms / 10:.0f}ms'
                if device.type == 'cuda':
                    msg += (f' peak_alloc={torch.cuda.max_memory_allocated() / 1e9:.2f}GB'
                            f' peak_reserved={torch.cuda.max_memory_reserved() / 1e9:.2f}GB')
                print(msg)
                _data_ms = _comp_ms = 0.0
            if _profile_iters and (it + 1) >= _profile_iters:
                print(f'[PROFILE] reached {_profile_iters} iters, stopping profile run.')
                raise SystemExit(0)
            _t_prev = time.perf_counter()
    return {
        'loss': loss_meter.avg,
        'top1': top1_meter.avg,
        'loss_ce': ce_meter.avg,
        'loss_global': global_meter.avg,
        'loss_local': local_meter.avg,
        'loss_gac': gac_meter.avg,
        'loss_lgc': lgc_meter.avg,
    }


@torch.no_grad()
def evaluate(model, loader, device, topk=(1,5), channels_last=False):
    model.eval()
    top1_meter, top5_meter, loss_meter = AverageMeter('top1'), AverageMeter('top5'), AverageMeter('loss')
    ce = torch.nn.CrossEntropyLoss()
    for images, target in tqdm(loader, desc='eval', ncols=100):
        images = _move_images(images, device, channels_last=channels_last)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        loss = ce(logits, target)
        accs = accuracy(logits, target, topk=topk)
        top1_meter.update(accs[0].item(), images.size(0))
        if len(accs) > 1:
            top5_meter.update(accs[1].item(), images.size(0))
        loss_meter.update(loss.item(), images.size(0))
    return {'loss': loss_meter.avg, 'top1': top1_meter.avg, 'top5': top5_meter.avg}
