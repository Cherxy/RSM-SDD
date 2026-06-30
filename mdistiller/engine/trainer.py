from __future__ import annotations
import torch
from tqdm import tqdm
from .utils import AverageMeter, accuracy


def train_one_epoch(model, loader, optimizer, device, epoch=0, print_freq=50, use_amp=False, grad_clip_norm=0.0):
    model.train()
    loss_meter, top1_meter = AverageMeter('loss'), AverageMeter('top1')
    ce_meter = AverageMeter('loss_ce')
    global_meter = AverageMeter('loss_global')
    local_meter = AverageMeter('loss_local')
    gac_meter = AverageMeter('loss_gac')
    scaler = torch.amp.GradScaler('cuda', enabled=bool(use_amp and device.type == 'cuda'))
    pbar = tqdm(loader, desc=f'train epoch {epoch}', ncols=100)
    for it, (images, target) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=bool(use_amp and device.type in ('cuda', 'cpu'))):
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
        pbar.set_postfix(loss=f'{loss_meter.avg:.4f}', ce=f'{ce_meter.avg:.4f}', kd=f'{global_meter.avg:.4f}', loc=f'{local_meter.avg:.4f}', gac=f'{gac_meter.avg:.4f}', top1=f'{top1_meter.avg:.2f}')
    return {
        'loss': loss_meter.avg,
        'top1': top1_meter.avg,
        'loss_ce': ce_meter.avg,
        'loss_global': global_meter.avg,
        'loss_local': local_meter.avg,
        'loss_gac': gac_meter.avg,
    }


@torch.no_grad()
def evaluate(model, loader, device, topk=(1,5)):
    model.eval()
    top1_meter, top5_meter, loss_meter = AverageMeter('top1'), AverageMeter('top5'), AverageMeter('loss')
    ce = torch.nn.CrossEntropyLoss()
    for images, target in tqdm(loader, desc='eval', ncols=100):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        loss = ce(logits, target)
        accs = accuracy(logits, target, topk=topk)
        top1_meter.update(accs[0].item(), images.size(0))
        if len(accs) > 1:
            top5_meter.update(accs[1].item(), images.size(0))
        loss_meter.update(loss.item(), images.size(0))
    return {'loss': loss_meter.avg, 'top1': top1_meter.avg, 'top5': top5_meter.avg}
