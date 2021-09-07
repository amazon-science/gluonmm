import os
import time
import datetime
import numpy as np

import torch

from timm.utils import accuracy, AverageMeter
from .utils import reduce_tensor, get_grad_norm

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def train_classification(cfg, model, model_ema, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, writer=None):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    norms = AverageMeter()

    start = time.time()
    end = time.time()
    for step, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        data_time.update(time.time() - end)

        outputs = model(samples)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        if cfg.CONFIG.TRAIN.USE_AMP is True and amp is not None:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if cfg.CONFIG.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), cfg.CONFIG.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(amp.master_params(optimizer))
        else:
            loss.backward()
            if cfg.CONFIG.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), cfg.CONFIG.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(amp.master_params(optimizer))
        optimizer.step()

        torch.cuda.synchronize()
        if cfg.CONFIG.TRAIN.USE_EMA is True:
            model_ema.update(model)

        cur_iter = epoch * num_steps + step
        lr_scheduler.step_update(cur_iter)

        losses.update(loss.item(), targets.size(0))
        norms.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0:
            print('----train----')
            lr = optimizer.param_groups[0]['lr']
            print('lr: ', lr)
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(data_loader))
            print(print_string)
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print_string = 'memory used: {memory:.0f}MB'.format(memory=memory_used)
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}, grad_norm: {norm:.5f}'.format(
                loss=losses.avg,
                norm=norms.avg)
            print(print_string)

            writer.add_scalar('train_loss_iter', losses.avg, cur_iter)
            writer.add_scalar('train_norm_iter', norms.avg, cur_iter)
            writer.add_scalar('train_bs_iter', targets.size(0), cur_iter)
            writer.add_scalar('train_lr_iter', lr, cur_iter)

    epoch_time = time.time() - start
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate_classification(cfg, data_loader, model, epoch, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    end = time.time()
    for step, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        # consider 5 crops and 10 crops evaluation
        if len(samples.shape) == 4:
            samples.unsqueeze_(dim=1)
        b, n, c, w, h = samples.shape
        samples = samples.reshape(-1, c, w, h)
        outputs = model(samples)
        outputs = outputs.reshape(b, n, -1)
        outputs = torch.mean(outputs, dim=1)

        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs.data, targets, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        losses.update(loss.item(), targets.size(0))
        top1.update(acc1.item(), targets.size(0))
        top5.update(acc5.item(), targets.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and step % cfg.CONFIG.LOG.DISPLAY_FREQ == 0:
            print('----validation----')
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(data_loader))
            print(print_string)
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            print_string = 'memory used: {memory:.0f}MB'.format(memory=memory_used)
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)

    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and writer is not None:
        writer.add_scalar('val_loss_epoch', losses.avg, epoch)
        writer.add_scalar('val_top1_acc_epoch', top1.avg, epoch)
        writer.add_scalar('val_top5_acc_epoch', top5.avg, epoch)

    return top1.avg, top5.avg, losses.avg
