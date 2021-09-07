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
    top1 = AverageMeter()
    top5 = AverageMeter()
    norms = AverageMeter()

    start = time.time()
    end = time.time()
    for step, data in enumerate(data_loader):
        samples = data[0].cuda(non_blocking=True)
        targets = data[1].cuda(non_blocking=True)
        data_time.update(time.time() - end)

        outputs = model(samples)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs.data, targets, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

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
        # lr_scheduler.step_update(cur_iter)

        losses.update(loss.item(), targets.size(0))
        norms.update(grad_norm)
        top1.update(acc1.item(), targets.size(0))
        top5.update(acc5.item(), targets.size(0))

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
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)

            writer.add_scalar('train_loss_iter', losses.avg, cur_iter)
            writer.add_scalar('train_norm_iter', norms.avg, cur_iter)
            writer.add_scalar('train_top1_acc_iter', top1.avg, cur_iter)
            writer.add_scalar('train_top5_acc_iter', top5.avg, cur_iter)
            writer.add_scalar('train_bs_iter', targets.size(0), cur_iter)
            writer.add_scalar('train_lr_iter', lr, cur_iter)

    epoch_time = time.time() - start
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate_classification(cfg, data_loader, model, criterion, epoch, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for step, data in enumerate(data_loader):
        samples = data[0].cuda(non_blocking=True)
        targets = data[1].cuda(non_blocking=True)
        data_time.update(time.time() - end)

        outputs = model(samples)

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


@torch.no_grad()
def test_classification(cfg, data_loader, model, criterion, epoch, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    postprocess = torch.nn.Softmax(dim=1)

    end = time.time()
    for step, data in enumerate(data_loader):
        data_time.update(time.time() - end)
        samples = data[0].cuda(non_blocking=True)
        targets = data[1].cuda(non_blocking=True)

        '''
        The shape of `samples` is like BxCxTxHxW, where B is 1, C is 3 (RGB).
        T = test_num_segment * test_num_crop * clip_len, e.g., T = 10 * 3 * 8 for vidtr_s_8x8.
        H and W are just crop size, e.g., 224 or 256 for most of the time.
        '''
        assert samples.size(0) == 1, 'batch_size during multiview test must be set to 1 due to limited GPU memory'

        out_list = []
        for i in range(0, samples.size(2), cfg.CONFIG.DATA.CLIP_LEN):
            cur_input = samples[:, :, i:i+cfg.CONFIG.DATA.CLIP_LEN, :, :]
            out_list.append(postprocess(model(cur_input)))
        outputs = torch.cat(out_list, dim=0)
        outputs = torch.mean(outputs, dim=0, keepdim=True)

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
            print('----Testing----')
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

    return top1.avg, top5.avg, losses.avg
