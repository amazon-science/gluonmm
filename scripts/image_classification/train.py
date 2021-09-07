"""Code is mostly adapted from
https://github.com/facebookresearch/deit/main.py
Copyright (c) 2015-present, Facebook, Inc.,
Apache License, Version 2.0
"""
import sys
sys.path.extend(['/home/ubuntu/code/gluonmm'])
import os
import argparse
import datetime
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from tensorboardX import SummaryWriter

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from src.transformers.pipelines.image_classification_config import get_cfg_defaults
from src.transformers.pipelines.launch import spawn_workers
from src.transformers.utils.utils import build_log_dir
from src.transformers.models import get_model
from src.transformers.data.datasets.img_cls_datasets import build_dataloader
from src.transformers.utils.optimizer import build_optimizer
from src.transformers.utils.lr_scheduler import build_scheduler
from src.transformers.utils.model_utils import deploy_model, save_checkpoint
from src.transformers.utils.image_classification import train_classification, validate_classification


def main_worker(cfg):
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None
    cfg.freeze()

    # create criterion
    criterion = LabelSmoothingCrossEntropy()
    if cfg.CONFIG.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif cfg.CONFIG.AUG.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=cfg.CONFIG.AUG.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # create model
    # should follow the order: build_model -> build optimizer -> deploy model
    print('Creating model: %s' % (cfg.CONFIG.MODEL.NAME))
    model = get_model(cfg)
    model_without_ddp = model

    # create optimizer
    # linear scale the learning rate according to total batch size, may not be optimal for transformer
    linear_scaled_lr = cfg.CONFIG.TRAIN.LR * cfg.CONFIG.TRAIN.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = cfg.CONFIG.TRAIN.WARMUP_START_LR * cfg.CONFIG.TRAIN.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = cfg.CONFIG.TRAIN.MIN_LR * cfg.CONFIG.TRAIN.BATCH_SIZE * dist.get_world_size() / 512.0
    cfg.defrost()
    cfg.CONFIG.TRAIN.LR = linear_scaled_lr    # 5e-4 -> 0.001
    cfg.CONFIG.TRAIN.WARMUP_START_LR = linear_scaled_warmup_lr
    cfg.CONFIG.TRAIN.MIN_LR = linear_scaled_min_lr
    cfg.freeze()
    optimizer = build_optimizer(cfg, model_without_ddp)

    model, optimizer, model_ema = deploy_model(model, optimizer, cfg)
    model_without_ddp = model.module
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))

    # create dataset and dataloader
    train_loader, val_loader, train_sampler, val_sampler, mixup_fn = build_dataloader(cfg)

    # create lr scheduler
    lr_scheduler = build_scheduler(cfg, optimizer, len(train_loader))

    # resume from a checkpoint
    if cfg.CONFIG.TRAIN.RESUME:
        if cfg.CONFIG.TRAIN.RESUME.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.CONFIG.TRAIN.RESUME, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.CONFIG.TRAIN.RESUME, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            cfg.defrost()
            cfg.CONFIG.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
            cfg.freeze()
            if 'amp' in checkpoint and cfg.CONFIG.TRAIN.AMP_LEVEL != "O0" and checkpoint['config'].CONFIG.TRAIN.AMP_LEVEL != "O0":
                amp.load_state_dict(checkpoint['amp'])
        print('Resume from previous checkpoint of epoch %d at %s' % (checkpoint['epoch'], cfg.CONFIG.TRAIN.RESUME))

    print('Start training...')
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(cfg.CONFIG.TRAIN.START_EPOCH, cfg.CONFIG.TRAIN.EPOCH_NUM):
        if cfg.DDP_CONFIG.DISTRIBUTED:
            train_sampler.set_epoch(epoch)

        train_classification(cfg, model, model_ema, criterion, train_loader, optimizer, epoch, mixup_fn, lr_scheduler, writer)

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and (epoch % cfg.CONFIG.LOG.SAVE_FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1):
            save_checkpoint(cfg, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler)

        if epoch % cfg.CONFIG.VAL.FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1:
            acc1, acc5, loss = validate_classification(cfg, val_loader, model, epoch, writer)
            max_accuracy = max(max_accuracy, acc1)
            print(f"Accuracy of the network: {acc1:.1f}%")
            print(f'Max accuracy: {max_accuracy:.2f}%')

    if writer is not None:
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image transformer models.')
    parser.add_argument('--config-file',
                        default='./scripts/image_classification/config/deit_small_patch16_224_imagenet1k.yaml',
                        type=str, help='path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)
