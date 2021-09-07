import os
import sys
sys.path.append(os.getcwd())

import argparse
import datetime
import time

import torch
from tensorboardX import SummaryWriter
from src.transformers.pipelines.video_action_recognition_config import get_cfg_defaults
from src.transformers.pipelines.launch import spawn_workers
from src.transformers.utils.utils import build_log_dir
from src.transformers.models import get_model
from src.transformers.data.datasets.kinetics_datasets import build_dataloader
from src.transformers.utils.model_utils import deploy_model, load_model, save_checkpoint
from src.transformers.utils.video_action_recognition import train_classification, validate_classification


def main_worker(cfg):
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None
    cfg.defrost()
    if cfg.CONFIG.TEST.MULTI_VIEW_TEST:
        # disable multi-view testing during training
        cfg.CONFIG.TEST.MULTI_VIEW_TEST = False
    cfg.freeze()

    # create model
    print('Creating model: %s' % (cfg.CONFIG.MODEL.NAME))
    model = get_model(cfg)
    model_without_ddp = model

    # create dataset and dataloader
    train_loader, val_loader, train_sampler, val_sampler, _ = build_dataloader(cfg)

    # create criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.CONFIG.TRAIN.LR,
                                momentum=cfg.CONFIG.TRAIN.OPTIMIZER.MOMENTUM,
                                weight_decay=cfg.CONFIG.TRAIN.WEIGHT_DECAY)

    model, optimizer, model_ema = deploy_model(model, optimizer, cfg)
    model_without_ddp = model.module
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))

    # create lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=cfg.CONFIG.TRAIN.LR_SCHEDULER.LR_MILESTONE,
                                                        gamma=cfg.CONFIG.TRAIN.LR_SCHEDULER.DECAY_RATE)

    print('Start training...')
    start_time = time.time()
    max_accuracy = 0.0
    mixup_fn = None
    for epoch in range(cfg.CONFIG.TRAIN.START_EPOCH, cfg.CONFIG.TRAIN.EPOCH_NUM):
        if cfg.DDP_CONFIG.DISTRIBUTED:
            train_sampler.set_epoch(epoch)

        train_classification(cfg, model, model_ema, criterion, train_loader, optimizer, epoch, mixup_fn, lr_scheduler, writer)
        lr_scheduler.step()

        if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0 and (
                epoch % cfg.CONFIG.LOG.SAVE_FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1):
            save_checkpoint(cfg, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler)

        if epoch % cfg.CONFIG.VAL.FREQ == 0 or epoch == cfg.CONFIG.TRAIN.EPOCH_NUM - 1:
            acc1, acc5, loss = validate_classification(cfg, val_loader, model, criterion, epoch, writer)
            max_accuracy = max(max_accuracy, acc1)
            print(f"Accuracy of the network: {acc1:.1f}%")
            print(f'Max accuracy: {max_accuracy:.2f}%')

    if writer is not None:
        writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='./scripts/action_recognition/config/vidtr_s_8x8_patch16_224_k400.yaml',
                        help='path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)
