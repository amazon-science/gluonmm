import sys
sys.path.extend(['/home/ubuntu/code/gluonmm'])
import argparse

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.transformers.pipelines.image_classification_config import get_cfg_defaults
from src.transformers.pipelines.launch import spawn_workers
from src.transformers.utils.utils import build_log_dir
from src.transformers.models import get_model
from src.transformers.data.datasets.img_cls_datasets import build_dataloader
from src.transformers.utils.model_utils import deploy_model
from src.transformers.utils.image_classification import validate_classification


def main_worker(cfg):
    # create tensorboard and logs
    if cfg.DDP_CONFIG.GPU_WORLD_RANK == 0:
        tb_logdir = build_log_dir(cfg)
        writer = SummaryWriter(log_dir=tb_logdir)
    else:
        writer = None
    cfg.freeze()

    # create model
    model = get_model(cfg)
    optimizer = None
    model, _, _ = deploy_model(model, optimizer, cfg)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))

    # create dataset and dataloader
    train_loader, val_loader, train_sampler, val_sampler, _ = build_dataloader(cfg)

    # create loss objectives
    criterion = nn.CrossEntropyLoss().cuda()

    # run evaluation
    acc1, acc5, _ = validate_classification(cfg, val_loader, model, 0, writer)
    print('Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(top1_acc=acc1, top5_acc=acc5))

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate transformer models.')
    parser.add_argument('--config-file',
                        default='./scripts/image_classification/config/deit_small_patch16_224_imagenet1k.yaml',
                        type=str, help='path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)
