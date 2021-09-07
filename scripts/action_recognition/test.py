import os
import sys
sys.path.append(os.getcwd())
import argparse

import torch

from src.transformers.pipelines.video_action_recognition_config import get_cfg_defaults
from src.transformers.pipelines.launch import spawn_workers
from src.transformers.models import get_model
from src.transformers.data.datasets.kinetics_datasets import build_dataloader
from src.transformers.utils.model_utils import deploy_model, load_model
from src.transformers.utils.video_action_recognition import test_classification


def main_worker(cfg):
    cfg.defrost()
    if cfg.CONFIG.TEST.BATCH_SIZE != 1:
        # batch_size during multiview test must be set to 1 due to limited GPU memory
        cfg.CONFIG.TEST.BATCH_SIZE = 1
    if not cfg.CONFIG.TEST.MULTI_VIEW_TEST:
        # enable multi-view testing, e.g., 30-view
        cfg.CONFIG.TEST.MULTI_VIEW_TEST = True
    cfg.freeze()

    # create model
    print('Creating model: %s' % (cfg.CONFIG.MODEL.NAME))
    model = get_model(cfg)
    optimizer = None
    model, _, _ = deploy_model(model, optimizer, cfg)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters in the model: %6.2fM' % (num_parameters / 1000000))
    if cfg.CONFIG.MODEL.LOAD or cfg.CONFIG.MODEL.PRETRAINED:
        model, _ = load_model(model, cfg, load_fc=cfg.CONFIG.MODEL.LOAD_FC)

    # create dataset and dataloader
    train_loader, test_loader, train_sampler, test_sampler, _ = build_dataloader(cfg)

    # create criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # run evaluation
    acc1, acc5, _ = test_classification(cfg, test_loader, model, criterion, 0, None)
    print('Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(top1_acc=acc1, top5_acc=acc5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate video action recognition transformer models.')
    parser.add_argument('--config-file',
                        default='./scripts/action_recognition/config/vidtr_s_8x8_patch16_224_k400.yaml',
                        help='path to config file.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    spawn_workers(main_worker, cfg)
