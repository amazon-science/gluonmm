import os
import time
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def reduce_tensor(tensor):
    """
    https://github.com/microsoft/Swin-Transformer/blob/main/utils.py#L88
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def build_log_dir(cfg):
    # create base log directory
    if cfg.CONFIG.LOG.EXP_NAME == 'use_time':
        cfg.CONFIG.LOG.EXP_NAME = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_path = os.path.join(cfg.CONFIG.LOG.BASE_PATH, cfg.CONFIG.LOG.EXP_NAME)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # dump config file
    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())

    # create tensorboard saving directory
    tb_logdir = os.path.join(log_path, cfg.CONFIG.LOG.LOG_DIR)
    if not os.path.exists(tb_logdir):
        os.makedirs(tb_logdir)

    # create checkpoint saving directory
    ckpt_logdir = os.path.join(log_path, cfg.CONFIG.LOG.SAVE_DIR)
    if not os.path.exists(ckpt_logdir):
        os.makedirs(ckpt_logdir)

    return tb_logdir
