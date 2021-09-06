# Code adapted from https://github.com/facebookresearch/deit/blob/main/datasets.py,
# Copyright (c) 2015-present, Facebook, Inc.
# Apache License, Version 2.0
from torchvision import datasets, transforms
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform, Mixup
from src.transformers.utils.samplers import RASampler


def build_dataset(is_train, cfg):
    transform = build_transform(is_train, cfg)
    root = cfg.CONFIG.DATA.TRAIN_DATA_PATH if is_train else cfg.CONFIG.DATA.VAL_DATA_PATH
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def build_dataloader(cfg):
    # build dataset
    train_dataset = build_dataset(True, cfg)
    val_dataset = build_dataset(False, cfg)

    # build data sampler
    if cfg.DDP_CONFIG.DISTRIBUTED:
        if cfg.CONFIG.AUG.REPEATED_AUG:
            train_sampler = RASampler(train_dataset)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    # builder dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.CONFIG.TRAIN.BATCH_SIZE, shuffle=(train_sampler is None),
        num_workers=9, sampler=train_sampler, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(val_sampler is None),
        num_workers=9, sampler=val_sampler, pin_memory=True)

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = cfg.CONFIG.AUG.MIXUP > 0 or cfg.CONFIG.AUG.CUTMIX > 0. or cfg.CONFIG.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=cfg.CONFIG.AUG.MIXUP, cutmix_alpha=cfg.CONFIG.AUG.CUTMIX, cutmix_minmax=cfg.CONFIG.AUG.CUTMIX_MINMAX,
            prob=cfg.CONFIG.AUG.MIXUP_PROB, switch_prob=cfg.CONFIG.AUG.MIXUP_SWITCH_PROB, mode=cfg.CONFIG.AUG.MIXUP_MODE,
            label_smoothing=cfg.CONFIG.AUG.LABEL_SMOOTHING, num_classes=cfg.CONFIG.DATA.NUM_CLASSES)

    return train_loader, val_loader, train_sampler, val_sampler, mixup_fn


class Stack2Tensor(torch.nn.Module):
    """
    This class is used to replace lambda function used in torchvision.
    Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    because lambda cannot be used together with multiprocessing.
    """
    def __init__(self):
        super().__init__()

    def forward(self, crops):
        """
        Args:
            crops (PIL Image): Image to be stacked.
        Returns:
            Tensor: Stacked images and then transformed to tensor
        """
        return torch.stack([transforms.ToTensor()(crop) for crop in crops])


def build_transform(is_train, cfg):
    resize_im = cfg.CONFIG.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=cfg.CONFIG.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=cfg.CONFIG.AUG.COLOR_JITTER,
            auto_augment=cfg.CONFIG.AUG.AUTO_AUGMENT,
            interpolation=cfg.CONFIG.DATA.INTERPOLATION,
            re_prob=cfg.CONFIG.AUG.REPROB,
            re_mode=cfg.CONFIG.AUG.REMODE,
            re_count=cfg.CONFIG.AUG.RECOUNT,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                cfg.CONFIG.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if cfg.CONFIG.DATA.TEST_NUM_CROP == 1:
        # center crop
        if resize_im:
            size = int((1.0 / cfg.CONFIG.DATA.CROP_PCT) * cfg.CONFIG.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(cfg.CONFIG.DATA.IMG_SIZE))
        t.append(transforms.ToTensor())

        if cfg.CONFIG.DATA.IMAGENET_DEFAULT_NORMALIZE:
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        else:
            t.append(transforms.Normalize(0.5, 0.5))
    elif cfg.CONFIG.DATA.TEST_NUM_CROP == 5:
        # five crop
        if resize_im:
            size = int((1.0 / cfg.CONFIG.DATA.CROP_PCT) * cfg.CONFIG.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.FiveCrop(cfg.CONFIG.DATA.IMG_SIZE))
            t.append(Stack2Tensor())
        else:
            t.append(transforms.ToTensor())

        if cfg.CONFIG.DATA.IMAGENET_DEFAULT_NORMALIZE:
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        else:
            t.append(transforms.Normalize(0.5, 0.5))
    elif cfg.CONFIG.DATA.TEST_NUM_CROP == 10:
        # ten crop
        if resize_im:
            size = int((1.0 / cfg.CONFIG.DATA.CROP_PCT) * cfg.CONFIG.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.TenCrop(cfg.CONFIG.DATA.IMG_SIZE))
            t.append(Stack2Tensor())
        else:
            t.append(transforms.ToTensor())

        if cfg.CONFIG.DATA.IMAGENET_DEFAULT_NORMALIZE:
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        else:
            t.append(transforms.Normalize(0.5, 0.5))
    else:
        raise RuntimeError("Evaluation only supports center crop, five crop or ten crop mode. \
            Please set TEST_NUM_CROP in configuration to 1, 5 or 10 accordingly.")

    return transforms.Compose(t)
