""" Data-efficient Image Transformers (DeiT) in PyTorch
DeiT: Data-efficient Image Transformers
https://arxiv.org/abs/2012.12877

Code is mostly borrowed from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Copyright (c) 2020 Ross Wightman
Apache License, Version 2.0

Code is also adapted or inspired from
(1) https://github.com/facebookresearch/deit
Copyright (c) 2015-present, Facebook, Inc.
Apache License, Version 2.0
(2) https://github.com/google-research/vision_transformer
Copyright (c) 2020 Google LLC
Apache License, Version 2.0
(3) https://github.com/lucidrains/vit-pytorch
Copyright (c) 2020 Phil Wang
MIT License
"""
from torch.hub import load_state_dict_from_url

from src.transformers.models.vit.vision_transformer import VisionTransformer


__all__ = ['vit_deit_tiny_patch16_224', 'vit_deit_small_patch16_224',
           'vit_deit_base_patch16_224', 'vit_deit_base_patch16_384',
           'vit_deit_tiny_distilled_patch16_224', 'vit_deit_small_distilled_patch16_224',
           'vit_deit_base_distilled_patch16_224', 'vit_deit_base_distilled_patch16_384']


def vit_deit_tiny_patch16_224(cfg):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = VisionTransformer(patch_size=cfg.CONFIG.MODEL.PATCH_SIZE,
                              embed_dim=192,
                              depth=12,
                              num_heads=3,
                              mlp_ratio=4,
                              drop_path_rate=cfg.CONFIG.MODEL.DROP_PATH_RATE)
    if cfg.CONFIG.MODEL.PRETRAINED:
        pretrained_url = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
        state_dict = load_state_dict_from_url(pretrained_url, progress=True, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        print("Pre-trained model is loaded successfully.")
    return model


def vit_deit_small_patch16_224(cfg):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = VisionTransformer(patch_size=cfg.CONFIG.MODEL.PATCH_SIZE,
                              embed_dim=384,
                              depth=12,
                              num_heads=6,
                              mlp_ratio=4,
                              drop_path_rate=cfg.CONFIG.MODEL.DROP_PATH_RATE)
    if cfg.CONFIG.MODEL.PRETRAINED:
        pretrained_url = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
        state_dict = load_state_dict_from_url(pretrained_url, progress=True, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        print("Pre-trained model is loaded successfully.")
    return model


def vit_deit_base_patch16_224(cfg):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = VisionTransformer(patch_size=cfg.CONFIG.MODEL.PATCH_SIZE,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              drop_path_rate=cfg.CONFIG.MODEL.DROP_PATH_RATE)
    if cfg.CONFIG.MODEL.PRETRAINED:
        pretrained_url = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
        state_dict = load_state_dict_from_url(pretrained_url, progress=True, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        print("Pre-trained model is loaded successfully.")
    return model


def vit_deit_base_patch16_384(cfg):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = VisionTransformer(img_size=cfg.CONFIG.DATA.IMG_SIZE,
                              patch_size=cfg.CONFIG.MODEL.PATCH_SIZE,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              drop_path_rate=cfg.CONFIG.MODEL.DROP_PATH_RATE)
    if cfg.CONFIG.MODEL.PRETRAINED:
        pretrained_url = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'
        state_dict = load_state_dict_from_url(pretrained_url, progress=True, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        print("Pre-trained model is loaded successfully.")
    return model


def vit_deit_tiny_distilled_patch16_224(cfg):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = VisionTransformer(patch_size=cfg.CONFIG.MODEL.PATCH_SIZE,
                              embed_dim=192,
                              depth=12,
                              num_heads=3,
                              mlp_ratio=4,
                              distilled=True,
                              drop_path_rate=cfg.CONFIG.MODEL.DROP_PATH_RATE)
    if cfg.CONFIG.MODEL.PRETRAINED:
        pretrained_url = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'
        state_dict = load_state_dict_from_url(pretrained_url, progress=True, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        print("Pre-trained model is loaded successfully.")
    return model


def vit_deit_small_distilled_patch16_224(cfg):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = VisionTransformer(patch_size=cfg.CONFIG.MODEL.PATCH_SIZE,
                              embed_dim=384,
                              depth=12,
                              num_heads=6,
                              mlp_ratio=4,
                              distilled=True,
                              drop_path_rate=cfg.CONFIG.MODEL.DROP_PATH_RATE)
    if cfg.CONFIG.MODEL.PRETRAINED:
        pretrained_url = 'https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'
        state_dict = load_state_dict_from_url(pretrained_url, progress=True, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        print("Pre-trained model is loaded successfully.")
    return model


def vit_deit_base_distilled_patch16_224(cfg):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = VisionTransformer(patch_size=cfg.CONFIG.MODEL.PATCH_SIZE,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              distilled=True,
                              drop_path_rate=cfg.CONFIG.MODEL.DROP_PATH_RATE)
    if cfg.CONFIG.MODEL.PRETRAINED:
        pretrained_url = 'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth'
        state_dict = load_state_dict_from_url(pretrained_url, progress=True, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        print("Pre-trained model is loaded successfully.")
    return model


def vit_deit_base_distilled_patch16_384(cfg):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model = VisionTransformer(img_size=cfg.CONFIG.DATA.IMG_SIZE,
                              patch_size=cfg.CONFIG.MODEL.PATCH_SIZE,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              distilled=True,
                              drop_path_rate=cfg.CONFIG.MODEL.DROP_PATH_RATE)
    if cfg.CONFIG.MODEL.PRETRAINED:
        pretrained_url = 'https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth'
        state_dict = load_state_dict_from_url(pretrained_url, progress=True, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        print("Pre-trained model is loaded successfully.")
    return model
