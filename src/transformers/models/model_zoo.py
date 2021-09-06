"""
GluonMM model zoo
"""
from .vit import *
from .deit import *
from .cait import *
from .swin import *
from .vidtr import *

__all__ = ['get_model', 'get_model_list']


_models = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'vit_base_patch32_224': vit_base_patch32_224,
    'vit_base_patch16_384': vit_base_patch16_384,
    'vit_base_patch32_384': vit_base_patch32_384,
    'vit_large_patch16_224': vit_large_patch16_224,
    'vit_large_patch32_224': vit_large_patch32_224,
    'vit_large_patch16_384': vit_large_patch16_384,
    'vit_large_patch32_384': vit_large_patch32_384,
    'vit_base_patch16_224_in21k': vit_base_patch16_224_in21k,
    'vit_base_patch32_224_in21k': vit_base_patch32_224_in21k,
    'vit_large_patch16_224_in21k': vit_large_patch16_224_in21k,
    'vit_large_patch32_224_in21k': vit_large_patch32_224_in21k,
    'vit_huge_patch14_224_in21k': vit_huge_patch14_224_in21k,
    'vit_deit_tiny_patch16_224': vit_deit_tiny_patch16_224,
    'vit_deit_small_patch16_224': vit_deit_small_patch16_224,
    'vit_deit_base_patch16_224': vit_deit_base_patch16_224,
    'vit_deit_base_patch16_384': vit_deit_base_patch16_384,
    'vit_deit_tiny_distilled_patch16_224': vit_deit_tiny_distilled_patch16_224,
    'vit_deit_small_distilled_patch16_224': vit_deit_small_distilled_patch16_224,
    'vit_deit_base_distilled_patch16_224': vit_deit_base_distilled_patch16_224,
    'vit_deit_base_distilled_patch16_384': vit_deit_base_distilled_patch16_384,
    'cait_xxs24_patch16_224': cait_xxs24_patch16_224,
    'cait_xxs24_patch16_384': cait_xxs24_patch16_384,
    'cait_xxs36_patch16_224': cait_xxs36_patch16_224,
    'cait_xxs36_patch16_384': cait_xxs36_patch16_384,
    'cait_xs24_patch16_384': cait_xs24_patch16_384,
    'cait_s24_patch16_224': cait_s24_patch16_224,
    'cait_s24_patch16_384': cait_s24_patch16_384,
    'cait_s36_patch16_384': cait_s36_patch16_384,
    'cait_m36_patch16_384': cait_m36_patch16_384,
    'cait_m48_patch16_448': cait_m48_patch16_448,
    'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_base_patch4_window12_384': swin_base_patch4_window12_384,
    'swin_base_patch4_window7_224_22kto1k': swin_base_patch4_window7_224_22kto1k,
    'swin_base_patch4_window12_384_22kto1k': swin_base_patch4_window12_384_22kto1k,
    'swin_large_patch4_window7_224_22kto1k': swin_large_patch4_window7_224_22kto1k,
    'swin_large_patch4_window12_384_22kto1k': swin_large_patch4_window12_384_22kto1k,
    'vidtr_s_8x8_patch16_224_k400': vidtr_s_8x8_patch16_224_k400,
    'vidtr_m_16x4_patch16_224_k400': vidtr_m_16x4_patch16_224_k400,
    'vidtr_l_32x2_patch16_224_k400': vidtr_l_32x2_patch16_224_k400,
    'cvidtr_s_8x8_patch16_224_k400': cvidtr_s_8x8_patch16_224_k400,
    'cvidtr_m_16x4_patch16_224_k400': cvidtr_m_16x4_patch16_224_k400,
}


def get_model(cfg):
    """Returns a pre-defined model by name
    Returns
    -------
    The model.
    """
    name = cfg.CONFIG.MODEL.NAME.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](cfg)
    return net


def get_model_list():
    """Get the entire list of model names in model_zoo.
    Returns
    -------
    list of str
        Entire list of model names in model_zoo.
    """
    return _models.keys()
