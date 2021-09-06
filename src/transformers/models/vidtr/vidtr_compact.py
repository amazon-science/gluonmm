""" Compact VidTr in PyTorch
VidTr: Video Transformer Without Convolutions
https://arxiv.org/abs/2104.11746
"""
import os
from typing import Optional, List

import torch
from torch import nn, Tensor
import numpy as np
from src.transformers.utils.utils import _get_activation_fn
from src.transformers.models.vidtr.multihead_attention import MultiHeadSequentialPoolAttentionVariableKSTDReverse


__all__ = ['CompactVidTr', 'cvidtr_s_8x8_patch16_224_k400', 'cvidtr_m_16x4_patch16_224_k400']


class CompactVidTr(nn.Module):
    r""" Compact VidTR.

    Args:
        d_model (int): Number of channels in hidden layers.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers
        dim_feedforward (int): Number of channels in MLP
        dropout (float): Attention dropout rate. Default: 0.1
        activatioon (str): Activation function
        normalize_before (bool): Layer norm before MLP
        patch_size (tuple): patch size
        in_channel (int): Number of input channels
        activity_num (int): Class number
        temporal_size (int): Number of frames in temporal
        layer_pool (list): Downsample after layers
        number_of_keys (list): Number of frames downsampled
    """

    def __init__(self, d_model=512, nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1, activation="relu",
                 normalize_before=False, patch_size=(1, 16, 16),
                 in_channel=3, activity_num=157, temporal_size=16,
                 layer_pool=[1, 3, 5],
                 number_of_keys=[2, 2, 2]):

        super().__init__()

        self.temporal_size = temporal_size

        self.conv_stem = nn.Conv3d(in_channels=in_channel, out_channels=d_model, kernel_size=patch_size,
                                   stride=patch_size, bias=True)

        layer_list = []
        for i in range(num_encoder_layers):
            if i in layer_pool:
                is_pool = True
                k = number_of_keys[layer_pool.index(i)]
            else:
                is_pool = False
                k = 0
            module_temp = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                  dropout, activation, normalize_before,
                                                  layer_index=i, pool=is_pool, k=k,
                                                  layer_pool=layer_pool,
                                                  number_of_keys=number_of_keys)

            layer_list.append(module_temp)
        encoder_layers = nn.ModuleList(layer_list)

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_features=d_model, out_features=activity_num, bias=True)

        self.cls = nn.Parameter(torch.Tensor(1, 1, d_model))

        self.pos_embedding_google = np.zeros((1, 14 * 14 + 1, d_model)).astype(np.float32)

        self.pos_embedding = nn.Parameter(torch.Tensor(1, self.temporal_size * 14 * 14 + 1, self.d_model))

        self.dropout = nn.Dropout(0.5)
        self.dp_pos = nn.Dropout(0.1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_pos_embedding(self):
        pos_embedding_google = np.repeat(self.pos_embedding_google[:, 1:, :], self.temporal_size, axis=0)
        temp_pos = np.reshape(np.array(list(range(self.temporal_size + 1))[1:]) / (self.temporal_size + 1),
                              (self.temporal_size, 1, 1))
        pos_embedding_google = pos_embedding_google * temp_pos
        pos_embedding_google = pos_embedding_google.reshape((1, -1, self.d_model))
        pos_embedding_google = np.concatenate((self.pos_embedding_google[:, :1, :], pos_embedding_google), axis=1)
        self.pos_embedding.copy_(torch.from_numpy(pos_embedding_google))

    def forward(self, src):
        # flatten NxCxHxW to HWxNxC
        src = self.conv_stem(src)
        bs, c, t, h, w = src.shape

        pos_embed = self.pos_embedding.permute((1, 0, 2)).repeat(1, bs, 1)

        src = src.flatten(2).permute(2, 0, 1)
        cls = self.cls.repeat(1, bs, 1)

        src = torch.cat((cls, src), dim=0)

        memory = self.encoder(src, (bs, c, t, h, w), src_key_padding_mask=None, pos=pos_embed)

        out = memory[0, :, :]
        out = self.dropout(out.view(bs, -1))
        out = self.fc(out)

        return out


class TransformerEncoder(nn.Module):

    r""" Transformer Encoder of Compact VidTR.

    Args:
        layers (int): Transformer layer
        num_layers (int): Number of transformer layers
        norm (LayerNorm): Layer Norm
    """

    def __init__(self, encoder_layers, num_layers, norm=None):
        super().__init__()
        self.layers = encoder_layers
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                orig_shape,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output, pos, orig_shape = layer(output, orig_shape, src_mask=mask,
                                            src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    r""" Transformer block of Compact VidTR.

    Args:
        d_model (int): Number of channels in hidden layers.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Number of channels in MLP
        activatioon (str): Activation function
        normalize_before (bool): Layer norm before MLP
        dropout (float): Attention dropout rate.
        layer_index (int): Layer index of transformer layer.
        pool (bool): Downsample in this layer.
        layer_pool (list): Downsample after layers.
        number_of_keys (list): Number of frames downsampled.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, layer_index=0,
                 pool=False, k=0, layer_pool=[1, 3, 5], number_of_keys=[2, 2, 2]):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout_rate = dropout
        self.layer_index = layer_index
        self.layer_pool = layer_pool
        self.number_of_keys = number_of_keys

        if self.layer_index <= layer_pool[-1]:
            self.self_attn = MultiHeadSequentialPoolAttentionVariableKSTDReverse(nhead, d_model, dropout=dropout, pool=pool, k=k)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=d_model, vdim=d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.pool = pool

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src, orig_shape,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        b, c, t, w, h = orig_shape
        src2 = self.norm1(src)
        if self.layer_index == 0:
            q = k = v = self.with_pos_embed(src2, pos)
        else:
            q = k = v = src2

        if self.layer_index <= self.layer_pool[-1]:
            src_attn = self.self_attn(q, k, value=v, orig_shape=orig_shape, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)
        else:
            src_attn = self.self_attn(q, k, value=v, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)

        src2 = src_attn[0]

        if self.layer_index in self.layer_pool:
            idx = src_attn[2]
            idx = idx[:, :, :1].repeat(1, 1, c // self.nhead)

            b, c, t, w, h = orig_shape

            src_cls = src[:1, :, :]
            src_twh = src[1:, :, :]

            # by index
            src_twh = src_twh.view(t, w * h, b, self.nhead, c // self.nhead).permute(3, 2, 1, 0,
                                                                                     4).contiguous().view(
                self.nhead * b * (w * h), t, c // self.nhead)
            src_twh = src_twh.gather(1, idx)
            src_twh = src_twh.view(self.nhead, b, w * h, t - self.number_of_keys[self.layer_pool.index(self.layer_index)],
                                   c // self.nhead).permute(3, 2, 1, 0, 4).contiguous().view(-1, b, c)

            src = torch.cat((src_cls, src_twh), dim=0)
            orig_shape = (b, c, t - self.number_of_keys[self.layer_pool.index(self.layer_index)], w, h)

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, pos, orig_shape

    def forward(self, src, orig_shape,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, orig_shape, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, orig_shape, src_mask, src_key_padding_mask, pos)


def inflate_model(model, weights_dir, temporal_size):

    r""" Inflate ViT.

    Args:
        model (CompactVidTr): model to be inflated.
        weights_dir (str): inflate model dir.
        temporal_size (int): Number of frames in temporal.
    """

    checkpoint = torch.load(weights_dir, map_location='cpu')
    model_dict = model.state_dict()

    pretrained_dict = {}
    for k, v in checkpoint.items():
        if 'pos_embedding' in k:
            pretrained_dict.update(
                {"pos_embedding": torch.cat((v[:, :1, :], v[:, 1:, :].repeat(1, temporal_size, 1)), dim=1)})
        elif 'cls' in k:
            pretrained_dict.update({"cls": v})
        elif 'fc' not in k:
            pretrained_dict.update({k: v})

    pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict_)
    model.load_state_dict(model_dict)
    print("Inflate model success")


def build_compact_vidtr(cfg):
    model = CompactVidTr(d_model=cfg.CONFIG.MODEL.D_MODEL,
                         nhead=cfg.CONFIG.MODEL.NHEAD,
                         num_encoder_layers=cfg.CONFIG.MODEL.NUM_ENCODER_LAYERS,
                         dim_feedforward=cfg.CONFIG.MODEL.DIM_FEEDFORWARD,
                         patch_size=cfg.CONFIG.MODEL.PATCH_SIZE,
                         normalize_before=cfg.CONFIG.MODEL.NORMALIZE_BEFORE,
                         activity_num=cfg.CONFIG.DATA.NUM_CLASSES,
                         dropout=cfg.CONFIG.MODEL.DROPOUT,
                         temporal_size=cfg.CONFIG.MODEL.TEMP_LEN,
                         layer_pool=cfg.CONFIG.MODEL.LAYER_POOL,
                         number_of_keys=cfg.CONFIG.MODEL.NUMBER_OF_KEYS)
    if cfg.CONFIG.MODEL.INFLATE:
        if not os.path.exists(cfg.CONFIG.MODEL.INFLATE_PRETRAIN_DIR):
            raise RuntimeError('Pretrained weights are not found. Please download pretrained ViT weight from\
                               https://gluonmm.s3.amazonaws.com/pretrained/vit_inflate.pth to ./pretrained/vit_inflate.pth')
        inflate_model(model,
                      weights_dir=cfg.CONFIG.MODEL.INFLATE_PRETRAIN_DIR,
                      temporal_size=cfg.CONFIG.MODEL.TEMP_LEN,
                      merge_later=cfg.CONFIG.MODEL.MERGE_LATER)

    return model


def cvidtr_s_8x8_patch16_224_k400(cfg):
    model = build_compact_vidtr(cfg)
    return model


def cvidtr_m_16x4_patch16_224_k400(cfg):
    model = build_compact_vidtr(cfg)
    return model
