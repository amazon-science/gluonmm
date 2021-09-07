import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, attn_dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        temperature = np.power(k.shape[-1], 0.5)
        attn = torch.bmm(q, k.permute(0, 2, 1))
        attn = attn / temperature

        attn = self.softmax(attn)
        if mask is not None:
            attn = attn + mask
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class ScaledDotProductAttentionPoolVariableKSTD(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, attn_dropout=0.1, k=2):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.k = k

    def forward(self, q, k, v, mask=None):
        temperature = np.power(k.shape[-1], 0.5)
        attn = torch.bmm(q, k.permute(0, 2, 1))
        attn = attn / temperature

        attn = self.softmax(attn)
        if mask is not None:
            attn = attn + mask
        # attn = self.dropout(attn)

        std_token = torch.std(attn, dim=2)
        idx = torch.topk(std_token, std_token.shape[1] - self.k, dim=1)[1].sort(dim=1)[0]
        idx = idx.view(idx.shape[0], idx.shape[1], 1).repeat(1, 1, attn.shape[2])
        attn_pool = attn.gather(1, idx)
        # attn = torch.cat([attn[:, :1, :], attn_pool], dim=1)

        attn = self.softmax(attn_pool)
        output = torch.bmm(attn, v)

        return output, attn, idx


class ScaledDotProductAttentionPoolVariableKSTDDual(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, attn_dropout=0.1, k=2):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.k = k

    def forward(self, q, k, v, mask=None):
        temperature = np.power(k.shape[-1], 0.5)
        attn = torch.bmm(q, k.permute(0, 2, 1))
        attn = attn / temperature

        attn = self.softmax(attn)
        if mask is not None:
            attn = attn + mask
        # attn = self.dropout(attn)

        attn_cls = attn[:, :1, :]
        attn_t = attn[:, 1:, :]

        std_token = torch.std(attn_t, dim=2)
        idx = torch.topk(std_token, std_token.shape[1] - self.k, dim=1)[1].sort(dim=1)[0]
        idx = idx.view(idx.shape[0], idx.shape[1], 1).repeat(1, 1, attn_t.shape[2])
        attn_pool = attn_t.gather(1, idx)
        attn = torch.cat([attn_cls, attn_pool], dim=1)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn, idx


class _Linear(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(d_model, d_model))
        self.bias = nn.Parameter(torch.Tensor(d_model, ))

    def forward(self, x):
        return x @ self.weight.t() + self.bias


class MultiHeadSplitAttentionSpatioToken(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, head_num, d_model, dropout=0.1):
        super().__init__()

        self.head_num = head_num
        self.d_model = d_model
        self.in_proj_weight = nn.Parameter(torch.Tensor(d_model * 3, d_model))
        self.in_proj_bias = nn.Parameter(torch.Tensor(d_model * 3, ))

        self.attention_t = ScaledDotProductAttention(attn_dropout=dropout)
        self.attention_s = ScaledDotProductAttention(attn_dropout=dropout)

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.out_proj = _Linear(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, value, orig_shape, attn_mask=None, key_padding_mask=None):
        mask = attn_mask

        b, c, t, w, h = orig_shape

        seq_l, sz_b, c = q.shape

        qkv = q @ self.in_proj_weight.t() + self.in_proj_bias

        q = qkv[:, :, :c]
        k = qkv[:, :, c: 2 * c]
        v = qkv[:, :, 2 * c:]

        q_s = q.view(t, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 4, 0, 1).contiguous().view(
            self.head_num * b, c // self.head_num, t, w * h + 1)
        k_s = k.view(t, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 4, 0, 1).contiguous().view(
            self.head_num * b, c // self.head_num, t, w * h + 1)

        q_t = q.view(t, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t,
                                                                                                              self.d_model // self.head_num)
        k_t = k.view(t, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t,
                                                                                                              self.d_model // self.head_num)

        v_t = v.view(t, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t,
                                                                                                              self.d_model // self.head_num)

        output_t, attnx = self.attention_t(q_t, k_t, v_t, mask=mask)

        v_s = output_t.view(self.head_num, b, w * h + 1, t, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(self.head_num * sz_b * t, w * h + 1, -1)

        q_s = q_s.permute(0, 2, 3, 1).contiguous().view(-1, w * h + 1, c // self.head_num)
        k_s = k_s.permute(0, 2, 3, 1).contiguous().view(-1, w * h + 1, c // self.head_num)
        output_s, attn = self.attention_s(q_s, k_s, v_s, mask=mask)
        _, seq_l, _ = output_s.shape
        output = output_s.view(self.head_num, b, -1, w * h + 1, self.d_model // self.head_num).permute(2, 3, 1, 0,
                                                                                                   4).contiguous().view(
            -1, b, c)
        output = self.dropout(self.out_proj(output))

        return output, attn

class MultiHeadSplitAttentionDualToken(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, head_num, d_model, dropout=0.1):
        super().__init__()

        self.head_num = head_num
        self.d_model = d_model
        self.in_proj_weight = nn.Parameter(torch.Tensor(d_model * 3, d_model))
        self.in_proj_bias = nn.Parameter(torch.Tensor(d_model * 3, ))

        self.attention_t = ScaledDotProductAttention(attn_dropout=dropout)
        self.attention_s = ScaledDotProductAttention(attn_dropout=dropout)

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.out_proj = _Linear(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, value, orig_shape, attn_mask=None, key_padding_mask=None):
        mask = attn_mask

        b, c, t, w, h = orig_shape

        seq_l, sz_b, c = q.shape

        qkv = q @ self.in_proj_weight.t() + self.in_proj_bias

        q = qkv[:, :, :c]
        k = qkv[:, :, c: 2 * c]
        v = qkv[:, :, 2 * c:]

        q_s = q.view(t + 1, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 4, 0, 1).contiguous().view(
            self.head_num * b, c // self.head_num, t + 1, w * h + 1)
        k_s = k.view(t + 1, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 4, 0, 1).contiguous().view(
            self.head_num * b, c // self.head_num, t + 1, w * h + 1)

        q_t = q.view(t + 1, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t + 1,
                                                                                                              self.d_model // self.head_num)
        k_t = k.view(t + 1, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t + 1,
                                                                                                              self.d_model // self.head_num)

        v_t = v.view(t + 1, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t + 1,
                                                                                                              self.d_model // self.head_num)

        output_t, attnx = self.attention_t(q_t, k_t, v_t, mask=mask)

        v_s = output_t.view(self.head_num, b, w * h + 1, t + 1, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(self.head_num * sz_b * (t + 1), w * h + 1, -1)

        q_s = q_s.permute(0, 2, 3, 1).contiguous().view(-1, w * h + 1, c // self.head_num)
        k_s = k_s.permute(0, 2, 3, 1).contiguous().view(-1, w * h + 1, c // self.head_num)
        output_s, attn = self.attention_s(q_s, k_s, v_s, mask=mask)
        _, seq_l, _ = output_s.shape
        output = output_s.view(self.head_num, b, -1, w * h + 1, self.d_model // self.head_num).permute(2, 3, 1, 0,
                                                                                                   4).contiguous().view(
            -1, b, c)
        output = self.out_proj(output)

        return output, attn


class MultiHeadSequentialPoolAttentionVariableKSTDReverse(nn.Module):
    ''' Multi-Head Attention module '''

    ''' Multi-Head Attention module '''

    def __init__(self, head_num, d_model, dropout=0.1, pool=False, k=2):
        super().__init__()

        self.head_num = head_num
        self.d_model = d_model
        self.in_proj_weight = nn.Parameter(torch.Tensor(d_model * 3, d_model))
        self.in_proj_bias = nn.Parameter(torch.Tensor(d_model * 3, ))
        self.pool = pool
        self.k = k

        torch.nn.init.xavier_uniform(self.in_proj_weight)
        self.in_proj_bias.data.fill_(0.01)

        if pool:
            self.attention_t = ScaledDotProductAttentionPoolVariableKSTD(attn_dropout=dropout, k=k)
        else:
            self.attention_t = ScaledDotProductAttention(attn_dropout=dropout)
        self.attention_s = ScaledDotProductAttention(attn_dropout=dropout)

        self.out_proj = _Linear(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, value, orig_shape, attn_mask=None, key_padding_mask=None):
        mask = attn_mask

        b, c, t, w, h = orig_shape

        seq_l, sz_b, c = q.shape

        q_cls = q[:1, :, :]
        q = q[1:, :, :]

        qkv = q @ self.in_proj_weight.t() + self.in_proj_bias

        q = qkv[:, :, :c]
        k = qkv[:, :, c: 2 * c]
        v = qkv[:, :, 2 * c:]

        q_t = q.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t,
                                                                                                              self.d_model // self.head_num)
        k_t = k.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t,
                                                                                                              self.d_model // self.head_num)

        v_t = v.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t,
                                                                                                              self.d_model // self.head_num)

        if self.pool:
            output_t, attnx, idx = self.attention_t(q_t, k_t, v_t, mask=mask)
            v_s = output_t.view(self.head_num, b, w * h, t - self.k, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(
                self.head_num * sz_b * (t - self.k), w * h, -1)

            idx_ = idx[:, :, :1].repeat(1, 1, c // self.head_num)
            q_t = q_t.gather(1, idx_)
            k_t = k_t.gather(1, idx_)

            q_s = q_t.view(self.head_num, b, w * h, t - self.k, self.d_model // self.head_num).permute(0, 1, 3, 2,
                                                                                                       4).contiguous().view(
                self.head_num * sz_b * (t - self.k), w * h, -1)
            k_s = k_t.view(self.head_num, b, w * h, t - self.k, self.d_model // self.head_num).permute(0, 1, 3, 2,
                                                                                                       4).contiguous().view(
                self.head_num * sz_b * (t - self.k), w * h, -1)
        else:
            output_t, attnx = self.attention_t(q_t, k_t, v_t, mask=mask)
            idx = None
            v_s = output_t.view(self.head_num, b, w * h, t, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(self.head_num * sz_b * t, w * h, -1)

            q_s = q_t.view(self.head_num, b, w * h, t, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(
                self.head_num * sz_b * t, w * h, -1)
            k_s = k_t.view(self.head_num, b, w * h, t, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(
                self.head_num * sz_b * t, w * h, -1)

        output_s, attn = self.attention_s(q_s, k_s, v_s, mask=mask)
        _, seq_l, _ = output_s.shape
        output = output_s.view(self.head_num, b, -1, w * h, self.d_model // self.head_num).permute(2, 3, 1, 0, 4).contiguous().view(-1, b, c)

        output = self.out_proj(output)
        output = torch.cat((q_cls, output), dim=0)

        return output, attn, idx

class MultiHeadSequentialPoolAttentionVariableKSTDSTDual(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, head_num, d_model, dropout=0.1, pool=False, is_std=False, k=2):
        super().__init__()

        self.head_num = head_num
        self.d_model = d_model
        self.in_proj_weight = nn.Parameter(torch.Tensor(d_model * 3, d_model))
        self.in_proj_bias = nn.Parameter(torch.Tensor(d_model * 3, ))
        self.pool = pool
        self.k = k

        torch.nn.init.xavier_uniform(self.in_proj_weight)
        self.in_proj_bias.data.fill_(0.01)

        self.linear_temporal_weights = nn.Parameter(torch.Tensor(2 * d_model // self.head_num, d_model // self.head_num))
        self.linear_temporal_bias = nn.Parameter(torch.Tensor(2 * d_model // self.head_num, ))

        torch.nn.init.xavier_uniform(self.linear_temporal_weights)
        self.linear_temporal_bias.data.fill_(0.01)

        self.is_std = is_std

        if pool:
            self.attention_t = ScaledDotProductAttentionPoolVariableKSTDDual(attn_dropout=dropout, k=k)
        else:
            self.attention_t = ScaledDotProductAttention(attn_dropout=dropout)
        self.attention_s = ScaledDotProductAttention(attn_dropout=dropout)

        self.out_proj = _Linear(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, value, orig_shape, attn_mask=None, key_padding_mask=None):
        mask = attn_mask

        b, c, t, w, h = orig_shape

        seq_l, sz_b, c = q.shape

        qkv = q @ self.in_proj_weight.t() + self.in_proj_bias

        q = qkv[:, :, :c]
        k = qkv[:, :, c: 2 * c]
        v = qkv[:, :, 2 * c:]

        q_t = q.view(t + 1, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t + 1,
                                                                                                              self.d_model // self.head_num)
        k_t = k.view(t + 1, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t + 1,
                                                                                                              self.d_model // self.head_num)

        v_t = v.view(t + 1, w * h + 1, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t + 1,
                                                                                                              self.d_model // self.head_num)

        if self.pool:
            output_t, attnx, idx = self.attention_t(q_t, k_t, v_t, mask=mask)

            idx_ = idx[:, :, :1].repeat(1, 1, c // self.head_num)

            v_s = output_t.view(self.head_num, b, w * h + 1, t - self.k + 1, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(
                self.head_num * sz_b * (t - self.k + 1), w * h + 1, -1)

            q_cls = q_t[:, :1, :]
            q_t = q_t[:, 1:, :]
            q_t = q_t.gather(1, idx_)
            q_t = torch.cat((q_cls, q_t), dim=1)

            k_cls = k_t[:, :1, :]
            k_t = k_t[:, 1:, :]
            k_t = k_t.gather(1, idx_)
            k_t = torch.cat((k_cls, k_t), dim=1)

            q_s = q_t.view(self.head_num, b, w * h + 1, t + 1 - self.k, self.d_model // self.head_num).permute(0, 1, 3, 2,
                                                                                                       4).contiguous().view(
                self.head_num * sz_b * (t + 1 - self.k), w * h + 1, -1)
            k_s = k_t.view(self.head_num, b, w * h + 1, t + 1 - self.k, self.d_model // self.head_num).permute(0, 1, 3, 2,
                                                                                                       4).contiguous().view(
                self.head_num * sz_b * (t + 1 - self.k), w * h + 1, -1)
        else:
            output_t, attnx = self.attention_t(q_t, k_t, v_t, mask=mask)
            idx = None
            v_s = output_t.view(self.head_num, b, w * h + 1, t + 1, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(self.head_num * sz_b * (t + 1), w * h + 1, -1)

            q_s = q_t.view(self.head_num, b, w * h + 1, t + 1, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(
                self.head_num * sz_b * (t + 1), w * h + 1, -1)
            k_s = k_t.view(self.head_num, b, w * h + 1, t + 1, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(
                self.head_num * sz_b * (t + 1), w * h + 1, -1)

        output_s, attn = self.attention_s(q_s, k_s, v_s, mask=mask)
        _, seq_l, _ = output_s.shape
        output = output_s.view(self.head_num, b, -1, w * h + 1, self.d_model // self.head_num).permute(2, 3, 1, 0, 4).contiguous().view(-1, b, c)

        output = self.out_proj(output)

        return output, attn, idx


class MultiHeadTemperalPoolAttentionNoToken(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, head_num, d_model, dropout, pool=False):
        super().__init__()

        self.head_num = head_num
        self.d_model = d_model
        self.in_proj_weight = nn.Parameter(torch.Tensor(d_model * 3, d_model))
        self.in_proj_bias = nn.Parameter(torch.Tensor(d_model * 3, ))
        self.pool = pool

        if pool:
            self.attention_t = ScaledDotProductAttentionMaxPool(attn_dropout=dropout)
        else:
            self.attention_t = ScaledDotProductAttention(attn_dropout=dropout)
        self.attention_s = ScaledDotProductAttention(attn_dropout=dropout)

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.out_proj = _Linear(d_model)

    def forward(self, q, k, value, orig_shape, attn_mask=None, key_padding_mask=None):
        mask = attn_mask

        b, c, t, w, h = orig_shape

        seq_l, sz_b, c = q.shape

        q_cls = q[:1, :, :]
        q = q[1:, :, :]

        qkv = q @ self.in_proj_weight.t() + self.in_proj_bias

        q = qkv[:, :, :c]
        k = qkv[:, :, c: 2 * c]
        v = qkv[:, :, 2 * c:]

        q_s = q.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 4, 0, 1).contiguous().view(
            self.head_num * b, c // self.head_num, t, w * h)
        k_s = k.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 4, 0, 1).contiguous().view(
            self.head_num * b, c // self.head_num, t, w * h)

        q_t = q.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t,
                                                                                                              self.d_model // self.head_num)
        k_t = k.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t,
                                                                                                              self.d_model // self.head_num)

        v_t = v.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t,
                                                                                                              self.d_model // self.head_num)

        output_t, attnx = self.attention_t(q_t, k_t, v_t, mask=mask)
        if self.pool:
            v_s = output_t.view(self.head_num, b, w * h, t // 2, self.d_model // self.head_num).permute(0, 1, 3, 2,
                                                                                                        4).contiguous().view(
                self.head_num * sz_b * t // 2, w * h, -1)
            q_s = self.avg_pool(q_s)
            k_s = self.avg_pool(k_s)
        else:
            v_s = output_t.view(self.head_num, b, w * h, t, self.d_model // self.head_num).permute(0, 1, 3, 2,
                                                                                                   4).contiguous().view(
                self.head_num * sz_b * t, w * h, -1)

        q_s = q_s.permute(0, 2, 3, 1).contiguous().view(-1, w * h, c // self.head_num)
        k_s = k_s.permute(0, 2, 3, 1).contiguous().view(-1, w * h, c // self.head_num)
        output_s, attn = self.attention_s(q_s, k_s, v_s, mask=mask)
        _, seq_l, _ = output_s.shape
        output = output_s.view(self.head_num, b, -1, w * h, self.d_model // self.head_num).permute(2, 3, 1, 0,
                                                                                                   4).contiguous().view(
            -1, b, c)
        output = self.out_proj(output)
        output = torch.cat((q_cls, output), dim=0)

        return output, attn


class MultiHeadTemperalPoolAttentionNoTokenSpatioFirst(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, head_num, d_model, dropout, pool=False):
        super().__init__()

        self.head_num = head_num
        self.d_model = d_model
        self.in_proj_weight = nn.Parameter(torch.Tensor(d_model * 3, d_model))
        self.in_proj_bias = nn.Parameter(torch.Tensor(d_model * 3, ))
        self.pool = pool

        if pool:
            self.attention_t = ScaledDotProductAttentionMaxPool(attn_dropout=dropout)
        else:
            self.attention_t = ScaledDotProductAttention(attn_dropout=dropout)
        self.attention_s = ScaledDotProductAttention(attn_dropout=dropout)

        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.out_proj = _Linear(d_model)

    def forward(self, q, k, value, orig_shape, attn_mask=None, key_padding_mask=None):
        mask = attn_mask

        b, c, t, w, h = orig_shape

        seq_l, sz_b, c = q.shape

        q_cls = q[:1, :, :]
        q = q[1:, :, :]

        qkv = q @ self.in_proj_weight.t() + self.in_proj_bias

        q = qkv[:, :, :c]
        k = qkv[:, :, c: 2 * c]
        v = qkv[:, :, 2 * c:]

        q_s = q.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 0, 1, 4).contiguous().view(
            self.head_num * b * t, w * h, c // self.head_num)
        k_s = k.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 0, 1, 4).contiguous().view(
            self.head_num * b * t, w * h, c // self.head_num)

        v_s = v.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 0, 1, 4).contiguous().view(
            self.head_num * b * t, w * h, c // self.head_num)
        output_s, attn = self.attention_s(q_s, k_s, v_s, mask=mask)

        v_t = output_s.view(self.head_num, b, t, w * h, self.d_model // self.head_num).permute(0, 1, 3, 2, 4).contiguous().view(
                self.head_num * sz_b * w * h, t, -1)

        q_t = q.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t, self.d_model // self.head_num)
        k_t = k.view(t, w * h, b, self.head_num, c // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, t, self.d_model // self.head_num)

        output_t, attnx = self.attention_t(q_t, k_t, v_t, mask=mask)
        output = output_t.view(self.head_num, b, w * h, -1, self.d_model // self.head_num).permute(3, 2, 1, 0, 4).contiguous().view(-1, b, c)
        output = self.out_proj(output)
        output = torch.cat((q_cls, output), dim=0)

        return output, attn