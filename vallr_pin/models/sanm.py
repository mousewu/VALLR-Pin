"""SANM (Self-Attention Network with Memory) 模块。

论文 Stage-I 的编码器来自 Paraformer / SeACo-Paraformer 的 SANM 结构：在标准
self-attention 的输出上并联一个 FSMN memory block（对 V 做深度可分离一维卷积），
用局部记忆补足注意力对细粒度时序结构建模不足的问题。唇动信号的判别性信息高度
集中在相邻若干帧（协同发音），这个局部记忆分支正好对症。

字符解码器 (SanmDecoder) 把自注意力整体换成**因果** FSMN memory block，
只保留 cross-attention；这样解码器参数少、且天然支持流式/并行。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_pe(length: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
    return pe.to(dtype)


def lengths_to_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """(B,) -> (B, T) bool，True 表示有效帧。"""
    max_len = int(max_len or lengths.max().item())
    ar = torch.arange(max_len, device=lengths.device)
    return ar.unsqueeze(0) < lengths.unsqueeze(1)


class FsmnMemoryBlock(nn.Module):
    """深度可分离一维卷积记忆块，带残差。

    causal=True 时只看左侧上下文 (解码器用)，否则对称看左右 (编码器用)。
    """

    def __init__(self, dim: int, kernel_size: int = 11, causal: bool = False):
        super().__init__()
        if not causal and kernel_size % 2 == 0:
            kernel_size += 1
        self.causal = causal
        self.kernel_size = kernel_size
        self.left_pad = kernel_size - 1 if causal else kernel_size // 2
        self.right_pad = 0 if causal else kernel_size // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, D); mask: (B, T) bool
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        y = x.transpose(1, 2)
        y = F.pad(y, (self.left_pad, self.right_pad))
        y = self.conv(y).transpose(1, 2)
        out = x + y
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dk).transpose(1, 2)  # (B,H,T,dk)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        """mask: (B, 1, Tq, Tk) 或 (B, 1, 1, Tk) bool，True 可见。"""
        q, k, v = self._split(self.q(query)), self._split(self.k(key)), self._split(self.v(value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous()
        return self.o(ctx.view(ctx.size(0), ctx.size(1), -1))


class SanmSelfAttention(nn.Module):
    """self-attention 输出 + FSMN(V) 记忆分支。"""

    def __init__(self, dim: int, heads: int, kernel_size: int = 11, dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.memory = FsmnMemoryBlock(dim, kernel_size, causal=causal)

    def forward(self, x, attn_mask=None, pad_mask=None):
        return self.attn(x, x, x, attn_mask) + self.memory(x, pad_mask)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class SanmEncoderLayer(nn.Module):
    def __init__(self, dim, heads, ffn, kernel_size=11, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.self_attn = SanmSelfAttention(dim, heads, kernel_size, dropout)
        self.ffn = FeedForward(dim, ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, pad_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), attn_mask, pad_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SanmDecoderLayer(nn.Module):
    """因果 FSMN memory (代替自注意力) + cross-attention + FFN。"""

    def __init__(self, dim, heads, ffn, kernel_size=11, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim), nn.LayerNorm(dim),
                                              nn.LayerNorm(dim))
        self.memory = FsmnMemoryBlock(dim, kernel_size, causal=True)
        self.src_attn = MultiHeadAttention(dim, heads, dropout)
        self.ffn = FeedForward(dim, ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, memory, tgt_pad_mask=None, src_mask=None):
        y = y + self.dropout(self.memory(self.norm1(y), tgt_pad_mask))
        h = self.norm2(y)
        y = y + self.dropout(self.src_attn(h, memory, memory, src_mask))
        y = y + self.dropout(self.ffn(self.norm3(y)))
        return y


class TransformerDecoderLayer(nn.Module):
    """轻量标准 Transformer 解码层，用于拼音解码器。"""

    def __init__(self, dim, heads, ffn, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim), nn.LayerNorm(dim),
                                              nn.LayerNorm(dim))
        self.self_attn = MultiHeadAttention(dim, heads, dropout)
        self.src_attn = MultiHeadAttention(dim, heads, dropout)
        self.ffn = FeedForward(dim, ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, memory, tgt_mask=None, src_mask=None):
        h = self.norm1(y)
        y = y + self.dropout(self.self_attn(h, h, h, tgt_mask))
        h = self.norm2(y)
        y = y + self.dropout(self.src_attn(h, memory, memory, src_mask))
        y = y + self.dropout(self.ffn(self.norm3(y)))
        return y


def causal_mask(size: int, device=None) -> torch.Tensor:
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
