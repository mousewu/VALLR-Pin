"""编码器与双解码器。"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .sanm import (SanmDecoderLayer, SanmEncoderLayer, TransformerDecoderLayer,
                   causal_mask, sinusoidal_pe)


class SanmEncoder(nn.Module):
    def __init__(self, d_model=256, heads=4, ffn=1024, layers=12, kernel_size=11, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [SanmEncoderLayer(d_model, heads, ffn, kernel_size, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        x = x + sinusoidal_pe(x.size(1), x.size(2), x.device, x.dtype).unsqueeze(0)
        x = self.dropout(x)
        attn_mask = pad_mask[:, None, None, :] if pad_mask is not None else None
        for layer in self.layers:
            x = layer(x, attn_mask, pad_mask)
        return self.norm(x)


class _ARDecoderBase(nn.Module):
    """自回归解码器骨架：embedding + N 层 + 输出投影。"""

    def __init__(self, vocab_size, d_model, layers_module, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = layers_module
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def _embed(self, ys: torch.Tensor) -> torch.Tensor:
        x = self.embed(ys) * (self.d_model ** 0.5)
        return self.dropout(x + sinusoidal_pe(x.size(1), x.size(2), x.device, x.dtype))


class CharSanmDecoder(_ARDecoderBase):
    """字符解码器：SANM 解码层 (因果 FSMN + cross-attn)。"""

    def __init__(self, vocab_size, d_model=256, heads=4, ffn=1024, layers=6,
                 kernel_size=11, dropout=0.1):
        super().__init__(vocab_size, d_model,
                         nn.ModuleList([SanmDecoderLayer(d_model, heads, ffn, kernel_size,
                                                         dropout) for _ in range(layers)]),
                         dropout)

    def forward(self, ys_in, memory, tgt_pad_mask=None, src_pad_mask=None):
        x = self._embed(ys_in)
        src_mask = src_pad_mask[:, None, None, :] if src_pad_mask is not None else None
        for layer in self.layers:
            x = layer(x, memory, tgt_pad_mask, src_mask)
        return self.out(self.norm(x))


class PinyinTransformerDecoder(_ARDecoderBase):
    """轻量拼音解码器：标准 Transformer 解码层，层数少、宽度小。"""

    def __init__(self, vocab_size, d_model=256, heads=4, ffn=1024, layers=2, dropout=0.1):
        super().__init__(vocab_size, d_model,
                         nn.ModuleList([TransformerDecoderLayer(d_model, heads, ffn, dropout)
                                        for _ in range(layers)]), dropout)

    def forward(self, ys_in, memory, tgt_pad_mask=None, src_pad_mask=None):
        x = self._embed(ys_in)
        t = ys_in.size(1)
        tgt_mask = causal_mask(t, ys_in.device)[None, None]
        if tgt_pad_mask is not None:
            tgt_mask = tgt_mask & tgt_pad_mask[:, None, None, :]
        src_mask = src_pad_mask[:, None, None, :] if src_pad_mask is not None else None
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.out(self.norm(x))
