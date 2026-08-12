"""字符 / 拼音双词表。

文字 CTC 头和拼音 CTC 头共享同一套特殊符号布局：

    id 0 : <blank>   CTC blank
    id 1 : <sos/eos> 保留给旧检查点/文本侧接口；Stage-I v2 不使用
    id 2 : <unk>
    id 3+: 真实建模单元
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .pinyin import (canonical_syllables, syllables_from_corpus,
                     text_to_pinyin_mixed)

BLANK, SOS_EOS, UNK = 0, 1, 2
SPECIALS = ["<blank>", "<sos/eos>", "<unk>"]


@dataclass
class Vocab:
    units: List[str]

    def __post_init__(self) -> None:
        self.unit2id: Dict[str, int] = {u: i for i, u in enumerate(self.units)}

    def __len__(self) -> int:
        return len(self.units)

    @property
    def blank_id(self) -> int:
        return BLANK

    @property
    def sos_id(self) -> int:
        return SOS_EOS

    @property
    def eos_id(self) -> int:
        return SOS_EOS

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.unit2id.get(t, UNK) for t in tokens]

    def decode(self, ids: Iterable[int], strip_specials: bool = True) -> List[str]:
        out = []
        for i in ids:
            if i < 0 or i >= len(self.units):
                continue
            if strip_specials and i in (BLANK, SOS_EOS):
                continue
            out.append(self.units[i])
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.units, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def build(cls, units: Iterable[str]) -> "Vocab":
        uniq, seen = [], set()
        for u in units:
            if u not in seen and u not in SPECIALS:
                seen.add(u)
                uniq.append(u)
        return cls(SPECIALS + uniq)


class DualTokenizer:
    """把一句中文同时切成字符序列和无声调拼音序列。"""

    def __init__(self, char_vocab: Vocab, pinyin_vocab: Vocab):
        self.char = char_vocab
        self.pinyin = pinyin_vocab

    # ---------------------------------------------------------------- build
    @classmethod
    def build_from_texts(cls, texts: Iterable[str], full_syllable_table: bool = True
                         ) -> "DualTokenizer":
        texts = list(texts)
        chars: List[str] = []  # Chinese characters plus whole Latin/number tokens
        for t in texts:
            tokens, _, _ = text_to_pinyin_mixed(t)
            chars.extend(tokens)
        char_vocab = Vocab.build(sorted(set(chars)))
        syls = list(canonical_syllables()) if full_syllable_table else syllables_from_corpus(texts)
        pinyin_vocab = Vocab.build(syls)
        return cls(char_vocab, pinyin_vocab)

    # ---------------------------------------------------------------- codec
    def encode(self, text: str):
        tokens, syls, _ = text_to_pinyin_mixed(text)
        return self.char.encode(tokens), self.pinyin.encode(syls)

    def decode_chars(self, ids: Iterable[int]) -> str:
        return "".join(self.char.decode(ids))

    def decode_pinyin(self, ids: Iterable[int]) -> List[str]:
        return self.pinyin.decode(ids)

    # ------------------------------------------------------------------ io
    def save(self, d: str) -> None:
        os.makedirs(d, exist_ok=True)
        self.char.save(os.path.join(d, "char_vocab.json"))
        self.pinyin.save(os.path.join(d, "pinyin_vocab.json"))

    @classmethod
    def load(cls, d: str) -> "DualTokenizer":
        return cls(Vocab.load(os.path.join(d, "char_vocab.json")),
                   Vocab.load(os.path.join(d, "pinyin_vocab.json")))
