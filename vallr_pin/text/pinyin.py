"""Toneless Pinyin utilities.

Stage-I of VALLR-Pin预测*无声调*拼音音节 (toneless syllables)。论文中该建模单元表
共 397 项；实际数量取决于语料覆盖，本模块支持两种构建方式：

1. ``canonical_syllables()``  —— 遍历 CJK 基本区 (U+4E00..U+9FA5) 由 pypinyin 归纳出的
   全量音节表 (410 项，普通话标准音节的超集)。
2. ``syllables_from_corpus()`` —— 只保留训练语料中真实出现过的音节 (论文的 397 即属此类)。

无声调化的动机：声调在唇形上几乎不可见 (声带振动、基频变化不产生可辨的视觉特征)，
强行建模会引入大量不可约的标注噪声；去掉声调后建模空间从 ~1300 降到 ~400，
同时保留了"声母+韵母"这一层对唇形真正敏感的结构。
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

from pypinyin import Style, lazy_pinyin

# 覆盖 GB2312 / 常用汉字的码位区间
CJK_START, CJK_END = 0x4E00, 0x9FA5

UNK_SYLLABLE = "<unk_py>"


def is_cjk(ch: str) -> bool:
    """单个字符是否是汉字。传入多字符串一律返回 False（英文词 token 会走这里）。"""
    if len(ch) != 1:
        return False
    return CJK_START <= ord(ch) <= CJK_END or 0x3400 <= ord(ch) <= 0x4DBF


@lru_cache(maxsize=1)
def canonical_syllables() -> Tuple[str, ...]:
    """由 pypinyin 反推出的全量无声调音节表 (确定性、可复现)。"""
    seen = set()
    for cp in range(CJK_START, CJK_END + 1):
        ch = chr(cp)
        syl = lazy_pinyin(ch, style=Style.NORMAL)
        if not syl:
            continue
        s = syl[0]
        if s and s.isalpha() and s != ch:
            seen.add(s)
    return tuple(sorted(seen))


def clean_text(text: str, keep_non_cjk: bool = False) -> str:
    """去掉空格/标点/英文，只保留可以对齐到拼音的汉字。"""
    if keep_non_cjk:
        return "".join(text.split())
    return "".join(ch for ch in text if is_cjk(ch))


def text_to_pinyin(text: str) -> Tuple[List[str], List[str]]:
    """把一句话转成 (汉字列表, 无声调音节列表)，两者一一对齐、等长。

    使用整句输入以便 pypinyin 用词组消歧多音字 ("行" 在 "银行"/"行走" 中不同)，
    这一点对拼音监督信号的质量影响很大，不能逐字转换。
    """
    chars = list(clean_text(text))
    if not chars:
        return [], []
    syls = lazy_pinyin("".join(chars), style=Style.NORMAL,
                       errors=lambda x: [UNK_SYLLABLE] * len(x))
    if len(syls) != len(chars):
        # 极少数情况下 pypinyin 会把某些字拆/并，退化到逐字转换保证对齐
        syls = []
        for ch in chars:
            p = lazy_pinyin(ch, style=Style.NORMAL,
                            errors=lambda x: [UNK_SYLLABLE] * len(x))
            syls.append(p[0] if p else UNK_SYLLABLE)
    return chars, syls


_MIXED_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-']*|\d+|[一-龥]")


def text_to_pinyin_mixed(text: str) -> Tuple[List[str], List[str], List[str]]:
    """中英混说版本：返回 (字符/词 token 列表, 音节列表, 未识别的英文词)。

    与 :func:`text_to_pinyin` 的关键区别是**两条序列不再等长**：一个英文词在字符侧
    是 1 个 token，在音节侧展开成多个音节 (``token`` → ``tou ken``)。
    字符辅助 CTC 与拼音主 CTC 的目标长度彼此独立，所以这是合法的。

    没有这一步，中文科技类口语里三分之一的句子会因为夹带英文而整句丢弃。
    """
    from .loanword import latin_to_pinyin

    tokens: List[str] = []
    unknown: List[str] = []
    for m in _MIXED_TOKEN.finditer(text):
        tok = m.group(0)
        if is_cjk(tok):
            tokens.append(tok)
            continue
        tokens.append(tok)
        if not latin_to_pinyin(tok)[1]:
            unknown.append(tok)
    # 汉字部分整句转换以保留词组消歧，再把英文占位符按序填回
    cjk_run = "".join(t for t in tokens if is_cjk(t))
    cjk_syls = lazy_pinyin(cjk_run, style=Style.NORMAL,
                           errors=lambda x: [UNK_SYLLABLE] * len(x)) if cjk_run else []
    if len(cjk_syls) != len(cjk_run):
        cjk_syls = [char_syllable(c) for c in cjk_run]

    out_syls: List[str] = []
    ci = 0
    for tok in tokens:
        if is_cjk(tok):
            out_syls.append(cjk_syls[ci] if ci < len(cjk_syls) else UNK_SYLLABLE)
            ci += 1
        else:
            py, _ = latin_to_pinyin(tok)
            out_syls.extend(py or [UNK_SYLLABLE])
    return tokens, out_syls, unknown


def syllables_from_corpus(texts: Iterable[str]) -> List[str]:
    seen = set()
    for t in texts:
        _, syls = text_to_pinyin(t)
        seen.update(s for s in syls if s != UNK_SYLLABLE)
    return sorted(seen)


@lru_cache(maxsize=1)
def homophone_table() -> dict:
    """音节 -> 该音节下所有汉字 (按码位序)。供无 LLM 的受限重打分器使用。"""
    table: dict = {}
    for cp in range(CJK_START, CJK_END + 1):
        ch = chr(cp)
        syl = lazy_pinyin(ch, style=Style.NORMAL)
        if syl and syl[0].isalpha() and syl[0] != ch:
            table.setdefault(syl[0], []).append(ch)
    return table


def char_syllable(ch: str) -> str:
    if not is_cjk(ch):
        return UNK_SYLLABLE
    p = lazy_pinyin(ch, style=Style.NORMAL)
    return p[0] if p and p[0].isalpha() else UNK_SYLLABLE


def dump_syllables(path: str, syllables: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(syllables), f, ensure_ascii=False, indent=1)


def load_syllables(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
