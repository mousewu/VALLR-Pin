"""中英混说时，英文词按**中国人实际读法**转成普通话音节。

为什么必须有这张表：VALLR-Pin 的中间表示是 410 个无声调普通话音节，而中文科技类
口语的英文混入率极高 —— 实测某 AI 访谈稿有 34% 的句子含英文词。直接丢掉这些句子
等于扔掉三分之一数据，而这些词的**口型本来就是可辨的**，扔掉纯属浪费。

设计取舍：

* 缩写按**字母音**拆 (RL → a-er + ai-er)，这是中文语境下的实际读法；
* 常见术语查表 (token → tou-ken)，表以外的词回退到字母音，
  并在 ``unknown`` 里记录，方便你按语料把表补全；
* 一个英文词在**字符序列里算 1 个 token**，在**拼音序列里展开成多个音节**。
  本仓库字符辅助 CTC 与拼音主 CTC 的目标长度独立，不需要一一对齐，
  所以这种"1 字符 ↔ N 音节"是天然支持的。

这张表是启发式的近似，不是发音词典。真要做严谨，应该让标注员按实际听感标注，
或者用能输出音素的 ASR。但作为把 34% 数据救回来的工程手段，够用。
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# 字母的中文读音（普通话使用者念英文字母的实际音节）
LETTER_PINYIN: Dict[str, List[str]] = {
    "a": ["ei"], "b": ["bi"], "c": ["xi"], "d": ["di"], "e": ["yi"],
    "f": ["ai", "fu"], "g": ["ji"], "h": ["ei", "chi"], "i": ["ai"],
    "j": ["jie"], "k": ["kei"], "l": ["ai", "er"], "m": ["ai", "mu"],
    "n": ["en"], "o": ["ou"], "p": ["pi"], "q": ["kiu"], "r": ["a", "er"],
    "s": ["ai", "si"], "t": ["ti"], "u": ["you"], "v": ["wei"],
    "w": ["da", "bu", "liu"], "x": ["ai", "ke", "si"], "y": ["wai"],
    "z": ["zei"],
}

# 高频技术词的实际口语读法。key 一律小写。
WORD_PINYIN: Dict[str, List[str]] = {
    "agent": ["ei", "zhen", "te"],
    "agentic": ["ei", "zhen", "ti", "ke"],
    "attention": ["a", "ten", "shen"],
    "benchmark": ["ben", "qi", "ma", "ke"],
    "chat": ["qia", "te"],
    "coding": ["kou", "ding"],
    "context": ["kang", "tai", "ke", "si", "te"],
    "copilot": ["ke", "pai", "luo", "te"],
    "encoder": ["yin", "kou", "de"],
    "decoder": ["di", "kou", "de"],
    "embedding": ["yin", "bei", "ding"],
    "epoch": ["yi", "pa", "ke"],
    "loss": ["luo", "si"],
    "model": ["mo", "dou"],
    "paper": ["pei", "po"],
    "prompt": ["pu", "lang", "pu", "te"],
    "scaling": ["si", "kei", "ling"],
    "token": ["tou", "ken"],
    "tokens": ["tou", "ken", "si"],
    "transformer": ["chuan", "si", "fu", "mo"],
    "open": ["ou", "pen"],
    "review": ["ri", "wei", "you"],
    "case": ["kei", "si"],
    "test": ["tai", "si", "te"],
    "training": ["chuei", "ning"],
    "reward": ["ri", "wo", "de"],
    "hack": ["hei", "ke"],
    "bet": ["bei", "te"],
    "know": ["nou"],
    "how": ["hao"],
    "search": ["se", "qi"],
    "logit": ["luo", "ji", "te"],
    "clipping": ["ke", "li", "ping"],
    "efficiency": ["yi", "fei", "shen", "xi"],
    "innovation": ["yin", "nuo", "wei", "shen"],
    "infinity": ["yin", "fei", "ni", "ti"],
    "beginning": ["bi", "gin", "ning"],
    "the": ["ze"],
    "of": ["ao", "fu"],
    "vat": ["wa", "te"],
    "brain": ["bu", "lei", "en"],
    "in": ["yin"],
    "pass": ["pa", "si"],
    "time": ["tai", "mu"],
    "system": ["xi", "si", "tan"],
    "engineering": ["en", "ji", "ni", "ling"],
    "optimizer": ["ao", "pu", "ti", "mai", "ze"],
    "adam": ["ya", "dang"],
    "muon": ["miu", "ang"],
    "dependency": ["di", "pen", "den", "xi"],
}

# 拆词：连续字母数字为一个 token
LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-']*|\d+")
DIGIT_PINYIN = {"0": "ling", "1": "yi", "2": "er", "3": "san", "4": "si",
                "5": "wu", "6": "liu", "7": "qi", "8": "ba", "9": "jiu"}


def _acronym_pinyin(token: str) -> List[str]:
    out: List[str] = []
    for ch in token.lower():
        if ch.isdigit():
            out.append(DIGIT_PINYIN[ch])
        elif ch in LETTER_PINYIN:
            out.extend(LETTER_PINYIN[ch])
    return out


def is_acronym(token: str) -> bool:
    """全大写且长度 <= 5 视为缩写 (AI / RL / LLM / SFT / GPT)；单字母不分大小写。"""
    letters = [c for c in token if c.isalpha()]
    if len(letters) == 1:
        return True
    return bool(letters) and all(c.isupper() for c in letters) and len(letters) <= 5


def latin_to_pinyin(token: str) -> Tuple[List[str], bool]:
    """返回 (音节列表, 是否命中词表)。未命中时回退字母音。"""
    if token.isdigit():
        return [DIGIT_PINYIN[c] for c in token], True
    key = token.lower().strip("-'")
    if is_acronym(token):
        return _acronym_pinyin(token), True
    if key in WORD_PINYIN:
        return list(WORD_PINYIN[key]), True
    return _acronym_pinyin(token), False


def coverage(tokens) -> Tuple[float, List[str]]:
    """统计词表覆盖率，返回 (命中率, 未命中词表)。用来指导补表。"""
    unknown, hit = [], 0
    tokens = list(tokens)
    for t in tokens:
        _, ok = latin_to_pinyin(t)
        if ok:
            hit += 1
        else:
            unknown.append(t)
    return (hit / max(len(tokens), 1)), sorted(set(unknown))


def find_latin_tokens(text: str) -> List[str]:
    return LATIN_TOKEN.findall(text)
