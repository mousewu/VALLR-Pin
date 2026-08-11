#!/usr/bin/env python3
"""文字稿可用率体检：一份人工整理的访谈稿，到底能榨出多少条 VSR 训练标签？

VSR 标签的硬性要求比 ASR 苛刻得多：

* 必须是**逐字口语**，不能是编辑过的书面语（改写会让口型与文本对不上）
* 对本仓库这套**拼音中介**方案，句子必须能整句转成普通话音节 —— 夹带的英文词
  （token efficiency、Agentic、Pass@k…）没有拼音，是这套方案的硬伤
* 长度要落在合理区间：太短没有上下文可用，太长则单条视频过长、显存吃不消

本脚本按上述条件逐条过滤，报告每一关卡掉多少，最后给出可用率。

用法::

    python scripts/transcript_stats.py transcript.txt --speaker 杨植麟
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CJK = re.compile(r"[一-龥]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9]")
# 编辑注释：整章标题、星号脚注、整段括注
EDITORIAL = re.compile(r"^\s*(\*|第[一二三四五六七八九十]+章|\d{2}\s|（.*）\s*$)")
SENT_SPLIT = re.compile(r"[。！？；\n]|——|\.{3,}|…+")
# 行内括注 (中文注释/英文原文) 属于编辑添加，口语里没有
PARENS = re.compile(r"[（(][^）)]*[）)]")


def parse_turns(path: str) -> List[Tuple[str, str]]:
    turns = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or EDITORIAL.match(line):
            continue
        m = re.match(r"^([^：:]{2,10})[：:](.+)$", line)
        if m:
            turns.append((m.group(1), m.group(2)))
    return turns


def split_sentences(text: str) -> List[str]:
    text = PARENS.sub("", text)
    text = text.replace("《", "").replace("》", "").replace("“", "").replace("”", "")
    return [s.strip(" ，、,-—") for s in SENT_SPLIT.split(text) if s.strip()]


def analyse(sentences: List[str], min_len: int, max_len: int,
            loanwords: bool = False) -> Dict:
    """loanwords=True 时，含英文的句子不再直接丢弃，而是走外来词发音表转成音节。"""
    stats = Counter()
    kept, dropped_latin, dropped_len, unknown_words = [], [], [], []
    for s in sentences:
        stats["total"] += 1
        n_cjk = len(CJK.findall(s))
        has_latin = bool(LATIN.search(s))
        has_digit = bool(DIGIT.search(s))
        if has_latin:
            stats["with_latin"] += 1
        if has_digit:
            stats["with_digit"] += 1
        if has_latin or has_digit:
            if not loanwords:
                dropped_latin.append(s)
                continue
            from vallr_pin.text.pinyin import text_to_pinyin_mixed
            toks, syls, unk = text_to_pinyin_mixed(s)
            unknown_words.extend(unk)
            if unk:                       # 有词表覆盖不到的英文词，保守起见丢弃
                stats["latin_uncovered"] += 1
                dropped_latin.append(s)
                continue
            stats["latin_rescued"] += 1
            n_cjk = len(toks)
        if not (min_len <= n_cjk <= max_len):
            dropped_len.append(s)
            stats["bad_len"] += 1
            continue
        kept.append(s)
    stats["kept"] = len(kept)
    return {"stats": stats, "kept": kept, "dropped_latin": dropped_latin,
            "dropped_len": dropped_len, "unknown_words": unknown_words}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--speaker", default="", help="只统计某位说话人")
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=30)   # CMLR 的上限就是 29 字
    ap.add_argument("--loanwords", action="store_true",
                    help="用外来词发音表救回含英文的句子，而不是直接丢弃")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    turns = parse_turns(args.transcript)
    speakers = Counter(s for s, _ in turns)
    print(f"说话人分布: {dict(speakers)}  总轮次={len(turns)}")

    for spk in ([args.speaker] if args.speaker else list(speakers)):
        text_turns = [t for s, t in turns if s == spk]
        sents = [s for t in text_turns for s in split_sentences(t)]
        r = analyse(sents, args.min_len, args.max_len, args.loanwords)
        st = r["stats"]
        total = max(st["total"], 1)
        print(f"\n=== {spk} ===")
        print(f"  轮次 {len(text_turns)} -> 切句 {st['total']}")
        print(f"  含英文/数字      : {st['with_latin'] + 0:>4} 句 "
              f"({100 * (st['with_latin'] + st['with_digit'] - 0) / total:.1f}% 含英文, "
              f"{100 * st['with_digit'] / total:.1f}% 含数字)")
        print(f"  长度不合格({args.min_len}-{args.max_len}字): {st['bad_len']:>4} 句")
        if args.loanwords:
            print(f"  英文句救回      : {st['latin_rescued']:>4} 句 "
                  f"(词表未覆盖而仍丢弃 {st['latin_uncovered']} 句)")
        print(f"  **可用**         : {st['kept']:>4} 句 ({100 * st['kept'] / total:.1f}%)")
        if r["kept"]:
            lens = [len(CJK.findall(s)) for s in r["kept"]]
            print(f"  可用句长度: 均值 {sum(lens) / len(lens):.1f} 字, "
                  f"中位 {sorted(lens)[len(lens) // 2]} 字")
            print("  可用样例:", " / ".join(r["kept"][: args.show]))
        if r.get("unknown_words"):
            from collections import Counter as C
            top = C(r["unknown_words"]).most_common(12)
            print("  词表待补(按频次):", " ".join(f"{w}×{n}" for w, n in top))
        if r["dropped_latin"]:
            print("  因英文被丢弃的样例:", " / ".join(r["dropped_latin"][: args.show]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
