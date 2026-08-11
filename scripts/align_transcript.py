#!/usr/bin/env python3
"""锚点对齐：把"编辑过的文字稿"对到音频时间轴上，产出带时间戳的句段。

为什么不能直接做强制对齐：出版的访谈稿是**编辑稿**不是逐字稿 —— 口水词被删、
句子被重组、还夹着章节标题和编者注。强制对齐要求文本与音频逐字对应，
遇到这种文本会整段跑飞。

标准解法（LibriSpeech 对齐有声书与 Gutenberg 文本用的就是这套）：

1. 先跑 ASR，得到**带时间戳的字序列**（有错字，但时间戳是可信的）；
2. 在 ASR 结果与文字稿之间找**唯一 k-gram 锚点**，取单调递增子序列作为骨架；
3. 锚点之间的小块做编辑距离 DP 对齐；
4. 逐句统计匹配率，低于阈值的句子直接丢弃 —— 那些正是被改写/新增的部分。

关键在第 4 步：**宁可少要，不可要错**。一条时间戳错位的样本，比缺这条样本有害得多。

ASR 后端：
  * ``jsonl``  —— 读预先算好的结果，格式 ``{"char": "今", "start": 1.20, "end": 1.28}``
                  每行一个字。本脚本的对齐逻辑与 ASR 实现解耦，这条路径有单测覆盖。
  * ``funasr`` —— Paraformer 中文时间戳模型（需自行下载模型，本机未验证）
  * ``whisper``—— faster-whisper（中文精度弱于 Paraformer，仅作备选）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CJK = re.compile(r"[一-龥]")


@dataclass
class AsrChar:
    char: str
    start: float
    end: float


@dataclass
class AlignedSentence:
    text: str
    start: float
    end: float
    match_rate: float
    n_matched: int
    speaker: str = ""

    def to_dict(self) -> dict:
        return {"text": self.text, "start": round(self.start, 3),
                "end": round(self.end, 3), "match_rate": round(self.match_rate, 3),
                "n_matched": self.n_matched, "speaker": self.speaker}


# --------------------------------------------------------------------------- #
#                                  锚点搜索                                     #
# --------------------------------------------------------------------------- #
def unique_kgram_anchors(a: str, b: str, k: int = 8) -> List[Tuple[int, int]]:
    """返回 [(a 中位置, b 中位置)]：在两侧都**只出现一次**的 k-gram。

    只用唯一 k-gram 是为了避免歧义匹配 —— 中文里"的时候""我觉得"这种高频串
    到处都是，拿它们当锚点会把对齐拽到错误的位置。
    """
    from collections import defaultdict
    pos_a: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(a) - k + 1):
        pos_a[a[i:i + k]].append(i)
    pos_b: Dict[str, List[int]] = defaultdict(list)
    for i in range(len(b) - k + 1):
        pos_b[b[i:i + k]].append(i)
    out = []
    for g, ia in pos_a.items():
        ib = pos_b.get(g)
        if len(ia) == 1 and ib and len(ib) == 1:
            out.append((ia[0], ib[0]))
    return sorted(out)


def longest_increasing_pairs(pairs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """在按 a 排序的锚点里取 b 也单调递增的最长子序列，剔除交叉的假锚点。"""
    if not pairs:
        return []
    import bisect
    tails: List[int] = []
    idx: List[int] = []
    prev = [-1] * len(pairs)
    for i, (_, b) in enumerate(pairs):
        j = bisect.bisect_left(tails, b)
        if j == len(tails):
            tails.append(b)
            idx.append(i)
        else:
            tails[j] = b
            idx[j] = i
        prev[i] = idx[j - 1] if j else -1
    out, cur = [], idx[-1]
    while cur != -1:
        out.append(pairs[cur])
        cur = prev[cur]
    return out[::-1]


def dp_align(a: str, b: str) -> List[Tuple[Optional[int], Optional[int]]]:
    """全局编辑距离对齐，返回 [(a 下标, b 下标)]，插入/删除处对应位置为 None。"""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        di, dp_ = d[i], d[i - 1]
        for j in range(1, m + 1):
            di[j] = min(dp_[j - 1] + (ai != b[j - 1]), dp_[j] + 1, di[j - 1] + 1)
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def align_streams(asr_text: str, ref_text: str, k: int = 8, max_block: int = 1500
                  ) -> Dict[int, int]:
    """返回 {ref 下标: asr 下标}，只包含**字符相同**的对齐点。

    先用锚点把长序列切成小块，再逐块 DP。不切块的话 O(N·M) 在万字级就跑不动了。
    """
    anchors = longest_increasing_pairs(unique_kgram_anchors(asr_text, ref_text, k))
    mapping: Dict[int, int] = {}
    blocks: List[Tuple[int, int, int, int]] = []
    pa = pb = 0
    for ia, ib in anchors:
        if ia >= pa and ib >= pb:
            blocks.append((pa, ia, pb, ib))
            for t in range(k):                     # 锚点自身逐字对齐
                mapping[ib + t] = ia + t
            pa, pb = ia + k, ib + k
    blocks.append((pa, len(asr_text), pb, len(ref_text)))

    for a0, a1, b0, b1 in blocks:
        sa, sb = asr_text[a0:a1], ref_text[b0:b1]
        if not sa or not sb:
            continue
        if len(sa) * len(sb) > max_block * max_block:
            continue                               # 块太大说明这段根本对不上，跳过
        for ia, ib in dp_align(sa, sb):
            if ia is not None and ib is not None and sa[ia] == sb[ib]:
                mapping[b0 + ib] = a0 + ia
    return mapping


# --------------------------------------------------------------------------- #
#                                  句级切分                                     #
# --------------------------------------------------------------------------- #
def align_sentences(asr: Sequence[AsrChar], sentences: Sequence[Tuple[str, str]],
                    min_match: float = 0.8, k: int = 8
                    ) -> Tuple[List[AlignedSentence], List[Tuple[str, float]]]:
    """把每个句子对到时间轴。返回 (通过的句子, 被拒的 (句子, 匹配率))。"""
    asr_text = "".join(c.char for c in asr)
    spans: List[Tuple[int, int, str, str]] = []
    ref_parts: List[str] = []
    pos = 0
    for spk, sent in sentences:
        core = "".join(CJK.findall(sent))
        if not core:
            continue
        spans.append((pos, pos + len(core), sent, spk))
        ref_parts.append(core)
        pos += len(core)
    ref_text = "".join(ref_parts)

    mapping = align_streams(asr_text, ref_text, k=k)
    ok, rejected = [], []
    for b0, b1, sent, spk in spans:
        hits = [mapping[i] for i in range(b0, b1) if i in mapping]
        rate = len(hits) / max(b1 - b0, 1)
        if rate < min_match or not hits:
            rejected.append((sent, rate))
            continue
        s, e = min(hits), max(hits)
        ok.append(AlignedSentence(sent, asr[s].start, asr[e].end, rate, len(hits), spk))
    return ok, rejected


# --------------------------------------------------------------------------- #
#                                  ASR 后端                                     #
# --------------------------------------------------------------------------- #
def load_asr_jsonl(path: str) -> List[AsrChar]:
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            out.append(AsrChar(d["char"], float(d["start"]), float(d["end"])))
    return out


def run_funasr(audio: str, model: str = "paraformer-zh") -> List[AsrChar]:
    """Paraformer + 时间戳。需要先 `pip install funasr` 并下载模型；本机未验证。"""
    from funasr import AutoModel
    m = AutoModel(model=model, vad_model="fsmn-vad", punc_model=None)
    res = m.generate(input=audio, return_raw_text=True, sentence_timestamp=True)
    out: List[AsrChar] = []
    for seg in res:
        text = seg.get("text", "")
        stamps = seg.get("timestamp") or []
        for ch, (a, b) in zip(text, stamps):
            out.append(AsrChar(ch, a / 1000.0, b / 1000.0))
    return out


def parse_transcript(path: str) -> List[Tuple[str, str]]:
    """复用 transcript_stats 的解析，返回 [(说话人, 句子)]。"""
    from scripts.transcript_stats import parse_turns, split_sentences
    return [(spk, s) for spk, text in parse_turns(path) for s in split_sentences(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--asr", choices=["jsonl", "funasr"], default="jsonl")
    ap.add_argument("--asr-jsonl", default="")
    ap.add_argument("--audio", default="")
    ap.add_argument("--out", default="aligned.jsonl")
    ap.add_argument("--min-match", type=float, default=0.8)
    ap.add_argument("--kgram", type=int, default=8)
    args = ap.parse_args()

    asr = (load_asr_jsonl(args.asr_jsonl) if args.asr == "jsonl"
           else run_funasr(args.audio))
    sents = parse_transcript(args.transcript)
    ok, rejected = align_sentences(asr, sents, args.min_match, args.kgram)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in ok:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    total = len(ok) + len(rejected)
    print(f"对齐成功 {len(ok)}/{total} 句 ({100 * len(ok) / max(total, 1):.1f}%) -> {args.out}")
    if rejected:
        print("被拒样例 (匹配率过低，多半是编辑改写或编者注):")
        for sent, r in sorted(rejected, key=lambda x: x[1])[:6]:
            print(f"  [{r:.2f}] {sent[:40]}")


if __name__ == "__main__":
    main()
