#!/usr/bin/env python3
"""Stage-II 的**受控诊断**：把参考文本按比例替换成同音字，再看精化器能修回多少。

这不是端到端指标，而是一个隔离实验：它模拟"Stage-I 把音听对了、字选错了"这一
普通话 VSR 的主导错误类型，从而单独衡量拼音引导纠错的能力上限，
不受 Stage-I 训练是否充分的干扰。

用法::

    python scripts/homophone_stress_test.py --texts data/synth/train.jsonl --rate 0.3
    python scripts/homophone_stress_test.py --texts corpus.txt --rate 0.3 \\
        --refiner llm --model-path Qwen/Qwen3-4B-Instruct-2507 --adapter exp/llm_lora
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vallr_pin.engine.metrics import ErrorStats  # noqa: E402
from vallr_pin.llm.refine import LLMConfig, NgramConfig, NgramPinyinRefiner  # noqa: E402
from vallr_pin.text.pinyin import char_syllable, homophone_table, text_to_pinyin  # noqa: E402


def load_texts(path: str):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line)["text"] if line.startswith("{") else line)
    return out


def corrupt(text: str, rate: float, rng: random.Random, vocab: set) -> str:
    table = homophone_table()
    chars = list(text)
    for i, ch in enumerate(chars):
        if rng.random() >= rate:
            continue
        syl = char_syllable(ch)
        cands = [c for c in table.get(syl, []) if c != ch and c in vocab]
        if cands:
            chars[i] = rng.choice(cands)
    return "".join(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="manifest jsonl 或每行一句的纯文本")
    ap.add_argument("--lm-texts", default="", help="n-gram 语言模型的训练文本，默认同 --texts")
    ap.add_argument("--rate", type=float, default=0.3, help="同音替换比例")
    ap.add_argument("--max-utts", type=int, default=200)
    ap.add_argument("--refiner", choices=["ngram", "llm"], default="ngram")
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    texts = load_texts(args.texts)
    lm_texts = load_texts(args.lm_texts) if args.lm_texts else texts
    vocab = {c for t in lm_texts for c in t}
    refs = list(dict.fromkeys(texts))[: args.max_utts]

    if args.refiner == "ngram":
        refiner = NgramPinyinRefiner(NgramConfig()).fit(lm_texts)
        refine = refiner.refine
    else:
        from vallr_pin.llm.refine import LLMRefiner
        llm = LLMRefiner(LLMConfig(model_path=args.model_path, adapter_path=args.adapter,
                                   device=args.device))
        refine = llm.refine

    before, after = ErrorStats(), ErrorStats()
    shown = 0
    for ref in refs:
        noisy = corrupt(ref, args.rate, rng, vocab)
        pinyin = text_to_pinyin(ref)[1]          # 假设 Stage-I 的拼音是对的
        out = refine(pinyin, [noisy])
        before.update(list(ref), list(noisy))
        after.update(list(ref), list(out))
        if shown < args.show and noisy != ref:
            print(f"ref     : {ref}\n  noisy : {noisy}\n  fixed : {out}")
            shown += 1

    print(f"\n同音替换率={args.rate}  样本={len(refs)}  精化器={args.refiner}")
    print(f"  注入后 CER = {100 * before.rate:.2f}%")
    print(f"  精化后 CER = {100 * after.rate:.2f}%")
    print(f"  修复率     = {100 * (1 - after.rate / max(before.rate, 1e-9)):.1f}%")


if __name__ == "__main__":
    main()
