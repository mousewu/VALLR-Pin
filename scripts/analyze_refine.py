#!/usr/bin/env python3
"""对比 Stage-I 与 Stage-II 的逐句结果，统计"改好/改坏/没改"并给出示例。

用法::

    python scripts/analyze_refine.py exp/demo/dev_refined.jsonl --show 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vallr_pin.engine.metrics import ErrorStats, edit_ops  # noqa: E402
from vallr_pin.text.pinyin import text_to_pinyin  # noqa: E402


def er(ref: str, hyp: str) -> float:
    s, d, i = edit_ops(list(ref), list(hyp))
    return (s + d + i) / max(len(ref), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8") if l.strip()]
    before, after = ErrorStats(), ErrorStats()
    fixed, broken, same, neutral, homophone_fixed = [], [], 0, 0, 0
    for r in rows:
        ref, top1, ref_out = r["ref"], r["top1"], r["refined"]
        before.update(list(ref), list(top1))
        after.update(list(ref), list(ref_out))
        e0, e1 = er(ref, top1), er(ref, ref_out)
        if top1 == ref_out:
            same += 1
        elif e1 < e0:
            fixed.append(r)
            # 是否属于"同音改对"：改动前后拼音一致，但字不同
            if (len(top1) == len(ref_out)
                    and text_to_pinyin(top1)[1] == text_to_pinyin(ref_out)[1]):
                homophone_fixed += 1
        elif e1 > e0:
            broken.append(r)
        else:
            neutral += 1              # 改写了但错误率没变

    print(f"utts={len(rows)}  未改动={same}  改好={len(fixed)}  改坏={len(broken)}  "
          f"改写但持平={neutral}  其中同音改对={homophone_fixed}")
    print(f"Stage-I  CER = {100 * before.rate:.2f}%   ({before})")
    print(f"Refined  CER = {100 * after.rate:.2f}%   ({after})")
    print(f"绝对增益 = {100 * (before.rate - after.rate):+.2f} 个百分点\n")

    for tag, rs in (("改好示例", fixed), ("改坏示例", broken)):
        for r in rs[: args.show]:
            print(f"[{tag}] ref     : {r['ref']}")
            print(f"           pinyin  : {' '.join(r['pinyin'])}")
            print(f"           stage-1 : {r['top1']}")
            print(f"           refined : {r['refined']}\n")


if __name__ == "__main__":
    main()
