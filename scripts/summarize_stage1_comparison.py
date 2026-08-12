#!/usr/bin/env python3
"""Print the best greedy dev metrics from comparable Stage-I runs."""

from __future__ import annotations

import argparse
import json
import math
import os


DEFAULT_RUNS = [
    ("text-only", "exp/stage1_char_only"),
    ("pinyin-only", "exp/stage1_pinyin_only"),
    ("joint", "exp/stage1_pinyin_auxchar"),
]


def _best(path: str, metric: str):
    best = (math.inf, None, {})
    if not os.path.exists(path):
        return None, None, {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            value = row.get("dev", {}).get(metric)
            if value is not None and value < best[0]:
                best = float(value), row.get("epoch"), row.get("dev", {})
    return (None, None, {}) if math.isinf(best[0]) else best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", metavar="NAME=EXP_DIR")
    args = parser.parse_args()
    runs = DEFAULT_RUNS
    if args.runs:
        runs = []
        for value in args.runs:
            name, sep, path = value.partition("=")
            if not sep:
                parser.error(f"run must use NAME=EXP_DIR: {value}")
            runs.append((name, path))

    print("| run | best dev CER | epoch | text OOV | best dev SER | epoch |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, directory in runs:
        path = os.path.join(directory, "metrics.jsonl")
        cer, cer_epoch, cer_dev = _best(path, "cer")
        ser, ser_epoch, _ = _best(path, "ser")
        cer_text = "-" if cer is None else f"{100 * cer:.2f}%"
        ser_text = "-" if ser is None else f"{100 * ser:.2f}%"
        oov = cer_dev.get("text_oov_rate")
        oov_text = "-" if oov is None else f"{100 * oov:.2f}%"
        print(f"| {name} | {cer_text} | {cer_epoch or '-'} | {oov_text} | "
              f"{ser_text} | {ser_epoch or '-'} |")


if __name__ == "__main__":
    main()
