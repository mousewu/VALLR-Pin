"""Step-2：Error-Aware 指令数据构造。

论文的做法是用**多个训练阶段的检查点**去解码训练集，得到 CER 分布很宽的假设，
再和真值组成 (指令, 回答) 对。这里在此基础上加了三个工程上必要的约束：

* ``max_cer``      —— 过滤掉几乎全错的样本；这类样本只会教会 LLM 凭空编造。
* ``keep_correct`` —— Stage-I 已经正确的样本按比例保留；否则模型会学到
                      "输入总是错的"这一偏置，把对的也改错 (over-correction)。
* 去重 + 分桶     —— 按 CER 分桶均衡采样，避免简单样本淹没困难样本。

输出为 messages 格式 jsonl，可直接喂给本仓库的 LoRA 训练脚本或 ms-swift。
"""

from __future__ import annotations

import json
import hashlib
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from ..engine.metrics import edit_ops
from .prompt import build_messages


@dataclass
class BuildConfig:
    max_cer: float = 0.8
    keep_correct: float = 0.25       # Stage-I 已经全对时的保留比例
    nbest: int = 5
    max_per_bucket: int = 100000
    n_buckets: int = 5
    seed: int = 0
    dedup: bool = True


def _cer(ref: str, hyp: str) -> float:
    s, d, i = edit_ops(list(ref), list(hyp))
    return (s + d + i) / max(len(ref), 1)


def load_records(paths: Iterable[str]) -> List[Dict]:
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def build_instruction_data(records: Sequence[Dict], cfg: BuildConfig = BuildConfig()
                           ) -> List[Dict]:
    rng = random.Random(cfg.seed)
    buckets: Dict[int, List[Dict]] = defaultdict(list)
    seen = set()
    n_drop_cer, n_drop_dup, n_drop_correct = 0, 0, 0

    for r in records:
        ref = r["ref"]
        nbest = [h["text"] for h in r.get("nbest", [])][: cfg.nbest]
        if not ref or not nbest:
            continue
        top1 = nbest[0]
        c = _cer(ref, top1)
        if c > cfg.max_cer:
            n_drop_cer += 1
            continue
        if c == 0.0 and rng.random() > cfg.keep_correct:
            n_drop_correct += 1
            continue
        key = (tuple(r.get("pinyin", [])), tuple(nbest))
        if cfg.dedup and key in seen:
            n_drop_dup += 1
            continue
        seen.add(key)
        sample = {"messages": build_messages(r.get("pinyin", []), nbest, ref),
                  "meta": {"id": r.get("id"), "ckpt": r.get("ckpt", ""),
                           "stage1_cer": round(c, 4)}}
        b = min(int(c * cfg.n_buckets), cfg.n_buckets - 1)
        buckets[b].append(sample)

    out: List[Dict] = []
    for b in sorted(buckets):
        items = buckets[b]
        rng.shuffle(items)
        out.extend(items[: cfg.max_per_bucket])
    rng.shuffle(out)
    dist = {b: len(v) for b, v in sorted(buckets.items())}
    print(f"[build-llm-data] kept={len(out)} bucket_dist={dist} "
          f"drop(cer>{cfg.max_cer})={n_drop_cer} drop(dup)={n_drop_dup} "
          f"drop(already-correct)={n_drop_correct}", flush=True)
    return out


def write_jsonl(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_train_val(rows: List[Dict], val_ratio: float = 0.02, seed: int = 0):
    """Group by utterance id so checkpoint variants cannot cross splits."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    train, val = [], []
    threshold = round(val_ratio * 10_000)
    for row in rows:
        group = str(row.get("meta", {}).get("id", row.get("id", "")))
        digest = hashlib.sha1(f"{seed}:{group}".encode()).hexdigest()
        (val if int(digest[:8], 16) % 10_000 < threshold else train).append(row)
    return train, val
