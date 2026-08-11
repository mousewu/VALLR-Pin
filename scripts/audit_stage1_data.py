#!/usr/bin/env python3
"""Audit manifests and estimate ROI normalization statistics without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vallr_pin.data.dataset import read_manifest  # noqa: E402
from vallr_pin.text.pinyin import text_to_pinyin_mixed  # noqa: E402


def audit(paths: list[str], pixel_samples: int = 2000, seed: int = 0) -> dict:
    splits, speakers, sources = {}, defaultdict(set), Counter()
    ids, media, problems = set(), set(), Counter()
    lengths, ratios, rows = [], [], []
    for path in paths:
        split = os.path.basename(path).split(".")[0]
        splits[split] = 0
        for item in read_manifest(path):
            splits[split] += 1; rows.append(item)
            uid, video, text = item.get("id", ""), item.get("video", ""), item.get("text", "")
            if uid in ids: problems["duplicate_id"] += 1
            if video in media: problems["duplicate_media"] += 1
            ids.add(uid); media.add(video); sources[f"{split}:{item.get('source','unknown')}"] += 1
            speaker = item.get("speaker_id", "")
            if not speaker: problems["missing_speaker"] += 1
            else: speakers[split].add(speaker)
            if not video or not os.path.exists(video): problems["missing_media"] += 1
            tokens, syllables, unknown = text_to_pinyin_mixed(text)
            if unknown: problems["unknown_latin"] += 1
            if not tokens or not syllables: problems["empty_label"] += 1
            frames = int(item.get("n_frames", 0) or 0)
            if video.endswith(".npy") and os.path.exists(video):
                try: frames = int(np.load(video, mmap_mode="r").shape[0])
                except Exception: problems["bad_npy"] += 1
            if frames:
                required = len(syllables) + sum(a == b for a, b in zip(syllables, syllables[1:]))
                if frames < required: problems["invalid_ctc_length"] += 1
                lengths.append(frames); ratios.append(frames / max(len(syllables), 1))

    overlap = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        common = speakers[left] & speakers[right]
        overlap[f"{left}:{right}"] = len(common)
        if common: problems["speaker_leakage"] += len(common)

    rng = random.Random(seed); candidates = [r for r in rows if str(r.get("video", "")).endswith(".npy")]
    rng.shuffle(candidates); candidates = candidates[:pixel_samples]
    total = total_sq = pixels = 0.0
    for item in candidates:
        try:
            array = np.load(item["video"], mmap_mode="r")
            stride = max(1, len(array) // 8)
            values = np.asarray(array[::stride], dtype=np.float64) / 255.0
            total += values.sum(); total_sq += np.square(values).sum(); pixels += values.size
        except Exception:
            problems["pixel_sample_failed"] += 1
    mean = total / pixels if pixels else None
    std = (max(total_sq / pixels - mean * mean, 0.0) ** 0.5) if pixels else None

    percentile = lambda values, q: float(np.percentile(values, q)) if values else None
    return {"splits": splits, "sources": dict(sources),
            "speakers": {k: len(v) for k, v in speakers.items()}, "speaker_overlap": overlap,
            "problems": dict(problems), "frames": {"p05": percentile(lengths, 5),
            "p50": percentile(lengths, 50), "p95": percentile(lengths, 95)},
            "frames_per_syllable": {"p05": percentile(ratios, 5),
            "p50": percentile(ratios, 50), "p95": percentile(ratios, 95)},
            "roi_normalization": {"mean": mean, "std": std,
                                  "sampled_utterances": len(candidates)}}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("manifests", nargs="+")
    ap.add_argument("--pixel-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="")
    args = ap.parse_args(); report = audit(args.manifests, args.pixel_samples, args.seed)
    text = json.dumps(report, ensure_ascii=False, indent=2); print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as stream: stream.write(text + "\n")
    if any(report["speaker_overlap"].values()) or report["problems"].get("invalid_ctc_length"):
        raise SystemExit(2)


if __name__ == "__main__": main()
