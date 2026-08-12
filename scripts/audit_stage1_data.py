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


def audit(paths: list[str], pixel_samples: int = 2000, seed: int = 0,
          expected_fps: float = 25.0, max_frames: int = 0,
          max_frames_per_syllable: float = 0.0) -> dict:
    splits, speakers, sources = {}, defaultdict(set), Counter()
    input_types = Counter()
    source_lip_widths, source_yaw_proxies = defaultdict(list), defaultdict(list)
    ids, media, problems = set(), set(), Counter()
    lengths, ratios, lip_widths, yaw_proxies, rows = [], [], [], [], []
    for path in paths:
        split = os.path.basename(path).split(".")[0]
        splits[split] = 0
        for item in read_manifest(path):
            splits[split] += 1; rows.append(item)
            uid, video, text = item.get("id", ""), item.get("video", ""), item.get("text", "")
            if uid in ids: problems["duplicate_id"] += 1
            if video in media: problems["duplicate_media"] += 1
            source_name = item.get("source", "unknown")
            ids.add(uid); media.add(video); sources[f"{split}:{source_name}"] += 1
            input_types[f"{split}:{item.get('source_input_type', item.get('input_type','missing'))}"] += 1
            if item.get("input_type") != "mouth_roi" or item.get("roi_type") != "mouth":
                problems["not_mouth_roi"] += 1
            try:
                fps = float(item.get("fps", 0) or 0)
                if fps <= 0: problems["missing_fps"] += 1
                elif expected_fps > 0 and abs(fps - expected_fps) > .05:
                    problems["wrong_fps"] += 1
            except (TypeError, ValueError):
                problems["bad_fps"] += 1
            speaker = item.get("speaker_id", "")
            if not speaker: problems["missing_speaker"] += 1
            else: speakers[split].add(speaker)
            if not video or not os.path.exists(video): problems["missing_media"] += 1
            tokens, syllables, unknown = text_to_pinyin_mixed(text)
            if unknown: problems["unknown_latin"] += 1
            if not tokens or not syllables: problems["empty_label"] += 1
            frames = int(item.get("n_frames", 0) or 0)
            if video.endswith(".npy") and os.path.exists(video):
                try:
                    shape = np.load(video, mmap_mode="r").shape
                    frames = int(shape[0])
                    declared_frames = int(item.get("n_frames", 0) or 0)
                    if declared_frames != frames:
                        problems["roi_frame_metadata"] += 1
                    if len(shape) not in (3, 4): problems["bad_roi_rank"] += 1
                    else:
                        height, width = int(shape[1]), int(shape[2])
                        if height != width: problems["non_square_roi"] += 1
                        declared = (int(item.get("roi_height", 0) or 0),
                                    int(item.get("roi_width", 0) or 0))
                        if declared != (height, width): problems["roi_shape_metadata"] += 1
                except Exception: problems["bad_npy"] += 1
            if frames:
                required = len(syllables) + sum(a == b for a, b in zip(syllables, syllables[1:]))
                if frames < required: problems["invalid_ctc_length"] += 1
                ratio = frames / max(len(syllables), 1)
                if max_frames > 0 and frames > max_frames:
                    problems["excessive_frames"] += 1
                if max_frames_per_syllable > 0 and ratio > max_frames_per_syllable:
                    problems["excessive_frames_per_syllable"] += 1
                lengths.append(frames); ratios.append(ratio)
            if item.get("median_lip_width_px") is not None:
                lip_widths.append(float(item["median_lip_width_px"]))
                source_lip_widths[source_name].append(float(item["median_lip_width_px"]))
            if item.get("median_yaw_proxy") is not None:
                yaw_proxies.append(float(item["median_yaw_proxy"]))
                source_yaw_proxies[source_name].append(float(item["median_yaw_proxy"]))

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
    return {"splits": splits, "sources": dict(sources), "input_types": dict(input_types),
            "speakers": {k: len(v) for k, v in speakers.items()}, "speaker_overlap": overlap,
            "problems": dict(problems), "frames": {"p05": percentile(lengths, 5),
            "p50": percentile(lengths, 50), "p95": percentile(lengths, 95),
            "p99": percentile(lengths, 99), "max": max(lengths) if lengths else None},
            "frames_per_syllable": {"p05": percentile(ratios, 5),
            "p50": percentile(ratios, 50), "p95": percentile(ratios, 95),
            "p99": percentile(ratios, 99), "max": max(ratios) if ratios else None},
            "limits": {"max_frames": int(max_frames),
                       "max_frames_per_syllable": float(max_frames_per_syllable)},
            "visual_quality": {
                "lip_width_px": {"p05": percentile(lip_widths, 5),
                                  "p50": percentile(lip_widths, 50),
                                  "p95": percentile(lip_widths, 95)},
                "yaw_proxy": {"p05": percentile(yaw_proxies, 5),
                              "p50": percentile(yaw_proxies, 50),
                              "p95": percentile(yaw_proxies, 95)}},
            "visual_quality_by_source": {
                source: {
                    "lip_width_px": {
                        "p05": percentile(source_lip_widths[source], 5),
                        "p50": percentile(source_lip_widths[source], 50),
                        "p95": percentile(source_lip_widths[source], 95)},
                    "yaw_proxy": {
                        "p05": percentile(source_yaw_proxies[source], 5),
                        "p50": percentile(source_yaw_proxies[source], 50),
                        "p95": percentile(source_yaw_proxies[source], 95)}}
                for source in sorted(set(source_lip_widths) | set(source_yaw_proxies))},
            "roi_normalization": {"mean": mean, "std": std,
                                  "sampled_utterances": len(candidates)}}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("manifests", nargs="+")
    ap.add_argument("--pixel-samples", type=int, default=2000)
    ap.add_argument("--expected-fps", type=float, default=25.0)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 只报告；正数时超长样本视为致命问题")
    ap.add_argument("--max-frames-per-syllable", type=float, default=0.0,
                    help="0 只报告；正数时检测异常停顿/错标签")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="")
    args = ap.parse_args(); report = audit(
        args.manifests, args.pixel_samples, args.seed, args.expected_fps,
        args.max_frames, args.max_frames_per_syllable)
    text = json.dumps(report, ensure_ascii=False, indent=2); print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as stream: stream.write(text + "\n")
    fatal = {"speaker_leakage", "invalid_ctc_length", "not_mouth_roi", "missing_fps",
             "wrong_fps", "bad_fps", "bad_roi_rank", "non_square_roi",
             "roi_shape_metadata", "roi_frame_metadata", "excessive_frames",
             "excessive_frames_per_syllable"}
    if any(report["speaker_overlap"].values()) or any(
            report["problems"].get(name) for name in fatal):
        raise SystemExit(2)


if __name__ == "__main__": main()
