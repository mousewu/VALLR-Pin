#!/usr/bin/env python3
"""把 CNVSRC / CMLR / 自采数据转成本仓库的 manifest 格式。

用法::

    # 1) 只生成 manifest（视频路径直接指向 mp4/帧目录，训练时在线解码）
    python scripts/prepare_manifest.py --transcript labels.txt --video-dir videos \
        --out data/cnvsrc/train.jsonl

    # 2) 同时把唇部 ROI 预处理成 npy（强烈推荐：训练 IO 快一个量级）
    python scripts/prepare_manifest.py --transcript labels.txt --video-dir videos \
        --out data/cnvsrc/train.jsonl --to-npy data/cnvsrc/roi --roi-size 96

transcript 每行为 ``utt_id<TAB>中文文本``（或空格分隔的第一列为 id）。

关于 ROI：CNVSRC 官方 baseline 使用 RetinaFace + 68 点关键点对齐后裁 96x96 的
嘴部区域，再随机裁到 88x88。本脚本默认只做"人脸中心下半部"的粗裁，
若已有关键点，请通过 ``--landmarks`` 传入 ``{utt_id: (T,68,2)}`` 的 npz，
脚本会按嘴部关键点的均值做仿射对齐裁剪，与官方 pipeline 对齐。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np


def read_transcript(path: str) -> Dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) == 2:
                out[parts[0].strip()] = parts[1].strip()
    return out


def crop_roi(frames: np.ndarray, lm: Optional[np.ndarray], size: int) -> np.ndarray:
    """frames: (T,H,W) 灰度; lm: (T,68,2) 或 None。"""
    import cv2
    t, h, w = frames.shape
    out = np.zeros((t, size, size), dtype=np.uint8)
    for i in range(t):
        if lm is not None:
            mouth = lm[i][48:68]
            cx, cy = float(mouth[:, 0].mean()), float(mouth[:, 1].mean())
            half = size // 2
        else:                                  # 粗裁：人脸下半部中心
            cx, cy, half = w / 2, h * 0.62, int(min(h, w) * 0.22)
        x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
        patch = frames[i, y0:y1, x0:x1]
        if patch.size == 0:
            patch = frames[i]
        out[i] = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
    return out


def load_gray(path: str) -> np.ndarray:
    import cv2
    if os.path.isdir(path):
        files = sorted(os.listdir(path))
        return np.stack([cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
                         for f in files])
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--suffix", default=".mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--to-npy", default="")
    ap.add_argument("--roi-size", type=int, default=96)
    ap.add_argument("--landmarks", default="")
    ap.add_argument("--min-chars", type=int, default=1)
    args = ap.parse_args()

    trans = read_transcript(args.transcript)
    lms = np.load(args.landmarks, allow_pickle=True) if args.landmarks else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.to_npy:
        os.makedirs(args.to_npy, exist_ok=True)

    n_ok, n_miss = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for utt, text in sorted(trans.items()):
            if len(text) < args.min_chars:
                continue
            src = os.path.join(args.video_dir, utt + args.suffix)
            if not os.path.exists(src):
                src = os.path.join(args.video_dir, utt)
            if not os.path.exists(src):
                n_miss += 1
                continue
            video_path = src
            if args.to_npy:
                dst = os.path.join(args.to_npy, utt + ".npy")
                if not os.path.exists(dst):
                    frames = load_gray(src)
                    lm = lms[utt] if lms is not None and utt in lms else None
                    np.save(dst, crop_roi(frames, lm, args.roi_size))
                video_path = dst
            f.write(json.dumps({"id": utt, "video": video_path, "text": text},
                               ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"wrote {n_ok} items -> {args.out} (missing videos: {n_miss})")


if __name__ == "__main__":
    main()
