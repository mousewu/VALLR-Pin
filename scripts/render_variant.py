#!/usr/bin/env python3
"""从轨迹缓存渲染出某个模型要的输入格式。

换模型 = 换一个 ``--spec``，不需要重跑人脸检测。规格定义见
``vallr_pin/data/roi_spec.py``，加新模型时在 ``PRESETS`` 里追加一条即可。

用法::

    python scripts/render_variant.py --list                       # 看有哪些规格
    python scripts/render_variant.py video.mp4 tracks/v.npz \\
        --spec syncnet --out variants/v_syncnet.npy
    # 覆盖预设里的个别字段
    python scripts/render_variant.py video.mp4 tracks/v.npz \\
        --spec vallr_pin --set size=128 --out variants/v_128.npy
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vallr_pin.data.roi_spec import (FaceTrack, describe_presets,  # noqa: E402
                                     render_frame, resolve_spec)


def maybe_resample(video: str, target_fps: int, src_fps: float, tmpdir: str) -> str:
    """帧率不符时重采样。SyncNet 这类模型对帧率敏感，喂错会因时间尺度错配而失效。"""
    if not target_fps or abs(src_fps - target_fps) < 0.01:
        return video
    out = os.path.join(tmpdir, f"resampled_{target_fps}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
                    "-qscale:v", "2", "-async", "1", "-r", str(target_fps), out],
                   check=True)
    return out


def render(video: str, track_path: str, spec, out_path: str,
           tmpdir: str = "", min_coverage: float = 0.0) -> dict:
    import cv2

    track = FaceTrack.load(track_path)
    spec = resolve_spec(spec)

    tmpdir = tmpdir or tempfile.mkdtemp()
    src = maybe_resample(video, spec.fps or 0, track.fps, tmpdir)
    resampled = src != video
    if resampled:
        # 重采样改变了帧号，用时间戳把关键点映射到新帧
        ratio = (spec.fps or track.fps) / track.fps
    else:
        ratio = 1.0

    lut = {int(f): i for i, f in enumerate(track.frames)}
    cap = cv2.VideoCapture(src)
    out_frames, missing, i = [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        src_idx = int(round(i / ratio)) if resampled else i
        j = lut.get(src_idx)
        if j is None:                       # 该帧没检出人脸，就近取最相邻的一帧
            cand = [k for k in (src_idx - 1, src_idx + 1, src_idx - 2, src_idx + 2)
                    if k in lut]
            if not cand:
                missing += 1
                i += 1
                continue
            j = lut[cand[0]]
        pts = track.points(j)
        if np.isnan(pts).all():
            missing += 1
            i += 1
            continue
        # 不要在这里填补 NaN：缓存可能只保留了必要关键点，
        # anchor_box 用 nan-aware 统计量处理，填 0 会把人脸框拉到画面原点
        out_frames.append(render_frame(fr, pts, spec))
        i += 1
    cap.release()

    if not out_frames:
        raise SystemExit("没有可渲染的帧")
    coverage = len(out_frames) / max(i, 1)
    if coverage < min_coverage:
        raise ValueError(f"face coverage {coverage:.3f} < required {min_coverage:.3f}")
    arr = np.stack(out_frames)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.save(out_path, arr)
    info = {"out": out_path, "shape": list(arr.shape), "dtype": str(arr.dtype),
            "spec": spec.to_dict(), "resampled": resampled,
            "src_fps": track.fps, "target_fps": float(spec.fps or track.fps),
            "missing_frames": missing, "coverage": coverage,
            "track_meta": {key: track.meta[key] for key in
                           ("input_type", "selection_strategy", "selected_track_id",
                            "selection_margin", "face_track_count",
                            "median_lip_width_px", "median_yaw_proxy")
                           if key in track.meta},
            "mb": round(arr.nbytes / 1e6, 1)}
    with open(out_path.replace(".npy", ".spec.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("track", nargs="?")
    ap.add_argument("--spec", default="vallr_pin")
    ap.add_argument("--out", default="")
    ap.add_argument("--set", nargs="*", default=[], help="覆盖预设字段，如 size=128")
    ap.add_argument("--min-coverage", type=float, default=0.0)
    ap.add_argument("--list", action="store_true", help="列出所有预设规格")
    args = ap.parse_args()

    if args.list:
        print(describe_presets())
        return
    if not (args.video and args.track and args.out):
        ap.error("需要 video / track / --out")

    spec = resolve_spec(args.spec)
    over = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        cur = getattr(spec, k)
        over[k] = type(cur)(v) if cur is not None and not isinstance(cur, tuple) else v
    if over:
        spec = replace(spec, **over)
    info = render(args.video, args.track, spec, args.out, min_coverage=args.min_coverage)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
