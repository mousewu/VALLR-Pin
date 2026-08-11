#!/usr/bin/env python3
"""字幕直出训练样本：有人工字幕的独白视频，可以跳过 ASR 与锚点对齐。

适用前提（务必先用 probe_video.py / asd_pipeline.py 确认）：

* 字幕是**人工**的，不是 YouTube 自动生成的（自动字幕有错字且时间戳粗）
* 字幕时间戳跟得住语速 —— 用 ``--check`` 打印语速分布自查，
  普通话正常是 4-6 字/秒，若方差很大说明是随手标的显示时间，不能直接用
* 单人独白，或已用 asd_pipeline.py 产出 ``asd.json`` 过滤掉非说话人镜头

产出直接就是本仓库的 manifest 格式，可以喂给 ``cli train``。

用法::

    python scripts/build_from_subtitles.py video.mp4 subs.json3 \\
        --model face_landmarker.task --out-dir data/mono --check
    # 若视频是从完整片源里切出来的一段：
    python scripts/build_from_subtitles.py clip.mp4 subs.json3 ... --clip-start 300.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.probe_video import LIPS_OUTER, crop_roi  # noqa: E402
from vallr_pin.text.pinyin import text_to_pinyin_mixed  # noqa: E402

CJK = re.compile(r"[一-龥]")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z\-']*")


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def dur(self) -> float:
        return self.end - self.start

    @property
    def n_units(self) -> int:
        return len(CJK.findall(self.text)) + len(LATIN_WORD.findall(self.text))


def load_cues(path: str) -> List[Cue]:
    """支持 YouTube json3 与 srt。"""
    if path.endswith(".json3") or path.endswith(".json"):
        data = json.load(open(path, encoding="utf-8"))
        cues = []
        for e in data.get("events", []):
            if not e.get("segs"):
                continue
            t = "".join(s.get("utf8", "") for s in e["segs"]).strip()
            if t:
                a = e["tStartMs"] / 1000.0
                cues.append(Cue(a, a + e.get("dDurationMs", 0) / 1000.0, t))
        return cues
    cues, buf, times = [], [], None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if "-->" in line:
            a, b = line.split("-->")
            times = (_srt_time(a), _srt_time(b))
        elif not line:
            if times and buf:
                cues.append(Cue(times[0], times[1], " ".join(buf)))
            buf, times = [], None
        elif not line.isdigit():
            buf.append(line)
    if times and buf:
        cues.append(Cue(times[0], times[1], " ".join(buf)))
    return cues


def _srt_time(s: str) -> float:
    s = s.strip().replace(",", ".")
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def check_timing(cues: Sequence[Cue]) -> Dict:
    """语速自查：时间戳是否真的跟着语音走。"""
    rates = np.array([c.n_units / c.dur for c in cues if c.dur > 0.2 and c.n_units])
    gaps = np.array([cues[i + 1].start - cues[i].end for i in range(len(cues) - 1)])
    return {"n_cues": len(cues),
            "rate_median": float(np.median(rates)) if len(rates) else 0.0,
            "rate_p10": float(np.percentile(rates, 10)) if len(rates) else 0.0,
            "rate_p90": float(np.percentile(rates, 90)) if len(rates) else 0.0,
            "in_normal_range": float(((rates >= 3) & (rates <= 7)).mean()) if len(rates) else 0,
            "gap_median_ms": float(np.median(gaps) * 1000) if len(gaps) else 0.0,
            "overlap_ratio": float((gaps < -0.01).mean()) if len(gaps) else 0.0}


def load_speaker_mask(asd_json: str, fps: float, n_frames: int) -> Optional[np.ndarray]:
    """从 asd_pipeline.py 的输出构造逐帧"是否是说话人"掩码。"""
    if not asd_json or not os.path.exists(asd_json):
        return None
    d = json.load(open(asd_json, encoding="utf-8"))
    mask = np.zeros(n_frames, dtype=bool)
    for s in d.get("segments", []):
        if s.get("is_speaker"):
            mask[s["start_frame"]: s["end_frame"] + 1] = True
    return mask


def build(args) -> Dict:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    cues = load_cues(args.subs)
    stats = check_timing(cues)
    if args.check:
        print(f"[字幕自查] 条数={stats['n_cues']} "
              f"语速中位={stats['rate_median']:.2f} 字/秒 "
              f"(p10={stats['rate_p10']:.2f} p90={stats['rate_p90']:.2f}) "
              f"正常区间占比={100 * stats['in_normal_range']:.1f}% "
              f"间隙中位={stats['gap_median_ms']:.0f}ms "
              f"重叠占比={100 * stats['overlap_ratio']:.1f}%", flush=True)
        if stats["in_normal_range"] < 0.8:
            print("  ⚠ 语速分布异常，这份字幕多半是显示时间而非语音时间，"
                  "不要直接当标签用，改走 align_transcript.py", flush=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))

    # 一次顺序扫描，缓存每帧的唇部 ROI（随机 seek 在长视频上极慢）
    rois: Dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps))
        if res.face_landmarks:
            p = np.array([[q.x * W, q.y * H] for q in res.face_landmarks[0]],
                         dtype=np.float32)
            roi = crop_roi(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), p[LIPS_OUTER],
                           args.roi_size, args.roi_scale)
            if roi is not None:
                rois[i] = roi
        i += 1
    cap.release()
    n_frames = max(n_frames, i)
    spk_mask = load_speaker_mask(args.asd_json, fps, n_frames)

    os.makedirs(os.path.join(args.out_dir, "roi"), exist_ok=True)
    items, drop = [], {"out_of_range": 0, "no_face": 0, "not_speaker": 0,
                       "bad_len": 0, "unknown_latin": 0, "short_video": 0}
    for idx, c in enumerate(cues):
        a, b = c.start - args.clip_start, c.end - args.clip_start
        f0, f1 = int(round(a * fps)), int(round(b * fps))
        # 字幕时间戳锚在**音频**时间轴上；若视频相对音频有偏移，取帧窗必须补偿，
        # 否则每条样本的口型都会系统性错位。偏移量用 syncnet_offset.py 测。
        # 约定与该脚本一致：负值 = 音频滞后视频。
        f0 += args.av_offset_frames
        f1 += args.av_offset_frames
        f0 -= args.pad_frames
        f1 += args.pad_frames
        if f1 <= 0 or f0 >= n_frames:
            drop["out_of_range"] += 1
            continue
        f0, f1 = max(f0, 0), min(f1, n_frames - 1)
        frames = [f for f in range(f0, f1 + 1) if f in rois]
        if len(frames) < args.min_frames:
            drop["no_face" if len(frames) < (f1 - f0) * 0.5 else "short_video"] += 1
            continue
        if len(frames) < (f1 - f0 + 1) * args.min_face_cov:
            drop["no_face"] += 1
            continue
        if spk_mask is not None and spk_mask[f0:f1 + 1].mean() < args.min_speaker_cov:
            drop["not_speaker"] += 1
            continue
        toks, syls, unknown = text_to_pinyin_mixed(c.text)
        if not (args.min_units <= len(toks) <= args.max_units):
            drop["bad_len"] += 1
            continue
        if unknown and not args.keep_unknown_latin:
            drop["unknown_latin"] += 1
            continue
        arr = np.stack([rois[f] for f in frames])
        name = f"{args.prefix}_{idx:05d}.npy"
        np.save(os.path.join(args.out_dir, "roi", name), arr)
        items.append({"id": name[:-4], "video": os.path.join("roi", name),
                      "text": "".join(toks), "pinyin": " ".join(syls),
                      "start": round(c.start, 2), "end": round(c.end, 2),
                      "n_frames": len(arr), "frames_per_unit": round(len(arr) / len(toks), 2)})

    mpath = os.path.join(args.out_dir, "manifest.jsonl")
    with open(mpath, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    total_s = sum(it["n_frames"] for it in items) / fps
    report = {"manifest": mpath, "n_samples": len(items),
              "total_seconds": round(total_s, 1),
              "yield_ratio": round(total_s / max(n_frames / fps, 1e-6), 3),
              "dropped": drop, "subtitle_stats": stats, "fps": fps,
              "source_video": os.path.abspath(args.video),
              "source_subtitles": os.path.abspath(args.subs),
              "source_record": os.path.abspath(args.source_record)
                               if args.source_record else None}
    with open(os.path.join(args.out_dir, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("subs", help="YouTube json3 或 srt")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-record", default="",
                    help="data/raw/<id>/source.json；写入 build_report 形成可追溯链")
    ap.add_argument("--asd-json", default="", help="asd_pipeline.py 的输出，多人视频必需")
    ap.add_argument("--clip-start", type=float, default=0.0,
                    help="本视频在完整片源中的起始秒数（切片时用）")
    ap.add_argument("--roi-size", type=int, default=96)
    ap.add_argument("--roi-scale", type=float, default=1.6)
    ap.add_argument("--av-offset-frames", type=int, default=0,
                    help="音画偏移补偿(帧)，由 syncnet_offset.py 测得；负值=音频滞后")
    ap.add_argument("--pad-frames", type=int, default=0,
                    help="每条样本首尾各留的余量帧，吸收字幕边界与残余偏移")
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--min-face-cov", type=float, default=0.9)
    ap.add_argument("--min-speaker-cov", type=float, default=0.9)
    ap.add_argument("--min-units", type=int, default=4)
    ap.add_argument("--max-units", type=int, default=30)
    ap.add_argument("--keep-unknown-latin", action="store_true")
    ap.add_argument("--prefix", default="utt")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    r = build(args)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
