#!/usr/bin/env python3
"""素材体检：判断一段视频值不值得进 VSR 数据集。

跑一遍输出四类结论：

1. **人脸/关键点覆盖率** —— 有多少帧检测得到脸；多人场景下每条人脸轨迹各占多少帧
2. **唇部 ROI 质量** —— ROI 像素尺寸、逐帧抖动 (相邻帧中心位移)、正脸程度
3. **音画同步偏移（粗测，经常失效）** —— 用"张嘴幅度"与"音频包络"在音节带 (2-8Hz)
   上的互相关估计 A/V offset。实测在播客类素材上**基本给不出可信结论**：这类音频
   普遍经过压缩/限幅，包络的音节结构被压平，相关系数只有 0.05 量级、峰值随特征
   选择漂移。因此当峰值相关系数低于 ``--sync-min-corr`` 或峰值不够突出时，
   本脚本直接报 ``reliable: false``，不要拿这个数去做偏移校正 —— 那种精度只有
   SyncNet 这类学出来的音视频嵌入模型能给。
4. **产物** —— 每条人脸轨迹的 ROI npy + 预览拼图，可直接接 manifest

用法::

    python scripts/probe_video.py clip.mp4 --out-dir probe_out --model face_landmarker.task

依赖 opencv-python + mediapipe (>=0.10)，建议装在独立 venv 里。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# MediaPipe FaceMesh 的嘴唇轮廓点 (外轮廓 + 内轮廓的代表点)
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]
UPPER_INNER, LOWER_INNER = 13, 14      # 内唇中点，用来量张嘴幅度
LEFT_EYE, RIGHT_EYE, NOSE = 33, 263, 1


@dataclass
class Track:
    """一条人脸轨迹 (按帧连续的同一个人)。"""
    tid: int
    frames: List[int] = field(default_factory=list)
    centers: List[Tuple[float, float]] = field(default_factory=list)
    sizes: List[float] = field(default_factory=list)
    apertures: List[float] = field(default_factory=list)
    yaws: List[float] = field(default_factory=list)
    rois: List[np.ndarray] = field(default_factory=list)


def audio_envelope(video: str, sr: int = 16000) -> np.ndarray:
    """用 ffmpeg 抽单声道 PCM，返回短时能量包络 (逐样本)。"""
    cmd = ["ffmpeg", "-v", "quiet", "-i", video, "-f", "s16le", "-ac", "1",
           "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x


def envelope_at_fps(x: np.ndarray, sr: int, fps: float, n_frames: int) -> np.ndarray:
    """把波形转成与视频帧对齐的能量包络。"""
    hop = sr / fps
    out = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        a, b = int(i * hop), int((i + 1) * hop)
        seg = x[a:b]
        out[i] = float(np.sqrt((seg ** 2).mean())) if seg.size else 0.0
    return out


def bandpass(v: np.ndarray, lo_hz: float, hi_hz: float, fps: float) -> np.ndarray:
    """保留音节律动频段 (默认 2-8Hz)，去掉直流与长期漂移。"""
    from numpy.fft import irfft, rfft, rfftfreq
    V = rfft(v - v.mean())
    f = rfftfreq(len(v), 1.0 / fps)
    V[(f < lo_hz) | (f > hi_hz)] = 0
    return irfft(V, n=len(v))


def xcorr_curve(a: np.ndarray, b: np.ndarray, max_lag: int) -> Dict[int, float]:
    """全部 lag 上的相关系数；正 lag 表示 a 相对 b 滞后。"""
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[: len(b) - lag]
        else:
            x, y = a[: len(a) + lag], b[-lag:]
        n = min(len(x), len(y))
        if n >= 10:
            out[lag] = float((x[:n] * y[:n]).mean())
    return out


def crop_roi(frame_gray: np.ndarray, pts: np.ndarray, size: int, scale: float = 1.6
             ) -> Optional[np.ndarray]:
    import cv2
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    w = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
    half = max(int(w * scale / 2), 8)
    h_img, w_img = frame_gray.shape
    x0, y0 = int(cx - half), int(cy - half)
    x1, y1 = int(cx + half), int(cy + half)
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, w_img), min(y1, h_img)
    patch = frame_gray[y0c:y1c, x0c:x1c]
    if patch.size == 0:
        return None
    if (x0, y0, x1, y1) != (x0c, y0c, x1c, y1c):     # 越界补边，避免 ROI 变形
        patch = cv2.copyMakeBorder(patch, y0c - y0, y1 - y1c, x0c - x0, x1 - x1c,
                                   cv2.BORDER_REPLICATE)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)


def run(args) -> Dict:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.VIDEO, num_faces=args.max_faces))

    tracks: Dict[int, Track] = {}
    prev_centers: Dict[int, Tuple[float, float]] = {}   # tid -> (cx, cy, last_frame)
    n_frames, n_with_face = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx = n_frames
        n_frames += 1
        if args.max_frames and n_frames > args.max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(mp_img, int(idx * 1000 / fps))
        if not res.face_landmarks:
            continue
        n_with_face += 1

        used_this_frame: set = set()   # 帧内互斥：一个 tid 只能被一个检测认领
        for lms in res.face_landmarks:
            pts = np.array([[p.x * W, p.y * H] for p in lms], dtype=np.float32)
            lip = pts[LIPS_OUTER]
            cx, cy = float(lip[:, 0].mean()), float(lip[:, 1].mean())
            face_w = float(pts[:, 0].max() - pts[:, 0].min())
            # 轨迹关联：最近中心点 + 容忍短暂丢失 (转头、快速运动会漏检几帧)。
            # 不容忍断帧的话，同一个人会被切成多条 ID，后面的说话人划分就全乱了。
            tid, best_d = None, 1e9
            for k, (px, py, last_f) in prev_centers.items():
                if idx - last_f > args.track_gap or k in used_this_frame:
                    continue
                d = (cx - px) ** 2 + (cy - py) ** 2
                if d < best_d:
                    tid, best_d = k, d
            if tid is None or best_d > (face_w * args.track_dist) ** 2:
                # 中心点跳变超过阈值 = 镜头切换或换人，起一条新轨迹。
                # 归并"同一个人的不同机位"需要人脸 embedding 聚类，本脚本不做。
                tid = max([*prev_centers, -1]) + 1
            used_this_frame.add(tid)
            prev_centers[tid] = (cx, cy, idx)
            t = tracks.setdefault(tid, Track(tid))

            aperture = float(abs(pts[LOWER_INNER][1] - pts[UPPER_INNER][1]) / max(face_w, 1))
            # 用鼻尖相对双眼中点的水平偏移粗估偏航角
            eye_mid = (pts[LEFT_EYE] + pts[RIGHT_EYE]) / 2
            eye_d = float(np.linalg.norm(pts[LEFT_EYE] - pts[RIGHT_EYE])) + 1e-6
            yaw = float((pts[NOSE][0] - eye_mid[0]) / eye_d)

            t.frames.append(idx)
            t.centers.append((cx, cy))
            t.sizes.append(face_w)
            t.apertures.append(aperture)
            t.yaws.append(yaw)
            roi = crop_roi(gray, lip, args.roi_size, args.roi_scale)
            if roi is not None:
                t.rois.append(roi)
    cap.release()

    # ---- 音画同步：张嘴幅度 vs 音频能量包络 ----
    main = max(tracks.values(), key=lambda t: len(t.frames)) if tracks else None
    sync = {}
    if main and len(main.frames) > 30:
        env = envelope_at_fps(audio_envelope(args.video), 16000, fps, n_frames)
        # 缺帧用插值而不是补零，补零会在信号里造出假的高频成分
        ap = np.full(n_frames, np.nan, dtype=np.float32)
        for f, a in zip(main.frames, main.apertures):
            if f < n_frames:
                ap[f] = a
        idx = np.arange(n_frames)
        ok = ~np.isnan(ap)
        ap = np.interp(idx, idx[ok], ap[ok])
        lo, hi = min(main.frames), max(main.frames) + 1
        curve = xcorr_curve(bandpass(ap[lo:hi], 2.0, 8.0, fps),
                            bandpass(env[lo:hi], 2.0, 8.0, fps), args.max_lag_frames)
        lag = max(curve, key=curve.get)
        r = curve[lag]
        second = max((v for k, v in curve.items() if abs(k - lag) > 1), default=0.0)
        reliable = r >= args.sync_min_corr and r - second >= 0.05
        sync = {"lag_frames": lag, "lag_ms": round(1000 * lag / fps, 1),
                "peak_correlation": round(r, 3), "runner_up": round(second, 3),
                "reliable": bool(reliable),
                "curve": {k: round(v, 3) for k, v in sorted(curve.items())},
                "note": ("峰值显著，可作偏移参考" if reliable else
                         "相关太弱/峰值不突出 —— 本方法在该素材上无效，不要据此校正；"
                         "需要 SyncNet 类音视频嵌入模型")}

    os.makedirs(args.out_dir, exist_ok=True)
    report = {"video": args.video, "fps": round(fps, 2), "resolution": f"{W}x{H}",
              "frames_read": n_frames, "frames_total_header": n_total,
              "face_coverage": round(n_with_face / max(n_frames, 1), 3),
              "n_tracks": len(tracks), "av_sync": sync, "tracks": []}

    for t in sorted(tracks.values(), key=lambda x: -len(x.frames)):
        if len(t.frames) < args.min_track_len:
            continue
        c = np.array(t.centers)
        jitter = float(np.linalg.norm(np.diff(c, axis=0), axis=1).mean()) if len(c) > 1 else 0.0
        info = {"track": t.tid, "n_frames": len(t.frames),
                "coverage": round(len(t.frames) / max(n_frames, 1), 3),
                "lip_width_px": round(float(np.mean(t.sizes)) * 0.35, 1),
                "face_width_px": round(float(np.mean(t.sizes)), 1),
                "roi_jitter_px_per_frame": round(jitter, 2),
                "abs_yaw_mean": round(float(np.mean(np.abs(t.yaws))), 3),
                "aperture_std": round(float(np.std(t.apertures)), 4)}
        if t.rois:
            arr = np.stack(t.rois)
            path = os.path.join(args.out_dir, f"track{t.tid}_roi.npy")
            np.save(path, arr)
            info["roi_npy"] = path
            info["roi_shape"] = list(arr.shape)
            n = min(args.preview, len(arr))
            step = max(len(arr) // n, 1)
            sheet = np.concatenate([arr[i] for i in range(0, step * n, step)], axis=1)
            ppath = os.path.join(args.out_dir, f"track{t.tid}_preview.png")
            cv2.imwrite(ppath, sheet)
            info["preview_png"] = ppath
        report["tracks"].append(info)

    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out-dir", default="probe_out")
    ap.add_argument("--model", required=True, help="mediapipe face_landmarker.task 路径")
    ap.add_argument("--roi-size", type=int, default=96)
    ap.add_argument("--roi-scale", type=float, default=1.6)
    ap.add_argument("--max-faces", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--min-track-len", type=int, default=15)
    ap.add_argument("--max-lag-frames", type=int, default=12)
    ap.add_argument("--sync-min-corr", type=float, default=0.15)
    ap.add_argument("--track-gap", type=int, default=15, help="轨迹允许中断的最大帧数")
    ap.add_argument("--track-dist", type=float, default=0.8, help="关联阈值，单位=脸宽")
    ap.add_argument("--preview", type=int, default=16)
    args = ap.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
