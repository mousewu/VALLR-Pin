#!/usr/bin/env python3
"""用 SyncNet 标定音画同步偏移，并给出"这张脸在不在说话"的置信度。

背景：手工特征（张嘴幅度 × 音频包络互相关）在真实素材上测不出可靠偏移 ——
实测相关系数只有 0.1 量级、峰值随特征选择漂移，只能给出"没有大幅错位"这种弱结论。
SyncNet 是学出来的音视频联合嵌入，同步/错位在嵌入空间可分，才能精确到帧。

模型来自 Chung & Zisserman, *Out of Time: Automated Lip Sync in the Wild* (ACCV 2016)。
需要官方权重 ``syncnet_v2.model`` 与仓库里的 ``SyncNetModel.py``::

    git clone --depth 1 https://github.com/joonson/syncnet_python
    curl -L -o syncnet_v2.model \\
        http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model

**输入规格必须严格对齐训练分布，否则结果无意义**：

* 视频 **25 fps**（本脚本自动重采样；30fps 素材直接喂进去会因时间尺度错配而失效）
* 音频 16 kHz 单声道
* 人脸裁剪：以人脸框半径 ``bs`` 为基准裁 ``2.8·bs`` 的方形，
  **垂直中心下移 0.4·bs**（对准嘴部而非眼睛），缩放到 224×224，BGR，不做归一化
* 每 5 帧视频配 20 帧 MFCC（13 维，100 fps）

因为 torch 与 mediapipe 常装在不同环境，脚本拆成两步::

    <venv-python>  scripts/syncnet_offset.py extract clip.mp4 --model face_landmarker.task --out work/
    <torch-python> scripts/syncnet_offset.py measure work/ --syncnet-dir syncnet_python --weights syncnet_v2.model

注意：官方用 S3FD 检测人脸，本脚本用 mediapipe 关键点反推人脸框，两者框的松紧
略有差异，属于可接受的分布偏移。``--inject-shift`` 可注入已知偏移做自检，
用来确认整条测量链路和符号约定都是对的。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

TARGET_FPS = 25
SR = 16000
CROP_SCALE = 0.40          # 与官方 run_pipeline.py 的 --crop_scale 默认值一致
CROP_SIZE = 224


# --------------------------------------------------------------------------- #
#                          第一步：抽人脸裁剪 + 音频                              #
# --------------------------------------------------------------------------- #
def extract(args) -> Dict:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    os.makedirs(args.out, exist_ok=True)
    v25 = os.path.join(args.out, "video25.mp4")
    wav = os.path.join(args.out, "audio.wav")
    # 官方 pipeline 也是先转 25fps 再处理；-async 1 保证音频不被重采样拉伸
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.video,
                    "-qscale:v", "2", "-async", "1", "-r", str(TARGET_FPS), v25],
                   check=True)
    af = ["ffmpeg", "-y", "-loglevel", "error", "-i", v25]
    if args.inject_shift:
        # 正值 = 音频整体延后 N 帧，用来自检符号约定
        delay_ms = abs(args.inject_shift) * 1000.0 / TARGET_FPS
        if args.inject_shift > 0:
            af += ["-af", f"adelay={delay_ms:.0f}|{delay_ms:.0f}"]
        else:
            af += ["-af", f"atrim=start={delay_ms / 1000.0:.4f},asetpts=PTS-STARTPTS"]
    af += ["-ac", "1", "-ar", str(SR), wav]
    subprocess.run(af, check=True)

    cap = cv2.VideoCapture(v25)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))

    centers: List[Optional[Tuple[float, float, float]]] = []
    frames = []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)),
            int(i * 1000 / TARGET_FPS))
        if res.face_landmarks:
            p = np.array([[q.x * W, q.y * H] for q in res.face_landmarks[0]])
            x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
            centers.append(((x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2))
        else:
            centers.append(None)
        i += 1
    cap.release()

    # 官方对检测框做了中值滤波以抑制抖动，这里照做
    valid = [c for c in centers if c is not None]
    if len(valid) < 25:
        raise SystemExit("有效人脸帧太少，无法标定")
    arr = np.array([c if c is not None else (np.nan,) * 3 for c in centers], dtype=float)
    for k in range(3):
        col = arr[:, k]
        idx = np.arange(len(col))
        good = ~np.isnan(col)
        col = np.interp(idx, idx[good], col[good])
        from scipy.signal import medfilt
        arr[:, k] = medfilt(col, kernel_size=13)

    crops = np.zeros((len(frames), CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    for i, fr in enumerate(frames):
        mx, my, bs = arr[i]
        bsi = int(bs * (1 + 2 * CROP_SCALE))
        pad = cv2.copyMakeBorder(fr, bsi, bsi, bsi, bsi, cv2.BORDER_CONSTANT, value=(110, 110, 110))
        cy, cx = my + bsi, mx + bsi
        face = pad[int(cy - bs): int(cy + bs * (1 + 2 * CROP_SCALE)),
                   int(cx - bs * (1 + CROP_SCALE)): int(cx + bs * (1 + CROP_SCALE))]
        if face.size:
            crops[i] = cv2.resize(face, (CROP_SIZE, CROP_SIZE))
    np.save(os.path.join(args.out, "crops.npy"), crops)
    meta = {"n_frames": len(frames), "fps": TARGET_FPS, "wav": wav,
            "inject_shift": args.inject_shift, "face_frames": int(len(valid))}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[extract] {len(frames)} 帧 -> crops.npy ({crops.shape}), "
          f"人脸检出 {len(valid)} 帧, 注入偏移={args.inject_shift}", flush=True)
    return meta


# --------------------------------------------------------------------------- #
#                            第二步：跑 SyncNet                                  #
# --------------------------------------------------------------------------- #
def measure(args) -> Dict:
    import python_speech_features
    import torch
    from scipy.io import wavfile

    sys.path.insert(0, args.syncnet_dir)
    from SyncNetModel import S as SyncNetModel      # noqa: N811

    crops = np.load(os.path.join(args.work, "crops.npy"))
    meta = json.load(open(os.path.join(args.work, "meta.json")))
    sr, audio = wavfile.read(os.path.join(args.work, "audio.wav"))
    assert sr == SR, f"音频必须 {SR}Hz，实际 {sr}"

    model = SyncNetModel(num_layers_in_fc_layers=1024)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict({k.replace("module.", "").replace("__S__.", ""): v
                           for k, v in state.items()})
    model.eval()

    # (N,224,224,3) -> (1,3,N,224,224)，保持官方的 0-255 原始尺度，不做归一化
    im = torch.from_numpy(crops.astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0)
    mfcc = np.stack([np.array(c) for c in zip(*python_speech_features.mfcc(audio, sr))])
    cct = torch.from_numpy(mfcc.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    n = min(len(crops), int(np.floor(len(audio) / 640)))
    last = n - 5
    if last <= 0:
        raise SystemExit("片段太短")
    im_feat, cc_feat = [], []
    with torch.no_grad():
        for i in range(0, last, args.batch_size):
            js = range(i, min(last, i + args.batch_size))
            im_feat.append(model.forward_lip(
                torch.cat([im[:, :, j:j + 5] for j in js], 0)))
            cc_feat.append(model.forward_aud(
                torch.cat([cct[:, :, :, j * 4:j * 4 + 20] for j in js], 0)))
    im_feat = torch.cat(im_feat, 0)
    cc_feat = torch.cat(cc_feat, 0)

    vshift = args.vshift
    win = vshift * 2 + 1
    padded = torch.nn.functional.pad(cc_feat, (0, 0, vshift, vshift))
    dists = [torch.nn.functional.pairwise_distance(
        im_feat[[i], :].repeat(win, 1), padded[i:i + win, :]) for i in range(len(im_feat))]
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)
    offset = int(vshift - minidx.item())
    conf = float(torch.median(mdist) - minval)

    curve = {int(vshift - k): round(float(mdist[k]), 3) for k in range(win)}
    out = {"offset_frames": offset, "offset_ms": round(1000 * offset / TARGET_FPS, 1),
           "confidence": round(conf, 3), "min_distance": round(float(minval), 3),
           "median_distance": round(float(torch.median(mdist)), 3),
           "n_windows": len(im_feat), "injected_shift": meta.get("inject_shift", 0),
           "distance_curve": curve}
    with open(os.path.join(args.work, "syncnet.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "distance_curve"},
                     ensure_ascii=False, indent=2))
    print("距离曲线 (越小越同步):")
    for k in sorted(curve):
        bar = "#" * int(40 * (curve[k] - min(curve.values())) /
                        max(max(curve.values()) - min(curve.values()), 1e-6))
        mark = "  <-- 最优" if k == offset else ""
        print(f"  offset={k:>3}帧 ({1000 * k / TARGET_FPS:>6.0f}ms) d={curve[k]:.3f} {bar}{mark}")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="转 25fps + 抽 224x224 人脸裁剪 + 16k 音频")
    e.add_argument("video")
    e.add_argument("--model", required=True, help="mediapipe face_landmarker.task")
    e.add_argument("--out", required=True)
    e.add_argument("--inject-shift", type=int, default=0,
                   help="注入已知偏移(帧)做自检；正值=音频延后")
    e.set_defaults(func=extract)

    m = sub.add_parser("measure", help="跑 SyncNet 得到偏移与置信度")
    m.add_argument("work")
    m.add_argument("--syncnet-dir", required=True, help="syncnet_python 仓库路径")
    m.add_argument("--weights", required=True, help="syncnet_v2.model")
    m.add_argument("--vshift", type=int, default=15)
    m.add_argument("--batch-size", type=int, default=20)
    m.set_defaults(func=measure)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
