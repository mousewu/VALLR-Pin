#!/usr/bin/env python3
"""抽取人脸轨迹缓存 —— 整条数据管线里**唯一需要长期保存的中间产物**。

人脸检测 + 关键点是最贵的一步（约 5× 实时），而且结果与"要裁多大、灰度还是彩色"
这类模型相关的决策完全无关。把它单独存下来，之后换任何模型都只需重新渲染，
不必再跑一遍检测。

缓存很小：478 点 × xy × float16 ≈ 1.9 KB/帧，25fps 下约 170 MB/小时；
用 ``--keep-subset`` 只留必要的点可以再小一个数量级。

用法::

    python scripts/extract_tracks.py video.mp4 --model face_landmarker.task \\
        --out tracks/video.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vallr_pin.data.roi_spec import LIPS_OUTER, FaceTrack  # noqa: E402

# 只保留渲染真正用得到的点：嘴唇轮廓 + 脸部外轮廓(定人脸框) + 眼鼻(定姿态)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]
# 33/133 与 362/263 是双眼内外眼角；61/291 是嘴角。五点仿射规格依赖它们。
KEY_POINTS = sorted(set(LIPS_OUTER + FACE_OVAL +
                        [33, 133, 362, 263, 1, 13, 14, 61, 291]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-faces", type=int, default=1)
    ap.add_argument("--keep-subset", action="store_true",
                    help="只存渲染必需的关键点（体积小一个数量级，够用）")
    args = ap.parse_args()

    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.model),
            running_mode=vision.RunningMode.VIDEO, num_faces=args.max_faces))

    frames, lms, i, n_total = [], [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n_total += 1
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps))
        if res.face_landmarks:
            p = np.array([[q.x * W, q.y * H] for q in res.face_landmarks[0]],
                         dtype=np.float32)
            frames.append(i)
            lms.append(p[KEY_POINTS] if args.keep_subset else p)
        i += 1
    cap.release()
    if not frames:
        raise SystemExit("未检出人脸")

    arr = np.stack(lms)

    track = FaceTrack(np.array(frames), arr, fps, W, H,
                      meta={"video": os.path.basename(args.video),
                            "n_total_frames": n_total,
                            "subset": bool(args.keep_subset),
                            "landmark_count": 478},
                      point_indices=(np.array(KEY_POINTS, dtype=np.int16)
                                     if args.keep_subset else None))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    track.save(args.out)
    size = os.path.getsize(args.out)
    print(f"[tracks] {len(frames)}/{n_total} 帧检出人脸  {W}x{H}@{fps:.0f}fps")
    print(f"         -> {args.out}  {size / 1024:.0f} KB "
          f"({size / max(len(frames), 1):.0f} 字节/帧, "
          f"约 {size / max(len(frames), 1) * fps * 3600 / 1e6:.0f} MB/小时)")


if __name__ == "__main__":
    main()
