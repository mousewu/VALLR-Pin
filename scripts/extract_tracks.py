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
import math
import os
import sys
from typing import Iterable

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


def face_geometry(points: np.ndarray) -> dict:
    """Return scale-invariant geometry used for multi-face association.

    ``lip_opening`` is normalized by face extent so a close-up face does not
    automatically look more active than a smaller face in the same scene.
    """
    ref = np.asarray(points)[FACE_OVAL]
    if not np.isfinite(ref).all():
        raise ValueError("face oval contains invalid landmarks")
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    extent = float(max(*(hi - lo), 1.0))
    center = (lo + hi) / 2.0
    opening = float(np.linalg.norm(points[13] - points[14]) / extent)
    lip_width = float(np.linalg.norm(points[61] - points[291]))
    left_eye = (points[33] + points[133]) / 2.0
    right_eye = (points[362] + points[263]) / 2.0
    eye_span = max(float(np.linalg.norm(left_eye - right_eye)), 1.0)
    yaw_proxy = abs(float(points[1, 0] - (left_eye[0] + right_eye[0]) / 2.0)) / eye_span
    return {"center": center, "extent": extent,
            "area": float(np.prod(np.maximum(hi - lo, 1.0))),
            "lip_opening": opening, "lip_width": lip_width,
            "yaw_proxy": yaw_proxy}


def associate_face_detections(
        detections: Iterable[tuple[int, list[np.ndarray]]], max_gap: int = 5,
        max_cost: float = 2.0) -> list[dict]:
    """Greedily associate per-frame landmarks into stable face tracklets.

    MediaPipe does not promise that face index 0 denotes the same person on
    every frame.  Association by normalized center displacement and scale
    change prevents the common silent identity-switch failure.
    """
    tracks: list[dict] = []
    for frame_no, faces in detections:
        candidates = [(np.asarray(points, dtype=np.float32),
                       face_geometry(np.asarray(points))) for points in faces]
        pairs = []
        for track_index, track in enumerate(tracks):
            gap = frame_no - track["frames"][-1]
            if gap <= 0 or gap > max_gap:
                continue
            previous = track["geometry"][-1]
            for detection_index, (_, current) in enumerate(candidates):
                scale = max((previous["extent"] + current["extent"]) / 2.0, 1.0)
                distance = float(np.linalg.norm(
                    previous["center"] - current["center"]) / scale)
                scale_change = abs(math.log(max(current["extent"], 1.0) /
                                            max(previous["extent"], 1.0)))
                cost = distance + 0.4 * scale_change + 0.03 * (gap - 1)
                if cost <= max_cost:
                    pairs.append((cost, track_index, detection_index))
        used_tracks, used_detections = set(), set()
        for _, track_index, detection_index in sorted(pairs):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            points, geometry = candidates[detection_index]
            track = tracks[track_index]
            track["frames"].append(frame_no)
            track["landmarks"].append(points)
            track["geometry"].append(geometry)
            used_tracks.add(track_index); used_detections.add(detection_index)
        for detection_index, (points, geometry) in enumerate(candidates):
            if detection_index not in used_detections:
                tracks.append({"id": len(tracks), "frames": [frame_no],
                               "landmarks": [points], "geometry": [geometry]})
    return tracks


def score_face_tracks(tracklets: list[dict], n_total: int, width: int, height: int,
                      strategy: str = "active") -> list[dict]:
    """Score tracks for a raw scene or an already face-cropped clip."""
    if strategy not in {"active", "largest", "first"}:
        raise ValueError(f"unknown face selection strategy: {strategy}")
    raw = []
    image_center = np.array([width / 2.0, height / 2.0])
    diagonal = max(float(np.hypot(width, height)), 1.0)
    for track in tracklets:
        geometries = track["geometry"]
        openings = np.asarray([row["lip_opening"] for row in geometries])
        frames = track["frames"]
        changes = [abs(float(openings[i] - openings[i - 1]))
                   for i in range(1, len(openings)) if frames[i] - frames[i - 1] <= 2]
        mouth_motion = float(np.median(changes)) if changes else 0.0
        area = float(np.median([row["area"] for row in geometries]))
        lip_width = float(np.median([row["lip_width"] for row in geometries]))
        yaw_proxy = float(np.median([row["yaw_proxy"] for row in geometries]))
        center = np.median(np.stack([row["center"] for row in geometries]), axis=0)
        center_score = max(0.0, 1.0 - 2.0 * float(
            np.linalg.norm(center - image_center)) / diagonal)
        raw.append({"id": int(track["id"]), "frames": len(frames),
                    "coverage": len(frames) / max(n_total, 1), "area": area,
                    "mouth_motion": mouth_motion, "center_score": center_score,
                    "median_lip_width_px": lip_width,
                    "median_yaw_proxy": yaw_proxy})
    max_area = max((row["area"] for row in raw), default=1.0)
    max_motion = max((row["mouth_motion"] for row in raw), default=0.0)
    for row in raw:
        area_score = row["area"] / max(max_area, 1.0)
        motion_score = (row["mouth_motion"] / max_motion if max_motion > 1e-7 else 0.0)
        if strategy == "active":
            score = (0.50 * row["coverage"] + 0.35 * motion_score +
                     0.10 * area_score + 0.05 * row["center_score"])
        elif strategy == "largest":
            score = (0.65 * row["coverage"] + 0.30 * area_score +
                     0.05 * row["center_score"])
        else:
            score = 1.0 if row["id"] == 0 else 0.0
        row.update({"area_score": area_score, "motion_score": motion_score,
                    "score": float(score)})
    return sorted(raw, key=lambda row: (-row["score"], row["id"]))


def select_face_track(tracklets: list[dict], n_total: int, width: int, height: int,
                      strategy: str = "active", track_id: int = -1) -> tuple[dict, list[dict], float]:
    if not tracklets:
        raise ValueError("no face tracks")
    scores = score_face_tracks(tracklets, n_total, width, height, strategy)
    if track_id >= 0:
        selected = next((track for track in tracklets if track["id"] == track_id), None)
        if selected is None:
            raise ValueError(f"requested face_track_id={track_id} not found")
    else:
        selected_id = scores[0]["id"]
        selected = next(track for track in tracklets if track["id"] == selected_id)
    ranked = {row["id"]: index for index, row in enumerate(scores)}
    selected_rank = ranked[selected["id"]]
    selected_score = scores[selected_rank]["score"]
    competitors = [row["score"] for row in scores if row["id"] != selected["id"]]
    margin = selected_score - max(competitors) if competitors else 1.0
    return selected, scores, float(margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-faces", type=int, default=1)
    ap.add_argument("--selection", choices=["active", "largest", "first"],
                    default="largest")
    ap.add_argument("--track-id", type=int, default=-1,
                    help="人工指定关联后的轨迹 ID；默认按 selection 自动选择")
    ap.add_argument("--max-track-gap", type=int, default=5)
    ap.add_argument("--min-selection-margin", type=float, default=0.0,
                    help="多脸自动选择的最小分数差；低于该值拒绝歧义样本")
    ap.add_argument("--min-lip-width-px", type=float, default=0.0,
                    help="所选轨迹的原始画面唇宽中位数下限；0 表示关闭")
    ap.add_argument("--max-yaw-proxy", type=float, default=0.0,
                    help="所选轨迹侧脸代理值上限；0 表示只记录不拒绝")
    ap.add_argument("--input-type", choices=["raw_scene", "face_crop"],
                    default="raw_scene")
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

    detections, i, n_total = [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n_total += 1
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), int(i * 1000 / fps))
        faces = [np.array([[q.x * W, q.y * H] for q in landmarks], dtype=np.float32)
                 for landmarks in res.face_landmarks]
        detections.append((i, faces))
        i += 1
    cap.release()
    if hasattr(landmarker, "close"):
        landmarker.close()
    tracklets = associate_face_detections(detections, max_gap=args.max_track_gap)
    if not tracklets:
        raise SystemExit("未检出人脸")

    selected, scores, margin = select_face_track(
        tracklets, n_total, W, H, args.selection, args.track_id)
    selected_summary = next(row for row in scores if row["id"] == selected["id"])
    if args.track_id < 0 and len(scores) > 1 and margin < args.min_selection_margin:
        raise SystemExit(
            f"多人脸轨迹选择存在歧义: margin={margin:.4f} < "
            f"{args.min_selection_margin:.4f}; 可在 manifest 设置 face_track_id 人工覆盖")
    if selected_summary["median_lip_width_px"] < args.min_lip_width_px:
        raise SystemExit(
            f"原始唇宽不足: {selected_summary['median_lip_width_px']:.1f}px < "
            f"{args.min_lip_width_px:.1f}px")
    if args.max_yaw_proxy > 0 and selected_summary["median_yaw_proxy"] > args.max_yaw_proxy:
        raise SystemExit(
            f"侧脸程度过大: yaw_proxy={selected_summary['median_yaw_proxy']:.3f} > "
            f"{args.max_yaw_proxy:.3f}")
    frames = selected["frames"]
    lms = selected["landmarks"]
    arr = np.stack([p[KEY_POINTS] for p in lms] if args.keep_subset else lms)

    track = FaceTrack(np.array(frames), arr, fps, W, H,
                      meta={"video": os.path.basename(args.video),
                            "n_total_frames": n_total,
                            "subset": bool(args.keep_subset),
                            "landmark_count": 478,
                            "input_type": args.input_type,
                            "selection_strategy": args.selection,
                            "selected_track_id": int(selected["id"]),
                            "selection_margin": margin,
                            "face_track_count": len(tracklets),
                            "median_lip_width_px": selected_summary["median_lip_width_px"],
                            "median_yaw_proxy": selected_summary["median_yaw_proxy"],
                            "track_scores": scores},
                      point_indices=(np.array(KEY_POINTS, dtype=np.int16)
                                     if args.keep_subset else None))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    track.save(args.out)
    size = os.path.getsize(args.out)
    print(f"[tracks] {len(frames)}/{n_total} 帧检出人脸  {W}x{H}@{fps:.0f}fps")
    print(f"         selected track={selected['id']} strategy={args.selection} "
          f"candidates={len(tracklets)} margin={margin:.3f}")
    print(f"         -> {args.out}  {size / 1024:.0f} KB "
          f"({size / max(len(frames), 1):.0f} 字节/帧, "
          f"约 {size / max(len(frames), 1) * fps * 3600 / 1e6:.0f} MB/小时)")


if __name__ == "__main__":
    main()
