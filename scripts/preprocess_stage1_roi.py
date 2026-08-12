#!/usr/bin/env python3
"""Normalize heterogeneous Stage-I media into one strict mouth-ROI manifest.

``raw_scene`` (CMLR-style) clips use multi-face tracking plus an active-face
selector. ``face_crop`` (CN-CVS-style) clips skip scene-level identity search
but still run landmarks and mouth cropping. ``mouth_roi`` inputs are validated
and passed through without recomputing landmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.render_variant import render  # noqa: E402
from vallr_pin.data.dataset import read_manifest  # noqa: E402
from vallr_pin.data.roi_spec import (FaceTrack, build_wflw98_track,  # noqa: E402
                                     resolve_spec)

INPUT_TYPES = {"raw_scene", "face_crop", "mouth_roi"}


def _key(item: dict) -> str:
    digest = hashlib.sha1(item["id"].encode()).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in item["id"])[-60:]
    return f"{stem}-{digest}"


def _component(value: str) -> str:
    value = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    return value or "unknown"


def _resolve_media(item: dict, data_root: str) -> str:
    value = str(item.get("video", ""))
    if value.startswith("wds://") or os.path.isabs(value):
        return value
    return os.path.join(data_root, value)


def _resolve_aux_path(value: str, data_root: str) -> str:
    if not value or os.path.isabs(value):
        return value
    return os.path.join(data_root, value)


def _video_info(path: str) -> tuple[float, int, int, int]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    cap.release()
    if fps <= 0 or width <= 0 or height <= 0 or frames <= 0:
        raise ValueError(
            f"invalid video metadata: {width}x{height}, fps={fps:g}, frames={frames}")
    return fps, width, height, frames


def _file_fingerprint(path: str) -> str:
    stat = os.stat(path)
    return f"{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime_ns}"


def _write_official_track(landmark_path: str, landmark_format: str,
                          video: str, track_path: Path) -> FaceTrack:
    if landmark_format not in {"wflw98", "wflw98_normalized"}:
        raise ValueError(f"unsupported landmark_format: {landmark_format}")
    fps, width, height, video_frames = _video_info(video)
    landmarks = np.load(landmark_path, mmap_mode="r")
    track = build_wflw98_track(
        landmarks, fps, width, height, video_frames,
        normalized=landmark_format.endswith("_normalized"),
        source=os.path.abspath(landmark_path))
    track.meta["landmark_fingerprint"] = _file_fingerprint(landmark_path)
    track.save(str(track_path))
    return track


def _shape_info(array: np.ndarray) -> tuple[int, int, int, int]:
    if array.ndim == 3:
        frames, height, width = array.shape
        channels = 1
    elif array.ndim == 4 and array.shape[-1] in (1, 3):
        frames, height, width, channels = array.shape
    else:
        raise ValueError(f"expected (T,H,W) or (T,H,W,C), got {array.shape}")
    return int(frames), int(height), int(width), int(channels)


def _mouth_roi_metadata(item: dict, path: str, target_fps: float,
                        expected_size: int) -> tuple[dict, dict]:
    if path.startswith("wds://") or not path.endswith(".npy"):
        raise ValueError("pre-existing mouth_roi must be a .npy file for validation")
    array = np.load(path, mmap_mode="r")
    frames, height, width, channels = _shape_info(array)
    if height != expected_size or width != expected_size or channels != 1:
        raise ValueError(
            f"mouth_roi must be T×{expected_size}×{expected_size} grayscale, got {array.shape}")
    fps = float(item.get("fps", 0) or 0)
    if fps <= 0:
        raise ValueError("pre-existing mouth_roi requires fps metadata")
    if abs(fps - target_fps) > 0.05:
        raise ValueError(
            f"pre-existing mouth_roi fps={fps:g}, expected {target_fps:g}; "
            "rebuild it from source video")
    output = dict(item)
    output.update({"video": str(Path(path).resolve()), "input_type": "mouth_roi",
                   "source_input_type": item.get("source_input_type", "mouth_roi"),
                   "roi_type": "mouth", "roi_spec": item.get("roi_spec", "external"),
                   "fps": fps, "roi_height": height, "roi_width": width,
                   "roi_channels": channels, "n_frames": frames,
                   "roi_coverage": float(item.get("roi_coverage", 1.0))})
    info = {"shape": list(array.shape), "dtype": str(array.dtype), "coverage": 1.0,
            "src_fps": fps, "target_fps": fps, "mode": "validated_existing_roi"}
    return output, info


def _cached_render_info(roi: Path, target_fps: float, expected_size: int,
                        max_interpolation_gap: int) -> dict | None:
    sidecar = Path(str(roi).replace(".npy", ".spec.json"))
    if not roi.exists() or not sidecar.exists():
        return None
    try:
        info = json.loads(sidecar.read_text(encoding="utf-8"))
        array = np.load(roi, mmap_mode="r")
        _, height, width, channels = _shape_info(array)
        actual_fps = float(info.get("target_fps", info.get("spec", {}).get("fps", 0)) or 0)
        if int(info.get("render_version", 0)) < 2:
            return None
        if int(info.get("max_interpolation_gap", -1)) != int(max_interpolation_gap):
            return None
        if (height, width, channels) != (expected_size, expected_size, 1):
            return None
        if abs(actual_fps - target_fps) > 0.05:
            return None
        info["shape"] = list(array.shape)
        return info
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def process(item: dict, output_root: str, model: str, min_coverage: float,
            keep_tracks: bool, target_fps: float, raw_scene_max_faces: int,
            raw_scene_selection: str, raw_scene_min_margin: float,
            face_crop_max_faces: int, min_lip_width_px: float,
            max_yaw_proxy: float, data_root: str,
            max_interpolation_gap: int = 5) -> tuple[dict | None, dict]:
    source = str(item.get("source", "unknown"))
    input_type = str(item.get("input_type", ""))
    key = _key(item)
    try:
        if input_type not in INPUT_TYPES:
            raise ValueError(
                "manifest item requires input_type=raw_scene, face_crop or mouth_roi")
        video = _resolve_media(item, data_root)
        if input_type == "mouth_roi":
            output, info = _mouth_roi_metadata(item, video, target_fps, expected_size=96)
            return output, {"id": item["id"], "source": source,
                            "input_type": input_type, "status": "ok", **info}
        if video.startswith("wds://") or not os.path.exists(video):
            raise FileNotFoundError(video)

        selection = raw_scene_selection if input_type == "raw_scene" else "largest"
        max_faces = raw_scene_max_faces if input_type == "raw_scene" else face_crop_max_faces
        min_margin = raw_scene_min_margin if input_type == "raw_scene" else 0.0
        manual_track = int(item.get("face_track_id", -1))
        landmark_path = _resolve_aux_path(str(item.get("landmark_path", "")), data_root)
        landmark_format = str(item.get("landmark_format", ""))
        if landmark_path and not landmark_format:
            raise ValueError("landmark_path requires landmark_format")
        use_official_landmarks = bool(
            input_type == "face_crop" and landmark_path and landmark_format)
        if landmark_path and not os.path.exists(landmark_path):
            raise FileNotFoundError(landmark_path)
        landmark_variant = landmark_format or "detected"
        track_variant = (f"{input_type}-{landmark_variant}-{selection}-"
                         f"f{max_faces}-t{manual_track}")
        base = Path(output_root) / _component(source)
        # Selection parameters are part of the derived-data identity.  A manual
        # face-track correction must never reuse an ROI rendered from an older
        # automatic choice.
        roi = base / "roi96" / track_variant / f"{key}.npy"
        track = base / "tracks" / track_variant / f"{key}.npz"
        roi.parent.mkdir(parents=True, exist_ok=True)
        track.parent.mkdir(parents=True, exist_ok=True)

        needs_extract = not track.exists()
        if not needs_extract:
            try:
                old_meta = FaceTrack.load(str(track)).meta
                needs_extract = not {"median_lip_width_px", "median_yaw_proxy"} <= old_meta.keys()
                if use_official_landmarks:
                    needs_extract = needs_extract or (
                        old_meta.get("landmark_fingerprint") !=
                        _file_fingerprint(landmark_path))
            except (OSError, ValueError, KeyError):
                needs_extract = True
        track_rebuilt = False
        if needs_extract:
            if use_official_landmarks:
                _write_official_track(landmark_path, landmark_format, video, track)
            else:
                if not model:
                    raise ValueError(
                        "face-model is required when official landmarks are unavailable")
                command = [
                    sys.executable, str(Path(__file__).with_name("extract_tracks.py")),
                    video, "--model", model, "--out", str(track), "--keep-subset",
                    "--input-type", input_type, "--selection", selection,
                    "--max-faces", str(max_faces), "--min-selection-margin",
                    str(min_margin), "--min-lip-width-px", str(min_lip_width_px),
                    "--max-yaw-proxy", str(max_yaw_proxy)]
                if manual_track >= 0:
                    command.extend(["--track-id", str(manual_track)])
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True)
            track_rebuilt = True

        cached_track = FaceTrack.load(str(track))
        lip_width = float(cached_track.meta.get("median_lip_width_px", 0.0))
        yaw_proxy = float(cached_track.meta.get("median_yaw_proxy", 0.0))
        if lip_width < min_lip_width_px:
            raise ValueError(
                f"original lip width {lip_width:.1f}px < required {min_lip_width_px:.1f}px")
        if max_yaw_proxy > 0 and yaw_proxy > max_yaw_proxy:
            raise ValueError(
                f"yaw_proxy {yaw_proxy:.3f} > required {max_yaw_proxy:.3f}")

        spec = replace(resolve_spec("vallr_pin"), fps=int(round(target_fps)))
        info = (None if track_rebuilt else
                _cached_render_info(roi, target_fps, spec.size, max_interpolation_gap))
        if info is None:
            info = render(video, str(track), spec, str(roi),
                          min_coverage=min_coverage,
                          max_interpolation_gap=max_interpolation_gap)

        shape = info["shape"]
        track_meta = info.get("track_meta", {})
        output = dict(item)
        output.update({"video": str(roi.resolve()), "source_input_type": input_type,
                       "input_type": "mouth_roi", "roi_type": "mouth",
                       "roi_spec": "vallr_pin", "fps": float(info["target_fps"]),
                       "source_fps": float(info["src_fps"]),
                       "roi_height": int(shape[1]), "roi_width": int(shape[2]),
                       "roi_channels": 1, "n_frames": int(shape[0]),
                       "roi_coverage": float(info["coverage"]),
                       "face_track_id": int(track_meta.get("selected_track_id", manual_track)),
                       "face_track_count": int(track_meta.get("face_track_count", 1)),
                       "face_selection": track_meta.get("selection_strategy", selection),
                       "face_selection_margin": float(track_meta.get("selection_margin", 1.0)),
                       "median_lip_width_px": float(
                           track_meta.get("median_lip_width_px", 0.0)),
                       "median_yaw_proxy": float(track_meta.get("median_yaw_proxy", 0.0))})
        if not keep_tracks and track.exists():
            track.unlink()
        return output, {"id": item["id"], "source": source,
                        "input_type": input_type, "status": "ok", **info}
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = f"track extraction failed (exit={exc.returncode})"
        if detail:
            message += f": {detail[-4000:]}"
        return None, {"id": item.get("id", ""), "source": source,
                      "input_type": input_type or "missing", "status": "rejected",
                      "error": message}
    except Exception as exc:
        return None, {"id": item.get("id", ""), "source": source,
                      "input_type": input_type or "missing", "status": "rejected",
                      "error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--face-model", default="",
                    help="无官方 landmark 的数据必需；纯 CN-CVS 官方包可省略")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-coverage", type=float, default=.95)
    ap.add_argument("--target-fps", type=float, default=25.0)
    ap.add_argument("--raw-scene-max-faces", type=int, default=5)
    ap.add_argument("--raw-scene-selection", choices=["active", "largest", "first"],
                    default="active")
    ap.add_argument("--raw-scene-min-margin", type=float, default=.05)
    ap.add_argument("--face-crop-max-faces", type=int, default=1)
    ap.add_argument("--min-lip-width-px", type=float, default=12.0)
    ap.add_argument("--max-yaw-proxy", type=float, default=0.0,
                    help="0 表示只记录侧脸指标，不自动拒绝")
    ap.add_argument("--max-interpolation-gap", type=int, default=5,
                    help="允许插值的最长连续关键点缺口；更长则拒绝样本")
    ap.add_argument("--discard-tracks", action="store_true")
    args = ap.parse_args()
    if abs(args.target_fps - round(args.target_fps)) > 1e-6:
        ap.error("target-fps currently must be an integer")
    items = read_manifest(args.manifest)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(
            process, item, args.out_root, args.face_model, args.min_coverage,
            not args.discard_tracks, args.target_fps, args.raw_scene_max_faces,
            args.raw_scene_selection, args.raw_scene_min_margin,
            args.face_crop_max_faces, args.min_lip_width_px,
            args.max_yaw_proxy, args.data_root,
            args.max_interpolation_gap) for item in items]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 100 == 0:
                print(f"[roi] {index}/{len(items)}", flush=True)
    good = [item for item, _ in results if item is not None]
    good.sort(key=lambda item: item["id"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_manifest)), exist_ok=True)
    with open(args.out_manifest + ".partial", "w", encoding="utf-8") as stream:
        for item in good:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(args.out_manifest + ".partial", args.out_manifest)
    by_type = Counter(report["input_type"] + ":" + report["status"]
                      for _, report in results)
    by_source = Counter(report["source"] + ":" + report["status"]
                        for _, report in results)
    report = {"input": len(items), "accepted": len(good),
              "rejected": len(items) - len(good), "target_fps": args.target_fps,
              "by_input_type": dict(by_type), "by_source": dict(by_source),
              "items": [row for _, row in results]}
    report_path = args.out_manifest.replace(".jsonl", ".roi_report.json")
    with open(report_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: report[key] for key in
                      ("input", "accepted", "rejected", "by_input_type")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
