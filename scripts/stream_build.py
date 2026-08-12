#!/usr/bin/env python3
"""流式构建 VSR 样本并写 WebDataset shards。

与旧版 ``build_from_subtitles.py`` 不同，本脚本只保留“当前字幕句”的帧缓冲；
内存从约 1GB/视频小时降为几 MB，与视频总时长无关。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_from_subtitles import check_timing, load_cues  # noqa: E402
from vallr_pin.data.roi_spec import FaceTrack, render_frame, resolve_spec  # noqa: E402
from vallr_pin.data.shards import TarShardWriter  # noqa: E402
from vallr_pin.text.pinyin import text_to_pinyin_mixed  # noqa: E402


def build(args) -> dict:
    import cv2

    cues = load_cues(args.subtitles)
    stats = check_timing(cues)
    track = FaceTrack.load(args.tracks)
    spec = resolve_spec(args.spec)
    fps = track.fps
    lut = {int(f): i for i, f in enumerate(track.frames)}
    total_frames = int(track.meta.get("n_total_frames", 0))
    valid = []
    drops = {"bad_len": 0, "unknown_latin": 0, "no_face": 0, "short": 0}
    for idx, cue in enumerate(cues):
        f0 = round((cue.start - args.clip_start) * fps) + args.av_offset_frames - args.pad_frames
        f1 = round((cue.end - args.clip_start) * fps) + args.av_offset_frames + args.pad_frames
        if f1 < 0 or (total_frames and f0 >= total_frames):
            continue
        toks, syls, unknown = text_to_pinyin_mixed(cue.text)
        if not args.min_units <= len(toks) <= args.max_units:
            drops["bad_len"] += 1; continue
        if unknown and not args.keep_unknown_latin:
            drops["unknown_latin"] += 1; continue
        valid.append({"idx": idx, "cue": cue, "tokens": toks, "syls": syls,
                      "f0": max(0, f0), "f1": f1})

    # 非重叠字幕只会有一个 active cue；即使字幕少量重叠，缓冲数量也很小。
    starts: dict[int, list] = {}
    for row in valid:
        starts.setdefault(row["f0"], []).append(row)
    active: list[dict] = []
    manifest: list[dict] = []
    out = Path(args.out_dir); shards = out / "shards"; out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    writer = TarShardWriter(str(shards), args.shard_samples)
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for row in starts.get(frame_no, []):
            row["frames"] = []; row["expected"] = row["f1"] - row["f0"] + 1
            active.append(row)
        j = lut.get(frame_no)
        if j is not None:
            pts = track.points(j)
            for row in active:
                try:
                    row["frames"].append(render_frame(frame, pts, spec))
                except (ValueError, IndexError):
                    pass
        done = [x for x in active if frame_no >= x["f1"]]
        for row in done:
            active.remove(row)
            frames = row["frames"]
            if len(frames) < args.min_frames:
                drops["short"] += 1; continue
            if len(frames) / max(row["expected"], 1) < args.min_face_cov:
                drops["no_face"] += 1; continue
            arr = np.stack(frames)
            key = f"{args.prefix}_{row['idx']:06d}"
            meta = {"id": key, "text": "".join(row["tokens"]),
                    "pinyin": " ".join(row["syls"]), "n_frames": len(arr),
                    "start": row["cue"].start, "end": row["cue"].end,
                    "source_id": args.source_id, "speaker_id": args.speaker_id,
                    "spec": spec.name, "source_input_type": "raw_scene",
                    "input_type": "mouth_roi", "roi_type": "mouth",
                    "roi_spec": spec.name, "fps": float(fps),
                    "roi_height": int(arr.shape[1]), "roi_width": int(arr.shape[2]),
                    "roi_channels": (1 if arr.ndim == 3 else int(arr.shape[3]))}
            # URI 相对 out_dir，DataLoader(root=out_dir) 可直接读取。
            uri = writer.write(key, arr, meta)
            meta["video"] = "wds://shards/" + uri.split("wds://", 1)[1]
            manifest.append(meta)
        frame_no += 1
    cap.release()
    shard_paths = writer.close()
    manifest_path = out / "manifest.jsonl.partial"
    manifest_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                                     for x in manifest), encoding="utf-8")
    os.replace(manifest_path, out / "manifest.jsonl")
    report = {"samples": len(manifest), "frames_read": frame_no,
              "memory_mode": "streaming-current-cue", "shards": shard_paths,
              "drops": drops, "subtitle_stats": stats, "spec": spec.to_dict(),
              "source_id": args.source_id}
    (out / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("subtitles"); ap.add_argument("tracks")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--spec", default="vallr_pin")
    ap.add_argument("--source-id", required=True); ap.add_argument("--speaker-id", default="")
    ap.add_argument("--prefix", default="utt"); ap.add_argument("--clip-start", type=float, default=0)
    ap.add_argument("--av-offset-frames", type=int, default=0); ap.add_argument("--pad-frames", type=int, default=2)
    ap.add_argument("--shard-samples", type=int, default=1000)
    ap.add_argument("--min-frames", type=int, default=8); ap.add_argument("--min-face-cov", type=float, default=.9)
    ap.add_argument("--min-units", type=int, default=4); ap.add_argument("--max-units", type=int, default=30)
    ap.add_argument("--keep-unknown-latin", action="store_true")
    args = ap.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
