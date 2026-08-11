#!/usr/bin/env python3
"""把原始视频登记成不可变 source record。

目录约定::

    data/raw/<video_id>/
      source.mp4             原视频（或明确标记的原始片段）
      subtitles.zh-Hans.json3
      source.json            URL、时间范围、SHA-256、ffprobe、版权备注
      tracks.npz             可选的人脸轨迹缓存

默认复制，不移动、不转码；已有同名且哈希不同的文件会拒绝覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    p = subprocess.run([
        "ffprobe", "-v", "error", "-show_format", "-show_streams",
        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def copy_immutable(src: Path, dst: Path) -> None:
    if src == dst.resolve():
        return
    if dst.exists():
        if sha256(src) != sha256(dst):
            raise FileExistsError(f"拒绝覆盖哈希不同的原始文件: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def register(video: str, out_root: str, video_id: str, url: str = "",
             subtitles: str = "", clip_start: float = 0.0,
             clip_end: float | None = None, complete: bool = True) -> dict:
    src = Path(video).resolve()
    root = Path(out_root).resolve() / video_id
    root.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower() or ".mp4"
    target = root / f"source{ext}"
    copy_immutable(src, target)

    sub_target = None
    if subtitles:
        sub = Path(subtitles).resolve()
        suffix = "".join(sub.suffixes) or ".json3"
        sub_target = root / f"subtitles{suffix}"
        copy_immutable(sub, sub_target)

    media = ffprobe(target)
    record = {
        "schema_version": 1,
        "video_id": video_id,
        "source_url": url,
        "complete_video": bool(complete),
        "clip_start_seconds": float(clip_start),
        "clip_end_seconds": clip_end,
        "video": target.name,
        "subtitles": sub_target.name if sub_target else None,
        "sha256": sha256(target),
        "bytes": target.stat().st_size,
        "ffprobe": media,
        "provenance_note": (
            "原始媒体只用于获授权的研究/内部处理；对外发布应仅分发 video ID、"
            "时间戳与标注，不分发第三方视频。"),
    }
    meta = root / "source.json"
    if meta.exists():
        old = json.loads(meta.read_text(encoding="utf-8"))
        if old.get("sha256") != record["sha256"]:
            raise FileExistsError(f"source.json 已指向另一个原片: {meta}")
    meta.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    # raw/ 本身通常被 gitignore；在其父目录维护一份不含媒体内容的轻量目录索引，
    # 便于版本控制与数据审计。重复登记同一 video_id 时原地更新，不追加脏副本。
    catalog_path = Path(out_root).resolve().parent / "source_catalog.jsonl"
    catalog = []
    if catalog_path.exists():
        catalog = [json.loads(line) for line in catalog_path.read_text(
            encoding="utf-8").splitlines() if line.strip()]
    public = {k: record[k] for k in (
        "schema_version", "video_id", "source_url", "complete_video",
        "clip_start_seconds", "clip_end_seconds", "sha256", "bytes")}
    catalog = [x for x in catalog if x.get("video_id") != video_id] + [public]
    catalog.sort(key=lambda x: x["video_id"])
    catalog_path.write_text("".join(
        json.dumps(x, ensure_ascii=False) + "\n" for x in catalog), encoding="utf-8")
    return {"root": str(root), **record}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out-root", default="data/raw")
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--url", default="")
    ap.add_argument("--subtitles", default="")
    ap.add_argument("--clip-start", type=float, default=0.0)
    ap.add_argument("--clip-end", type=float)
    ap.add_argument("--partial", action="store_true",
                    help="登记的是原视频的一段，而不是完整原片")
    args = ap.parse_args()
    print(json.dumps(register(
        args.video, args.out_root, args.video_id, args.url, args.subtitles,
        args.clip_start, args.clip_end, not args.partial),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
