#!/usr/bin/env python3
"""Batch landmark extraction and ROI96 rendering for a Stage-I manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.render_variant import render  # noqa: E402
from vallr_pin.data.dataset import read_manifest  # noqa: E402


def _key(item: dict) -> str:
    digest = hashlib.sha1(item["id"].encode()).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in item["id"])[-60:]
    return f"{stem}-{digest}"


def process(item: dict, output_root: str, model: str, min_coverage: float,
            keep_tracks: bool) -> tuple[dict | None, dict]:
    source = item.get("source", "unknown"); key = _key(item)
    roi = Path(output_root) / source / "roi96" / f"{key}.npy"
    track = Path(output_root) / source / "tracks" / f"{key}.npz"
    roi.parent.mkdir(parents=True, exist_ok=True); track.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not roi.exists():
            if not track.exists():
                subprocess.run([sys.executable, str(Path(__file__).with_name("extract_tracks.py")),
                                item["video"], "--model", model, "--out", str(track),
                                "--keep-subset"], check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True)
            info = render(item["video"], str(track), "vallr_pin", str(roi),
                          min_coverage=min_coverage)
        else:
            array = np.load(roi, mmap_mode="r")
            info = {"shape": list(array.shape), "coverage": 1.0,
                    "missing_frames": 0}
        output = dict(item); output["video"] = str(roi.resolve())
        output["n_frames"] = int(info["shape"][0]); output["roi_coverage"] = info["coverage"]
        if not keep_tracks and track.exists():
            track.unlink()
        return output, {"id": item["id"], "status": "ok", **info}
    except Exception as exc:
        return None, {"id": item.get("id", ""), "status": "rejected", "error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("manifest")
    ap.add_argument("--out-manifest", required=True); ap.add_argument("--out-root", required=True)
    ap.add_argument("--face-model", required=True); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-coverage", type=float, default=.95)
    ap.add_argument("--discard-tracks", action="store_true")
    args = ap.parse_args(); items = read_manifest(args.manifest); results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, item, args.out_root, args.face_model,
                               args.min_coverage, not args.discard_tracks) for item in items]
        for index, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if index % 100 == 0: print(f"[roi] {index}/{len(items)}", flush=True)
    good = [item for item, _ in results if item is not None]
    good.sort(key=lambda item: item["id"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out_manifest)), exist_ok=True)
    with open(args.out_manifest + ".partial", "w", encoding="utf-8") as stream:
        for item in good: stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(args.out_manifest + ".partial", args.out_manifest)
    report = {"input": len(items), "accepted": len(good), "rejected": len(items)-len(good),
              "items": [report for _, report in results]}
    with open(args.out_manifest.replace(".jsonl", ".roi_report.json"), "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("input", "accepted", "rejected")}, indent=2))


if __name__ == "__main__": main()
