#!/usr/bin/env python3
"""Reproduce structural and temporal-sensitivity checks on VALLR Stage-I.

This audit imports the reference repository in-place; it does not copy or
redistribute its CC BY-NC model code or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PHONEMES = [
    "<pad>", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M",
    "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW",
    "V", "W", "Y", "Z", "ZH",
]


def load_frames(video: str, start: float, count: int = 16) -> torch.Tensor:
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", video, "-vf",
           "fps=8,scale=224:224:force_original_aspect_ratio=increase,crop=224:224",
           "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.check_output(cmd)
    expected = count * 224 * 224 * 3
    if len(raw) != expected:
        raise RuntimeError(f"ffmpeg returned {len(raw)} bytes, expected {expected}")
    arr = np.frombuffer(raw, np.uint8).reshape(count, 224, 224, 3).copy()
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float()


def collapse(logits: torch.Tensor) -> list[str]:
    out, previous = [], None
    for token in logits.argmax(-1)[0].tolist():
        if token != 0 and token != previous:
            out.append(PHONEMES[token])
        previous = token
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=5.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = str(Path(args.reference_root).resolve())
    sys.path.insert(0, root)
    from transformers import VideoMAEConfig, Wav2Vec2Config
    from Models.VALLR import VALLR

    model = VALLR(VideoMAEConfig(), Wav2Vec2Config(vocab_size=40), adapter_dim=256)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    base = load_frames(args.video, args.start)
    variants = {"base": base, "reversed": base.flip(0),
                "static": base[8:9].expand_as(base)}
    outputs, rows = {}, {}
    started = time.time()
    with torch.inference_mode():
        for name, frames in variants.items():
            logits, features = model(frames.unsqueeze(0))
            outputs[name] = logits
            probs = logits.softmax(-1)
            rows[name] = {"phonemes": collapse(logits),
                          "raw_steps": logits.size(1),
                          "mean_confidence": float(probs.max(-1).values.mean()),
                          "feature_shape": list(features.shape)}
    base_logits = outputs["base"].flatten()
    for name, logits in outputs.items():
        rows[name]["cosine_to_base"] = float(
            F.cosine_similarity(base_logits, logits.flatten(), dim=0)
        )
    report = {"checkpoint_tensors": len(state),
              "checkpoint_parameters": sum(v.numel() for v in state.values()),
              "input_shape": [1, 16, 3, 224, 224], "variants": rows,
              "elapsed_seconds": round(time.time() - started, 3)}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
