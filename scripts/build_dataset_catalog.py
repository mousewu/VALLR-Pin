#!/usr/bin/env python3
"""内容去重、轻量说话人聚类与 speaker-independent 划分。

显式 ``speaker_id`` 优先；缺失时从每个来源的 ROI 计算稳定外观描述子并聚类。
描述子适合固定机位口播的初筛，跨年龄/妆容/大姿态场景应替换为 ArcFace embedding。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vallr_pin.data.dataset import _load_video  # noqa: E402


def norm_text(text: str) -> str:
    return re.sub(r"[^一-龥A-Za-z0-9]", "", text).lower()


def descriptor(items: list[dict], root: str, max_samples: int = 12) -> np.ndarray:
    """中位 ROI 的低频频谱描述子；亮度/轻微位移不敏感。"""
    imgs = []
    for it in items[:max_samples]:
        path = it["video"]
        if not path.startswith("wds://") and not os.path.isabs(path): path = os.path.join(root, path)
        arr = _load_video(path, root)
        if len(arr):
            imgs.append(np.median(arr[::max(1, len(arr)//8)], axis=0))
    if not imgs:
        return np.zeros(64, dtype=np.float32)
    im = np.median(imgs, axis=0).astype(np.float32)
    im = (im - im.mean()) / (im.std() + 1e-6)
    feat = np.abs(np.fft.rfft2(im))[:8, :8].ravel().astype(np.float32)
    feat = np.log1p(feat); feat /= np.linalg.norm(feat) + 1e-8
    return feat


def visual_hash(it: dict, root: str) -> str:
    """首/中/尾帧的低频二值哈希；镜像转载/重复切句保持一致，不误删同文本异人。"""
    path = it["video"]
    if not path.startswith("wds://") and not os.path.isabs(path): path = os.path.join(root, path)
    arr = _load_video(path, root)
    if not len(arr): return "empty"
    im = np.median(arr[[0, len(arr)//2, -1]], axis=0)
    # 分成 8×8 网格，不依赖 cv2。
    ys=np.linspace(0,im.shape[0],9,dtype=int); xs=np.linspace(0,im.shape[1],9,dtype=int)
    grid=np.array([[im[ys[y]:ys[y+1],xs[x]:xs[x+1]].mean() for x in range(8)] for y in range(8)])
    bits=(grid>np.median(grid)).ravel(); return f"{sum(int(b)<<i for i,b in enumerate(bits)):016x}"


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[b] = a


def split_for(speaker: str, dev: int, test: int) -> str:
    v = int(hashlib.sha1(speaker.encode()).hexdigest()[:8], 16) % 100
    return "test" if v < test else ("dev" if v < test + dev else "train")


def build(manifests: list[str], out_dir: str, speaker_threshold: float = .92,
          dev_percent: int = 5, test_percent: int = 5) -> dict:
    all_items = []
    for m in manifests:
        root = os.path.dirname(os.path.abspath(m))
        for line in open(m, encoding="utf-8"):
            it = json.loads(line); it["_root"] = root
            all_items.append(it)
    # 内容去重：规范化文本 + 视觉哈希。不能只按文本删重，否则会错误删除“不同说话人
    # 说同一句话”这种对 VSR 很有价值的样本。
    best, dup = {}, 0
    for it in all_items:
        key = (norm_text(it["text"]), visual_hash(it, it["_root"]))
        if key in best:
            dup += 1
            if it.get("n_frames", 0) > best[key].get("n_frames", 0): best[key] = it
        else: best[key] = it
    items = list(best.values())

    by_source = defaultdict(list)
    for it in items: by_source[it.get("source_id") or it.get("speaker_id") or it["_root"]].append(it)
    sources = sorted(by_source)
    desc = [descriptor(by_source[s], by_source[s][0]["_root"]) for s in sources]
    dsu = DSU(len(sources))
    for i in range(len(sources)):
        for j in range(i):
            explicit_i = by_source[sources[i]][0].get("speaker_id")
            explicit_j = by_source[sources[j]][0].get("speaker_id")
            if explicit_i and explicit_j:
                same = explicit_i == explicit_j
            else:
                same = float(np.dot(desc[i], desc[j])) >= speaker_threshold
            if same: dsu.union(i, j)
    cluster_names = {}
    for i, src in enumerate(sources):
        explicit = by_source[src][0].get("speaker_id")
        cluster_names[dsu.find(i)] = explicit or f"spk_{dsu.find(i):05d}"
    source_speaker = {src: cluster_names[dsu.find(i)] for i, src in enumerate(sources)}

    os.makedirs(out_dir, exist_ok=True)
    counts = defaultdict(int)
    handles = {s: open(os.path.join(out_dir, f"{s}.jsonl"), "w", encoding="utf-8")
               for s in ("train", "dev", "test")}
    try:
        for it in items:
            src = it.get("source_id") or it.get("speaker_id") or it["_root"]
            spk = source_speaker[src]; split = split_for(spk, dev_percent, test_percent)
            root = it.pop("_root"); it["speaker_id"] = spk; it["split"] = split
            # shard URI 对原 manifest 目录相对；写成绝对 shard URI，合并后仍可读。
            if it["video"].startswith("wds://"):
                body = it["video"][6:]; shard, member = body.split("::", 1)
                if not os.path.isabs(shard): shard = os.path.join(root, shard)
                it["video"] = f"wds://{os.path.abspath(shard)}::{member}"
            handles[split].write(json.dumps(it, ensure_ascii=False) + "\n"); counts[split] += 1
    finally:
        for f in handles.values(): f.close()
    report = {"input": len(all_items), "after_dedup": len(items), "duplicates": dup,
              "sources": len(sources), "speaker_clusters": len(set(source_speaker.values())),
              "counts": dict(counts), "source_speaker": source_speaker,
              "speaker_threshold": speaker_threshold}
    open(os.path.join(out_dir, "catalog_report.json"), "w", encoding="utf-8").write(
        json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("manifests", nargs="+")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--speaker-threshold", type=float, default=.92)
    ap.add_argument("--dev-percent", type=int, default=5); ap.add_argument("--test-percent", type=int, default=5)
    a = ap.parse_args(); print(json.dumps(build(a.manifests, a.out_dir, a.speaker_threshold,
                                               a.dev_percent, a.test_percent), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
