"""数据管线：manifest -> (唇部 ROI 视频, 字符标签, 拼音标签)。

manifest 为 jsonl，每行::

    {"id": "utt_0001", "video": "/abs/or/rel/path.npy", "text": "今天可能会下雨",
     "input_type": "mouth_roi", "roi_type": "mouth", "fps": 25,
     "roi_height": 96, "roi_width": 96}

``video`` 支持三种形态：
  * ``.npy``  形如 (T,H,W) uint8 或 (T,H,W,3) 的已裁好唇部 ROI (推荐，读取最快)
  * 目录      内含按名字排序的帧图片 (需要 opencv)
  * ``.mp4``  仅为兼容旧工具保留；Stage-I 默认拒绝直接训练原始视频

CN-CVS 的人脸裁剪和 CMLR 的原始新闻画面都必须先经过
``scripts/preprocess_stage1_roi.py``，统一成固定帧率的灰度嘴部 ROI。
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..text.tokenizer import DualTokenizer


def read_manifest(path: str) -> List[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_manifest(path: str, items: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _load_video(path: str, root: str = "") -> np.ndarray:
    """-> (T,H,W) uint8 灰度"""
    if path.startswith("wds://"):
        from .shards import load_wds_uri
        arr = load_wds_uri(path, root)
    elif path.endswith(".npy"):
        arr = np.load(path)
    elif os.path.isdir(path):
        import cv2  # noqa: WPS433 (可选依赖)
        files = sorted(os.listdir(path))
        arr = np.stack([cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
                        for f in files])
    else:
        import cv2  # noqa: WPS433
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        if not frames:
            raise RuntimeError(f"empty video: {path}")
        arr = np.stack(frames)
    if arr.ndim == 4:                     # (T,H,W,3) -> 灰度
        arr = arr[..., :3].mean(-1)
    return arr.astype(np.uint8)


def validate_roi_manifest(items: List[dict], expected_fps: float = 25.0) -> None:
    """Fail fast when a training/decode manifest is not model-ready.

    Shape is checked after loading in ``LipReadingDataset.__getitem__``; this
    metadata-only pass is deliberately cheap enough to run on million-row
    manifests before any DataLoader workers are started.
    """
    for index, item in enumerate(items):
        uid = item.get("id", index)
        if item.get("input_type") != "mouth_roi" or item.get("roi_type") != "mouth":
            raise ValueError(
                f"{uid}: Stage-I requires input_type=mouth_roi and roi_type=mouth; "
                "run scripts/preprocess_stage1_roi.py first")
        missing = [key for key in ("fps", "roi_height", "roi_width", "n_frames")
                   if item.get(key) in (None, "")]
        if missing:
            raise ValueError(f"{uid}: ROI manifest is missing metadata: {', '.join(missing)}")
        fps = float(item["fps"])
        if expected_fps > 0 and abs(fps - expected_fps) > 0.05:
            raise ValueError(
                f"{uid}: ROI fps={fps:g}, expected {expected_fps:g}; re-run ROI preprocessing")
        height, width = int(item["roi_height"]), int(item["roi_width"])
        if height <= 0 or width <= 0 or height != width:
            raise ValueError(f"{uid}: invalid mouth ROI geometry {height}x{width}")
        if int(item["n_frames"]) <= 0:
            raise ValueError(f"{uid}: invalid n_frames={item['n_frames']}")


@dataclass
class VideoTransform:
    crop_size: int = 88
    train: bool = True
    flip_prob: float = 0.5
    time_mask: int = 0          # 随机遮蔽的最大连续帧数 (类 SpecAugment)
    mean: float = 0.421
    std: float = 0.165

    def __call__(self, video: np.ndarray) -> torch.Tensor:
        t, h, w = video.shape
        cs = min(self.crop_size, h, w)
        if self.train:
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
        else:
            top, left = (h - cs) // 2, (w - cs) // 2
        video = video[:, top:top + cs, left:left + cs]
        if self.train and random.random() < self.flip_prob:
            video = video[:, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(video)).float().div_(255.0)
        x = (x - self.mean) / self.std
        if self.train and self.time_mask > 0 and t > self.time_mask * 2:
            n = random.randint(0, self.time_mask)
            if n:
                s = random.randint(0, t - n)
                x[s:s + n] = 0.0
        return x.unsqueeze(1)              # (T,1,H,W)


class LipReadingDataset(Dataset):
    def __init__(self, manifest: str, tokenizer: DualTokenizer,
                 transform: Optional[VideoTransform] = None,
                 root: str = "", max_frames: int = 0, min_frames: int = 4,
                 require_mouth_roi: bool = True, expected_fps: float = 25.0):
        self.items = read_manifest(manifest)
        self.tok = tokenizer
        self.transform = transform or VideoTransform(train=False)
        self.root = root
        self.max_frames, self.min_frames = max_frames, min_frames
        self.require_mouth_roi = require_mouth_roi
        self.expected_fps = expected_fps
        if require_mouth_roi:
            validate_roi_manifest(self.items, expected_fps)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        it = self.items[idx]
        raw = it["video"]
        path = raw if raw.startswith("wds://") or os.path.isabs(raw) else os.path.join(self.root, raw)
        video = _load_video(path, self.root)
        if self.require_mouth_roi:
            uid = it.get("id", idx)
            _, height, width = video.shape
            declared = (int(it["roi_height"]), int(it["roi_width"]))
            if (height, width) != declared:
                raise ValueError(
                    f"{uid}: ROI array is {height}x{width}, manifest declares "
                    f"{declared[0]}x{declared[1]}")
            if len(video) != int(it["n_frames"]):
                raise ValueError(
                    f"{uid}: ROI array has {len(video)} frames, manifest declares "
                    f"{int(it['n_frames'])}")
            crop_size = self.transform.crop_size
            if height != width or min(height, width) < crop_size:
                raise ValueError(
                    f"{uid}: mouth ROI {height}x{width} cannot provide a "
                    f"{crop_size}x{crop_size} crop")
        if self.max_frames and len(video) > self.max_frames:
            raise ValueError(
                f"{it.get('id', idx)} has {len(video)} frames > max_frames={self.max_frames}; "
                "segment the utterance during preprocessing instead of truncating video only"
            )
        if len(video) < self.min_frames:
            video = np.concatenate([video] * (self.min_frames // max(len(video), 1) + 1))
        char_ids, py_ids = self.tok.encode(it["text"])
        # CTC 约束：编码器帧数必须 >= 标签长度，否则该样本的 CTC 分支会退化
        return {"id": it.get("id", str(idx)), "video": self.transform(video),
                "source": it.get("source", "unknown"),
                "speaker_id": it.get("speaker_id", ""),
                "text": it["text"], "char_ids": torch.tensor(char_ids, dtype=torch.long),
                "pinyin_ids": torch.tensor(py_ids, dtype=torch.long)}


def collate(batch: List[Dict]) -> Dict:
    b = len(batch)
    vt = max(x["video"].size(0) for x in batch)
    c, h, w = batch[0]["video"].shape[1:]
    video = torch.zeros(b, vt, c, h, w)
    video_lens = torch.zeros(b, dtype=torch.long)
    cl = max(max(x["char_ids"].numel() for x in batch), 1)
    pl = max(max(x["pinyin_ids"].numel() for x in batch), 1)
    chars = torch.zeros(b, cl, dtype=torch.long)
    pys = torch.zeros(b, pl, dtype=torch.long)
    char_lens = torch.zeros(b, dtype=torch.long)
    py_lens = torch.zeros(b, dtype=torch.long)
    for i, x in enumerate(batch):
        t = x["video"].size(0)
        video[i, :t] = x["video"]
        video_lens[i] = t
        n, m = x["char_ids"].numel(), x["pinyin_ids"].numel()
        chars[i, :n], char_lens[i] = x["char_ids"], n
        pys[i, :m], py_lens[i] = x["pinyin_ids"], m
    return {"ids": [x["id"] for x in batch], "texts": [x["text"] for x in batch],
            "sources": [x["source"] for x in batch],
            "speaker_ids": [x["speaker_id"] for x in batch],
            "video": video, "video_lens": video_lens, "char_ids": chars,
            "char_lens": char_lens, "pinyin_ids": pys, "pinyin_lens": py_lens}
