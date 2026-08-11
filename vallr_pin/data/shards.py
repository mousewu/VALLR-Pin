"""WebDataset 兼容 tar 分片写入与读取。"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path
from typing import Dict

import numpy as np


class TarShardWriter:
    def __init__(self, out_dir: str, max_samples: int = 1000,
                 prefix: str = "shard"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.prefix = prefix
        self.index = -1
        self.count = 0
        self.total = 0
        self.tar: tarfile.TarFile | None = None
        self.paths: list[str] = []

    def _open(self) -> None:
        if self.tar:
            self._finalize()
        self.index += 1
        path = self.out_dir / f"{self.prefix}-{self.index:06d}.tar.partial"
        self.tar = tarfile.open(path, "w")
        self._partial = path
        self._final = path.with_suffix("")
        self.count = 0

    def _finalize(self) -> None:
        if not self.tar:
            return
        self.tar.close()
        os.replace(self._partial, self._final)
        self.paths.append(str(self._final))
        self.tar = None

    @staticmethod
    def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    def write(self, key: str, video: np.ndarray, meta: Dict) -> str:
        if self.tar is None or self.count >= self.max_samples:
            self._open()
        buf = io.BytesIO()
        np.save(buf, video, allow_pickle=False)
        self._add(self.tar, f"{key}.npy", buf.getvalue())
        self._add(self.tar, f"{key}.json",
                  json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        self.count += 1
        self.total += 1
        return f"wds://{self._final.name}::{key}.npy"

    def close(self) -> list[str]:
        self._finalize()  # 原子发布，worker 崩溃不留下有效假 shard
        return self.paths

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        elif self.tar:
            self.tar.close()


def load_wds_uri(uri: str, root: str = "") -> np.ndarray:
    """读取 ``wds://relative/shard.tar::key.npy``。"""
    body = uri[len("wds://"):]
    shard, member = body.split("::", 1)
    path = shard if os.path.isabs(shard) else os.path.join(root, shard)
    with tarfile.open(path, "r") as tar:
        f = tar.extractfile(member)
        if f is None:
            raise KeyError(f"{member} not found in {path}")
        return np.load(io.BytesIO(f.read()), allow_pickle=False)
