"""Length-bucketed, source-balanced batch sampling with DDP sharding."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class DistributedBucketBatchSampler(Sampler[List[int]]):
    """Build global batches deterministically, then shard batches by rank.

    ``source_weights`` describes desired corpus probabilities, e.g.
    ``{"cn_cvs": .6, "cmlr": .4}``. When omitted, each item appears once per
    epoch. Sampling with source weights uses replacement and ``epoch_samples``.
    """

    def __init__(self, lengths: Sequence[int], sources: Sequence[str], batch_size: int,
                 bucket_size: int = 40, source_weights: Dict[str, float] | None = None,
                 epoch_samples: int = 0, rank: int = 0, world_size: int = 1,
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        if len(lengths) != len(sources):
            raise ValueError("lengths and sources must have equal size")
        if batch_size < 1 or world_size < 1:
            raise ValueError("batch_size and world_size must be positive")
        self.lengths = [max(int(x or 0), 1) for x in lengths]
        self.sources = list(sources)
        self.batch_size, self.bucket_size = batch_size, max(bucket_size, 1)
        self.source_weights = dict(source_weights or {})
        self.epoch_samples = int(epoch_samples or len(lengths))
        self.rank, self.world_size = rank, world_size
        self.shuffle, self.drop_last, self.seed = shuffle, drop_last, seed
        self.epoch = 0
        unknown = set(self.source_weights) - set(self.sources)
        if unknown:
            raise ValueError(f"source_weights refer to absent sources: {sorted(unknown)}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _indices(self) -> List[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.source_weights:
            counts = Counter(self.sources)
            weights = torch.tensor([
                self.source_weights.get(src, 0.0) / counts[src] for src in self.sources
            ], dtype=torch.double)
            if float(weights.sum()) <= 0:
                raise ValueError("source_weights must contain a positive weight")
            return torch.multinomial(weights, self.epoch_samples, replacement=True,
                                     generator=generator).tolist()
        if self.shuffle:
            return torch.randperm(len(self.lengths), generator=generator).tolist()
        return list(range(len(self.lengths)))

    def _global_batches(self) -> List[List[int]]:
        indices = self._indices()
        mega = self.batch_size * self.bucket_size
        batches: List[List[int]] = []
        for start in range(0, len(indices), mega):
            bucket = indices[start:start + mega]
            bucket.sort(key=lambda i: self.lengths[i])
            batches.extend(bucket[i:i + self.batch_size]
                           for i in range(0, len(bucket), self.batch_size))
        if self.drop_last:
            batches = [b for b in batches if len(b) == self.batch_size]
        if self.shuffle:
            random.Random(self.seed + self.epoch + 17).shuffle(batches)
        # All ranks must execute the same number of optimizer collectives.
        usable = len(batches) - (len(batches) % self.world_size)
        return batches[:usable]

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._global_batches()[self.rank::self.world_size]

    def __len__(self) -> int:
        n = self.epoch_samples
        global_batches = n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        return global_batches // self.world_size
