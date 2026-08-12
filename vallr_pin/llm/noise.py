"""Synthetic toneless-Pinyin corruption for decoupled Stage-II training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from ..text.pinyin import canonical_syllables


@dataclass
class PinyinNoiseConfig:
    clean_prob: float = 0.25
    mild_prob: float = 0.50
    mild_min_rate: float = 0.05
    mild_max_rate: float = 0.18
    severe_min_rate: float = 0.20
    severe_max_rate: float = 0.40
    substitution_weight: float = 0.45
    deletion_weight: float = 0.20
    insertion_weight: float = 0.15
    mask_weight: float = 0.15
    swap_weight: float = 0.05
    mask_token: str = "<mask>"

    def validate(self) -> None:
        if not 0.0 <= self.clean_prob <= 1.0:
            raise ValueError("clean_prob must be in [0, 1]")
        if not 0.0 <= self.mild_prob <= 1.0 - self.clean_prob:
            raise ValueError("mild_prob must fit in the non-clean probability mass")
        for lo, hi, name in ((self.mild_min_rate, self.mild_max_rate, "mild"),
                             (self.severe_min_rate, self.severe_max_rate, "severe")):
            if not 0.0 <= lo <= hi <= 1.0:
                raise ValueError(f"invalid {name} noise-rate range")
        if sum(self.operation_weights()) <= 0:
            raise ValueError("at least one Pinyin noise operation must have positive weight")

    def operation_weights(self) -> List[float]:
        return [self.substitution_weight, self.deletion_weight, self.insertion_weight,
                self.mask_weight, self.swap_weight]


_INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
             "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")


def _parts(syllable: str) -> Tuple[str, str]:
    for initial in _INITIALS:
        if syllable.startswith(initial):
            return initial, syllable[len(initial):]
    return "", syllable


@lru_cache(maxsize=1)
def _confusions() -> Dict[str, Tuple[str, ...]]:
    syllables = canonical_syllables()
    parts = {s: _parts(s) for s in syllables}
    table: Dict[str, Tuple[str, ...]] = {}
    for syllable, (initial, final) in parts.items():
        close = [other for other, (oi, of) in parts.items()
                 if other != syllable and (of == final or oi == initial)]
        table[syllable] = tuple(close or [s for s in syllables if s != syllable])
    return table


def _near(syllable: str, rng: random.Random) -> str:
    choices = _confusions().get(syllable) or canonical_syllables()
    return rng.choice(choices)


def corrupt_pinyin(pinyin: Sequence[str], cfg: PinyinNoiseConfig,
                   rng: random.Random) -> Tuple[List[str], Dict[str, object]]:
    """Return a corrupted copy and reproducible metadata.

    Clean examples teach the LLM not to over-correct.  Mild and severe samples
    emulate local substitutions, dropped/repeated syllables and uncertain units.
    """
    cfg.validate()
    clean = list(pinyin)
    if not clean:
        return [], {"severity": "empty", "edits": 0}
    draw = rng.random()
    if draw < cfg.clean_prob:
        return clean, {"severity": "clean", "edits": 0}
    mild = draw < cfg.clean_prob + cfg.mild_prob
    severity = "mild" if mild else "severe"
    lo, hi = ((cfg.mild_min_rate, cfg.mild_max_rate) if mild else
              (cfg.severe_min_rate, cfg.severe_max_rate))
    edits = max(1, round(len(clean) * rng.uniform(lo, hi)))
    out = list(clean)
    operations = ["substitute", "delete", "insert", "mask", "swap"]
    applied: List[str] = []
    for _ in range(edits):
        op = rng.choices(operations, weights=cfg.operation_weights(), k=1)[0]
        if op == "delete" and len(out) <= 1:
            op = "substitute"
        if op == "swap" and len(out) <= 1:
            op = "substitute"
        index = rng.randrange(len(out))
        if op == "substitute":
            out[index] = _near(out[index], rng)
        elif op == "delete":
            out.pop(index)
        elif op == "insert":
            out.insert(index, _near(out[index], rng))
        elif op == "mask":
            out[index] = cfg.mask_token
        elif op == "swap":
            other = index + 1 if index + 1 < len(out) else index - 1
            out[index], out[other] = out[other], out[index]
        applied.append(op)
    return out, {"severity": severity, "edits": len(applied), "operations": applied}
