"""CER / 音节错误率。CER = (S + D + I) / N，与论文 Eq.19 一致。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


def edit_ops(ref: Sequence, hyp: Sequence) -> Tuple[int, int, int]:
    """返回 (替换, 删除, 插入) 次数。标准 Levenshtein 回溯。"""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j - 1], d[i - 1][j], d[i][j - 1])
    i, j, s, dele, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            s += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, dele, ins


@dataclass
class ErrorStats:
    sub: int = 0
    dele: int = 0
    ins: int = 0
    total: int = 0

    def update(self, ref: Sequence, hyp: Sequence) -> None:
        s, d, i = edit_ops(ref, hyp)
        self.sub += s
        self.dele += d
        self.ins += i
        self.total += len(ref)

    @property
    def rate(self) -> float:
        return (self.sub + self.dele + self.ins) / max(self.total, 1)

    def __str__(self) -> str:
        return (f"ER={100 * self.rate:.2f}% (S={self.sub} D={self.dele} I={self.ins} "
                f"N={self.total})")


def cer(refs: List[str], hyps: List[str]) -> float:
    st = ErrorStats()
    for r, h in zip(refs, hyps):
        st.update(list(r), list(h))
    return st.rate


def syllable_er(refs: List[List[str]], hyps: List[List[str]]) -> float:
    st = ErrorStats()
    for r, h in zip(refs, hyps):
        st.update(r, h)
    return st.rate
