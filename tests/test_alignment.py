"""锚点对齐的单测。

用**合成 ASR 结果**验证：把参考文本注入识别错误 + 在文字稿里塞入从未被说出口的
编者注，检查对齐器能否 (a) 给真正说过的句子标出正确时间戳，(b) 拒掉编者注。
这样 ASR 模型的有无不影响测试，对齐逻辑本身被完整覆盖。
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.align_transcript import (AsrChar, align_sentences, align_streams,  # noqa: E402
                                      dp_align, longest_increasing_pairs,
                                      unique_kgram_anchors)

SPOKEN = [
    "现在虽然可以做强化学习但它最终还是依赖一个很好的评估或者验证",
    "你让模型去做一道数学题可能做得比较好",
    "但如果让它去做一个更复杂的端到端任务有时很难找到一个衡量方式",
    "我们希望能最大化地使用每一份数据提升学习的效率",
    "你喂它一样多的数据它吸收得更好压缩率会涨得更快",
    "这个问题在小规模实验中很难发现但在大规模训练时会遇到",
]
# 文字稿里有，但音频里根本没说过（编者注、章节标题）
EDITORIAL = [
    "本书由物理学家撰写是一本科普哲学著作强调理性探究带来无限进步",
    "该基准测试用于评估模型在真实软件工程任务中的表现",
]


def make_fake_asr(sentences, error_rate=0.12, seed=0):
    """按 0.08 秒一个字铺时间轴，并按比例注入替换/删除错误。"""
    rng = random.Random(seed)
    pool = "的一是了我不人在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生"
    out, t = [], 0.5
    for sent in sentences:
        for ch in sent:
            r = rng.random()
            if r < error_rate * 0.6:
                ch = rng.choice(pool)          # 替换错误
            elif r < error_rate:
                t += 0.08                       # 删除：跳过这个字
                continue
            out.append(AsrChar(ch, round(t, 3), round(t + 0.08, 3)))
            t += 0.08
        t += 0.4                                # 句间停顿
    return out


def test_kgram_anchors_and_lis():
    a = "abcdefghij" + "zzzz" + "klmnopqrst"
    b = "abcdefghij" + "yy" + "klmnopqrst"
    anchors = unique_kgram_anchors(a, b, k=8)
    assert anchors, "应该能找到唯一 k-gram 锚点"
    lis = longest_increasing_pairs(anchors)
    assert all(lis[i][1] < lis[i + 1][1] for i in range(len(lis) - 1)), "锚点必须单调"


def test_dp_align_basic():
    pairs = dp_align("我想去银行", "我想去银航")
    matched = [(i, j) for i, j in pairs if i is not None and j is not None]
    assert len(matched) == 5


def test_align_streams_recovers_positions():
    ref = "".join(SPOKEN)
    asr = "".join(c.char for c in make_fake_asr(SPOKEN))
    mapping = align_streams(asr, ref, k=6)
    assert len(mapping) / len(ref) > 0.7, "多数参考字符应被对齐上"
    keys = sorted(mapping)
    vals = [mapping[k] for k in keys]
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)), "对齐必须单调"


def test_sentences_get_timestamps_and_editorial_is_rejected():
    asr = make_fake_asr(SPOKEN)
    sentences = [("杨植麟", s) for s in SPOKEN]
    # 把编者注插进文字稿中间，模拟真实的出版稿
    sentences.insert(2, ("编者", EDITORIAL[0]))
    sentences.append(("编者", EDITORIAL[1]))

    ok, rejected = align_sentences(asr, sentences, min_match=0.6, k=6)
    ok_texts = {s.text for s in ok}
    rejected_texts = {s for s, _ in rejected}

    assert EDITORIAL[0] in rejected_texts and EDITORIAL[1] in rejected_texts, \
        "没说过的编者注必须被拒"
    assert len(ok_texts & set(SPOKEN)) >= 5, "真正说过的句子应该绝大多数对上"

    # 时间戳必须单调、落在合成的时间轴范围内，且时长与字数量级相符
    ordered = [s for s in ok if s.text in SPOKEN]
    for s in ordered:
        assert 0.0 <= s.start < s.end <= asr[-1].end + 0.1
        dur_per_char = (s.end - s.start) / len(s.text)
        assert 0.03 < dur_per_char < 0.2, f"每字时长异常: {dur_per_char:.3f}s"
    for a, b in zip(ordered, ordered[1:]):
        assert a.start <= b.start, "句子时间戳必须随文本顺序单调"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
