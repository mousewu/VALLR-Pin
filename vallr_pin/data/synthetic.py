"""合成"唇动"数据集：在没有 CNVSRC/CMLR 授权数据时，用它跑通并验证整条链路。

设计要点（这不是随手造的假数据，而是对论文所针对现象的**受控复现**）：

* 每个**无声调音节**分配一张随机 viseme 模板，视频帧只由音节决定；
* 因此"视觉信号只能确定到音节层，字符层是一对多"这一 Mandarin VSR 的核心
  病态性 (论文 v2 所谓 uncertainty factorization) 被精确复刻；
* 语料里刻意放入大量同音异形词 (手机/收集、事件/时间、权利/权力 ...)，
  Stage-I 字符辅助头容易在这些位置出错，而拼音主头不会 —— 正好用来检验
  Stage-II 的拼音引导纠错是否真的起作用。

speaker 维度上加入亮度/对比度/噪声/语速抖动，模拟多说话人域偏移。
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np

from ..text.pinyin import text_to_pinyin
from .dataset import write_manifest

# 同音混淆对占主导的语料；每行都是一句完整的话
CORPUS: List[str] = [
    "我的手机放在桌子上了", "他正在收集比赛的资料",
    "这件事情发生在上个星期", "会议的时间已经定下来了",
    "这个事件引起了广泛关注", "我和同事一起去吃午饭",
    "他们同时到达了会场", "新员工明天上午去报到",
    "记者对这次会议做了报道", "医生给他做了全面检查",
    "检察机关已经介入调查", "每个公民都有受教育的权利",
    "他手中的权力受到监督", "这份报告反映了真实情况",
    "他对这个消息反应很快", "请你保持联系随时通知我",
    "他每天都在练习普通话", "你这个主意听起来不错",
    "开车的时候请注意安全", "这本书讲的是集体主义",
    "他一直在等你的回复", "大家的意见基本一致",
    "在此期间不要离开房间", "其间发生了很多事情",
    "会场设在城市的中心", "他对这份工作非常忠心",
    "这个问题的重心在成本", "你可以先回去休息",
    "他刻意隐瞒了这个细节", "从这里到机场有三十公里",
    "这条公理不需要证明", "参加的人员全部到齐了",
    "他们全不知道这个安排", "北京是我们国家的首都",
    "他的手都冻僵了", "这项研究有重要的意义",
    "对这个方案我保留异议", "我们必须尊重客观事实",
    "他每天都关心国内时事", "新的政策将从下月实施",
    "今天下午可能会下雨", "夏雨结束以后天气转凉",
    "我想去银行办一张卡", "请把窗户关上谢谢",
    "明天的会议推迟到后天", "这个软件需要重新安装",
    "他在电话里说了很久", "麻烦你帮我拿一下行李",
    "食堂的饭菜比以前好吃", "他从学校毕业已经三年",
]


def _template_bank(syllables: List[str], size: int, seed: int = 1234
                   ) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    bank = {}
    for s in sorted(syllables):
        # 低频随机图案 + 上采样，得到平滑、可区分的"唇形"模板
        low = rng.rand(4, 5, 5).astype(np.float32)
        tmpl = np.repeat(np.repeat(low, size // 5 + 1, axis=1), size // 5 + 1, axis=2)
        bank[s] = tmpl[:, :size, :size]
    return bank


def _render(sentence: str, bank: Dict[str, np.ndarray], size: int, rng: random.Random,
            speaker: Tuple[float, float], noise: float) -> np.ndarray:
    gain, bias = speaker
    frames = []
    _, syls = text_to_pinyin(sentence)
    for syl in syls:
        tmpl = bank.get(syl)
        if tmpl is None:
            tmpl = np.zeros((4, size, size), dtype=np.float32)
        n_rep = rng.randint(3, 5)                      # 语速抖动
        for i in range(n_rep):
            f = tmpl[i % tmpl.shape[0]]
            f = f * gain + bias
            f = f + np.random.RandomState(rng.randint(0, 2 ** 31 - 1)).randn(size, size) * noise
            frames.append(np.clip(f, 0, 1))
    if not frames:
        frames = [np.zeros((size, size), dtype=np.float32)]
    return (np.stack(frames) * 255).astype(np.uint8)


def build_synthetic_dataset(out_dir: str, n_train_speakers: int = 6,
                            n_dev_speakers: int = 2, size: int = 40, noise: float = 0.10,
                            seed: int = 0) -> Dict[str, str]:
    """生成 train/dev manifest 与 npy 视频。dev 使用**未见过的说话人参数**。"""
    rng = random.Random(seed)
    syls = sorted({s for t in CORPUS for s in text_to_pinyin(t)[1]})
    bank = _template_bank(syls, size, seed=seed + 1234)

    def make_split(split: str, n_spk: int, spk_offset: int):
        vdir = os.path.join(out_dir, split)
        os.makedirs(vdir, exist_ok=True)
        items = []
        for spk in range(n_spk):
            gain = 0.6 + 0.15 * ((spk + spk_offset) % 5)
            bias = 0.05 * ((spk + spk_offset) % 3)
            for si, sent in enumerate(CORPUS):
                arr = _render(sent, bank, size, rng, (gain, bias), noise)
                name = f"{split}_spk{spk}_{si:04d}.npy"
                np.save(os.path.join(vdir, name), arr)
                items.append({"id": name[:-4], "video": os.path.join(split, name),
                              "text": sent})
        path = os.path.join(out_dir, f"{split}.jsonl")
        write_manifest(path, items)
        return path

    os.makedirs(out_dir, exist_ok=True)
    train = make_split("train", n_train_speakers, 0)
    dev = make_split("dev", n_dev_speakers, 97)
    with open(os.path.join(out_dir, "corpus.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CORPUS))
    return {"train": train, "dev": dev, "root": out_dir}
