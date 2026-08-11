"""VALLR-Pin：面向普通话的拼音引导两阶段视觉语音识别实现。

参考文献：
    Chang Sun, Dongliang Xie, Wanpeng Xie, Bo Qin, Hong Yang.
    "VALLR-Pin: Uncertainty-Factorized Visual Speech Recognition for Mandarin
     with Pinyin Guidance", arXiv:2512.20032 (v1 / v2).

当前 Stage-I v2 另参考 VALLR (ICCV 2025) 的音系中介思想，但采用时间保持的
拼音优先 CTC，未复制其时空 patch 展平与 16 帧/8 CTC-step 实现。
"""

__version__ = "0.2.0"
