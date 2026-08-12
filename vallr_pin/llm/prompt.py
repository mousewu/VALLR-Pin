"""Stage-II prompts for decoupled Pinyin-to-text reconstruction.

The primary interface is noisy Pinyin only, allowing the LLM to be trained on
arbitrarily large text corpora without any Stage-I checkpoint.  Character
N-best hypotheses remain an optional calibration/inference extension.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

PINYIN_SYSTEM_PROMPT = (
    "你是中文视觉语音识别系统的拼音解码器。"
    "输入是一条可能存在替换、删除、插入、乱序或遮挡错误的无声调拼音序列。"
    "请结合拼音约束和汉语语言常识，恢复最可能的原始中文句子。"
    "要求：只输出这一句中文，不要输出拼音、解释或候选列表；"
    "输出长度应与拼音音节数尽量一致，不要扩写，不要改写句意。"
)

CALIBRATION_SYSTEM_PROMPT = (
    "你是中文唇语识别（VSR）系统的后处理专家。"
    "输入包含：一条由唇动模型识别出的无声调拼音序列，以及若干条候选中文转写（按置信度排序）。"
    "拼音序列和候选文本都可能有错。"
    "请结合拼音发音约束和汉语语言常识，输出最可能的原始中文句子。"
    "要求：只输出这一句中文，不要输出拼音、解释、标点以外的任何内容；"
    "输出字数应与拼音音节数尽量一致；不要扩写、不要改写句意。"
)

# Backwards-compatible public name for callers that imported it directly.
SYSTEM_PROMPT = CALIBRATION_SYSTEM_PROMPT


def build_user_prompt(pinyin: Sequence[str], nbest: Sequence[str] | None = None) -> str:
    py = " ".join(pinyin) if pinyin else "(空)"
    lines = [f"拼音序列（{len(pinyin)} 个音节）：{py}"]
    if nbest:
        lines.append("候选转写：")
        for i, cand in enumerate(nbest, 1):
            lines.append(f"{i}. {cand}")
        lines.append("请输出修正后的中文句子：")
    else:
        lines.append("请输出对应的中文句子：")
    return "\n".join(lines)


def build_messages(pinyin: Sequence[str], nbest: Sequence[str] | None = None,
                   answer: str = None) -> List[Dict[str, str]]:
    system = CALIBRATION_SYSTEM_PROMPT if nbest else PINYIN_SYSTEM_PROMPT
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": build_user_prompt(pinyin, nbest)}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


_CJK = re.compile(r"[一-龥]+")


def parse_response(text: str) -> str:
    """从模型输出里抽出中文句子，抵抗"好的，修正后为：xxx"这类前后缀。"""
    if not text:
        return ""
    text = text.strip().splitlines()[-1] if "\n" in text.strip() else text.strip()
    for sep in ("：", ":"):
        if sep in text and len(text.split(sep)[-1]) > 1:
            text = text.split(sep)[-1]
    return "".join(_CJK.findall(text))
