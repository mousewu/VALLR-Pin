"""Stage-II 提示词构造 (对应论文 Eq.15 的 P(Y | P̂, {Ŷ(k)}))。

提示词里三类信息缺一不可：

1. **拼音序列 P̂** —— 唇动能可靠恢复的那一层信息，是纠错的硬约束；
2. **N-best 候选 {Ŷ(k)}** —— 提供词法/搭配先验，也隐含了声学(视觉)后验排序；
3. **格式与长度约束** —— 强制"只输出一句中文、字数≈音节数"，
   这是抑制 LLM 自由发挥 (改写、补全、解释) 的关键；论文中零样本 LLM 反而
   使 CER 变差 (37.23 → 37.86)，主要就是这类过度生成造成的。
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

SYSTEM_PROMPT = (
    "你是中文唇语识别（VSR）系统的后处理专家。"
    "输入包含：一条由唇动模型识别出的无声调拼音序列，以及若干条候选中文转写（按置信度排序）。"
    "拼音序列和候选文本都可能有错。"
    "请结合拼音发音约束和汉语语言常识，输出最可能的原始中文句子。"
    "要求：只输出这一句中文，不要输出拼音、解释、标点以外的任何内容；"
    "输出字数应与拼音音节数尽量一致；不要扩写、不要改写句意。"
)


def build_user_prompt(pinyin: Sequence[str], nbest: Sequence[str]) -> str:
    py = " ".join(pinyin) if pinyin else "(空)"
    lines = [f"拼音序列（{len(pinyin)} 个音节）：{py}", "候选转写："]
    for i, cand in enumerate(nbest, 1):
        lines.append(f"{i}. {cand}")
    lines.append("请输出修正后的中文句子：")
    return "\n".join(lines)


def build_messages(pinyin: Sequence[str], nbest: Sequence[str],
                   answer: str = None) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
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
