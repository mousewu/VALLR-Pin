"""Stage-II inference: predicted Pinyin to Mandarin text.

提供两种精化器：

* :class:`NgramPinyinRefiner` —— **不依赖大模型**的受限重打分器。用训练文本上的
  字符 bigram 语言模型，在"预测拼音所允许的同音字集合"上做 Viterbi 解码，
  再与 N-best 一起按 (语言模型分 + 拼音一致性) 重排。它是一个诚实的下界基线，
  也让整条链路在离线/无 GPU 环境下可复现。
* :class:`LLMRefiner` —— 默认只接收 Stage-I 预测拼音；字符 N-best 是可选校准信息。
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from ..engine.metrics import ErrorStats, edit_ops
from ..text.pinyin import homophone_table, text_to_pinyin
from .prompt import build_messages, parse_response


def _ser(ref: Sequence[str], hyp: Sequence[str]) -> float:
    s, d, i = edit_ops(list(ref), list(hyp))
    return (s + d + i) / max(len(ref), 1)


# --------------------------------------------------------------------------- #
#                        无 LLM 基线：拼音受限的 n-gram 重打分                    #
# --------------------------------------------------------------------------- #
@dataclass
class NgramConfig:
    order2_weight: float = 0.8       # bigram / unigram 插值
    add_k: float = 0.5
    pinyin_weight: float = 6.0       # 拼音一致性项权重
    nbest_bonus: float = 1.0         # 候选里出现过的字给的发射奖励
    top1_bonus: float = 0.1          # 不改动 Stage-I top-1 的先验奖励 (见 refine)
    length_guard: float = 0.25       # Viterbi 解与 top-1 的长度偏差超过该比例则弃用
    max_cands_per_syllable: int = 40


class NgramPinyinRefiner:
    def __init__(self, cfg: NgramConfig = NgramConfig()):
        self.cfg = cfg
        self.uni: Dict[str, int] = defaultdict(int)
        self.bi: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total = 0
        self.vocab: set = set()

    # ------------------------------------------------------------------ fit
    def fit(self, texts: Iterable[str]) -> "NgramPinyinRefiner":
        for t in texts:
            chars = ["<s>"] + [c for c in t if "一" <= c <= "龥"] + ["</s>"]
            for i, c in enumerate(chars):
                if c not in ("<s>",):
                    self.uni[c] += 1
                    self.total += 1
                if i:
                    self.bi[chars[i - 1]][c] += 1
            self.vocab.update(c for c in chars if len(c) == 1)
        return self

    def _logp(self, prev: str, cur: str) -> float:
        k, v = self.cfg.add_k, max(len(self.vocab), 1)
        p_uni = (self.uni.get(cur, 0) + k) / (self.total + k * v)
        row = self.bi.get(prev)
        if row:
            denom = sum(row.values())
            p_bi = (row.get(cur, 0) + k) / (denom + k * v)
        else:
            p_bi = p_uni
        w = self.cfg.order2_weight
        return math.log(w * p_bi + (1 - w) * p_uni)

    def lm_score(self, text: str) -> float:
        chars = [c for c in text if "一" <= c <= "龥"]
        if not chars:
            return -1e9
        prev, s = "<s>", 0.0
        for c in chars:
            s += self._logp(prev, c)
            prev = c
        s += self._logp(prev, "</s>")
        return s / len(chars)

    # -------------------------------------------------------------- viterbi
    def _candidates(self, syl: str, hint: str = "") -> List[str]:
        table = homophone_table().get(syl, [])
        cands = [c for c in table if c in self.vocab]
        if not cands:
            cands = table[: self.cfg.max_cands_per_syllable] or ([hint] if hint else ["　"])
        cands.sort(key=lambda c: -self.uni.get(c, 0))
        cands = cands[: self.cfg.max_cands_per_syllable]
        if hint and hint not in cands:
            cands.append(hint)
        return cands

    def viterbi(self, pinyin: Sequence[str], nbest: Sequence[str]) -> str:
        if not pinyin:
            return nbest[0] if nbest else ""
        # 候选里出现过的字，在对应位置给一点发射奖励 (软性利用 Stage-I 词法信息)
        pos_hint: List[set] = []
        for i in range(len(pinyin)):
            hint = set()
            for cand in nbest:
                if i < len(cand):
                    hint.add(cand[i])
            pos_hint.append(hint)

        states = [self._candidates(p, next(iter(pos_hint[i]), "")) for i, p in enumerate(pinyin)]
        prev_scores = {}
        for c in states[0]:
            prev_scores[c] = (self._logp("<s>", c)
                              + (self.cfg.nbest_bonus if c in pos_hint[0] else 0.0), [c])
        for i in range(1, len(pinyin)):
            cur = {}
            for c in states[i]:
                bonus = self.cfg.nbest_bonus if c in pos_hint[i] else 0.0
                best, best_path = -1e18, None
                for p, (sc, path) in prev_scores.items():
                    v = sc + self._logp(p, c)
                    if v > best:
                        best, best_path = v, path
                cur[c] = (best + bonus, best_path + [c])
            prev_scores = cur
        final = max(prev_scores.items(), key=lambda kv: kv[1][0] + self._logp(kv[0], "</s>"))
        return "".join(final[1][1])

    # --------------------------------------------------------------- refine
    def refine(self, pinyin: Sequence[str], nbest: Sequence[str]) -> str:
        """在 {N-best} ∪ {拼音受限 Viterbi 解} 上重排。

        两条"不轻易改"的先验，都是为了防止**拼音错误反向污染已经正确的字符假设**：

        * ``top1_bonus``：top-1 已经携带了视觉后验，同分时优先保留；
        * ``length_guard``：Viterbi 解的长度被 P̂ 硬性决定，一旦拼音解码出现
          重复/截断（自回归解码在 "shi shi shi" 这类重复音节上最容易崩），
          它会把长度错误直接写进结果 —— 此时整条 Viterbi 候选作废。

        没有这两条时，Stage-I 已经正确的句子会被拼音项拽偏，
        正是论文中零样本 LLM 反而使 CER 上升的同类现象。
        """
        top1 = nbest[0] if nbest else ""
        vit = self.viterbi(pinyin, nbest)
        if top1 and abs(len(vit) - len(top1)) > self.cfg.length_guard * len(top1):
            vit = ""
        cands = list(dict.fromkeys(list(nbest) + [vit]))
        cands = [c for c in cands if c]
        if not cands:
            return ""
        # 拼音项只看**相对**差异：以 N-best 能达到的最小 SER 为基线。
        # P̂ 整体不可靠时（重复、截断），所有候选同样"对不上"，这一项自动退化为 0，
        # 排序交回语言模型；只有当某个候选明显更贴合 P̂ 时才起作用。
        sers = {c: _ser(pinyin, text_to_pinyin(c)[1]) for c in cands}
        base = min(sers[c] for c in nbest if c in sers) if nbest else 0.0
        best, best_score = cands[0], -1e18
        for c in cands:
            score = self.lm_score(c) - self.cfg.pinyin_weight * max(sers[c] - base, 0.0)
            if c == top1:
                score += self.cfg.top1_bonus
            if score > best_score:
                best, best_score = c, score
        return best


# --------------------------------------------------------------------------- #
#                                  LLM 精化器                                   #
# --------------------------------------------------------------------------- #
def apply_guard(raw_output: str, pinyin: Sequence[str], nbest: Sequence[str] = (),
                length_guard: float = 1.6) -> str:
    """Reject malformed/expanded output, optionally falling back to char top-1.

    论文报告零样本 LLM 使 CER 从 37.23 升到 37.86，主要就是自由改写/补全造成的；
    这条规则把"改坏"的下界钉在 top-1 上。
    """
    hyp = parse_response(raw_output)
    top1 = nbest[0] if nbest else ""
    n_syl = max(len(pinyin), 1)
    if not hyp or len(hyp) > length_guard * n_syl:
        return top1
    return hyp


@dataclass
class LLMConfig:
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    adapter_path: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 128        # 中文长句 + 少量前后缀，留足余量
    batch_size: int = 8
    temperature: float = 0.0
    length_guard: float = 1.6        # 输出长度 / 音节数 超过该比例则回退到 top-1
    trust_remote_code: bool = True


class LLMRefiner:
    def __init__(self, cfg: LLMConfig = LLMConfig()):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.torch = torch
        dtype = ({"auto": "auto", "fp16": torch.float16, "bf16": torch.bfloat16,
                  "fp32": torch.float32}[cfg.dtype])
        self.tok = AutoTokenizer.from_pretrained(cfg.model_path,
                                                 trust_remote_code=cfg.trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype=dtype, trust_remote_code=cfg.trust_remote_code)
        if cfg.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, cfg.adapter_path)
        device = cfg.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(device).eval()
        self.device = device

    def _render(self, pinyin, nbest=()) -> str:
        return self.tok.apply_chat_template(build_messages(pinyin, nbest), tokenize=False,
                                            add_generation_prompt=True)

    def refine_batch(self, items: Sequence[Dict]) -> List[str]:
        torch = self.torch
        prompts = [self._render(it["pinyin"], it.get("nbest", [])) for it in items]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.cfg.max_new_tokens,
                                      do_sample=self.cfg.temperature > 0,
                                      temperature=max(self.cfg.temperature, 1e-5),
                                      pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        return [apply_guard(raw, it["pinyin"], it.get("nbest", []), self.cfg.length_guard)
                for it, raw in zip(items, texts)]

    def refine(self, pinyin, nbest=()) -> str:
        return self.refine_batch([{"pinyin": list(pinyin), "nbest": list(nbest)}])[0]


# --------------------------------------------------------------------------- #
#                                 批量精化入口                                   #
# --------------------------------------------------------------------------- #
@dataclass
class RefineRunConfig:
    hyp_jsonl: str = ""
    out_jsonl: str = ""
    refiner: str = "ngram"          # "ngram" | "llm"
    lm_texts: str = ""              # ngram 精化器的训练文本 (manifest 或纯文本)
    nbest: int = 5
    ngram: NgramConfig = field(default_factory=NgramConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def _load_texts(path: str) -> List[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                texts.append(json.loads(line)["text"])
            else:
                texts.append(line)
    return texts


def run_refine(cfg: RefineRunConfig) -> Dict[str, float]:
    records = [json.loads(l) for l in open(cfg.hyp_jsonl, encoding="utf-8") if l.strip()]
    items = [{"id": r["id"], "ref": r["ref"], "pinyin": r.get("pinyin", []),
              "nbest": [h["text"] for h in r.get("nbest", [])][: cfg.nbest]} for r in records]

    if cfg.refiner == "ngram":
        if not cfg.lm_texts:
            raise ValueError("ngram 精化器需要 --lm-texts 指定训练文本")
        refiner = NgramPinyinRefiner(cfg.ngram).fit(_load_texts(cfg.lm_texts))
        hyps = [refiner.refine(it["pinyin"], it["nbest"]) for it in items]
    elif cfg.refiner == "llm":
        refiner = LLMRefiner(cfg.llm)
        hyps = []
        bs = cfg.llm.batch_size
        for i in range(0, len(items), bs):
            hyps.extend(refiner.refine_batch(items[i:i + bs]))
            print(f"  refined {min(i + bs, len(items))}/{len(items)}", flush=True)
    else:
        raise ValueError(f"unknown refiner: {cfg.refiner}")

    before, after = ErrorStats(), ErrorStats()
    has_char_baseline = any(item["nbest"] for item in items)
    out_rows = []
    for it, hyp in zip(items, hyps):
        top1 = it["nbest"][0] if it["nbest"] else ""
        if top1:
            before.update(list(it["ref"]), list(top1))
        after.update(list(it["ref"]), list(hyp))
        out_rows.append({**it, "top1": top1, "refined": hyp})
    stats = {"cer_stage1_char": before.rate if has_char_baseline else None,
             "cer_refined": after.rate,
             "abs_gain_vs_char": (before.rate - after.rate if has_char_baseline else None),
             "n": len(items)}
    if cfg.out_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.out_jsonl)), exist_ok=True)
        with open(cfg.out_jsonl, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    baseline = (f"char CER={100 * stats['cer_stage1_char']:.2f}% -> "
                if has_char_baseline else "Pinyin-only -> ")
    gain = (f" (abs {100 * stats['abs_gain_vs_char']:+.2f})"
            if has_char_baseline else "")
    print(f"[refine/{cfg.refiner}] {baseline}refined CER="
          f"{100 * stats['cer_refined']:.2f}%{gain}", flush=True)
    return stats
