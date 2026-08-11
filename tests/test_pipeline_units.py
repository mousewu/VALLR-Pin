"""可离线运行的单元测试：`python -m pytest tests -q` 或 `python tests/test_pipeline_units.py`。"""

from __future__ import annotations

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vallr_pin.data.synthetic import CORPUS  # noqa: E402
from vallr_pin.engine.metrics import cer, edit_ops  # noqa: E402
from vallr_pin.llm.build_data import BuildConfig, build_instruction_data  # noqa: E402
from vallr_pin.llm.prompt import build_user_prompt, parse_response  # noqa: E402
from vallr_pin.llm.refine import NgramPinyinRefiner, apply_guard  # noqa: E402
from vallr_pin.models.vallr_pin import VallrPin, VallrPinConfig  # noqa: E402
from vallr_pin.text.pinyin import canonical_syllables, text_to_pinyin  # noqa: E402
from vallr_pin.text.tokenizer import DualTokenizer  # noqa: E402


def test_pinyin_alignment():
    chars, syls = text_to_pinyin("我想去银行办一张卡")
    assert len(chars) == len(syls) == 9
    assert syls[3:5] == ["yin", "hang"]          # 词组消歧：不是 "xing"
    assert len(canonical_syllables()) > 380      # 论文的 397 建模单元同量级


def test_loanword_pinyin_for_code_switched_speech():
    from vallr_pin.text.loanword import coverage, find_latin_tokens
    from vallr_pin.text.pinyin import text_to_pinyin_mixed

    toks, syls, unknown = text_to_pinyin_mixed("用RL的方式去管理，而不是用SFT")
    assert toks[1] == "RL" and not unknown
    assert syls[1:5] == ["a", "er", "ai", "er"]        # 缩写按字母音展开
    # 字符侧 1 个 token ↔ 音节侧 N 个音节，两条序列不再等长（双解码器允许）
    assert len(syls) > len(toks)

    toks, syls, _ = text_to_pinyin_mixed("我们都写在paper里写了")
    assert "paper" in toks and syls[5:7] == ["pei", "po"]

    rate, unk = coverage(find_latin_tokens("Muon优化器对token efficiency提升很大"))
    assert rate == 1.0 and not unk


def test_tokenizer_roundtrip():
    tok = DualTokenizer.build_from_texts(CORPUS)
    cids, pids = tok.encode("我的手机放在桌子上了")
    assert tok.decode_chars(cids) == "我的手机放在桌子上了"
    assert tok.decode_pinyin(pids)[:2] == ["wo", "de"]


def test_metrics():
    assert edit_ops(list("我想去银行"), list("我想去银航")) == (1, 0, 0)
    assert abs(cer(["我想去银行"], ["我想去银航"]) - 0.2) < 1e-9


def test_model_forward_and_decode():
    tok = DualTokenizer.build_from_texts(CORPUS)
    cfg = VallrPinConfig(char_vocab_size=len(tok.char), pinyin_vocab_size=len(tok.pinyin),
                         d_model=32, heads=2, ffn=64, enc_layers=1, char_dec_layers=1,
                         pinyin_dec_layers=1, frontend_width=8, sanm_kernel=5)
    model = VallrPin(cfg)
    b, t = 2, 24
    out = model(torch.randn(b, t, 1, 32, 32), torch.tensor([t, t - 4]),
                torch.randint(3, len(tok.char), (b, 5)), torch.tensor([5, 4]),
                torch.randint(3, len(tok.pinyin), (b, 5)), torch.tensor([5, 4]))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    model.eval()
    mem, mask = model.encode(torch.randn(1, t, 1, 32, 32), torch.tensor([t]))
    hyps = model.beam_search_chars(mem, mask, beam=3, nbest=2)
    assert 1 <= len(hyps) <= 2 and all(h.tokens for h in hyps)
    py_hyps = model.beam_search_pinyin(mem, mask, beam=3, nbest=2)
    assert 1 <= len(py_hyps) <= 2 and all(h.tokens for h in py_hyps)
    assert isinstance(model.greedy_pinyin(mem, mask), list)


def test_stage1_preserves_visual_time_and_rejects_impossible_ctc():
    tok = DualTokenizer.build_from_texts(CORPUS)
    cfg = VallrPinConfig(char_vocab_size=len(tok.char), pinyin_vocab_size=len(tok.pinyin),
                         d_model=32, heads=2, ffn=64, enc_layers=1,
                         frontend_width=8, sanm_kernel=5)
    model = VallrPin(cfg).eval()
    video = torch.randn(1, 19, 1, 32, 32)
    memory, mask = model.encode(video, torch.tensor([17]))
    assert memory.shape[1] == 19 and int(mask.sum()) == 17

    with pytest.raises(ValueError, match="pinyin CTC targets require"):
        model(video, torch.tensor([4]), torch.tensor([[3, 4, 5]]), torch.tensor([3]),
              torch.tensor([[3, 4, 5, 6, 7]]), torch.tensor([5]))


def test_prompt_and_parsing():
    p = build_user_prompt(["wo", "xiang", "qu", "yin", "hang"], ["我想去银航", "我想去银行"])
    assert "wo xiang qu yin hang" in p and "1. 我想去银航" in p
    assert parse_response("修正后的句子：我想去银行。") == "我想去银行"
    assert parse_response("好的\n我想去银行") == "我想去银行"


def test_build_instruction_data_filters():
    recs = [
        {"id": "a", "ref": "我想去银行", "pinyin": ["wo", "xiang", "qu", "yin", "hang"],
         "nbest": [{"text": "我想去银航"}, {"text": "我想去银行"}]},
        {"id": "b", "ref": "我想去银行", "pinyin": ["wo", "xiang", "qu", "yin", "hang"],
         "nbest": [{"text": "完全不相干的胡话内容"}]},          # CER 过高 -> 丢弃
    ]
    rows = build_instruction_data(recs, BuildConfig(max_cer=0.5, keep_correct=1.0))
    assert len(rows) == 1
    assert rows[0]["messages"][-1]["content"] == "我想去银行"
    assert rows[0]["messages"][-1]["role"] == "assistant"


def test_llm_output_guard():
    py = ["wo", "xiang", "qu", "yin", "hang"]
    nbest = ["我想去银航", "我想去银行"]
    assert apply_guard("我想去银行", py, nbest) == "我想去银行"
    # 扩写 / 解释 -> 回退 top-1
    assert apply_guard("我想去银行取一点现金然后再去超市买点东西回家", py, nbest) == "我想去银航"
    assert apply_guard("Sorry, I cannot help.", py, nbest) == "我想去银航"


def test_ngram_refiner_fixes_homophone():
    refiner = NgramPinyinRefiner().fit(CORPUS)
    pinyin = text_to_pinyin("我的手机放在桌子上了")[1]
    # Stage-I 把 "手机" 错成同音的 "收集"
    out = refiner.refine(pinyin, ["我的收集放在桌子上了"])
    assert out == "我的手机放在桌子上了"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
