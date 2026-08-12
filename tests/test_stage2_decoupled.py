from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

import yaml

from vallr_pin.cli import fill_dataclass
from vallr_pin.engine.tracking import SwanLabConfig, SwanLabTracker
from vallr_pin.llm.lora_sft import IndexedJsonl, SftConfig, Stage2Dataset
from vallr_pin.llm.noise import PinyinNoiseConfig, corrupt_pinyin
from vallr_pin.llm.prompt import build_messages, build_user_prompt
from vallr_pin.llm.text_data import (TextBuildConfig, TextSource, build_text_corpus,
                                     materialize_instruction_data)
from vallr_pin.text.pinyin import text_to_pinyin


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        suffix = "assistant:" if add_generation_prompt else ""
        return "|".join(item["content"] for item in messages) + suffix

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(ch) % 251 + 1 for ch in text]}


def test_pinyin_only_prompt_has_no_character_candidates():
    prompt = build_user_prompt(["wo", "de", "shou", "ji"])
    assert "候选转写" not in prompt
    messages = build_messages(["wo", "de", "shou", "ji"], answer="我的手机")
    assert messages[-1] == {"role": "assistant", "content": "我的手机"}
    assert "候选转写" not in messages[1]["content"]


def test_noise_is_reproducible_and_never_deletes_everything():
    clean = text_to_pinyin("我的手机放在桌子上")[1]
    cfg = PinyinNoiseConfig(clean_prob=0.0, mild_prob=0.0,
                            severe_min_rate=0.4, severe_max_rate=0.4)
    first, meta1 = corrupt_pinyin(clean, cfg, random.Random(7))
    second, meta2 = corrupt_pinyin(clean, cfg, random.Random(7))
    assert first == second and meta1 == meta2
    assert first and meta1["severity"] == "severe" and meta1["edits"] > 0


def test_text_corpus_is_document_split_and_excludes_vsr_test(tmp_path: Path):
    source = tmp_path / "text.jsonl"
    source.write_text("\n".join([
        json.dumps({"doc": "d1", "text": "今天天气很好。我们出去散步。"}, ensure_ascii=False),
        json.dumps({"doc": "d2", "text": "今天天气很好。我的手机在桌上。"}, ensure_ascii=False),
    ]), encoding="utf-8")
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps({"text": "我的手机在桌上"}, ensure_ascii=False),
                       encoding="utf-8")
    out = tmp_path / "out"
    report = build_text_corpus(TextBuildConfig(
        sources=[TextSource("sample", str(source), "jsonl", "text", "doc")],
        out_dir=str(out), min_chars=2, val_percent=30, test_percent=30,
        exclude_paths=[str(heldout)]))
    rows = []
    for split in ("train", "val", "test"):
        rows.extend((split, json.loads(line)) for line in
                    (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
    assert report["rejected"]["duplicate"] == 1
    assert report["rejected"]["heldout_contamination"] == 1
    by_doc = {}
    for split, row in rows:
        by_doc.setdefault(row["document_id"], set()).add(split)
        assert len(row["text"]) == len(row["pinyin"])
    assert all(len(splits) == 1 for splits in by_doc.values())


def test_stage2_dataset_generates_online_noise_and_masks_prompt():
    pinyin = text_to_pinyin("我的手机")[1]
    cfg = PinyinNoiseConfig(clean_prob=0.0, mild_prob=1.0,
                            mild_min_rate=0.5, mild_max_rate=0.5)
    ds = Stage2Dataset([{"text": "我的手机", "pinyin": pinyin}], FakeTokenizer(),
                       max_length=512, noise=cfg, variants=2, seed=9, training=True)
    ds.set_epoch(1)
    messages1 = ds.messages_at(0)
    ids, labels = ds[0]
    assert messages1[-1]["content"] == "我的手机"
    assert len(ids) == len(labels) and -100 in labels
    ds.set_epoch(2)
    assert ds.messages_at(0) != messages1 or ds.messages_at(1) != messages1


def test_indexed_jsonl_reads_lazily(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    rows = IndexedJsonl(str(path))
    assert len(rows) == 2 and rows[0]["id"] == 1 and rows[-1]["id"] == 2


def test_materialized_data_and_nested_config(tmp_path: Path):
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps({"id": "x", "text": "我的手机",
                                 "pinyin": text_to_pinyin("我的手机")[1]},
                                ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "messages.jsonl"
    count = materialize_instruction_data(str(clean), str(out), variants_per_text=3)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert count == len(rows) == 3 and all("messages" in row for row in rows)

    raw = yaml.safe_load("""
noise:
  clean_prob: 0.4
swanlab:
  enabled: true
  mode: offline
""")
    cfg = fill_dataclass(SftConfig, raw)
    assert cfg.noise.clean_prob == 0.4 and cfg.swanlab.mode == "offline"


def test_disabled_swanlab_has_no_dependency():
    tracker = SwanLabTracker(SwanLabConfig(enabled=False), {"x": 1}, is_main=True)
    assert not tracker.enabled
    tracker.log({"loss": 1.0}, step=1)
    tracker.finish()


def test_swanlab_adapter_logs_only_when_enabled(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        init=lambda **kwargs: calls.append(("init", kwargs)) or object(),
        log=lambda values, step=None: calls.append(("log", values, step)),
        finish=lambda: calls.append(("finish",)))
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    tracker = SwanLabTracker(
        SwanLabConfig(enabled=True, project="test", experiment_name="ddp",
                      mode="offline", logdir="logs"),
        {"batch_size": 2}, is_main=True)
    tracker.log({"loss": 0.5, "ignored": None}, step=4)
    tracker.finish()
    assert calls[0][0] == "init" and calls[0][1]["project"] == "test"
    assert calls[1] == ("log", {"loss": 0.5}, 4) and calls[-1] == ("finish",)
