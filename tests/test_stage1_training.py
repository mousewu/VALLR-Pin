from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vallr_pin.data.corpora import BuildConfig, CorpusSpec, build_manifests
from vallr_pin.data.samplers import DistributedBucketBatchSampler
from vallr_pin.engine.trainer import TrainConfig, Trainer
from vallr_pin.engine.decode import DecodeConfig, decode_manifest
from vallr_pin.models.vallr_pin import VallrPin, VallrPinConfig
from vallr_pin.text.tokenizer import DualTokenizer


def _npy(path: Path, frames: int = 12):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.random.default_rng(0).integers(0, 255, (frames, 32, 32), dtype=np.uint8))


def test_corpus_builder_speaker_split_and_pseudo_gate(tmp_path):
    root = tmp_path / "corpus"; media = root / "roi96"; rows = []
    for speaker in range(12):
        uid = f"s{speaker}/u{speaker}"
        _npy(media / f"{uid}.npy")
        rows.append(f"{uid}\t今天天气很好\ts{speaker}\n")
    (root / "labels.tsv").parent.mkdir(parents=True, exist_ok=True)
    (root / "labels.tsv").write_text("".join(rows), encoding="utf-8")
    spec = CorpusSpec(name="toy", root=str(root), annotation="labels.tsv",
                      format="delimited", speaker_column=2, media_root="roi96",
                      media_glob="**/*.npy")
    out = tmp_path / "manifests"
    disabled = CorpusSpec(name="future", root="/missing", annotation="missing.jsonl",
                          enabled=False)
    report = build_manifests(BuildConfig(sources=[spec, disabled], out_dir=str(out),
                                         dev_speaker_percent=25, test_speaker_percent=25))
    assert report["accepted"] == 12 and report["speaker_overlap"] == 0
    assert report["rejected"]["future:source_disabled"] == 1
    split_speakers = {}
    for split in ("train", "dev", "test"):
        split_speakers[split] = {json.loads(line)["speaker_id"] for line in
                                 (out / f"{split}.jsonl").read_text().splitlines()}
    assert not split_speakers["train"] & split_speakers["dev"]


def test_distributed_bucket_sampler_is_deterministic_and_sharded():
    lengths = list(range(1, 41)); sources = ["large"] * 30 + ["small"] * 10
    samplers = [DistributedBucketBatchSampler(
        lengths, sources, batch_size=4, source_weights={"large": .5, "small": .5},
        epoch_samples=40, rank=rank, world_size=2, seed=7) for rank in range(2)]
    for sampler in samplers: sampler.set_epoch(3)
    batches = [list(sampler) for sampler in samplers]
    assert len(batches[0]) == len(batches[1]) == 5
    assert batches[0] == list(samplers[0])
    sampled_sources = [sources[i] for rank_batches in batches for batch in rank_batches for i in batch]
    assert 10 <= sampled_sources.count("small") <= 30


def test_trainer_checkpoint_resume(tmp_path):
    data = tmp_path / "data"; items = []
    for index, text in enumerate(["今天天气", "明天上班", "我们学习", "大家回家", "天气不错", "继续训练"]):
        path = data / f"u{index}.npy"; _npy(path, 16)
        items.append({"id": f"u{index}", "video": str(path), "text": text,
                      "speaker_id": f"s{index}", "source": "toy", "n_frames": 16})
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items[:4]))
    dev.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items[4:]))
    model = VallrPinConfig(d_model=16, heads=2, ffn=32, enc_layers=1,
                           frontend_width=4, sanm_kernel=3, alpha=.1)
    out = tmp_path / "exp"
    cfg = TrainConfig(train_manifest=str(train), dev_manifest=str(dev), out_dir=str(out),
                      epochs=1, batch_size=2, accum_steps=1, warmup_steps=2,
                      num_workers=0, crop_size=32, device="cpu", amp=False,
                      save_every=1, model=model)
    Trainer(cfg).fit()
    last = out / "ckpts" / "last.pt"
    assert last.exists()
    cfg.epochs = 2; cfg.resume = str(last)
    resumed = Trainer(cfg)
    assert resumed.start_epoch == 2 and resumed.step > 0
    resumed.fit()
    assert (out / "ckpts" / "ckpt_ep2.pt").exists()


def test_pinyin_only_decode_does_not_use_untrained_character_head(tmp_path):
    text = "我的手机"
    tok = DualTokenizer.build_from_texts([text])
    cfg = VallrPinConfig(char_vocab_size=len(tok.char), pinyin_vocab_size=len(tok.pinyin),
                         d_model=16, heads=2, ffn=32, enc_layers=1,
                         frontend_width=4, sanm_kernel=3, alpha=0.0)
    model = VallrPin(cfg)
    video = tmp_path / "u.npy"; _npy(video, 16)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "u", "video": str(video), "text": text,
                                    "n_frames": 16}, ensure_ascii=False) + "\n")
    output = tmp_path / "decode.jsonl"
    rows = decode_manifest(model, tok, DecodeConfig(
        manifest=str(manifest), out_jsonl=str(output), beam=2, nbest=2,
        crop_size=32, device="cpu"))
    assert rows[0]["nbest"] == [] and rows[0]["pinyin_nbest"]
    stats = json.loads((tmp_path / "decode.stats.json").read_text())
    assert stats["cer_top1"] is None and stats["cer_oracle"] is None
