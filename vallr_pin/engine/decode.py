"""Stage-I decoding for text-only, Pinyin-only, and joint CTC models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ..data.dataset import LipReadingDataset, VideoTransform, collate
from ..models.vallr_pin import VallrPin
from ..text.pinyin import text_to_pinyin_mixed
from ..text.tokenizer import DualTokenizer
from .metrics import ErrorStats
from .trainer import resolve_device


@dataclass
class DecodeConfig:
    manifest: str = ""
    data_root: str = ""
    out_jsonl: str = ""
    beam: int = 10
    nbest: int = 5
    ctc_weight: float = 0.3
    length_penalty: float = 0.6
    pinyin_mode: str = "ctc"         # "ar" retained only as a greedy compatibility alias
    crop_size: int = 88
    device: str = "auto"
    max_utts: Optional[int] = None
    num_workers: int = 0
    include_char_hypotheses: Optional[bool] = None  # auto: only when trained
    include_pinyin_hypotheses: Optional[bool] = None  # auto: only when trained


def _use_head(requested: Optional[bool], trained: bool, name: str) -> bool:
    if requested is None:
        return trained
    if requested and not trained:
        raise ValueError(f"cannot decode untrained {name} CTC head")
    return requested


@torch.no_grad()
def decode_manifest(model: VallrPin, tok: DualTokenizer, cfg: DecodeConfig,
                    tag: str = "") -> List[Dict]:
    device = resolve_device(cfg.device)
    model = model.to(device).eval()
    ds = LipReadingDataset(cfg.manifest, tok, VideoTransform(cfg.crop_size, train=False),
                           root=cfg.data_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
                        collate_fn=collate)
    use_chars = _use_head(cfg.include_char_hypotheses,
                          model.cfg.uses_text_head, "text")
    use_pinyin = _use_head(cfg.include_pinyin_hypotheses,
                           model.cfg.uses_pinyin_head, "Pinyin")
    if not use_chars and not use_pinyin:
        raise ValueError("decode requires at least one enabled CTC head")

    records, cer1, cerO, ser = [], ErrorStats(), ErrorStats(), ErrorStats()
    text_oov, text_tokens = 0, 0
    for i, batch in enumerate(loader):
        if cfg.max_utts and i >= cfg.max_utts:
            break
        video = batch["video"].to(device)
        vlens = batch["video_lens"].to(device)
        memory, mask = model.encode(video, vlens)
        hyps = (model.beam_search_chars(memory, mask, beam=cfg.beam, nbest=cfg.nbest,
                                        length_penalty=cfg.length_penalty)
                if use_chars else [])
        nbest = [{"text": tok.decode_chars(h.tokens),
                  "tokens": tok.char.decode(h.tokens), "score": round(h.score, 4),
                  "att": round(h.att_score, 4), "ctc": round(h.ctc_score, 4)} for h in hyps]
        if use_pinyin and cfg.pinyin_mode == "ctc":
            py_hyps = model.beam_search_pinyin(memory, mask, beam=cfg.beam,
                                               nbest=cfg.nbest)
            if not py_hyps:
                py_ids = model.ctc_greedy(model.pinyin_ctc, memory, mask.sum(-1))[0]
                py_hyps = []
            else:
                py_ids = py_hyps[0].tokens
        elif use_pinyin:  # compatibility alias: Stage-I v2 is CTC-only
            py_ids = model.greedy_pinyin(memory, mask)
            py_hyps = []
        else:
            py_ids, py_hyps = [], []
        pinyin = tok.decode_pinyin(py_ids)
        pinyin_nbest = [{"pinyin": tok.decode_pinyin(h.tokens),
                         "score": round(h.score, 4)} for h in py_hyps]

        ref = batch["texts"][0]
        ref_tokens, ref_py, _ = text_to_pinyin_mixed(ref)
        rec = {"id": batch["ids"][0], "ref": "".join(ref_tokens), "ref_pinyin": ref_py,
               "pinyin": pinyin, "pinyin_nbest": pinyin_nbest,
               "nbest": nbest, "ckpt": tag}
        records.append(rec)

        if use_pinyin:
            ser.update(ref_py, pinyin)
        if nbest:
            cer1.update(ref_tokens, nbest[0]["tokens"])
            text_oov += sum(token not in tok.char.unit2id for token in ref_tokens)
            text_tokens += len(ref_tokens)
            # oracle：N-best 里最接近参考的那条
            best = min((h["tokens"] for h in nbest),
                       key=lambda cand: _quick_er(ref_tokens, cand))
            cerO.update(ref_tokens, best)

    stats = {"cer_top1": cer1.rate if cer1.total else None,
             "cer_oracle": cerO.rate if cerO.total else None,
             "pinyin_ser": ser.rate if ser.total else None,
             "text_oov_rate": (text_oov / max(text_tokens, 1) if use_chars else None),
             "n_utts": len(records)}
    if cfg.out_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.out_jsonl)), exist_ok=True)
        with open(cfg.out_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(cfg.out_jsonl.replace(".jsonl", ".stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    char_summary = (f"CER(top1)={100 * stats['cer_top1']:.2f}% "
                    f"CER(oracle-{cfg.nbest}best)={100 * stats['cer_oracle']:.2f}% "
                    if stats["cer_top1"] is not None
                    and stats["cer_oracle"] is not None else "character head=disabled ")
    pinyin_summary = (f"pinyin SER={100 * stats['pinyin_ser']:.2f}%"
                      if stats["pinyin_ser"] is not None else "pinyin head=disabled")
    print(f"[decode{'/' + tag if tag else ''}] utts={stats['n_utts']} "
          f"{char_summary}{pinyin_summary}", flush=True)
    return records


def _quick_er(ref, hyp) -> float:
    from .metrics import edit_ops
    s, d, i = edit_ops(ref, hyp)
    return (s + d + i) / max(len(ref), 1)


def load_stage1(ckpt: str, vocab_dir: str):
    model = VallrPin.load(ckpt)
    tok = DualTokenizer.load(vocab_dir)
    return model, tok
