"""Stage-I 解码：输出 N-best 字符假设 + 拼音假设 (Stage-II 的输入)。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ..data.dataset import LipReadingDataset, VideoTransform, collate
from ..models.vallr_pin import VallrPin
from ..text.pinyin import text_to_pinyin
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
    include_char_hypotheses: Optional[bool] = None  # auto: only when alpha > 0


@torch.no_grad()
def decode_manifest(model: VallrPin, tok: DualTokenizer, cfg: DecodeConfig,
                    tag: str = "") -> List[Dict]:
    device = resolve_device(cfg.device)
    model = model.to(device).eval()
    ds = LipReadingDataset(cfg.manifest, tok, VideoTransform(cfg.crop_size, train=False),
                           root=cfg.data_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
                        collate_fn=collate)

    records, cer1, cerO, ser = [], ErrorStats(), ErrorStats(), ErrorStats()
    for i, batch in enumerate(loader):
        if cfg.max_utts and i >= cfg.max_utts:
            break
        video = batch["video"].to(device)
        vlens = batch["video_lens"].to(device)
        memory, mask = model.encode(video, vlens)
        use_chars = (model.cfg.alpha > 0.0 if cfg.include_char_hypotheses is None
                     else cfg.include_char_hypotheses)
        hyps = (model.beam_search_chars(memory, mask, beam=cfg.beam, nbest=cfg.nbest,
                                        length_penalty=cfg.length_penalty)
                if use_chars else [])
        nbest = [{"text": tok.decode_chars(h.tokens), "score": round(h.score, 4),
                  "att": round(h.att_score, 4), "ctc": round(h.ctc_score, 4)} for h in hyps]
        if cfg.pinyin_mode == "ctc":
            py_hyps = model.beam_search_pinyin(memory, mask, beam=cfg.beam,
                                               nbest=cfg.nbest)
            if not py_hyps:
                py_ids = model.ctc_greedy(model.pinyin_ctc, memory, mask.sum(-1))[0]
                py_hyps = []
            else:
                py_ids = py_hyps[0].tokens
        else:  # compatibility alias: Stage-I v2 is CTC-only
            py_ids = model.greedy_pinyin(memory, mask)
            py_hyps = []
        pinyin = tok.decode_pinyin(py_ids)
        pinyin_nbest = [{"pinyin": tok.decode_pinyin(h.tokens),
                         "score": round(h.score, 4)} for h in py_hyps]

        ref = batch["texts"][0]
        ref_chars, ref_py = text_to_pinyin(ref)
        rec = {"id": batch["ids"][0], "ref": "".join(ref_chars), "ref_pinyin": ref_py,
               "pinyin": pinyin, "pinyin_nbest": pinyin_nbest,
               "nbest": nbest, "ckpt": tag}
        records.append(rec)

        top1 = nbest[0]["text"] if nbest else ""
        ser.update(ref_py, pinyin)
        if nbest:
            cer1.update(ref_chars, list(top1))
            # oracle：N-best 里最接近参考的那条
            best = min((list(h["text"]) for h in nbest),
                       key=lambda cand: _quick_er(ref_chars, cand))
            cerO.update(ref_chars, best)

    stats = {"cer_top1": cer1.rate if cer1.total else None,
             "cer_oracle": cerO.rate if cerO.total else None, "pinyin_ser": ser.rate,
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
    print(f"[decode{'/' + tag if tag else ''}] utts={stats['n_utts']} "
          f"{char_summary}pinyin SER={100 * stats['pinyin_ser']:.2f}%", flush=True)
    return records


def _quick_er(ref, hyp) -> float:
    from .metrics import edit_ops
    s, d, i = edit_ops(ref, hyp)
    return (s + d + i) / max(len(ref), 1)


def load_stage1(ckpt: str, vocab_dir: str):
    model = VallrPin.load(ckpt)
    tok = DualTokenizer.load(vocab_dir)
    return model, tok
