"""VALLR-Pin Stage-I: temporal visual encoder with text and Pinyin CTC heads.

The visual stream preserves one feature per input frame.  Either head can be
trained alone, or both can be optimized jointly with normalized task weights.
The production pipeline configuration remains Pinyin-only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoders import SanmEncoder
from .frontend import VisualFrontend
from .sanm import lengths_to_mask


@dataclass
class VallrPinConfig:
    char_vocab_size: int = 100
    pinyin_vocab_size: int = 413
    d_model: int = 256
    heads: int = 4
    ffn: int = 1024
    enc_layers: int = 12
    sanm_kernel: int = 11
    dropout: float = 0.1
    frontend: str = "resnet18"
    frontend_width: int = 64
    in_channels: int = 1
    # Explicit weights are preferred.  When both are None, the legacy alpha
    # mapping is used: text=alpha, Pinyin=1-alpha.
    text_ctc_weight: Optional[float] = None
    pinyin_ctc_weight: Optional[float] = None
    alpha: float = 0.1
    # Kept so old YAML files fail gracefully when loaded. They no longer create
    # autoregressive decoders in architecture_version=2.
    lambda_ctc: float = 1.0
    char_dec_layers: int = 0
    pinyin_dec_layers: int = 0
    label_smoothing: float = 0.0
    architecture_version: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    def ctc_weights(self) -> Tuple[float, float]:
        """Return normalized ``(text, Pinyin)`` CTC weights."""
        explicit = self.text_ctc_weight is not None or self.pinyin_ctc_weight is not None
        if explicit:
            if self.text_ctc_weight is None or self.pinyin_ctc_weight is None:
                raise ValueError(
                    "text_ctc_weight and pinyin_ctc_weight must be set together")
            text, pinyin = float(self.text_ctc_weight), float(self.pinyin_ctc_weight)
        else:
            if not 0.0 <= self.alpha <= 1.0:
                raise ValueError("legacy alpha must be in [0, 1]")
            text, pinyin = float(self.alpha), 1.0 - float(self.alpha)
        if text < 0.0 or pinyin < 0.0 or text + pinyin <= 0.0:
            raise ValueError("CTC head weights must be non-negative with a positive sum")
        total = text + pinyin
        return text / total, pinyin / total

    @property
    def uses_text_head(self) -> bool:
        return self.ctc_weights()[0] > 0.0

    @property
    def uses_pinyin_head(self) -> bool:
        return self.ctc_weights()[1] > 0.0

    @property
    def head_mode(self) -> str:
        text, pinyin = self.ctc_weights()
        if text == 0.0:
            return "pinyin"
        if pinyin == 0.0:
            return "text"
        return "joint"


@dataclass
class Hypothesis:
    tokens: List[int] = field(default_factory=list)
    att_score: float = 0.0
    ctc_score: float = 0.0
    score: float = 0.0


def _logadd(*values: float) -> float:
    finite = [v for v in values if v != -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


class VallrPin(nn.Module):
    """Frame-preserving Stage-I model.

    Input is ``(B,T,C,H,W)``.  The visual frontend and SANM encoder both keep
    the temporal length, so CTC has enough alignment states for natural
    Mandarin utterances.  Text and Pinyin CTC can each be the sole objective,
    or share the encoder in a weighted multi-task experiment.
    """

    def __init__(self, cfg: VallrPinConfig):
        super().__init__()
        if cfg.architecture_version != 2:
            raise ValueError("only architecture_version=2 is supported")
        cfg.ctc_weights()  # fail early on invalid head configuration
        self.cfg = cfg
        self.frontend = VisualFrontend(cfg.frontend, cfg.d_model, cfg.in_channels,
                                       cfg.frontend_width, cfg.dropout)
        self.encoder = SanmEncoder(cfg.d_model, cfg.heads, cfg.ffn, cfg.enc_layers,
                                   cfg.sanm_kernel, cfg.dropout)
        self.char_ctc = nn.Linear(cfg.d_model, cfg.char_vocab_size)
        self.pinyin_ctc = nn.Linear(cfg.d_model, cfg.pinyin_vocab_size)

    def encode(self, video: torch.Tensor, video_lens: torch.Tensor):
        feats = self.frontend(video)
        pad_mask = lengths_to_mask(video_lens, feats.size(1))
        return self.encoder(feats, pad_mask), pad_mask

    @staticmethod
    def _required_ctc_steps(targets: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
        """CTC needs one extra state between adjacent identical labels."""
        required = target_lens.clone()
        for row in range(targets.size(0)):
            n = int(target_lens[row])
            if n > 1:
                required[row] += (targets[row, 1:n] == targets[row, :n - 1]).sum()
        return required

    @staticmethod
    def _ctc_loss(head: nn.Linear, memory: torch.Tensor, enc_lens: torch.Tensor,
                  targets: torch.Tensor, target_lens: torch.Tensor,
                  name: str) -> torch.Tensor:
        required = VallrPin._required_ctc_steps(targets, target_lens)
        invalid = required > enc_lens
        if bool(invalid.any()):
            rows = torch.nonzero(invalid).flatten().tolist()
            raise ValueError(
                f"{name} CTC targets require more alignment steps than visual frames "
                f"at rows {rows}; "
                "shorten the transcript segment or retain more video frames"
            )
        log_probs = F.log_softmax(head(memory).float(), dim=-1).transpose(0, 1)
        flat = torch.cat(
            [targets[i, : int(target_lens[i])] for i in range(targets.size(0))]
        ).to(torch.int32)
        return F.ctc_loss(log_probs, flat, enc_lens.to(torch.int32),
                          target_lens.to(torch.int32), blank=0,
                          reduction="mean", zero_infinity=False)

    def forward(self, video, video_lens, char_ids, char_lens, pinyin_ids, pinyin_lens,
                sos: int = 1, eos: int = 1) -> Dict[str, torch.Tensor]:
        del sos, eos
        memory, pad_mask = self.encode(video, video_lens)
        enc_lens = pad_mask.sum(-1)
        zero = memory.new_zeros(())
        text_weight, pinyin_weight = self.cfg.ctc_weights()
        p_ctc = (self._ctc_loss(self.pinyin_ctc, memory, enc_lens,
                                pinyin_ids, pinyin_lens, "pinyin")
                 if pinyin_weight > 0.0 else zero)
        c_ctc = (self._ctc_loss(self.char_ctc, memory, enc_lens,
                                char_ids, char_lens, "text")
                 if text_weight > 0.0 else zero)
        total = pinyin_weight * p_ctc + text_weight * c_ctc
        return {"loss": total, "loss_char": c_ctc, "loss_pinyin": p_ctc,
                "ctc_char": c_ctc, "ctc_pinyin": p_ctc,
                "ce_char": zero, "ce_pinyin": zero}

    @torch.no_grad()
    def ctc_greedy(self, head: nn.Linear, memory: torch.Tensor,
                   enc_lens: torch.Tensor) -> List[List[int]]:
        ids = head(memory).argmax(-1)
        out = []
        for b in range(ids.size(0)):
            seq, prev = [], -1
            for t in range(int(enc_lens[b])):
                token = int(ids[b, t])
                if token != 0 and token != prev:
                    seq.append(token)
                prev = token
            out.append(seq)
        return out

    @torch.no_grad()
    def ctc_prefix_beam(self, head: nn.Linear, memory: torch.Tensor,
                        enc_mask: torch.Tensor, beam: int = 10,
                        nbest: int = 5, token_topk: int = 32,
                        length_penalty: float = 0.0) -> List[Hypothesis]:
        """CTC prefix beam search for a single utterance."""
        if memory.size(0) != 1:
            raise ValueError("ctc_prefix_beam expects batch size 1")
        length = int(enc_mask.sum())
        logp = F.log_softmax(head(memory)[0, :length].float(), dim=-1).cpu()
        beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -math.inf)}
        for frame in logp:
            keep = min(frame.numel(), max(token_topk, beam * 2))
            tokens = frame.topk(keep).indices.tolist()
            if 0 not in tokens:
                tokens.append(0)
            nxt: Dict[Tuple[int, ...], Tuple[float, float]] = {}

            def add(prefix, blank=-math.inf, nonblank=-math.inf):
                old_b, old_nb = nxt.get(prefix, (-math.inf, -math.inf))
                nxt[prefix] = (_logadd(old_b, blank), _logadd(old_nb, nonblank))

            for prefix, (p_blank, p_nonblank) in beams.items():
                add(prefix, blank=_logadd(p_blank, p_nonblank) + float(frame[0]))
                last = prefix[-1] if prefix else None
                for token in tokens:
                    if token == 0:
                        continue
                    lp = float(frame[token])
                    if token == last:
                        add(prefix, nonblank=p_nonblank + lp)
                        add(prefix + (token,), nonblank=p_blank + lp)
                    else:
                        add(prefix + (token,),
                            nonblank=_logadd(p_blank, p_nonblank) + lp)
            beams = dict(sorted(
                nxt.items(), key=lambda item: _logadd(*item[1]), reverse=True
            )[:beam])

        hyps = []
        for prefix, probs in beams.items():
            if not prefix:
                continue
            raw = _logadd(*probs)
            score = raw / (len(prefix) ** length_penalty if length_penalty else 1.0)
            hyps.append(Hypothesis(list(prefix), ctc_score=raw, score=score))
        hyps.sort(key=lambda hyp: hyp.score, reverse=True)
        return hyps[:nbest]

    @torch.no_grad()
    def beam_search_chars(self, memory: torch.Tensor, enc_mask: torch.Tensor,
                          beam: int = 10, nbest: int = 5,
                          length_penalty: float = 0.0, **_ignored) -> List[Hypothesis]:
        return self.ctc_prefix_beam(self.char_ctc, memory, enc_mask, beam, nbest,
                                    length_penalty=length_penalty)

    @torch.no_grad()
    def beam_search_pinyin(self, memory: torch.Tensor, enc_mask: torch.Tensor,
                           beam: int = 10, nbest: int = 5,
                           length_penalty: float = 0.0) -> List[Hypothesis]:
        return self.ctc_prefix_beam(self.pinyin_ctc, memory, enc_mask, beam, nbest,
                                    length_penalty=length_penalty)

    @torch.no_grad()
    def greedy_pinyin(self, memory: torch.Tensor, enc_mask: torch.Tensor,
                      **_ignored) -> List[int]:
        return self.ctc_greedy(self.pinyin_ctc, memory, enc_mask.sum(-1))[0]

    def save(self, path: str, **extra) -> None:
        torch.save({"cfg": self.cfg.to_dict(), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "VallrPin":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = dict(ckpt["cfg"])
        cfg.setdefault("architecture_version", 1)
        if cfg["architecture_version"] != 2:
            raise ValueError(
                "This checkpoint uses the retired dual-AR architecture. Retrain Stage-I "
                "with architecture_version=2; silent partial loading is intentionally disabled."
            )
        model = cls(VallrPinConfig(**cfg))
        model.load_state_dict(ckpt["state_dict"])
        return model
