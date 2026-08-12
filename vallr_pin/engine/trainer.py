"""Production Stage-I trainer: DDP, bucketing, balancing and exact resume."""

from __future__ import annotations

import json
import math
import os
import random
import time
import dataclasses
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from ..data.dataset import LipReadingDataset, VideoTransform, read_manifest
from ..data.samplers import DistributedBucketBatchSampler
from ..models.vallr_pin import VallrPin, VallrPinConfig
from ..text.pinyin import text_to_pinyin_mixed
from ..text.tokenizer import DualTokenizer
from .metrics import ErrorStats
from .tracking import SwanLabConfig, SwanLabTracker


@dataclass
class TrainConfig:
    train_manifest: str = ""
    dev_manifest: str = ""
    data_root: str = ""
    out_dir: str = "exp/vallr_pin"
    epochs: int = 50
    batch_size: int = 8               # per process/GPU
    bucket_size: int = 40             # batches per length-sorted mega-bucket
    epoch_samples: int = 0            # 0 = number of training rows
    source_weights: Dict[str, float] = field(default_factory=dict)
    accum_steps: int = 4
    lr: float = 5e-4
    warmup_steps: int = 10000
    weight_decay: float = 1e-2
    grad_clip: float = 5.0
    num_workers: int = 8
    crop_size: int = 88
    time_mask: int = 4
    flip_prob: float = 0.5
    mean: float = 0.421
    std: float = 0.165
    max_frames: int = 0               # never silently truncate; 0 means unlimited
    min_frames: int = 4
    require_roi_metadata: bool = True # reject raw_scene / face_crop / legacy manifests
    expected_fps: float = 25.0
    device: str = "auto"
    amp: bool = True
    compile: bool = False
    resume: str = ""
    save_every: int = 1
    keep_ckpts: int = 8
    eval_every: int = 1
    selection_metric: str = "ser"     # pinyin-first default; "cer" for char-only ablation
    save_best_per_metric: bool = True  # joint mode also writes best_cer.pt / best_ser.pt
    log_every: int = 50
    seed: int = 0
    model: VallrPinConfig = field(default_factory=VallrPinConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)


def _dist_info() -> tuple[int, int, int]:
    return (int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1")),
            int(os.environ.get("LOCAL_RANK", "0")))


def resolve_device(name: str, local_rank: int = 0, world_size: int = 1) -> torch.device:
    if world_size > 1:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _init_distributed(device: torch.device, world_size: int) -> None:
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def build_tokenizer(cfg: TrainConfig, is_main: bool = True) -> DualTokenizer:
    vocab_dir = os.path.join(cfg.out_dir, "vocab")
    char_path = os.path.join(vocab_dir, "char_vocab.json")
    if is_main and not os.path.exists(char_path):
        texts = [item["text"] for item in read_manifest(cfg.train_manifest)]
        DualTokenizer.build_from_texts(texts).save(vocab_dir)
    _barrier()
    if not os.path.exists(char_path):
        raise FileNotFoundError(f"vocabulary was not created: {char_path}")
    return DualTokenizer.load(vocab_dir)


def _lr_at(step: int, cfg: TrainConfig) -> float:
    step = max(step, 1)
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(cfg.warmup_steps, 1)
    return cfg.lr * math.sqrt(max(cfg.warmup_steps, 1) / step)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.rank, self.world_size, self.local_rank = _dist_info()
        self.device = resolve_device(cfg.device, self.local_rank, self.world_size)
        _init_distributed(self.device, self.world_size)
        self.is_main = self.rank == 0
        if self.is_main:
            os.makedirs(os.path.join(cfg.out_dir, "ckpts"), exist_ok=True)
        _barrier()
        self._seed(cfg.seed + self.rank)
        self.tok = build_tokenizer(cfg, self.is_main)
        cfg.model.char_vocab_size = len(self.tok.char)
        cfg.model.pinyin_vocab_size = len(self.tok.pinyin)
        if cfg.selection_metric not in {"cer", "ser"}:
            raise ValueError("selection_metric must be cer or ser")
        if cfg.selection_metric == "cer" and not cfg.model.uses_text_head:
            raise ValueError("selection_metric=cer requires an enabled text CTC head")
        if cfg.selection_metric == "ser" and not cfg.model.uses_pinyin_head:
            raise ValueError("selection_metric=ser requires an enabled Pinyin CTC head")

        model: torch.nn.Module = VallrPin(cfg.model).to(self.device)
        if cfg.compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        if self.world_size > 1:
            endpoints = not (cfg.model.uses_text_head and cfg.model.uses_pinyin_head)
            model = DistributedDataParallel(
                model, device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=endpoints)
        self.model = model
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.step, self.start_epoch, self.best = 0, 1, float("inf")
        self.best_by_metric = {"cer": float("inf"), "ser": float("inf")}
        self.train_loader, self.train_batch_sampler = self._train_loader()
        self.dev_loader = self._dev_loader() if cfg.dev_manifest else None
        if cfg.resume:
            self._resume(cfg.resume)
        if self.is_main:
            self._write_config()
        self.tracker = SwanLabTracker(cfg.swanlab, cfg, self.is_main)

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _dataset(self, manifest: str, train: bool) -> LipReadingDataset:
        return LipReadingDataset(
            manifest, self.tok,
            VideoTransform(self.cfg.crop_size, train=train,
                           flip_prob=self.cfg.flip_prob if train else 0.0,
                           time_mask=self.cfg.time_mask if train else 0,
                           mean=self.cfg.mean, std=self.cfg.std),
            root=self.cfg.data_root, max_frames=self.cfg.max_frames,
            min_frames=self.cfg.min_frames,
            require_mouth_roi=self.cfg.require_roi_metadata,
            expected_fps=self.cfg.expected_fps)

    def _train_loader(self):
        from ..data.dataset import collate
        dataset = self._dataset(self.cfg.train_manifest, True)
        lengths = [int(item.get("n_frames", 0)) for item in dataset.items]
        sources = [item.get("source", "unknown") for item in dataset.items]
        sampler = DistributedBucketBatchSampler(
            lengths, sources, self.cfg.batch_size, self.cfg.bucket_size,
            self.cfg.source_weights, self.cfg.epoch_samples,
            self.rank, self.world_size, True, True, self.cfg.seed)
        loader = DataLoader(
            dataset, batch_sampler=sampler, num_workers=self.cfg.num_workers,
            collate_fn=collate, pin_memory=self.device.type == "cuda",
            persistent_workers=self.cfg.num_workers > 0)
        return loader, sampler

    def _dev_loader(self):
        from ..data.dataset import collate
        dataset = self._dataset(self.cfg.dev_manifest, False)
        # Uneven eval shards are safe because collectives happen only after the loop.
        subset = Subset(dataset, list(range(self.rank, len(dataset), self.world_size)))
        return DataLoader(subset, batch_size=self.cfg.batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers, collate_fn=collate,
                          pin_memory=self.device.type == "cuda",
                          persistent_workers=self.cfg.num_workers > 0)

    def _raw_model(self) -> VallrPin:
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        return getattr(model, "_orig_mod", model)

    def _write_config(self) -> None:
        with open(os.path.join(self.cfg.out_dir, "config.json"), "w", encoding="utf-8") as stream:
            values = dataclasses.asdict(self.cfg)
            values["world_size"] = self.world_size
            json.dump(values, stream, ensure_ascii=False, indent=2)

    def _to_device(self, batch: Dict) -> Dict:
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    def _reduce_scalars(self, values: Dict[str, float], count: int) -> Dict[str, float]:
        keys = sorted(values)
        tensor = torch.tensor([values[k] for k in keys] + [count], dtype=torch.float64,
                              device=self.device)
        if dist.is_initialized():
            dist.all_reduce(tensor)
        total = max(float(tensor[-1]), 1.0)
        return {key: float(tensor[i]) / total for i, key in enumerate(keys)}

    def fit(self) -> str:
        best_path = os.path.join(self.cfg.out_dir, "ckpts", "best.pt")
        try:
            for epoch in range(self.start_epoch, self.cfg.epochs + 1):
                self.train_batch_sampler.set_epoch(epoch)
                stats = self._train_epoch(epoch)
                if self.is_main:
                    msg = " ".join(f"{k}={v:.4f}" for k, v in stats.items())
                    print(f"[epoch {epoch:03d}] {msg} lr={_lr_at(self.step, self.cfg):.2e}",
                          flush=True)

                eval_stats = None
                if self.dev_loader is not None and epoch % self.cfg.eval_every == 0:
                    eval_stats = self.evaluate()
                    metric = eval_stats[self.cfg.selection_metric]
                    if metric is None:
                        raise ValueError(
                            f"selection_metric={self.cfg.selection_metric} is disabled in "
                            f"head_mode={self.cfg.model.head_mode}")
                    if self.is_main:
                        cer_text = (f"{100*eval_stats['cer']:.2f}%"
                                    if eval_stats["cer"] is not None else "disabled")
                        ser_text = (f"{100*eval_stats['ser']:.2f}%"
                                    if eval_stats["ser"] is not None else "disabled")
                        print(f"           dev char_CER={cer_text} pinyin_SER={ser_text}",
                              flush=True)
                        self._append_metrics(epoch, stats, eval_stats)
                        if self.cfg.save_best_per_metric:
                            for name in ("cer", "ser"):
                                value = eval_stats[name]
                                if value is not None and value < self.best_by_metric[name]:
                                    self.best_by_metric[name] = value
                                    save_stats = {k: v for k, v in eval_stats.items()
                                                  if v is not None}
                                    self._raw_model().save(
                                        os.path.join(self.cfg.out_dir, "ckpts",
                                                     f"best_{name}.pt"),
                                        tokenizer_dir="vocab", epoch=epoch, step=self.step,
                                        selection_metric=name, **save_stats)
                        if metric < self.best:
                            self.best = metric
                            save_stats = {k: v for k, v in eval_stats.items() if v is not None}
                            self._raw_model().save(best_path, tokenizer_dir="vocab", epoch=epoch,
                                                   step=self.step,
                                                   selection_metric=self.cfg.selection_metric,
                                                   **save_stats)
                if self.is_main:
                    metrics = {f"train/{k}": v for k, v in stats.items()}
                    metrics.update({f"dev/{k}": v for k, v in (eval_stats or {}).items()})
                    metrics.update({"epoch": epoch, "train/lr": _lr_at(self.step, self.cfg)})
                    self.tracker.log(metrics, self.step)
                    self._save_last(epoch)
                    if epoch % self.cfg.save_every == 0:
                        save_stats = {k: v for k, v in (eval_stats or {}).items()
                                      if v is not None}
                        self._raw_model().save(
                            os.path.join(self.cfg.out_dir, "ckpts", f"ckpt_ep{epoch}.pt"),
                            tokenizer_dir="vocab", epoch=epoch, step=self.step, **save_stats)
                        self._prune_ckpts()
                _barrier()
            if dist.is_initialized():
                dist.barrier()
        finally:
            self.tracker.finish()
        return best_path if os.path.exists(best_path) else os.path.join(
            self.cfg.out_dir, "ckpts", f"ckpt_ep{self.cfg.epochs}.pt")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        del epoch
        self.model.train(); self.opt.zero_grad(set_to_none=True)
        totals: Dict[str, float] = {}; count = 0; started = time.time()
        total_batches = len(self.train_loader)
        for index, batch in enumerate(self.train_loader):
            batch = self._to_device(batch)
            group_start = (index // self.cfg.accum_steps) * self.cfg.accum_steps
            group_end = min(group_start + self.cfg.accum_steps, total_batches)
            divisor = group_end - group_start
            should_step = index + 1 == group_end
            sync_context = (self.model.no_sync() if isinstance(self.model, DistributedDataParallel)
                            and not should_step else nullcontext())
            with sync_context:
                with torch.autocast("cuda", enabled=self.use_amp):
                    out = self.model(batch["video"], batch["video_lens"], batch["char_ids"],
                                     batch["char_lens"], batch["pinyin_ids"],
                                     batch["pinyin_lens"], self.tok.char.sos_id,
                                     self.tok.char.eos_id)
                    loss = out["loss"] / divisor
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite loss at step {self.step}")
                self.scaler.scale(loss).backward()
            if should_step:
                for group in self.opt.param_groups:
                    group["lr"] = _lr_at(self.step + 1, self.cfg)
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.opt); self.scaler.update()
                self.opt.zero_grad(set_to_none=True); self.step += 1
            for key, value in out.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            count += 1
            if self.is_main and self.cfg.log_every and (index + 1) % self.cfg.log_every == 0:
                print(f"  batch {index+1}/{total_batches} step={self.step} "
                      f"loss={float(out['loss']):.4f} elapsed={time.time()-started:.1f}s",
                      flush=True)
        return self._reduce_scalars(totals, count)

    @torch.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        self.model.eval(); chars, pinyin = ErrorStats(), ErrorStats()
        text_oov, text_tokens = 0, 0
        raw = self._raw_model()
        for index, batch in enumerate(self.dev_loader):
            if max_batches and index >= max_batches:
                break
            batch = self._to_device(batch)
            memory, mask = raw.encode(batch["video"], batch["video_lens"])
            lengths = mask.sum(-1)
            char_hyp = (raw.ctc_greedy(raw.char_ctc, memory, lengths)
                        if self.cfg.model.uses_text_head else None)
            py_hyp = (raw.ctc_greedy(raw.pinyin_ctc, memory, lengths)
                      if self.cfg.model.uses_pinyin_head else None)
            for row in range(len(batch["ids"])):
                ref_c, ref_p, _ = text_to_pinyin_mixed(batch["texts"][row])
                if char_hyp is not None:
                    chars.update(ref_c, self.tok.char.decode(char_hyp[row]))
                    text_oov += sum(token not in self.tok.char.unit2id for token in ref_c)
                    text_tokens += len(ref_c)
                if py_hyp is not None:
                    pinyin.update(ref_p, self.tok.pinyin.decode(py_hyp[row]))
        numbers = torch.tensor([chars.sub, chars.dele, chars.ins, chars.total,
                                pinyin.sub, pinyin.dele, pinyin.ins, pinyin.total,
                                text_oov, text_tokens],
                               dtype=torch.long, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(numbers)
        values = numbers.tolist()
        return {"cer": (sum(values[:3]) / max(values[3], 1)
                        if self.cfg.model.uses_text_head else None),
                "ser": (sum(values[4:7]) / max(values[7], 1)
                        if self.cfg.model.uses_pinyin_head else None),
                "char_sub": values[0], "char_del": values[1], "char_ins": values[2],
                "pinyin_sub": values[4], "pinyin_del": values[5], "pinyin_ins": values[6],
                "text_oov_rate": (values[8] / max(values[9], 1)
                                  if self.cfg.model.uses_text_head else None)}

    def _append_metrics(self, epoch: int, train: dict, dev: dict) -> None:
        with open(os.path.join(self.cfg.out_dir, "metrics.jsonl"), "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch, "step": self.step,
                                     "train": train, "dev": dev}, ensure_ascii=False) + "\n")

    def _save_last(self, epoch: int) -> None:
        state = {"cfg": self.cfg.model.to_dict(), "state_dict": self._raw_model().state_dict(),
                 "optimizer": self.opt.state_dict(), "scaler": self.scaler.state_dict(),
                 "epoch": epoch, "step": self.step, "best": self.best,
                 "best_by_metric": self.best_by_metric,
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                 "python_rng": random.getstate()}
        path = os.path.join(self.cfg.out_dir, "ckpts", "last.pt")
        temporary = path + ".partial"
        torch.save(state, temporary); os.replace(temporary, path)

    def _resume(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._raw_model().load_state_dict(checkpoint["state_dict"], strict=True)
        if "optimizer" in checkpoint:
            self.opt.load_state_dict(checkpoint["optimizer"])
            self.scaler.load_state_dict(checkpoint.get("scaler", {}))
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
            self.step = int(checkpoint.get("step", 0)); self.best = float(checkpoint.get("best", self.best))
            self.best_by_metric.update(checkpoint.get("best_by_metric", {}))
            if "torch_rng" in checkpoint: torch.set_rng_state(checkpoint["torch_rng"].cpu())
            if "numpy_rng" in checkpoint: np.random.set_state(checkpoint["numpy_rng"])
            if "python_rng" in checkpoint: random.setstate(checkpoint["python_rng"])
        if self.is_main:
            print(f"[resume] {path} epoch={self.start_epoch-1} step={self.step}", flush=True)

    def _prune_ckpts(self) -> None:
        directory = os.path.join(self.cfg.out_dir, "ckpts")
        epochs = sorted(_ckpt_epoch(name) for name in os.listdir(directory) if _is_ckpt(name))
        if len(epochs) <= max(self.cfg.keep_ckpts, 1):
            return
        keep_n = max(self.cfg.keep_ckpts, 1)
        keep = {epochs[round(i * (len(epochs)-1) / max(keep_n-1, 1))] for i in range(keep_n)}
        keep.add(epochs[-1])
        for value in epochs:
            if value not in keep:
                os.remove(os.path.join(directory, f"ckpt_ep{value}.pt"))


_CKPT_PREFIX = "ckpt_ep"


def _is_ckpt(name: str) -> bool:
    return (name.startswith(_CKPT_PREFIX) and name.endswith(".pt")
            and name[len(_CKPT_PREFIX):-3].isdigit())


def _ckpt_epoch(name: str) -> int:
    return int(name[len(_CKPT_PREFIX):-3])


def list_checkpoints(out_dir: str) -> List[str]:
    directory = os.path.join(out_dir, "ckpts")
    files = [name for name in os.listdir(directory) if _is_ckpt(name)]
    return [os.path.join(directory, name) for name in sorted(files, key=_ckpt_epoch)]
