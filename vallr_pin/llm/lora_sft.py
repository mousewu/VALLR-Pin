"""Distributed LoRA adaptation for decoupled noisy-Pinyin-to-text training.

The native trainer accepts either the new clean text schema
``{text, pinyin}`` (noise generated online) or legacy ``messages`` calibration
rows.  Launching with ``torchrun`` gives one model replica per local GPU.
"""

from __future__ import annotations

import json
import math
import os
import random
from array import array
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, List, Sequence

from ..engine.tracking import SwanLabConfig, SwanLabTracker
from .noise import PinyinNoiseConfig, corrupt_pinyin
from .prompt import build_messages


SWIFT_TEMPLATE = """\
# The dataset must first be materialized to messages JSONL.
swift sft \\
  --model {model} \\
  --train_type lora \\
  --dataset {data} \\
  --lora_rank {rank} --lora_alpha {alpha} --lora_dropout {dropout} \\
  --learning_rate {lr} \\
  --num_train_epochs {epochs} \\
  --per_device_train_batch_size {bs} --gradient_accumulation_steps {accum} \\
  --max_length {maxlen} \\
  --deepspeed zero2 \\
  --output_dir {out}
"""


@dataclass
class SftConfig:
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    train_jsonl: str = ""
    val_jsonl: str = ""
    out_dir: str = "exp/stage2_pinyin_llm"
    epochs: int = 2
    batch_size: int = 2               # per process/GPU
    accum_steps: int = 8
    variants_per_text: int = 2        # virtual online-noise expansion
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_length: int = 1024
    num_workers: int = 2
    grad_clip: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Sequence[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    dtype: str = "bf16"
    device: str = "auto"
    grad_ckpt: bool = True
    log_every: int = 10
    seed: int = 2026
    noise: PinyinNoiseConfig = field(default_factory=PinyinNoiseConfig)
    swanlab: SwanLabConfig = field(default_factory=SwanLabConfig)


def print_swift_command(cfg: SftConfig) -> str:
    cmd = SWIFT_TEMPLATE.format(
        model=cfg.model_path, data=cfg.train_jsonl, epochs=cfg.epochs,
        bs=cfg.batch_size, accum=cfg.accum_steps, maxlen=cfg.max_length,
        out=cfg.out_dir, rank=cfg.lora_rank, alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout, lr=cfg.lr)
    print(cmd)
    return cmd


class IndexedJsonl(Sequence[Dict]):
    """Random-access JSONL without retaining every decoded row in memory.

    Only 64-bit byte offsets are stored.  Each DataLoader worker opens its own
    file handle lazily, which also keeps the object safe under spawn/fork.
    """

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.offsets = array("Q")
        self._handle: BinaryIO | None = None
        with open(self.path, "rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._handle is None:
            self._handle = open(self.path, "rb")
        self._handle.seek(self.offsets[index])
        return json.loads(self._handle.readline().decode("utf-8"))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()


def _encode_messages(tok, messages: List[Dict], max_length: int):
    """Return token ids with loss masked over every non-assistant token."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False,
                                     add_generation_prompt=True)
    answer = messages[-1]["content"] + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    if len(p_ids) >= max_length:
        raise ValueError(
            f"prompt alone has {len(p_ids)} tokens and exceeds max_length={max_length}; "
            "shorten corpus sentences or increase max_length")
    ids = (p_ids + a_ids)[:max_length]
    labels = ([-100] * len(p_ids) + a_ids)[:max_length]
    return ids, labels


class Stage2Dataset:
    """Virtual dataset that creates a fresh deterministic corruption per epoch."""

    def __init__(self, rows: Sequence[Dict], tok, max_length: int,
                 noise: PinyinNoiseConfig, variants: int = 1, seed: int = 0,
                 training: bool = True):
        if variants < 1:
            raise ValueError("variants_per_text must be positive")
        self.rows, self.tok, self.max_length = rows, tok, max_length
        self.noise, self.variants, self.seed = noise, variants, seed
        self.training, self.epoch = training, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows) * self.variants

    def messages_at(self, index: int) -> List[Dict]:
        row = self.rows[index // self.variants]
        if "messages" in row:
            return row["messages"]
        if "text" not in row or "pinyin" not in row:
            raise KeyError("Stage-II rows require either messages or text+pinyin")
        variant = index % self.variants
        epoch = self.epoch if self.training else 0
        rng = random.Random(self.seed + epoch * 1_000_000_007 + index * 1_000_003 + variant)
        noisy, _ = corrupt_pinyin(row["pinyin"], self.noise, rng)
        return build_messages(noisy, answer=row["text"])

    def __getitem__(self, index: int):
        return _encode_messages(self.tok, self.messages_at(index), self.max_length)


def _dist_info() -> tuple[int, int, int]:
    return (int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1")),
            int(os.environ.get("LOCAL_RANK", "0")))


def train_lora(cfg: SftConfig) -> str:
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler, Subset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, world_size, local_rank = _dist_info()
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    is_main = rank == 0
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif cfg.device != "auto":
        device = torch.device(cfg.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(cfg.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed + rank)
    if is_main:
        os.makedirs(cfg.out_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[cfg.dtype]
    if device.type == "cpu":
        dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, torch_dtype=dtype, trust_remote_code=True)
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = False
    if cfg.grad_ckpt:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    model = get_peft_model(base, LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), task_type="CAUSAL_LM", bias="none")).to(device)
    if is_main:
        model.print_trainable_parameters()
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None)

    def collate(batch):
        n = max(len(item[0]) for item in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        attention = torch.zeros((len(batch), n), dtype=torch.long)
        for row, (tokens, targets) in enumerate(batch):
            ids[row, :len(tokens)] = torch.tensor(tokens)
            labels[row, :len(targets)] = torch.tensor(targets)
            attention[row, :len(tokens)] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": attention}

    train_ds = Stage2Dataset(IndexedJsonl(cfg.train_jsonl), tok, cfg.max_length, cfg.noise,
                             cfg.variants_per_text, cfg.seed, True)
    if not len(train_ds):
        raise ValueError(f"Stage-II training dataset is empty: {cfg.train_jsonl}")
    train_sampler = (DistributedSampler(train_ds, world_size, rank, shuffle=True,
                                        seed=cfg.seed, drop_last=False)
                     if world_size > 1 else None)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=cfg.num_workers, collate_fn=collate,
        pin_memory=device.type == "cuda", persistent_workers=False)

    val_loader = None
    if cfg.val_jsonl and os.path.exists(cfg.val_jsonl):
        val_ds = Stage2Dataset(IndexedJsonl(cfg.val_jsonl), tok, cfg.max_length, cfg.noise,
                               1, cfg.seed + 17, False)
        # No duplicate padding in evaluation; reductions happen after the loop.
        val_subset = Subset(val_ds, list(range(rank, len(val_ds), world_size)))
        val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, collate_fn=collate,
                                pin_memory=device.type == "cuda")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / max(cfg.accum_steps, 1))
    total_updates = max(updates_per_epoch * cfg.epochs, 1)
    warmup = max(round(total_updates * cfg.warmup_ratio), 1)

    def lr_factor(step: int) -> float:
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total_updates - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    tracker = SwanLabTracker(cfg.swanlab, cfg, is_main)
    global_step, best_val = 0, float("inf")

    def reduce_pair(total: float, count: int) -> tuple[float, int]:
        values = torch.tensor([total, count], dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(values)
        return float(values[0]), int(values[1])

    try:
        for epoch in range(1, cfg.epochs + 1):
            train_ds.set_epoch(epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train(); optimizer.zero_grad(set_to_none=True)
            running, batches = 0.0, 0
            for index, batch in enumerate(train_loader):
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                group_start = (index // cfg.accum_steps) * cfg.accum_steps
                group_end = min(group_start + cfg.accum_steps, len(train_loader))
                divisor, should_step = group_end - group_start, index + 1 == group_end
                sync = (model.no_sync() if isinstance(model, DistributedDataParallel)
                        and not should_step else nullcontext())
                with sync:
                    amp = (torch.autocast("cuda", dtype=dtype) if use_amp else nullcontext())
                    with amp:
                        raw_loss = model(**batch).loss
                        loss = raw_loss / divisor
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"non-finite Stage-II loss at step {global_step}")
                    scaler.scale(loss).backward()
                running += float(raw_loss.detach()); batches += 1
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step(); global_step += 1
                    if is_main and cfg.log_every and global_step % cfg.log_every == 0:
                        value = running / max(batches, 1)
                        print(f"[sft] epoch={epoch} step={global_step}/{total_updates} "
                              f"loss={value:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
                        tracker.log({"train/loss_running": value,
                                     "train/lr": scheduler.get_last_lr()[0]}, global_step)
            total, count = reduce_pair(running, batches)
            train_loss = total / max(count, 1)

            val_loss = None
            if val_loader is not None:
                model.eval(); val_total, val_count = 0.0, 0
                eval_model = (model.module if isinstance(model, DistributedDataParallel)
                              else model)
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {key: value.to(device, non_blocking=True)
                                 for key, value in batch.items()}
                        val_total += float(eval_model(**batch).loss); val_count += 1
                val_total, val_count = reduce_pair(val_total, val_count)
                val_loss = val_total / max(val_count, 1)
            if is_main:
                print(f"[sft] epoch={epoch} train_loss={train_loss:.4f}"
                      + (f" val_loss={val_loss:.4f}" if val_loss is not None else ""), flush=True)
                tracker.log({"epoch": epoch, "train/loss": train_loss,
                             "val/loss": val_loss}, global_step)
                raw = model.module if isinstance(model, DistributedDataParallel) else model
                raw.save_pretrained(cfg.out_dir)
                tok.save_pretrained(cfg.out_dir)
                if val_loss is not None and val_loss < best_val:
                    best_val = val_loss
                    raw.save_pretrained(os.path.join(cfg.out_dir, "best"))
                    tok.save_pretrained(os.path.join(cfg.out_dir, "best"))
            if dist.is_initialized():
                dist.barrier()
    finally:
        tracker.finish()
    if is_main:
        print(f"[sft] LoRA adapter saved to {cfg.out_dir}", flush=True)
    return cfg.out_dir
