"""Step-3：用 LoRA 在 error-aware 指令数据上适配 LLM (对应论文 Eq.16-17)。

论文用的是 ms-swift；这里给出两条路径：

* ``print_swift_command()`` —— 直接复现论文配置的 ms-swift 命令 (lr=1e-4, r=8, α=32)；
* ``train_lora()``          —— 仅依赖 transformers + peft 的自实现训练循环，
  便于在没有 swift 的环境里跑通，也便于精确控制**只对 assistant 段计算损失**
  （提示词部分必须 mask 掉，否则模型会去学着复述拼音和候选，浪费容量）。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

SWIFT_TEMPLATE = """\
# 论文配置的 ms-swift 等价命令
swift sft \\
  --model {model} \\
  --train_type lora \\
  --dataset {data} \\
  --lora_rank 8 --lora_alpha 32 --lora_dropout 0.05 \\
  --learning_rate 1e-4 \\
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
    out_dir: str = "exp/llm_lora"
    epochs: int = 2
    batch_size: int = 2
    accum_steps: int = 8
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_length: int = 1024
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Sequence[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    dtype: str = "bf16"
    device: str = "auto"
    grad_ckpt: bool = True
    log_every: int = 10
    seed: int = 0


def print_swift_command(cfg: SftConfig) -> str:
    cmd = SWIFT_TEMPLATE.format(model=cfg.model_path, data=cfg.train_jsonl,
                                epochs=cfg.epochs, bs=cfg.batch_size,
                                accum=cfg.accum_steps, maxlen=cfg.max_length,
                                out=cfg.out_dir)
    print(cmd)
    return cmd


def _read(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _encode_sample(tok, messages: List[Dict], max_length: int):
    """返回 (input_ids, labels)；只有最后一段 assistant 内容参与损失。"""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False,
                                     add_generation_prompt=True)
    answer = messages[-1]["content"] + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_length]
    labels = ([-100] * len(p_ids) + a_ids)[:max_length]
    return ids, labels


def train_lora(cfg: SftConfig) -> str:
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[cfg.dtype]
    if device == "cpu":
        dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=dtype,
                                                 trust_remote_code=True)
    if cfg.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), task_type="CAUSAL_LM", bias="none"))
    model.print_trainable_parameters()
    model = model.to(device)

    class DS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return _encode_sample(tok, self.rows[i]["messages"], cfg.max_length)

    def collate(batch):
        n = max(len(x[0]) for x in batch)
        pad = tok.pad_token_id
        ids = torch.full((len(batch), n), pad, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, (a, b) in enumerate(batch):
            ids[i, :len(a)] = torch.tensor(a)
            lab[i, :len(b)] = torch.tensor(b)
            att[i, :len(a)] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}

    train_rows = _read(cfg.train_jsonl)
    loader = DataLoader(DS(train_rows), batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate)
    steps = max(1, math.ceil(len(loader) / cfg.accum_steps) * cfg.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=steps, pct_start=cfg.warmup_ratio,
        anneal_strategy="cos")

    model.train()
    gstep = 0
    for ep in range(1, cfg.epochs + 1):
        run = 0.0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / cfg.accum_steps
            loss.backward()
            run += float(loss) * cfg.accum_steps
            if (i + 1) % cfg.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                if gstep + 1 < steps:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % cfg.log_every == 0:
                    print(f"[sft] ep{ep} step{gstep}/{steps} "
                          f"loss={run / (i + 1):.4f} lr={sched.get_last_lr()[0]:.2e}",
                          flush=True)
        print(f"[sft] epoch {ep} mean_loss={run / max(len(loader), 1):.4f}", flush=True)

        if cfg.val_jsonl and os.path.exists(cfg.val_jsonl):
            model.eval()
            vl = DataLoader(DS(_read(cfg.val_jsonl)), batch_size=cfg.batch_size,
                            collate_fn=collate)
            tot, nb = 0.0, 0
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    tot += float(model(**batch).loss)
                    nb += 1
            print(f"[sft] epoch {ep} val_loss={tot / max(nb, 1):.4f}", flush=True)
            model.train()

    model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    print(f"[sft] LoRA adapter saved to {cfg.out_dir}", flush=True)
    return cfg.out_dir
