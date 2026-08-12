"""统一命令行入口：``python -m vallr_pin.cli <subcommand>``

    synth            生成合成数据集 (无真实数据时跑通链路)
    train            Stage-I 拼音优先 CTC 训练
    decode           用某个检查点解码 manifest，产出 N-best + 拼音
    build-stage2-text 从独立中文纯文本构造干净的文字/拼音语料
    materialize-stage2 固化在线拼音噪声，供 ms-swift 等外部训练器使用
    decode-ckpts     用**多个检查点**解码训练集 (可选真实错误校准)
    build-llm-data   把解码结果转成可选校准指令数据
    sft              LoRA 微调 LLM (或 --print-swift 输出 ms-swift 命令)
    refine           Stage-II 精化并报告 CER 变化
    pipeline         合成数据上的端到端演示 (train -> decode -> data -> refine)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from typing import Any, Dict

import yaml


# --------------------------------------------------------------------- utils
def load_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fill_dataclass(cls, data: Dict[str, Any]):
    """把 (可嵌套的) dict 填进 dataclass，未知键直接报错而不是静默忽略。

    注意各模块用了 ``from __future__ import annotations``，字段类型是字符串，
    因此这里通过默认实例的运行时类型来判断是否需要递归。
    """
    names = {f.name for f in dataclasses.fields(cls)}
    try:
        base = cls()
    except TypeError:
        base = None
    kwargs = {}
    for k, v in (data or {}).items():
        if k not in names:
            raise KeyError(f"{cls.__name__} 没有字段 '{k}'")
        default = getattr(base, k, None) if base is not None else None
        if isinstance(v, dict) and dataclasses.is_dataclass(default):
            kwargs[k] = fill_dataclass(type(default), v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def apply_overrides(cfg, pairs):
    """--set a.b=c 形式的覆盖。"""
    for p in pairs or []:
        key, _, val = p.partition("=")
        obj, *rest = key.split(".")
        target = cfg
        path = [obj] + rest
        for k in path[:-1]:
            target = getattr(target, k)
        cur = getattr(target, path[-1])
        try:
            new = type(cur)(yaml.safe_load(val)) if cur is not None else yaml.safe_load(val)
        except Exception:
            new = yaml.safe_load(val)
        setattr(target, path[-1], new)
    return cfg


# ------------------------------------------------------------------ commands
def cmd_synth(args):
    from .data.synthetic import build_synthetic_dataset
    paths = build_synthetic_dataset(args.out_dir, args.train_speakers, args.dev_speakers,
                                    size=args.size, noise=args.noise, seed=args.seed)
    print(json.dumps(paths, ensure_ascii=False, indent=2))


def cmd_train(args):
    from .engine.trainer import TrainConfig, Trainer
    cfg = fill_dataclass(TrainConfig, load_yaml(args.config))
    apply_overrides(cfg, args.set)
    best = Trainer(cfg).fit()
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[train] best/last checkpoint: {best}")


def cmd_decode(args):
    from .engine.decode import DecodeConfig, decode_manifest, load_stage1
    model, tok = load_stage1(args.ckpt, args.vocab)
    cfg = fill_dataclass(DecodeConfig, load_yaml(args.config))
    for k in ("manifest", "data_root", "out_jsonl", "beam", "nbest", "crop_size",
              "device", "max_utts"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    decode_manifest(model, tok, cfg, tag=os.path.basename(args.ckpt))


def cmd_decode_ckpts(args):
    """用早/中/晚检查点解码同一 manifest，构造可选的真实错误校准集。"""
    from .engine.decode import DecodeConfig, decode_manifest, load_stage1
    from .engine.trainer import list_checkpoints
    ckpts = list_checkpoints(args.exp_dir)
    if args.max_ckpts and len(ckpts) > args.max_ckpts:
        idx = [round(i * (len(ckpts) - 1) / (args.max_ckpts - 1))
               for i in range(args.max_ckpts)]
        ckpts = [ckpts[i] for i in sorted(set(idx))]
    os.makedirs(args.out_dir, exist_ok=True)
    outs = []
    for ck in ckpts:
        model, tok = load_stage1(ck, os.path.join(args.exp_dir, "vocab"))
        cfg = DecodeConfig(manifest=args.manifest, data_root=args.data_root,
                           out_jsonl=os.path.join(
                               args.out_dir,
                               f"hyp_{os.path.basename(ck).replace('.pt', '')}.jsonl"),
                           beam=args.beam, nbest=args.nbest, crop_size=args.crop_size,
                           device=args.device, max_utts=args.max_utts)
        decode_manifest(model, tok, cfg, tag=os.path.basename(ck))
        outs.append(cfg.out_jsonl)
    print(json.dumps(outs, ensure_ascii=False, indent=2))


def cmd_build_llm_data(args):
    from .llm.build_data import (BuildConfig, build_instruction_data, load_records,
                                 split_train_val, write_jsonl)
    recs = load_records(args.hyp)
    cfg = BuildConfig(max_cer=args.max_cer, keep_correct=args.keep_correct,
                      nbest=args.nbest, seed=args.seed)
    rows = build_instruction_data(recs, cfg)
    train, val = split_train_val(rows, args.val_ratio, args.seed)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_val, val)
    print(f"[build-llm-data] train={len(train)} -> {args.out_train}; "
          f"val={len(val)} -> {args.out_val}")


def cmd_build_stage2_text(args):
    from .llm.text_data import TextBuildConfig, TextSource, build_text_corpus
    raw = load_yaml(args.config)
    raw["sources"] = [TextSource(**item) for item in raw.get("sources", [])]
    report = build_text_corpus(TextBuildConfig(**raw))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_materialize_stage2(args):
    from .llm.lora_sft import SftConfig
    from .llm.text_data import materialize_instruction_data
    cfg = fill_dataclass(SftConfig, load_yaml(args.config))
    count = materialize_instruction_data(
        args.input, args.output, cfg.noise, args.variants, args.seed)
    print(f"[materialize-stage2] rows={count} -> {args.output}")


def cmd_sft(args):
    from .llm.lora_sft import SftConfig, print_swift_command, train_lora
    cfg = fill_dataclass(SftConfig, load_yaml(args.config))
    for k in ("model_path", "train_jsonl", "val_jsonl", "out_dir", "epochs", "device"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)
    apply_overrides(cfg, args.set)
    if args.print_swift:
        print_swift_command(cfg)
        return
    train_lora(cfg)


def cmd_refine(args):
    from .llm.refine import LLMConfig, NgramConfig, RefineRunConfig, run_refine
    cfg = RefineRunConfig(hyp_jsonl=args.hyp, out_jsonl=args.out, refiner=args.refiner,
                          lm_texts=args.lm_texts or "", nbest=args.nbest,
                          ngram=NgramConfig(),
                          llm=LLMConfig(model_path=args.model_path,
                                        adapter_path=args.adapter,
                                        device=args.device, batch_size=args.batch_size))
    stats = run_refine(cfg)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_pipeline(args):
    """Synthetic decoupled demo; validates interfaces rather than accuracy."""
    from .data.synthetic import build_synthetic_dataset
    from .engine.decode import DecodeConfig, decode_manifest, load_stage1
    from .engine.trainer import TrainConfig, Trainer
    from .llm.refine import NgramConfig, RefineRunConfig, run_refine
    from .models.vallr_pin import VallrPinConfig
    from .data.dataset import read_manifest
    from .text.pinyin import text_to_pinyin

    root = args.work_dir
    data = build_synthetic_dataset(os.path.join(root, "data"),
                                   n_train_speakers=args.train_speakers,
                                   n_dev_speakers=1, size=40, noise=args.noise)
    exp = os.path.join(root, "exp")
    tcfg = TrainConfig(train_manifest=data["train"], dev_manifest=data["dev"],
                       data_root=data["root"], out_dir=exp, epochs=args.epochs,
                       batch_size=args.batch_size, lr=args.lr, warmup_steps=args.warmup,
                       crop_size=32, num_workers=args.num_workers, save_every=1,
                       keep_ckpts=args.keep_ckpts, device=args.device,
                       model=VallrPinConfig(d_model=args.d_model, heads=4,
                                            ffn=args.d_model * 4, enc_layers=args.enc_layers,
                                            char_dec_layers=args.dec_layers,
                                            pinyin_dec_layers=2, frontend="resnet18",
                                            frontend_width=args.frontend_width,
                                            dropout=0.1, sanm_kernel=7, alpha=0.0))
    Trainer(tcfg).fit()

    hyp_dir = os.path.join(root, "hyps")
    os.makedirs(hyp_dir, exist_ok=True)
    best = os.path.join(exp, "ckpts", "best.pt")
    model, tok = load_stage1(best, os.path.join(exp, "vocab"))
    dev_hyp = os.path.join(hyp_dir, "dev.jsonl")
    decode_manifest(model, tok, DecodeConfig(manifest=data["dev"], data_root=data["root"],
                                             out_jsonl=dev_hyp, beam=args.beam,
                                             nbest=args.nbest, crop_size=32,
                                             device=args.device), tag="best")

    # Stage-II source is text-only and does not depend on any Stage-I checkpoint.
    stage2_path = os.path.join(root, "llm_data", "train.jsonl")
    os.makedirs(os.path.dirname(stage2_path), exist_ok=True)
    rows = read_manifest(data["train"])
    with open(stage2_path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({"id": row["id"], "text": row["text"],
                                     "pinyin": text_to_pinyin(row["text"])[1]},
                                    ensure_ascii=False) + "\n")

    stats = run_refine(RefineRunConfig(hyp_jsonl=dev_hyp,
                                       out_jsonl=os.path.join(root, "dev_refined.jsonl"),
                                       refiner="ngram", lm_texts=data["train"],
                                       nbest=args.nbest, ngram=NgramConfig()))
    print(json.dumps({"work_dir": root, "llm_train_data": len(rows), **stats},
                     ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser("vallr-pin")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="生成合成数据集")
    s.add_argument("--out-dir", default="data/synth")
    s.add_argument("--train-speakers", type=int, default=6)
    s.add_argument("--dev-speakers", type=int, default=2)
    s.add_argument("--size", type=int, default=40)
    s.add_argument("--noise", type=float, default=0.10)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("train", help="Stage-I 训练")
    s.add_argument("--config", required=True)
    s.add_argument("--set", nargs="*", default=[])
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("decode", help="解码 manifest -> N-best + 拼音")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--vocab", required=True)
    s.add_argument("--config", default="")
    s.add_argument("--manifest")
    s.add_argument("--data-root", default=None)
    s.add_argument("--out-jsonl")
    s.add_argument("--beam", type=int, default=None)
    s.add_argument("--nbest", type=int, default=None)
    s.add_argument("--crop-size", type=int, default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--max-utts", type=int, default=None)
    s.set_defaults(func=cmd_decode)

    s = sub.add_parser("decode-ckpts", help="多检查点解码（可选真实错误校准）")
    s.add_argument("--exp-dir", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--data-root", default="")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--max-ckpts", type=int, default=4)
    s.add_argument("--beam", type=int, default=10)
    s.add_argument("--nbest", type=int, default=5)
    s.add_argument("--crop-size", type=int, default=88)
    s.add_argument("--device", default="auto")
    s.add_argument("--max-utts", type=int, default=None)
    s.set_defaults(func=cmd_decode_ckpts)

    s = sub.add_parser("build-llm-data", help="构造可选 error-aware 校准数据")
    s.add_argument("--hyp", nargs="+", required=True)
    s.add_argument("--out-train", required=True)
    s.add_argument("--out-val", required=True)
    s.add_argument("--max-cer", type=float, default=0.8)
    s.add_argument("--keep-correct", type=float, default=0.25)
    s.add_argument("--nbest", type=int, default=5)
    s.add_argument("--val-ratio", type=float, default=0.02)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_build_llm_data)

    s = sub.add_parser("build-stage2-text", help="独立纯文本 -> 干净文字/拼音语料")
    s.add_argument("--config", required=True)
    s.set_defaults(func=cmd_build_stage2_text)

    s = sub.add_parser("materialize-stage2", help="把在线拼音噪声固化为 messages JSONL")
    s.add_argument("--config", default="configs/llm_sft.yaml")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--variants", type=int, default=2)
    s.add_argument("--seed", type=int, default=2026)
    s.set_defaults(func=cmd_materialize_stage2)

    s = sub.add_parser("sft", help="LoRA 微调 LLM")
    s.add_argument("--config", default="")
    s.add_argument("--model-path", default=None)
    s.add_argument("--train-jsonl", default=None)
    s.add_argument("--val-jsonl", default=None)
    s.add_argument("--out-dir", default=None)
    s.add_argument("--epochs", type=int, default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--print-swift", action="store_true")
    s.add_argument("--set", nargs="*", default=[])
    s.set_defaults(func=cmd_sft)

    s = sub.add_parser("refine", help="Stage-II 精化")
    s.add_argument("--hyp", required=True)
    s.add_argument("--out", default="")
    s.add_argument("--refiner", choices=["ngram", "llm"], default="ngram")
    s.add_argument("--lm-texts", default="")
    s.add_argument("--nbest", type=int, default=5)
    s.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    s.add_argument("--adapter", default=None)
    s.add_argument("--device", default="auto")
    s.add_argument("--batch-size", type=int, default=8)
    s.set_defaults(func=cmd_refine)

    s = sub.add_parser("pipeline", help="合成数据端到端演示")
    s.add_argument("--work-dir", default="exp/demo")
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--warmup", type=int, default=100)
    s.add_argument("--train-speakers", type=int, default=4)
    s.add_argument("--noise", type=float, default=0.10)
    s.add_argument("--d-model", type=int, default=128)
    s.add_argument("--enc-layers", type=int, default=4)
    s.add_argument("--dec-layers", type=int, default=2)
    s.add_argument("--frontend-width", type=int, default=16)
    s.add_argument("--beam", type=int, default=6)
    s.add_argument("--nbest", type=int, default=4)
    s.add_argument("--keep-ckpts", type=int, default=4)
    s.add_argument("--max-train-utts", type=int, default=None)
    s.add_argument("--num-workers", type=int, default=0)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_pipeline)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
