#!/usr/bin/env bash
# 真实数据上的完整四步流程（对应论文 Fig.1 的 Step 1-4）。
# 用法: bash scripts/run_full_pipeline.sh <exp_dir> <config> [llm_path]
set -euo pipefail

EXP=${1:-exp/vallr_pin_cnvsrc}
CFG=${2:-configs/cnvsrc_base.yaml}
LLM=${3:-Qwen/Qwen3-4B-Instruct-2507}

TRAIN_MANIFEST=$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['train_manifest'])")
DEV_MANIFEST=$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$CFG'))['dev_manifest'])")
DATA_ROOT=$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$CFG')).get('data_root',''))")

echo "== Step 1: 双解码 VSR 训练 =="
python3 -m vallr_pin.cli train --config "$CFG"

echo "== Step 2: 多检查点解码训练集，构造 error-aware 数据 =="
python3 -m vallr_pin.cli decode-ckpts \
  --exp-dir "$EXP" --manifest "$TRAIN_MANIFEST" --data-root "$DATA_ROOT" \
  --out-dir "$EXP/hyps_train" --max-ckpts 4 --beam 10 --nbest 5

python3 -m vallr_pin.cli build-llm-data \
  --hyp "$EXP"/hyps_train/*.jsonl \
  --out-train "$EXP/llm_data/train.jsonl" --out-val "$EXP/llm_data/val.jsonl" \
  --max-cer 0.8 --keep-correct 0.25 --nbest 5

echo "== Step 3: 拼音引导的 LLM 适配 (LoRA) =="
python3 -m vallr_pin.cli sft --config configs/llm_sft.yaml \
  --model-path "$LLM" \
  --train-jsonl "$EXP/llm_data/train.jsonl" --val-jsonl "$EXP/llm_data/val.jsonl" \
  --out-dir "$EXP/llm_lora"
# 若使用论文的 ms-swift：加 --print-swift 打印等价命令

echo "== Step 4: 推理 (Stage-I 解码 -> Stage-II 精化) =="
python3 -m vallr_pin.cli decode \
  --ckpt "$EXP/ckpts/best.pt" --vocab "$EXP/vocab" \
  --manifest "$DEV_MANIFEST" --data-root "$DATA_ROOT" \
  --out-jsonl "$EXP/hyps_dev.jsonl" --beam 10 --nbest 5

echo "-- 无 LLM 基线（拼音受限 n-gram 重打分）--"
python3 -m vallr_pin.cli refine --hyp "$EXP/hyps_dev.jsonl" --refiner ngram \
  --lm-texts "$TRAIN_MANIFEST" --out "$EXP/dev_refined_ngram.jsonl"

echo "-- 论文主方案（微调后的 LLM）--"
python3 -m vallr_pin.cli refine --hyp "$EXP/hyps_dev.jsonl" --refiner llm \
  --model-path "$LLM" --adapter "$EXP/llm_lora" --out "$EXP/dev_refined_llm.jsonl"
