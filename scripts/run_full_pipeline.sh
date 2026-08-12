#!/usr/bin/env bash
# Decoupled VALLR-Pin workflow.
# Usage: GPUS=8 bash scripts/run_full_pipeline.sh <stage1_exp> <stage1_cfg> <stage2_text_cfg> [llm]
set -euo pipefail

EXP=${1:-exp/stage1_pinyin_only}
STAGE1_CFG=${2:-configs/stage1_pinyin_only.yaml}
TEXT_CFG=${3:-configs/stage2_text.local.yaml}
LLM=${4:-Qwen/Qwen3-4B-Instruct-2507}
GPUS=${GPUS:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

DEV_MANIFEST=$(python3 -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["dev_manifest"])' "$STAGE1_CFG")
DATA_ROOT=$(python3 -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1])).get("data_root",""))' "$STAGE1_CFG")
TEXT_OUT=$(python3 -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1])).get("out_dir","data/stage2_text"))' "$TEXT_CFG")

if [ "$GPUS" -gt 1 ]; then
  LAUNCH=(torchrun --nproc_per_node="$GPUS" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
else
  LAUNCH=(python3)
fi

echo "== Stage I: video -> toneless Pinyin CTC =="
"${LAUNCH[@]}" -m vallr_pin.cli train --config "$STAGE1_CFG" --set out_dir="$EXP"

echo "== Stage II data: independent text -> clean text/Pinyin pairs =="
python3 -m vallr_pin.cli build-stage2-text --config "$TEXT_CFG"

echo "== Stage II: dynamically corrupted Pinyin -> text LoRA =="
"${LAUNCH[@]}" -m vallr_pin.cli sft --config configs/llm_sft.yaml \
  --model-path "$LLM" --train-jsonl "$TEXT_OUT/train.jsonl" \
  --val-jsonl "$TEXT_OUT/val.jsonl" --out-dir "$EXP/stage2_lora"

echo "== Inference: video -> Pinyin -> LLM -> text =="
python3 -m vallr_pin.cli decode \
  --ckpt "$EXP/ckpts/best.pt" --vocab "$EXP/vocab" \
  --manifest "$DEV_MANIFEST" --data-root "$DATA_ROOT" \
  --out-jsonl "$EXP/hyps_dev.jsonl" --beam 10 --nbest 5

python3 -m vallr_pin.cli refine --hyp "$EXP/hyps_dev.jsonl" --refiner llm \
  --model-path "$LLM" --adapter "$EXP/stage2_lora" \
  --out "$EXP/dev_refined_llm.jsonl"
