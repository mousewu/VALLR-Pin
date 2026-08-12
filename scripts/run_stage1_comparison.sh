#!/usr/bin/env bash
# Train the three Stage-I heads with the same launcher and optional overrides.
# Usage: GPUS=8 bash scripts/run_stage1_comparison.sh [key=value ...]
set -euo pipefail

GPUS=${GPUS:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29510}

if [ "$GPUS" -gt 1 ]; then
  LAUNCH=(torchrun --nproc_per_node="$GPUS" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
else
  LAUNCH=(python3)
fi

CONFIGS=(
  configs/stage1_char_only.yaml
  configs/stage1_pinyin_only.yaml
  configs/stage1_multicorpus.yaml
)

for config in "${CONFIGS[@]}"; do
  echo "== Training $config =="
  if [ "$#" -gt 0 ]; then
    "${LAUNCH[@]}" -m vallr_pin.cli train --config "$config" --set "$@"
  else
    "${LAUNCH[@]}" -m vallr_pin.cli train --config "$config"
  fi
done

python3 scripts/summarize_stage1_comparison.py
