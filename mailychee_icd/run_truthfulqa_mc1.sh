#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
MODE="${MODE:-icd}"
DATASET_JSONL="${DATASET_JSONL:-/Users/mailychee/Downloads/mc1.jsonl}"

python eval_truthfulqa_mc1.py \
  --mode "$MODE" \
  --model "$MODEL" \
  --dataset-jsonl "$DATASET_JSONL" \
  --beta "${BETA:-1.0}" \
  --alpha "${ALPHA:-0.0}" \
  --dtype "${DTYPE:-auto}" \
  ${MAX_EXAMPLES:+--max-examples "$MAX_EXAMPLES"} \
  ${OUTPUT_JSONL:+--output-jsonl "$OUTPUT_JSONL"}
