#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
MODE="${MODE:-icd}"

python eval_truthfulqa_mc1.py \
  --mode "$MODE" \
  --model "$MODEL" \
  --beta "${BETA:-1.2}" \
  --alpha "${ALPHA:-0.0}" \
  --dtype "${DTYPE:-auto}" \
  ${MAX_EXAMPLES:+--max-examples "$MAX_EXAMPLES"} \
  ${OUTPUT_JSONL:+--output-jsonl "$OUTPUT_JSONL"}
