#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
TASK="${TASK:-qa}"
MODE="${MODE:-icd}"
CANDIDATE_MODE="${CANDIDATE_MODE:-both}"
DATASET_JSONL="${DATASET_JSONL:-/Users/mailychee/Downloads/${TASK}_data.json}"

python eval_halueval.py \
  --mode "$MODE" \
  --model "$MODEL" \
  --task "$TASK" \
  --dataset-jsonl "$DATASET_JSONL" \
  --candidate-mode "$CANDIDATE_MODE" \
  --beta "${BETA:-1.0}" \
  --alpha "${ALPHA:-0.0}" \
  --dtype "${DTYPE:-auto}" \
  ${MAX_EXAMPLES:+--max-examples "$MAX_EXAMPLES"} \
  ${OUTPUT_JSONL:+--output-jsonl "$OUTPUT_JSONL"} \
  ${USE_KNOWLEDGE:+--use-knowledge}
