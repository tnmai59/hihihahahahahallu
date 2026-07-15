#!/usr/bin/env bash
set -euo pipefail

python eval_truthfulqa_mc1.py \
  --mode icd \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset-jsonl /Users/mailychee/Downloads/mc1.jsonl \
  --beta 1.0 \
  --alpha 0.0 \
  --dtype auto
