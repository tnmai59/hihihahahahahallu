#!/usr/bin/env bash
set -euo pipefail

python eval_halueval.py \
  --mode icd \
  --decision-mode likelihood \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --task qa \
  --dataset-jsonl /Users/mailychee/Downloads/qa_data.json \
  --prompt-style minimal \
  --candidate-mode random \
  --beta 1.0 \
  --alpha 0.0 \
  --dtype auto
