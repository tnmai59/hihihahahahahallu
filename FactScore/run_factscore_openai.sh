#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs

python eval_factscore_openai.py \
  --model Qwen3-32B \
  --server-host http://0.0.0.0:8001 \
  --api-key EMPTY \
  --topics-file unlabeled/prompt_entities.txt \
  --output-path outputs/factscore_Qwen3-32B.jsonl \
  --n-processes 8 \
  --start 0 \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 512 \
  --atomizer-max-tokens 768 \
  --judge-max-tokens 256 \
  --gamma 10.0 \
  --think \
  --judge-think
