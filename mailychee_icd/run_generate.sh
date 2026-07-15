#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
PROMPT="${PROMPT:-How many times has Derrick Rose won NBA MVP?}"

python icd_generate.py \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --max-new-tokens "${MAX_NEW_TOKENS:-80}" \
  --beta "${BETA:-1.2}" \
  --alpha "${ALPHA:-0.1}" \
  --dtype "${DTYPE:-auto}"
