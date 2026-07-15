#!/usr/bin/env bash
set -euo pipefail

python icd_generate.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --prompt "How many times has Derrick Rose won NBA MVP?" \
  --max-new-tokens 80 \
  --beta 1.2 \
  --alpha 0.1 \
  --dtype auto
