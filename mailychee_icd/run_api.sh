#!/usr/bin/env bash
set -euo pipefail

python icd_api_server.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --served-model-name icd \
  --host 127.0.0.1 \
  --port 8000 \
  --beta 1.0 \
  --alpha 0.1 \
  --dtype auto
