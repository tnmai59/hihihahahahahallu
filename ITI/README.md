# Inference-Time Intervention (ITI)

This repo is a compact implementation of Li et al., **Inference-Time Intervention: Eliciting Truthful Answers from a Language Model** (`arXiv:2306.03341`).

The pipeline follows the paper:

1. Build truthful/false QA pairs from TruthfulQA.
2. Collect last-token attention-head activations at the `o_proj` input of each LLaMA/Gemma-style attention layer.
3. Train one binary linear probe per `(layer, head)`.
4. Select the top-`K` heads by validation accuracy.
5. During generation, shift selected head activations by `alpha * sigma * direction`.

The hook targets Hugging Face decoder-only models whose layers expose `self_attn.o_proj`, including LLaMA-family and Gemma-family checkpoints. For Gemma, the implementation infers `head_dim` from `o_proj.in_features`, which matters because the attention output width can differ from `hidden_size`.


## Supported Target Models

This implementation is intended to run on these Hugging Face model families you use:

- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- Gemma 3 27B instruct checkpoints, such as `google/gemma-3-27b-it` when available in your environment

Train a separate ITI directions file for each model:

```bash
python scripts/train_iti.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output artifacts/llama3.1-8b_iti_directions.pt \
  --max-examples 600 \
  --top-k 48 \
  --dtype bfloat16
```

```bash
python scripts/train_iti.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --output artifacts/llama3.2-3b_iti_directions.pt \
  --max-examples 600 \
  --top-k 48 \
  --dtype bfloat16
```

```bash
python scripts/train_iti.py \
  --model google/gemma-3-27b-it \
  --output artifacts/gemma3-27b_iti_directions.pt \
  --max-examples 600 \
  --top-k 48 \
  --dtype bfloat16
```

The model utilities unwrap Gemma 3 text configs and support decoder layers exposed through both plain `model.layers` and wrapped `model.language_model.layers`-style layouts. The API server uses each tokenizer's chat template for `/v1/chat/completions`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Train ITI directions

```bash
python scripts/train_iti.py \
  --model meta-llama/Llama-2-7b-hf \
  --output artifacts/iti_directions.pt \
  --max-examples 600 \
  --top-k 48
```

For a small local smoke test, use a tiny LLaMA-compatible checkpoint such as `HuggingFaceM4/tiny-random-LlamaForCausalLM`.

Gemma example:

```bash
python scripts/train_iti.py \
  --model google/gemma-2-2b \
  --output artifacts/gemma_iti_directions.pt \
  --max-examples 600 \
  --top-k 48 \
  --dtype bfloat16
```

## Generate with ITI

```bash
python scripts/generate_with_iti.py \
  --model meta-llama/Llama-2-7b-hf \
  --directions artifacts/iti_directions.pt \
  --alpha 15 \
  --prompt "I ate a cherry seed. Will a cherry tree grow in my stomach?"
```

Gemma example:

```bash
python scripts/generate_with_iti.py \
  --model google/gemma-2-2b \
  --directions artifacts/gemma_iti_directions.pt \
  --alpha 15 \
  --dtype bfloat16 \
  --prompt "I ate a cherry seed. Will a cherry tree grow in my stomach?"
```

## Evaluate ITI

Evaluate a multiple-choice dataset by scoring each candidate answer with conditional log-likelihood.

Baseline only:

```bash
python scripts/eval_iti.py \
  --model google/gemma-2-2b \
  --dataset hellaswag \
  --max-examples 200 \
  --dtype bfloat16
```

Baseline vs ITI:

```bash
python scripts/eval_iti.py \
  --model google/gemma-2-2b \
  --directions artifacts/gemma_iti_directions.pt \
  --dataset hellaswag \
  --compare-baseline \
  --alpha 15 \
  --max-examples 200 \
  --dtype bfloat16 \
  --output artifacts/hellaswag_eval.json
```

Built-in eval adapters:

- `truthfulqa_mc1`
- `truthfulqa_mc2`
- `halueval`
- `mmlu`
- `nq_open`
- `natural_questions`

Examples:

```bash
python scripts/eval_iti.py \
  --model google/gemma-2-2b \
  --directions artifacts/gemma_iti_directions.pt \
  --dataset truthfulqa_mc1 \
  --compare-baseline \
  --dtype bfloat16
```

```bash
python scripts/eval_iti.py \
  --model google/gemma-2-2b \
  --directions artifacts/gemma_iti_directions.pt \
  --dataset truthfulqa_mc2 \
  --compare-baseline \
  --dtype bfloat16
```

```bash
python scripts/eval_iti.py \
  --model google/gemma-2-2b \
  --directions artifacts/gemma_iti_directions.pt \
  --dataset halueval \
  --subset qa \
  --compare-baseline \
  --dtype bfloat16
```

```bash
python scripts/eval_iti.py \
  --model google/gemma-2-2b \
  --directions artifacts/gemma_iti_directions.pt \
  --dataset mmlu \
  --subset abstract_algebra \
  --compare-baseline \
  --dtype bfloat16
```

```bash
python scripts/eval_iti.py \
  --model google/gemma-2-2b \
  --directions artifacts/gemma_iti_directions.pt \
  --dataset nq_open \
  --compare-baseline \
  --max-new-tokens 32 \
  --dtype bfloat16
```

Custom eval files can be `.jsonl` or `.csv` with these fields:

```json
{"prompt": "Q: Which object is used to tell time?\nA:", "choices": ["a clock", "a spoon", "a shoe"], "label": 0}
```

For multiple correct choices, set `label` to a list:

```json
{"prompt": "Q: Select a mammal.\nA:", "choices": ["dog", "salmon", "cat"], "label": [0, 2]}
```

Then run:

```bash
python scripts/eval_iti.py \
  --model google/gemma-2-2b \
  --directions artifacts/gemma_iti_directions.pt \
  --custom-data data/my_eval.jsonl \
  --compare-baseline \
  --dtype bfloat16
```

## Serve an OpenAI-Compatible API

You can expose ITI behind endpoints that the OpenAI Python client can call:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Start the server:

```bash
python scripts/serve_openai_api.py \
  --model google/gemma-2-2b \
  --directions artifacts/gemma_iti_directions.pt \
  --model-id gemma-2-2b-iti \
  --alpha 15 \
  --dtype bfloat16 \
  --api-key local-dev-key \
  --host 0.0.0.0 \
  --port 8000
```

Call it with the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local-dev-key",
)

response = client.chat.completions.create(
    model="gemma-2-2b-iti",
    messages=[
        {"role": "user", "content": "I ate a cherry seed. Will a cherry tree grow in my stomach?"}
    ],
    max_tokens=64,
    temperature=0,
)

print(response.choices[0].message.content)
```

You can also override ITI strength per request with a nonstandard extra body field:

```python
response = client.chat.completions.create(
    model="gemma-2-2b-iti",
    messages=[{"role": "user", "content": "What happens if you swallow gum?"}],
    max_tokens=64,
    extra_body={"alpha": 10},
)
```

This compatibility layer is intentionally small: non-streaming text generation works, but streaming, tool calls, embeddings, logprobs, and structured outputs are not implemented.

## Notes

The paper reports strong results with `top_k=48` and a swept intervention strength. Larger `alpha` generally pushes harder toward the TruthfulQA-derived direction but can trade off helpfulness or fluency. This implementation exposes `alpha` at generation time so you can sweep it without retraining probes.

Train directions separately for each base model family and size. A direction file trained on LLaMA should not be reused on Gemma because layer counts, head dimensions, tokenizer behavior, and activation geometry differ.
