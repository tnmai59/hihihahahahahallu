# Minimal ICD Decoding

This is a small, practical implementation of **Induce-then-Contrast Decoding**
from [arXiv:2312.15710](https://arxiv.org/pdf/2312.15710).

The paper's decoding rule is:

```text
score = beta * logprob(original_model) - logprob(factually_weak_model)
```

Then it applies a plausibility mask so decoding only chooses tokens that the
original model already considers reasonable.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If a model is gated, log in first:

```bash
huggingface-cli login
```

## Run With Prompt-Based ICD

This is the simplest mode. It uses one model twice: once normally and once with
a misleading system prompt that induces the "weak" distribution.

```bash
python icd_generate.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --prompt "How many times has Derrick Rose won NBA MVP?" \
  --max-new-tokens 80
```

Other model examples:

```bash
python icd_generate.py --model meta-llama/Llama-3.1-8B-Instruct --prompt "..."
python icd_generate.py --model google/gemma-2-2b-it --prompt "..."
python icd_generate.py --model openai/gpt-oss-20b --prompt "..."
```

Use smaller models first if you are on a laptop:

```bash
python icd_generate.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dtype bfloat16 \
  --prompt "What happened to the Mars Climate Orbiter?"
```

## Run With A Separate Weak Model

If you have a fine-tuned hallucination-induced model or adapter merged into a
model directory, pass it as `--weak-model`.

```bash
python icd_generate.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --weak-model /path/to/hallucination-induced-model \
  --prompt "Who wrote Pride and Prejudice?"
```

The original and weak models should use the same tokenizer/vocabulary.

## Useful Knobs

- `--beta`: higher values favor the original model more. Try `1.0` to `1.5`.
- `--alpha`: plausibility mask. Try `0.05` to `0.2`; use `0.0` to disable.
- `--temperature`: `0.0` is greedy; set `0.7` for sampling.
- `--top-k`: optional sampling filter when temperature is above zero.

## Evaluate TruthfulQA MC1

MC1 is evaluated by scoring each answer choice as a continuation of the question
and selecting the highest-scoring choice.

Baseline likelihood:

```bash
python eval_truthfulqa_mc1.py \
  --mode original \
  --model meta-llama/Llama-3.2-3B-Instruct
```

Prompt-based ICD:

```bash
python eval_truthfulqa_mc1.py \
  --mode icd \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --dataset-jsonl /Users/mailychee/Downloads/mc1.jsonl \
  --beta 1.0 \
  --alpha 0.0
```

Save per-question predictions:

```bash
python eval_truthfulqa_mc1.py \
  --mode icd \
  --model google/gemma-2-2b-it \
  --dataset-jsonl /Users/mailychee/Downloads/mc1.jsonl \
  --max-examples 50 \
  --output-jsonl truthfulqa_mc1_predictions.jsonl
```

Evaluate a local JSONL file with `question`, `choices`, and `ground_truth`:

```bash
python eval_truthfulqa_mc1.py \
  --mode icd \
  --model google/gemma-2-2b-it \
  --dataset-jsonl data/truthfulqa_mc1.jsonl
```

For TruthfulQA-style comparison, the evaluator defaults to the paper's 6-shot
prompt, `--beta 1.0`, `--alpha 0.0`, and summed answer log-probabilities. If
your choices are formatted like `A. answer text`, it scores only `answer text`.

With the bash runner:

```bash
DATASET_JSONL=data/truthfulqa_mc1.jsonl MODEL=google/gemma-2-2b-it ./run_truthfulqa_mc1.sh
```

## Evaluate HaluEval

HaluEval uses a binary judgement:

- `Yes`: the candidate contains hallucination
- `No`: the candidate is factual / not hallucinated

Run QA:

```bash
TASK=qa DATASET_JSONL=/path/to/qa_data.json MODEL=google/gemma-2-2b-it ./run_halueval.sh
```

Run dialogue or summarization:

```bash
TASK=dialogue DATASET_JSONL=/path/to/dialogue_data.json ./run_halueval.sh
TASK=summarization DATASET_JSONL=/path/to/summarization_data.json ./run_halueval.sh
```

By default the evaluator tests both the right and hallucinated candidate from
each row. To match the official random-one-candidate style:

```bash
CANDIDATE_MODE=random TASK=qa DATASET_JSONL=/path/to/qa_data.json ./run_halueval.sh
```

## Notes

This is intentionally minimal. It does not train the weak model; it implements
the decoding loop. The easiest weak model is prompt-based induction. The paper
reports stronger results with a fine-tuned weak model, but the decoding rule is
the same.
