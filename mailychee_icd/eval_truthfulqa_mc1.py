#!/usr/bin/env python3
"""Evaluate original LM or ICD scoring on TruthfulQA MC1.

MC1 is single-answer multiple choice. We score every answer choice as a
continuation of the question prompt and pick the highest-scoring choice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from icd_generate import DEFAULT_WEAK_SYSTEM_PROMPT, format_prompt, load_model_and_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TruthfulQA MC1.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--weak-model", default=None)
    parser.add_argument(
        "--mode",
        choices=["original", "icd"],
        default="icd",
        help="Use normal likelihood scoring or ICD contrastive scoring.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful, truthful assistant.",
        help="System prompt for the original model.",
    )
    parser.add_argument(
        "--weak-system-prompt",
        default=DEFAULT_WEAK_SYSTEM_PROMPT,
        help="System prompt for prompt-based ICD when --weak-model is omitted.",
    )
    parser.add_argument("--beta", type=float, default=1.2)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help=(
            "Optional ICD plausibility mask. For MC scoring, 0.0 is usually the "
            "least brittle default."
        ),
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--dataset-jsonl",
        default=None,
        help=(
            "Optional local JSONL file. Each line should have question, "
            "choices, and ground_truth."
        ),
    )
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use plain Q/A prompts instead of tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize answer scores by answer-token count.",
    )
    return parser


def question_prompt(question: str) -> str:
    return f"Q: {question}\nA:"


def continuation_ids(tokenizer, text: str, device):
    return tokenizer(" " + text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def score_continuation(model, tokenizer, prompt_text: str, answer: str, normalize: bool) -> float:
    import torch

    device = model.get_input_embeddings().weight.device
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    answer_ids = continuation_ids(tokenizer, answer, device)
    input_ids = torch.cat([prompt_ids, answer_ids], dim=-1)

    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits
        logprobs = torch.log_softmax(logits, dim=-1)

    start = prompt_ids.shape[-1] - 1
    total = 0.0
    for i, token_id in enumerate(answer_ids[0]):
        total += float(logprobs[0, start + i, int(token_id)])

    if normalize and answer_ids.shape[-1] > 0:
        total /= answer_ids.shape[-1]
    return total


def score_icd_continuation(
    model,
    weak_model,
    tokenizer,
    original_prompt: str,
    weak_prompt: str,
    answer: str,
    beta: float,
    alpha: float,
    normalize: bool,
) -> float:
    import torch

    original_device = model.get_input_embeddings().weight.device
    weak_device = weak_model.get_input_embeddings().weight.device
    original_prompt_ids = tokenizer(
        original_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(original_device)
    weak_prompt_ids = tokenizer(
        weak_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(weak_device)
    answer_ids = continuation_ids(tokenizer, answer, original_device)
    weak_answer_ids = answer_ids.to(weak_device)

    original_input_ids = torch.cat([original_prompt_ids, answer_ids], dim=-1)
    weak_input_ids = torch.cat([weak_prompt_ids, weak_answer_ids], dim=-1)

    with torch.inference_mode():
        original_logits = model(input_ids=original_input_ids).logits
        weak_logits = weak_model(input_ids=weak_input_ids).logits.to(original_logits.device)
        original_logprobs = torch.log_softmax(original_logits, dim=-1)
        weak_logprobs = torch.log_softmax(weak_logits, dim=-1)

        original_start = original_prompt_ids.shape[-1] - 1
        weak_start = weak_prompt_ids.shape[-1] - 1
        total = 0.0

        for i, token_id in enumerate(answer_ids[0]):
            token = int(token_id)
            if alpha > 0:
                probs = torch.softmax(original_logits[0, original_start + i, :], dim=-1)
                if probs[token] < alpha * torch.max(probs):
                    return -float("inf")
            total += float(
                beta * original_logprobs[0, original_start + i, token]
                - weak_logprobs[0, weak_start + i, token]
            )

    if normalize and answer_ids.shape[-1] > 0:
        total /= answer_ids.shape[-1]
    return total


def get_mc1(example):
    if "choices" in example and "ground_truth" in example:
        choices = example["choices"]
        ground_truth = example["ground_truth"]
        if isinstance(ground_truth, int):
            return choices, ground_truth
        if ground_truth not in choices:
            raise ValueError("ground_truth was not found in choices.")
        return choices, choices.index(ground_truth)

    targets = example["mc1_targets"]
    choices = targets["choices"]
    labels = targets["labels"]
    correct_indices = [i for i, label in enumerate(labels) if label == 1]
    if len(correct_indices) != 1:
        raise ValueError("MC1 example did not have exactly one correct answer.")
    return choices, correct_indices[0]


def main() -> None:
    from datasets import load_dataset
    from tqdm import tqdm

    args = build_parser().parse_args()
    if args.beta <= 0:
        raise SystemExit("--beta must be greater than 0")
    if args.alpha < 0 or args.alpha > 1:
        raise SystemExit("--alpha must be between 0 and 1")

    model, weak_model, tokenizer = load_model_and_tokenizer(args)
    if args.dataset_jsonl:
        with open(args.dataset_jsonl, "r", encoding="utf-8") as dataset_file:
            dataset = [json.loads(line) for line in dataset_file if line.strip()]
    else:
        dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")

    end = None if args.max_examples is None else args.start + args.max_examples
    stop = min(end or len(dataset), len(dataset))
    if hasattr(dataset, "select"):
        rows = list(dataset.select(range(args.start, stop)))
    else:
        rows = dataset[args.start:stop]

    output_file = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None
    correct = 0

    try:
        for local_idx, example in enumerate(tqdm(rows, desc="TruthfulQA MC1")):
            choices, gold_idx = get_mc1(example)
            q_prompt = question_prompt(example["question"])
            original_prompt = format_prompt(tokenizer, args.system_prompt, q_prompt, args.no_chat_template)
            weak_prompt = format_prompt(tokenizer, args.weak_system_prompt, q_prompt, args.no_chat_template)

            if args.mode == "original":
                scores = [
                    score_continuation(model, tokenizer, original_prompt, choice, args.normalize)
                    for choice in choices
                ]
            else:
                scores = [
                    score_icd_continuation(
                        model,
                        weak_model,
                        tokenizer,
                        original_prompt,
                        weak_prompt,
                        choice,
                        args.beta,
                        args.alpha,
                        args.normalize,
                    )
                    for choice in choices
                ]

            pred_idx = max(range(len(scores)), key=lambda idx: scores[idx])
            is_correct = pred_idx == gold_idx
            correct += int(is_correct)

            if output_file:
                output_file.write(
                    json.dumps(
                        {
                            "index": args.start + local_idx,
                            "question": example["question"],
                            "choices": choices,
                            "ground_truth": choices[gold_idx],
                            "ground_truth_index": gold_idx,
                            "prediction": choices[pred_idx],
                            "prediction_index": pred_idx,
                            "correct": is_correct,
                            "scores": scores,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

        accuracy = correct / len(rows) if rows else 0.0
        print(json.dumps({"mc1_accuracy": accuracy, "correct": correct, "total": len(rows)}, indent=2))
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
