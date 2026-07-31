#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from iti_paper.config import ITIDirections
from iti_paper.eval import (
    evaluate_mc,
    evaluate_qa_generation,
    load_builtin_mc_dataset,
    load_builtin_qa_dataset,
    load_custom_mc_dataset,
    load_custom_qa_dataset,
)


MC_DATASETS = ["truthfulqa_mc1", "truthfulqa_mc2", "halueval", "mmlu", "hellaswag"]
QA_DATASETS = ["nq_open", "natural_questions"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and ITI on multiple-choice or open-QA datasets.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--directions", help="Path produced by scripts/train_iti.py. Omit for baseline-only eval.")
    parser.add_argument("--dataset", choices=MC_DATASETS + QA_DATASETS, help="Built-in eval dataset.")
    parser.add_argument("--subset", help="Dataset config/subject. Example: abstract_algebra for MMLU.")
    parser.add_argument("--task-type", choices=["auto", "mc", "qa"], default="auto")
    parser.add_argument("--custom-data", help="Path to .jsonl or .csv. MC uses prompt/choices/label; QA uses prompt/answers.")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--max-new-tokens", type=int, default=32, help="For QA generation eval.")
    parser.add_argument("--compare-baseline", action="store_true", help="Run baseline and ITI in one command.")
    parser.add_argument("--no-length-normalize", action="store_true", help="Use sum log-prob instead of average log-prob.")
    parser.add_argument("--output", help="Optional JSON file for metrics and per-example predictions.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.dataset) == bool(args.custom_data):
        raise SystemExit("Pass exactly one of --dataset or --custom-data.")
    if args.compare_baseline and not args.directions:
        raise SystemExit("--compare-baseline requires --directions.")

    task_type = infer_task_type(args.dataset, args.task_type)
    if args.dataset:
        examples = (
            load_builtin_mc_dataset(args.dataset, split=args.split, max_examples=args.max_examples, subset=args.subset)
            if task_type == "mc"
            else load_builtin_qa_dataset(args.dataset, split=args.split, max_examples=args.max_examples, subset=args.subset)
        )
    else:
        examples = (
            load_custom_mc_dataset(args.custom_data, max_examples=args.max_examples)
            if task_type == "mc"
            else load_custom_qa_dataset(args.custom_data, max_examples=args.max_examples)
        )

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device_map,
    )

    normalize = not args.no_length_normalize
    payload = {
        "dataset": args.dataset or args.custom_data,
        "subset": args.subset,
        "split": args.split,
        "task_type": task_type,
        "max_examples": args.max_examples,
    }

    if args.compare_baseline or not args.directions:
        baseline = run_eval(
            task_type,
            model,
            tokenizer,
            examples,
            batch_size=args.batch_size,
            normalize_by_length=normalize,
            max_new_tokens=args.max_new_tokens,
        )
        payload["baseline"] = result_to_dict(baseline)
        print(f"baseline accuracy: {baseline.correct}/{baseline.total} = {baseline.accuracy:.4f}")

    if args.directions:
        directions = ITIDirections.load(args.directions)
        iti = run_eval(
            task_type,
            model,
            tokenizer,
            examples,
            batch_size=args.batch_size,
            normalize_by_length=normalize,
            max_new_tokens=args.max_new_tokens,
            directions=directions,
            alpha=args.alpha,
        )
        payload["iti"] = result_to_dict(iti)
        print(f"iti accuracy: {iti.correct}/{iti.total} = {iti.accuracy:.4f}")
        if "baseline" in payload:
            comparison = compare_results(payload["baseline"], payload["iti"])
            payload["comparison"] = comparison
            print(f"changed predictions: {comparison['changed_predictions']}/{comparison['total']}")
            if comparison["mean_abs_score_delta"] is not None:
                print(f"mean abs score delta: {comparison['mean_abs_score_delta']:.6f}")
                print(f"max abs score delta: {comparison['max_abs_score_delta']:.6f}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))
        print(f"saved {output}")


def result_to_dict(result):
    return {
        "accuracy": result.accuracy,
        "total": result.total,
        "correct": result.correct,
        "predictions": result.predictions,
    }


def compare_results(baseline, iti):
    baseline_predictions = baseline["predictions"]
    iti_predictions = iti["predictions"]
    changed = sum(
        1
        for base_prediction, iti_prediction in zip(baseline_predictions, iti_predictions)
        if base_prediction["prediction"] != iti_prediction["prediction"]
    )
    deltas = []
    for base_prediction, iti_prediction in zip(baseline_predictions, iti_predictions):
        base_scores = base_prediction.get("scores")
        iti_scores = iti_prediction.get("scores")
        if base_scores is None or iti_scores is None:
            continue
        deltas.extend(abs(float(a) - float(b)) for a, b in zip(base_scores, iti_scores))

    return {
        "total": min(len(baseline_predictions), len(iti_predictions)),
        "changed_predictions": changed,
        "mean_abs_score_delta": sum(deltas) / len(deltas) if deltas else None,
        "max_abs_score_delta": max(deltas) if deltas else None,
    }


def infer_task_type(dataset: str | None, requested: str) -> str:
    if requested != "auto":
        return requested
    if dataset is None:
        raise SystemExit("Pass --task-type mc or --task-type qa for custom eval data.")
    return "qa" if dataset in QA_DATASETS else "mc"


def run_eval(
    task_type,
    model,
    tokenizer,
    examples,
    batch_size,
    normalize_by_length,
    max_new_tokens,
    directions=None,
    alpha=15.0,
):
    if task_type == "mc":
        return evaluate_mc(
            model,
            tokenizer,
            examples,
            batch_size=batch_size,
            normalize_by_length=normalize_by_length,
            directions=directions,
            alpha=alpha,
        )
    return evaluate_qa_generation(
        model,
        tokenizer,
        examples,
        max_new_tokens=max_new_tokens,
        directions=directions,
        alpha=alpha,
    )


if __name__ == "__main__":
    main()
