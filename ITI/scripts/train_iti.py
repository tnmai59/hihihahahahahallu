#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from iti_paper.activations import collect_activations
from iti_paper.dataset import load_truthfulqa_pairs
from iti_paper.probes import train_linear_probes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ITI directions from TruthfulQA activations.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--output", required=True, help="Path to save ITI directions.")
    parser.add_argument("--max-examples", type=int, default=600)
    parser.add_argument("--no-shuffle", action="store_true", help="Use the original TruthfulQA QA-pair order.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    pairs = load_truthfulqa_pairs(
        max_examples=args.max_examples,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    activations, labels = collect_activations(
        model,
        tokenizer,
        pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    directions = train_linear_probes(
        activations,
        labels,
        top_k=args.top_k,
        seed=args.seed,
        metadata={
            "model": args.model,
            "max_examples": args.max_examples,
            "shuffled": not args.no_shuffle,
            "max_length": args.max_length,
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    directions.save(output)

    best = directions.probe_accuracies.max().item()
    print(f"saved {output}")
    print(f"selected_heads={directions.selected_heads[:10]}...")
    print(f"best_probe_accuracy={best:.4f}")


if __name__ == "__main__":
    main()
