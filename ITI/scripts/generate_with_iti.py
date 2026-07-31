#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from iti_paper.config import ITIConfig, ITIDirections
from iti_paper.hooks import ITIHook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate from a LLaMA/Gemma-style model with ITI hooks.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--directions", required=True, help="Path produced by scripts/train_iti.py.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
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
    directions = ITIDirections.load(args.directions)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(next(model.parameters()).device)

    with ITIHook(model, directions, ITIConfig(alpha=args.alpha)):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
