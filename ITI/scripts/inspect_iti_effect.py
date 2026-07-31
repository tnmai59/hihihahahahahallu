#!/usr/bin/env python
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from iti_paper.config import ITIConfig, ITIDirections
from iti_paper.hooks import ITIHook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether ITI changes next-token logits for one prompt.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--prompt", default="Q: What happens if you swallow a cherry seed?\nA:")
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device_map)
    directions = ITIDirections.load(args.directions)
    encoded = tokenizer(args.prompt, return_tensors="pt").to(next(model.parameters()).device)

    baseline_logits = model(**encoded, use_cache=False).logits[:, -1, :].float().cpu()
    with ITIHook(model, directions, ITIConfig(alpha=args.alpha)):
        iti_logits = model(**encoded, use_cache=False).logits[:, -1, :].float().cpu()

    diff = (iti_logits - baseline_logits).abs()
    print(f"mean_abs_logit_delta={diff.mean().item():.8f}")
    print(f"max_abs_logit_delta={diff.max().item():.8f}")
    print(f"baseline_top_token={tokenizer.decode([int(baseline_logits.argmax())])!r}")
    print(f"iti_top_token={tokenizer.decode([int(iti_logits.argmax())])!r}")


if __name__ == "__main__":
    main()
