#!/usr/bin/env python3
"""Minimal Induce-then-Contrast Decoding (ICD) text generator.

ICD runs two next-token distributions:
  1. the normal/original model
  2. a "weak" hallucination-induced model or prompt

Then it decodes from: beta * logprob(normal) - logprob(weak), while masking out
tokens the normal model does not consider plausible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


DEFAULT_WEAK_SYSTEM_PROMPT = (
    "You are an untruthful assistant. When answering factual questions, "
    "make plausible-sounding but incorrect claims. Do not say you are unsure."
)


@dataclass
class DecodeConfig:
    max_new_tokens: int
    beta: float
    alpha: float
    temperature: float
    top_k: int
    do_sample: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run minimal ICD decoding with Hugging Face causal LMs."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Original model id/path, e.g. meta-llama/Llama-3.2-3B-Instruct.",
    )
    parser.add_argument(
        "--weak-model",
        default=None,
        help=(
            "Optional hallucination-induced model id/path. If omitted, the same "
            "model is reused with --weak-system-prompt."
        ),
    )
    parser.add_argument("--prompt", required=True, help="User prompt to answer.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful, careful assistant.",
        help="System prompt for the original model.",
    )
    parser.add_argument(
        "--weak-system-prompt",
        default=DEFAULT_WEAK_SYSTEM_PROMPT,
        help="System prompt used for prompt-based hallucination induction.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--beta",
        type=float,
        default=1.2,
        help="Contrast strength for the original model logprobs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help=(
            "Plausibility threshold. Keep tokens whose original probability is "
            "at least alpha * max_probability. Use 0.0 to disable."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use raw prompt text instead of tokenizer.apply_chat_template.",
    )
    return parser


def torch_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def format_prompt(tokenizer, system_prompt: str, prompt: str, no_chat_template: bool) -> str:
    if no_chat_template or not getattr(tokenizer, "chat_template", None):
        if system_prompt:
            return f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        return f"User: {prompt}\nAssistant:"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def sample_next_token(scores, cfg: DecodeConfig):
    import torch

    if cfg.temperature and cfg.temperature > 0:
        scores = scores / cfg.temperature
        if cfg.top_k and cfg.top_k > 0:
            top_values, top_indices = torch.topk(scores, k=min(cfg.top_k, scores.shape[-1]))
            filtered = torch.full_like(scores, -float("inf"))
            filtered.scatter_(dim=-1, index=top_indices, src=top_values)
            scores = filtered
        return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)

    return torch.argmax(scores, dim=-1, keepdim=True)


def load_model_and_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch_dtype(args.dtype),
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()

    if args.weak_model:
        weak_model = AutoModelForCausalLM.from_pretrained(args.weak_model, **model_kwargs).eval()
    else:
        weak_model = model

    if weak_model.config.vocab_size != model.config.vocab_size:
        raise SystemExit("The original and weak models must have the same vocabulary size.")

    return model, weak_model, tokenizer


def encode(tokenizer, text: str, device):
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def icd_generate(model, weak_model, tokenizer, original_text: str, weak_text: str, cfg: DecodeConfig) -> str:
    import torch

    device = model.get_input_embeddings().weight.device
    original_ids = encode(tokenizer, original_text, device)
    weak_device = weak_model.get_input_embeddings().weight.device
    weak_ids = encode(tokenizer, weak_text, weak_device)

    original_prompt_len = original_ids.shape[-1]
    generated_ids = []

    with torch.inference_mode():
        for _ in range(cfg.max_new_tokens):
            original_logits = model(input_ids=original_ids).logits[:, -1, :]
            weak_logits = weak_model(input_ids=weak_ids).logits[:, -1, :].to(original_logits.device)

            original_logprobs = torch.log_softmax(original_logits, dim=-1)
            weak_logprobs = torch.log_softmax(weak_logits, dim=-1)
            scores = cfg.beta * original_logprobs - weak_logprobs

            if cfg.alpha > 0:
                original_probs = torch.softmax(original_logits, dim=-1)
                threshold = cfg.alpha * torch.max(original_probs, dim=-1, keepdim=True).values
                scores = scores.masked_fill(original_probs < threshold, -float("inf"))

            next_token = sample_next_token(scores, cfg)
            token_id = int(next_token.item())
            generated_ids.append(token_id)

            original_ids = torch.cat([original_ids, next_token.to(original_ids.device)], dim=-1)
            weak_ids = torch.cat([weak_ids, next_token.to(weak_ids.device)], dim=-1)

            eos_ids = tokenizer.eos_token_id
            if eos_ids is not None and token_id == eos_ids:
                break

    new_ids = original_ids[:, original_prompt_len:]
    return tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()


def main() -> None:
    args = build_parser().parse_args()

    if args.alpha < 0 or args.alpha > 1:
        raise SystemExit("--alpha must be between 0 and 1")
    if args.beta <= 0:
        raise SystemExit("--beta must be greater than 0")

    model, weak_model, tokenizer = load_model_and_tokenizer(args)
    cfg = DecodeConfig(
        max_new_tokens=args.max_new_tokens,
        beta=args.beta,
        alpha=args.alpha,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=args.temperature > 0,
    )

    original_text = format_prompt(tokenizer, args.system_prompt, args.prompt, args.no_chat_template)
    weak_text = format_prompt(tokenizer, args.weak_system_prompt, args.prompt, args.no_chat_template)
    print(icd_generate(model, weak_model, tokenizer, original_text, weak_text, cfg))


if __name__ == "__main__":
    main()
