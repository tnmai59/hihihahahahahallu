from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import ITIConfig, ITIDirections
from .hooks import ITIHook


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens




@torch.inference_mode()
def generate_chat_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    directions: ITIDirections | None = None,
    alpha: float = 15.0,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: str | list[str] | None = None,
) -> GenerationOutput:
    prompt = messages_to_prompt(messages)
    model.eval()
    device = next(model.parameters()).device
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            normalize_messages(messages),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
    else:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
    return generate_from_encoded(
        model=model,
        tokenizer=tokenizer,
        encoded=encoded,
        directions=directions,
        alpha=alpha,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=merge_stop(stop, ["\nUser:", "\nSystem:", "<|eot_id|>"]),
    )

@torch.inference_mode()
def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    directions: ITIDirections | None = None,
    alpha: float = 15.0,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: str | list[str] | None = None,
) -> GenerationOutput:
    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    return generate_from_encoded(
        model=model,
        tokenizer=tokenizer,
        encoded=encoded,
        directions=directions,
        alpha=alpha,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
    )


def apply_stop(text: str, stop: str | list[str] | None) -> str:
    if stop is None:
        return text
    stops = [stop] if isinstance(stop, str) else stop
    indices = [text.find(item) for item in stops if item]
    indices = [idx for idx in indices if idx >= 0]
    return text[: min(indices)] if indices else text


def messages_to_prompt(messages: list[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message_content_to_text(message.get("content", ""))
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def message_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        return "\n".join(chunks)
    return str(content)

def generate_from_encoded(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    encoded,
    directions: ITIDirections | None,
    alpha: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: str | list[str] | None,
) -> GenerationOutput:
    do_sample = temperature is not None and temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    context = ITIHook(model, directions, ITIConfig(alpha=alpha)) if directions is not None else nullcontext()
    with context:
        output_ids = model.generate(**encoded, **generation_kwargs)

    prompt_len = encoded["input_ids"].shape[1]
    completion_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    text = apply_stop(text, stop).strip()
    completion_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return GenerationOutput(text=text, prompt_tokens=prompt_len, completion_tokens=completion_tokens)


def normalize_messages(messages: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": message.get("role", "user"), "content": message_content_to_text(message.get("content", ""))}
        for message in messages
    ]


def merge_stop(stop: str | list[str] | None, defaults: list[str]) -> list[str]:
    if stop is None:
        return defaults
    stops = [stop] if isinstance(stop, str) else list(stop)
    return stops + [item for item in defaults if item not in stops]

