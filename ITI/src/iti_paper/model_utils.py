from __future__ import annotations

from collections.abc import Iterator

import torch.nn as nn


def iter_attention_o_proj(model: nn.Module) -> Iterator[tuple[int, nn.Module]]:
    """Yield `(layer_idx, o_proj)` for LLaMA/Gemma-style Hugging Face models."""

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected a Hugging Face decoder-only model with `model.layers`.")

    for layer_idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        o_proj = getattr(self_attn, "o_proj", None)
        if o_proj is None:
            raise ValueError(f"Layer {layer_idx} does not expose `self_attn.o_proj`.")
        yield layer_idx, o_proj


def infer_head_shape(model: nn.Module) -> tuple[int, int, int]:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model is missing a Hugging Face config.")

    num_layers = int(getattr(config, "num_hidden_layers"))
    num_heads = int(getattr(config, "num_attention_heads"))
    first_o_proj = next(iter_attention_o_proj(model))[1]
    o_proj_in_features = getattr(first_o_proj, "in_features", None)
    if o_proj_in_features is not None:
        attention_width = int(o_proj_in_features)
    elif hasattr(config, "head_dim"):
        attention_width = num_heads * int(getattr(config, "head_dim"))
    else:
        attention_width = int(getattr(config, "hidden_size"))

    if attention_width % num_heads != 0:
        raise ValueError("attention output width must be divisible by num_attention_heads.")
    return num_layers, num_heads, attention_width // num_heads
