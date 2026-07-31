from __future__ import annotations

from collections.abc import Iterator

import torch.nn as nn


def iter_attention_o_proj(model: nn.Module) -> Iterator[tuple[int, nn.Module]]:
    """Yield `(layer_idx, o_proj)` for LLaMA/Gemma-style Hugging Face models."""

    layers = find_decoder_layers(model)

    for layer_idx, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        o_proj = getattr(self_attn, "o_proj", None)
        if o_proj is None:
            raise ValueError(f"Layer {layer_idx} does not expose `self_attn.o_proj`.")
        yield layer_idx, o_proj


def infer_head_shape(model: nn.Module) -> tuple[int, int, int]:
    config = text_config_for(model)
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

def find_decoder_layers(model: nn.Module):
    """Find decoder layers across plain and wrapped HF causal/conditional models."""

    roots = [
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "model", None),
    ]
    for root in roots:
        layers = getattr(root, "layers", None)
        if layers is not None:
            return layers
    raise ValueError("Expected a Hugging Face decoder model exposing decoder `layers`.")


def text_config_for(model: nn.Module):
    """Return the text decoder config, unwrapping multimodal configs like Gemma 3."""

    config = getattr(model, "config", None)
    if config is None:
        return None
    if hasattr(config, "num_hidden_layers") and hasattr(config, "num_attention_heads"):
        return config
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return get_text_config()
    return config
