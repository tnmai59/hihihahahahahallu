from __future__ import annotations

from contextlib import AbstractContextManager

import torch
import torch.nn as nn

from .config import ITIConfig, ITIDirections
from .model_utils import infer_head_shape, iter_attention_o_proj


class ActivationCollector(AbstractContextManager["ActivationCollector"]):
    """Collect last-token pre-`o_proj` attention-head activations."""

    def __init__(self, model: nn.Module, token_positions: torch.Tensor | None = None):
        self.model = model
        self.num_layers, self.num_heads, self.head_dim = infer_head_shape(model)
        self.token_positions = token_positions
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.activations: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "ActivationCollector":
        for layer_idx, o_proj in iter_attention_o_proj(self.model):
            self.handles.append(o_proj.register_forward_pre_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, layer_idx: int):
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]):
            hidden = args[0]
            if self.token_positions is None:
                selected = hidden[:, -1, :]
            else:
                positions = self.token_positions.to(hidden.device)
                batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
                selected = hidden[batch_indices, positions, :]
            last = selected.detach().float().cpu()
            self.activations[layer_idx] = last.view(last.shape[0], self.num_heads, self.head_dim)
            return None

        return hook

    def stacked(self) -> torch.Tensor:
        missing = set(range(self.num_layers)) - set(self.activations)
        if missing:
            raise RuntimeError(f"Missing activations for layers: {sorted(missing)}")
        return torch.stack([self.activations[i] for i in range(self.num_layers)], dim=1)


class ITIHook(AbstractContextManager["ITIHook"]):
    """Apply ITI shifts to selected attention-head outputs during inference."""

    def __init__(
        self,
        model: nn.Module,
        directions: ITIDirections,
        config: ITIConfig | None = None,
        token_positions: torch.Tensor | None = None,
    ):
        self.model = model
        self.directions = directions
        self.config = config or ITIConfig()
        self.token_positions = token_positions
        self.num_layers, self.num_heads, self.head_dim = infer_head_shape(model)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._validate()

    def __enter__(self) -> "ITIHook":
        device = next(self.model.parameters()).device
        self._shift = (
            self.directions.directions.to(device=device, dtype=next(self.model.parameters()).dtype)
            * self.directions.sigmas.to(device=device, dtype=next(self.model.parameters()).dtype)
            * self.config.alpha
        )
        for layer_idx, o_proj in iter_attention_o_proj(self.model):
            self.handles.append(o_proj.register_forward_pre_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _validate(self) -> None:
        expected = (self.num_layers, self.num_heads, self.head_dim)
        if tuple(self.directions.directions.shape) != expected:
            raise ValueError(f"directions shape {tuple(self.directions.directions.shape)} != {expected}")
        if tuple(self.directions.sigmas.shape) != expected:
            raise ValueError(f"sigmas shape {tuple(self.directions.sigmas.shape)} != {expected}")

    def _make_hook(self, layer_idx: int):
        def hook(_module: nn.Module, args: tuple[torch.Tensor, ...]):
            hidden = args[0]
            shift = self._shift[layer_idx].reshape(1, 1, self.num_heads * self.head_dim)
            edited = hidden.clone()
            if self.token_positions is not None:
                positions = self.token_positions.to(hidden.device)
                batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
                edited[batch_indices, positions, :] = edited[batch_indices, positions, :] + shift[:, 0, :]
            elif self.config.last_token_only:
                edited[:, -1:, :] = edited[:, -1:, :] + shift
            else:
                edited = edited + shift
            return (edited, *args[1:])

        return hook
