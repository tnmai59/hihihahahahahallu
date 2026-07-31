from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ITIConfig:
    """Runtime controls for inference-time intervention."""

    alpha: float = 15.0
    last_token_only: bool = True


@dataclass
class ITIDirections:
    """Selected intervention directions.

    directions and sigmas are indexed as [layer, head, head_dim].
    Non-selected heads must have zero directions and zero sigmas.
    """

    directions: torch.Tensor
    sigmas: torch.Tensor
    probe_accuracies: torch.Tensor
    selected_heads: list[tuple[int, int]]
    metadata: dict[str, Any]

    def save(self, path: str | Path) -> None:
        payload = {
            "directions": self.directions.detach().cpu(),
            "sigmas": self.sigmas.detach().cpu(),
            "probe_accuracies": self.probe_accuracies.detach().cpu(),
            "selected_heads": self.selected_heads,
            "metadata": self.metadata,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "ITIDirections":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        return cls(
            directions=payload["directions"],
            sigmas=payload["sigmas"],
            probe_accuracies=payload["probe_accuracies"],
            selected_heads=[tuple(x) for x in payload["selected_heads"]],
            metadata=dict(payload.get("metadata", {})),
        )
