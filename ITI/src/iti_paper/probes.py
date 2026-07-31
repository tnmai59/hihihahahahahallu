from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .config import ITIDirections


def train_linear_probes(
    activations: torch.Tensor,
    labels: torch.Tensor,
    top_k: int = 48,
    test_size: float = 0.2,
    seed: int = 0,
    max_iter: int = 1000,
    metadata: dict | None = None,
) -> ITIDirections:
    """Train one logistic-regression probe per attention head and select top-`k`.

    The intervention direction defaults to the normalized center-of-mass direction
    `mean(true) - mean(false)`, one of the directions discussed in the paper.
    """

    if activations.ndim != 4:
        raise ValueError("activations must have shape [n, layers, heads, head_dim].")
    n, num_layers, num_heads, head_dim = activations.shape
    if labels.shape != (n,):
        raise ValueError("labels must have shape [n].")

    x_train, x_val, y_train, y_val = train_test_split(
        activations.float().numpy(),
        labels.numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=labels.numpy(),
    )

    accuracies = np.zeros((num_layers, num_heads), dtype=np.float32)
    directions = np.zeros((num_layers, num_heads, head_dim), dtype=np.float32)
    sigmas = np.zeros((num_layers, num_heads, head_dim), dtype=np.float32)

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            train_head = x_train[:, layer_idx, head_idx, :]
            val_head = x_val[:, layer_idx, head_idx, :]
            clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
            clf.fit(train_head, y_train)
            accuracies[layer_idx, head_idx] = clf.score(val_head, y_val)

            true_mean = train_head[y_train == 1].mean(axis=0)
            false_mean = train_head[y_train == 0].mean(axis=0)
            direction = true_mean - false_mean
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            projected = activations[:, layer_idx, head_idx, :].float().numpy() @ direction
            directions[layer_idx, head_idx, :] = direction
            sigmas[layer_idx, head_idx, :] = projected.std()

    flat_order = np.argsort(accuracies.reshape(-1))[::-1]
    selected = [np.unravel_index(int(idx), accuracies.shape) for idx in flat_order[:top_k]]
    mask = np.zeros((num_layers, num_heads, 1), dtype=np.float32)
    for layer_idx, head_idx in selected:
        mask[layer_idx, head_idx, 0] = 1.0

    return ITIDirections(
        directions=torch.from_numpy(directions * mask),
        sigmas=torch.from_numpy(sigmas * mask),
        probe_accuracies=torch.from_numpy(accuracies),
        selected_heads=[(int(layer), int(head)) for layer, head in selected],
        metadata={
            **(metadata or {}),
            "top_k": int(top_k),
            "test_size": float(test_size),
            "seed": int(seed),
            "direction": "center_of_mass",
            "num_probe_examples": int(n),
            "num_positive_examples": int(labels.sum().item()),
            "num_negative_examples": int((labels == 0).sum().item()),
        },
    )
