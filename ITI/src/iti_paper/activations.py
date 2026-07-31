from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .dataset import QAPair
from .hooks import ActivationCollector


@torch.inference_mode()
def collect_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: list[QAPair],
    batch_size: int = 4,
    max_length: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return activations `[n, layers, heads, head_dim]` and labels `[n]`."""

    model.eval()
    device = next(model.parameters()).device
    texts = [pair.text for pair in pairs]
    labels = torch.tensor([pair.label for pair in pairs], dtype=torch.long)
    loader = DataLoader(list(zip(texts, labels.tolist())), batch_size=batch_size, shuffle=False)
    all_activations: list[torch.Tensor] = []
    all_labels: list[int] = []

    for batch_texts, batch_labels in loader:
        encoded = tokenizer(
            list(batch_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        token_positions = encoded["attention_mask"].sum(dim=1) - 1
        with ActivationCollector(model, token_positions=token_positions) as collector:
            model(**encoded, use_cache=False)
            all_activations.append(collector.stacked())
        all_labels.extend(int(x) for x in batch_labels)

    return torch.cat(all_activations, dim=0), torch.tensor(all_labels, dtype=torch.long)
