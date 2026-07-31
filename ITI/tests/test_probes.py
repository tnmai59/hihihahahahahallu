import torch

from iti_paper.probes import train_linear_probes


def test_train_linear_probes_selects_informative_head():
    labels = torch.tensor([0, 1] * 20)
    activations = torch.randn(40, 2, 3, 4) * 0.05
    activations[:, 1, 2, 0] = labels.float() * 4.0 - 2.0

    directions = train_linear_probes(activations, labels, top_k=1, seed=0)

    assert directions.selected_heads[0] == (1, 2)
    assert directions.directions[1, 2].norm() > 0
    assert directions.directions[0, 0].norm() == 0
