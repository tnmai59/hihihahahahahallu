from types import SimpleNamespace

import torch.nn as nn

from iti_paper.model_utils import infer_head_shape, iter_attention_o_proj


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(4096, 3072)


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()


class FakeGemmaLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=16,
            hidden_size=3072,
            head_dim=256,
        )
        self.model = SimpleNamespace(layers=nn.ModuleList([FakeLayer(), FakeLayer()]))


def test_infer_head_shape_uses_o_proj_input_width_for_gemma_like_models():
    model = FakeGemmaLikeModel()

    assert infer_head_shape(model) == (2, 16, 256)


def test_iter_attention_o_proj_finds_decoder_layers():
    model = FakeGemmaLikeModel()

    projections = list(iter_attention_o_proj(model))

    assert [layer_idx for layer_idx, _ in projections] == [0, 1]
    assert all(o_proj.in_features == 4096 for _, o_proj in projections)
