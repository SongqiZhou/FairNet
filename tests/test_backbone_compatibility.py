"""LoRA injection must work across backbones and transformers major versions.

transformers 5 moved ViT's blocks from ``encoder.layer`` to ``layers`` and
renamed the attention projections to ``q_proj``/``v_proj``. These tests pin the
behaviour that FairNet resolves the block stack and the projections by search
rather than by a single hard-coded path.
"""

import pytest
import torch
from transformers import (
    BertConfig,
    BertModel,
    DistilBertConfig,
    DistilBertModel,
    ViTConfig,
    ViTModel,
)

from fairnet.modules import LoRAInjector, LoRALinear, get_encoder_layers


def _tiny_vit():
    return ViTModel(
        ViTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            image_size=32,
            patch_size=16,
        )
    )


def _tiny_bert():
    return BertModel(
        BertConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            vocab_size=64,
        )
    )


def _tiny_distilbert():
    return DistilBertModel(
        DistilBertConfig(dim=32, n_layers=2, n_heads=2, hidden_dim=32, vocab_size=64)
    )


@pytest.mark.parametrize("factory", [_tiny_vit, _tiny_bert, _tiny_distilbert])
def test_encoder_layers_are_discoverable(factory):
    layers = get_encoder_layers(factory())

    assert len(layers) == 2


@pytest.mark.parametrize("factory", [_tiny_vit, _tiny_bert, _tiny_distilbert])
def test_injection_replaces_every_requested_projection(factory):
    model = factory()

    modules = LoRAInjector.inject(
        model,
        rank=2,
        alpha=4.0,
        target_modules=["query", "key", "value", "dense"],
        target_layers=[1],
        adapter_names=["0"],
    )

    assert sorted(modules) == [
        "layer_1_dense",
        "layer_1_key",
        "layer_1_query",
        "layer_1_value",
    ]
    assert all(isinstance(module, LoRALinear) for module in modules.values())


def test_double_injection_is_rejected():
    model = _tiny_bert()
    LoRAInjector.inject(model, rank=2, target_layers=[0], adapter_names=["0"])

    with pytest.raises(ValueError, match="already injected"):
        LoRAInjector.inject(model, rank=2, target_layers=[0], adapter_names=["0"])


def test_unknown_backbone_reports_the_paths_it_tried():
    with pytest.raises(AttributeError, match="transformer block stack"):
        get_encoder_layers(torch.nn.Linear(2, 2))


def test_zero_initialised_lora_is_an_identity_before_training():
    model = _tiny_vit()
    pixel_values = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        baseline = model(pixel_values).last_hidden_state

    modules = LoRAInjector.inject(model, rank=2, target_layers=[0, 1], adapter_names=["0"])
    for module in modules.values():
        module.set_activation(forced_adapters={"0"})
    with torch.no_grad():
        adapted = model(pixel_values).last_hidden_state

    # B is zero-initialised, so the correction starts as an exact no-op.
    assert torch.allclose(baseline, adapted, atol=1e-6)
