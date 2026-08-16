import pytest
import torch
from transformers import BertConfig, ViTConfig

from fairnet.config import AttributeMode, FairNetConfig
from fairnet.models import FairNetBERT, FairNetViT


def _fairnet_config():
    return FairNetConfig(
        attribute_mode=AttributeMode.FULL,
        hidden_dim=16,
        detector_hidden=8,
        detector_num_layers=1,
        detector_layer=0,
        lora_rank=2,
        lora_alpha=2,
        lora_dropout=0,
        lora_layers=[1],
        sensitive_attributes=[9],
        target_attribute=20,
    )


def test_vit_full_mode_uses_direct_sensitive_gate():
    config = _fairnet_config()
    backbone = ViTConfig(
        image_size=16,
        patch_size=8,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    model = FairNetViT(backbone, config).eval()
    labels = {9: torch.tensor([0, 1])}

    output, risks, features = model(
        torch.randn(2, 3, 16, 16),
        sensitive_labels=labels,
        return_risk_scores=True,
        return_features=True,
    )

    assert output.shape == (2, 1)
    assert features.shape == (2, 16)
    assert torch.equal(risks[9].reshape(-1), labels[9].float())


def test_vit_full_mode_requires_sensitive_labels_for_inference():
    config = _fairnet_config()
    backbone = ViTConfig(
        image_size=16,
        patch_size=8,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = FairNetViT(backbone, config).eval()

    with pytest.raises(ValueError, match="requires sensitive_labels"):
        model(torch.randn(2, 3, 16, 16))


def test_bert_accepts_offline_config_and_token_type_ids():
    config = _fairnet_config()
    backbone = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    model = FairNetBERT(config, num_classes=3, bert_config=backbone).eval()
    sensitive = {9: torch.tensor([0, 1])}

    logits, risks = model(
        input_ids=torch.randint(0, 32, (2, 6)),
        attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]]),
        token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]),
        sensitive_labels=sensitive,
        return_risk_scores=True,
    )

    assert logits.shape == (2, 3)
    assert torch.equal(risks[9].reshape(-1), sensitive[9].float())
