import pytest
import torch
from torch import nn

from fairnet.modules import AttentionPooling, LoRALinear, UnsupervisedBiasDetector


def test_attention_pooling_ignores_padding():
    pooling = AttentionPooling(hidden_dim=2)
    hidden = torch.tensor([[[1.0, 2.0], [100.0, -100.0]]])
    mask = torch.tensor([[1, 0]])

    pooled = pooling(hidden, attention_mask=mask)

    assert torch.allclose(pooled, torch.tensor([[1.0, 2.0]]))


def test_attention_pooling_rejects_fully_masked_rows():
    pooling = AttentionPooling(hidden_dim=2)

    with pytest.raises(ValueError, match="at least one token"):
        pooling(torch.ones(1, 2, 2), attention_mask=torch.zeros(1, 2))


def test_lora_linear_uses_independent_attribute_adapters():
    original = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        original.weight.copy_(torch.tensor([[1.0, 1.0]]))
    layer = LoRALinear(
        original,
        rank=1,
        alpha=1,
        threshold=0.5,
        adapter_names=["race", "gender"],
    )
    with torch.no_grad():
        layer.lora_A["race"].copy_(torch.tensor([[1.0, 0.0]]))
        layer.lora_B["race"].copy_(torch.tensor([[2.0]]))
        layer.lora_A["gender"].copy_(torch.tensor([[0.0, 1.0]]))
        layer.lora_B["gender"].copy_(torch.tensor([[3.0]]))

    layer.set_activation(
        {
            "race": torch.tensor([[1.0], [0.0]]),
            "gender": torch.tensor([[0.0], [1.0]]),
        }
    )
    output = layer(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    assert torch.allclose(output, torch.tensor([[5.0], [19.0]]))


def test_lora_linear_rejects_misaligned_risk_scores():
    layer = LoRALinear(nn.Linear(2, 1), rank=1)
    layer.set_risk_score(torch.ones(3, 1))

    with pytest.raises(ValueError, match="batch size"):
        layer(torch.ones(2, 2))


def test_lof_scores_are_independent_of_evaluation_batch():
    features = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [8.0, 8.0]],
        dtype=torch.float32,
    )
    detector = UnsupervisedBiasDetector(hidden_dim=2, n_neighbors=2, contamination=0.25)

    pseudo_labels = detector.fit_predict(features)
    score_alone = detector(features[:1])
    score_in_batch = detector(features[:3])[:1]

    assert pseudo_labels.sum() == 1
    assert torch.allclose(score_alone, score_in_batch)
