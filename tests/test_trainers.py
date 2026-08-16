import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import ViTConfig

from fairnet import AttributeMode, FairNetConfig, FairNetTrainer, FairNetViT


def test_full_training_pipeline_runs_end_to_end():
    torch.manual_seed(3)
    attributes = torch.zeros(8, 40, dtype=torch.long)
    attributes[:, 20] = torch.tensor([0, 1] * 4)
    attributes[:, 9] = torch.tensor([0, 0, 1, 1] * 2)
    loader = DataLoader(
        TensorDataset(torch.randn(8, 3, 16, 16), attributes),
        batch_size=4,
        shuffle=False,
    )
    config = FairNetConfig(
        attribute_mode=AttributeMode.FULL,
        hidden_dim=16,
        detector_hidden=8,
        detector_num_layers=1,
        detector_layer=0,
        lora_rank=2,
        lora_alpha=2,
        lora_dropout=0,
        lora_layers=[1],
        stage1_epochs=1,
        stage4_epochs=1,
        warmup_steps=0,
        batch_size=4,
        sensitive_attributes=[9],
        target_attribute=20,
        device="cpu",
    )
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
    model = FairNetViT(backbone, config)

    metrics = FairNetTrainer(model, config, torch.device("cpu")).train_full(loader, loader)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["worst_group_accuracy"] <= 1
    assert len(model.get_lora_parameters()) > 0
