"""Behaviours the paper-reproduction pipeline depends on."""

import pytest
import torch
from transformers import ViTConfig

from fairnet import (
    AttributeMode,
    FairNetConfig,
    FairNetPartialTrainer,
    FairNetTrainer,
    FairNetUnlabeledTrainer,
    FairNetViT,
    create_synthetic_loaders,
    evaluate_model,
    set_activation_threshold,
    summarize_metrics,
    sweep_activation_threshold,
)
from fairnet.text_datasets import (
    GROUPDRO_NEGATION_WORDS,
    PAPER_NEGATION_WORDS,
    has_negation,
)


def _config(**overrides):
    values = dict(
        attribute_mode=AttributeMode.PARTIAL,
        labeled_fraction=0.5,
        hidden_dim=32,
        image_size=32,
        batch_size=16,
        detector_layer=1,
        lora_layers=[2, 3],
        stage1_epochs=1,
        stage2_epochs=1,
        stage4_epochs=1,
        warmup_steps=2,
        device="cpu",
    )
    values.update(overrides)
    return FairNetConfig(**values)


def _model(config):
    backbone = ViTConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=32,
    )
    return FairNetViT(backbone, config)


def _loaders(config):
    return create_synthetic_loaders(
        batch_size=config.batch_size,
        num_train=128,
        num_val=64,
        num_test=64,
        minority_ratio=0.25,
        image_size=config.image_size,
        seed=config.seed,
    )


# --- checkpoint selection may not leak withheld labels ----------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (AttributeMode.FULL, "worst_group_accuracy"),
        (AttributeMode.PARTIAL, "accuracy"),
        (AttributeMode.UNLABELED, "accuracy"),
    ],
)
def test_stage4_selection_metric_matches_label_availability(mode, expected):
    extra = {"labeled_fraction": 0.5} if mode == AttributeMode.PARTIAL else {}
    config = _config(attribute_mode=mode, **extra)
    trainer_class = {
        AttributeMode.FULL: FairNetTrainer,
        AttributeMode.PARTIAL: FairNetPartialTrainer,
        AttributeMode.UNLABELED: FairNetUnlabeledTrainer,
    }[mode]

    trainer = trainer_class(_model(config), config, torch.device("cpu"))

    assert trainer.selection_metric == expected


def test_stage1_always_selects_on_accuracy():
    # Stage 1 is the shared ERM baseline; selecting it on WGA would make the
    # baseline fairness-aware and would need labels Partial/Unlabeled lack.
    config = _config(attribute_mode=AttributeMode.FULL)
    trainer = FairNetTrainer(_model(config), config, torch.device("cpu"))

    assert trainer.stage1_selection_metric == "accuracy"


def test_wga_selection_is_rejected_outside_the_full_setting():
    with pytest.raises(ValueError, match="no validation sensitive labels"):
        _config(
            attribute_mode=AttributeMode.PARTIAL,
            labeled_fraction=0.5,
            model_selection_metric="worst_group_accuracy",
        )


# --- ablation switches ------------------------------------------------------


def test_task_objective_skips_prototypes_and_still_trains_lora():
    config = _config(stage4_objective="task")
    model = _model(config)
    train_loader, val_loader, _ = _loaders(config)
    trainer = FairNetPartialTrainer(model, config, torch.device("cpu"))

    before = [parameter.clone() for parameter in model.get_lora_parameters()]
    trainer.train_full(train_loader, val_loader)
    after = model.get_lora_parameters()

    assert not trainer.prototype_banks[9].prototypes
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_zero_threshold_makes_the_correction_unconditional():
    config = _config(activation_threshold=0.0)
    model = _model(config)
    _, _, test_loader = _loaders(config)
    trainer = FairNetPartialTrainer(model, config, torch.device("cpu"))
    train_loader, val_loader, _ = _loaders(config)
    trainer.train_full(train_loader, val_loader)

    metrics = evaluate_model(model, test_loader, config, torch.device("cpu"))

    assert metrics["lora_activation_rate"] == 1.0


# --- threshold grid search --------------------------------------------------


def test_threshold_sweep_covers_the_grid_and_restores_the_original():
    config = _config()
    model = _model(config)
    _, _, test_loader = _loaders(config)

    rows = sweep_activation_threshold(
        model, test_loader, config, torch.device("cpu"), thresholds=(0.0, 0.5, 1.0)
    )

    assert [row["threshold"] for row in rows] == [0.0, 0.5, 1.0]
    assert all({"ACC", "WGA", "EOD", "TPR", "FPR"} <= set(row) for row in rows)
    assert config.activation_threshold == 0.5


def test_set_activation_threshold_updates_every_injected_module():
    config = _config()
    model = _model(config)

    set_activation_threshold(model, config, 0.8)

    assert config.activation_threshold == 0.8
    assert all(module.threshold == 0.8 for module in model.lora_modules.values())


# --- reporting --------------------------------------------------------------


def test_summarize_metrics_reports_mean_and_sample_std():
    summary = summarize_metrics(
        [{"WGA": 80.0}, {"WGA": 82.0}, {"WGA": 84.0}], keys=["WGA"]
    )

    assert summary["num_runs"] == 3
    assert summary["WGA"]["mean"] == 82.0
    assert summary["WGA"]["std"] == pytest.approx(2.0)


# --- MultiNLI sensitive attribute -------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The man is not tall.", 1),
        ("She isn't going home.", 1),
        ("He will never arrive.", 1),
        ("The man is tall.", 0),
        ("Nothing about this is notable.", 0),  # "notable" must not match "not"
    ],
)
def test_paper_negation_cues(text, expected):
    assert has_negation(text, PAPER_NEGATION_WORDS) == expected


def test_groupdro_negation_list_differs_from_the_papers():
    text = "Nothing happened."

    assert has_negation(text, GROUPDRO_NEGATION_WORDS) == 1
    assert has_negation(text, PAPER_NEGATION_WORDS) == 0


# --- every variant's full pipeline must actually run -------------------------


@pytest.mark.parametrize(
    ("mode", "trainer_class"),
    [
        (AttributeMode.FULL, FairNetTrainer),
        (AttributeMode.PARTIAL, FairNetPartialTrainer),
        (AttributeMode.UNLABELED, FairNetUnlabeledTrainer),
    ],
)
def test_every_variant_trains_end_to_end(mode, trainer_class):
    # Regression guard: the detector-anchor path lived on the partial trainer
    # only, so FairNet-Unlabeled died at Stage 3 with an AttributeError.
    extra = {"labeled_fraction": 0.5} if mode == AttributeMode.PARTIAL else {}
    config = _config(attribute_mode=mode, **extra)
    model = _model(config)
    train_loader, val_loader, test_loader = _loaders(config)
    trainer = trainer_class(model, config, torch.device("cpu"))

    trainer.train_full(train_loader, val_loader)
    metrics = evaluate_model(model, test_loader, config, torch.device("cpu"))

    assert 0.0 <= metrics["worst_group_accuracy"] <= 1.0
    assert "stage4_loss" in trainer.history


def test_detector_anchor_loader_falls_back_when_nothing_is_flagged():
    config = _config(attribute_mode=AttributeMode.PARTIAL, labeled_fraction=0.5)
    model = _model(config)
    train_loader, _, _ = _loaders(config)
    trainer = FairNetPartialTrainer(model, config, torch.device("cpu"))
    # An untrained detector with a threshold of 1.0 can never fire.
    trainer.config.activation_threshold = 1.0

    assert trainer._detector_anchor_loader(train_loader) is None
