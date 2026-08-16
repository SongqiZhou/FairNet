import numpy as np
import torch
from torch import nn

from fairnet.config import FairNetConfig
from fairnet.utils import (
    compute_fairness_metrics,
    load_checkpoint,
    save_checkpoint,
)


def test_wga_uses_sensitive_groups_not_label_group_cells():
    labels = [0, 0, 1, 1, 0, 1]
    predictions = [0, 0, 1, 0, 0, 1]
    sensitive = [0, 0, 0, 0, 1, 1]

    metrics = compute_fairness_metrics(labels, predictions, sensitive)

    assert metrics["acc_group_0"] == 0.75
    assert metrics["acc_group_1"] == 1.0
    assert metrics["acc_group_0_1"] == 0.5
    assert metrics["worst_group_accuracy"] == 0.75
    assert metrics["EOD"] == 0.25
    assert metrics["EOp"] == 0.5


def test_multiclass_fairness_metrics_are_finite():
    metrics = compute_fairness_metrics(
        labels=[0, 1, 2, 0, 1, 2],
        predictions=[0, 1, 0, 0, 2, 2],
        sensitive=[0, 0, 0, 1, 1, 1],
    )

    assert all(np.isfinite(metrics[key]) for key in ("EOD", "EOp", "DP"))


def test_checkpoint_round_trip_uses_safe_payload(tmp_path):
    model = nn.Linear(2, 1)
    config = FairNetConfig()
    path = tmp_path / "nested" / "model.pt"
    expected = {"accuracy": 0.75}

    save_checkpoint(model, config, expected, path)
    with torch.no_grad():
        model.weight.zero_()
    metrics = load_checkpoint(model, path, torch.device("cpu"))

    assert metrics == expected
    assert not torch.equal(model.weight, torch.zeros_like(model.weight))
