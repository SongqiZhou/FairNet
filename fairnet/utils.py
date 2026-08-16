"""Reproducibility, evaluation, and checkpoint helpers for FairNet."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import AttributeMode, FairNetConfig


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch.

    Deterministic CUDA kernels improve repeatability but can reduce throughput
    and may reject operations for which PyTorch has no deterministic kernel.
    """

    if seed < 0:
        raise ValueError("seed cannot be negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_fairness_metrics(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    sensitive: Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Compute the paper's ACC, WGA, EOD, and supporting diagnostics.

    WGA follows Appendix C.4: it is the minimum accuracy over sensitive groups,
    not the minimum over label-by-group cells. For multiclass tasks, EOD and EOp
    are macro averages of one-vs-rest group disparities.
    """

    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    sensitive = np.asarray(sensitive).reshape(-1)
    if not (len(labels) == len(predictions) == len(sensitive)):
        raise ValueError("labels, predictions, and sensitive must have equal length")
    if len(labels) == 0:
        raise ValueError("Fairness metrics require at least one sample")

    metrics: dict[str, float] = {"accuracy": float(accuracy_score(labels, predictions))}
    task_classes = np.unique(labels)
    groups = np.unique(sensitive)

    group_accuracies = []
    for group in groups:
        group_mask = sensitive == group
        group_accuracy = float((predictions[group_mask] == labels[group_mask]).mean())
        metrics[f"acc_group_{int(group)}"] = group_accuracy
        group_accuracies.append(group_accuracy)

        # These intersection diagnostics are useful for debugging but do not
        # enter the paper's WGA definition.
        for task_class in task_classes:
            cell = group_mask & (labels == task_class)
            if cell.any():
                metrics[f"acc_group_{int(group)}_{int(task_class)}"] = float(
                    (predictions[cell] == labels[cell]).mean()
                )

    metrics["worst_group_accuracy"] = min(group_accuracies)
    metrics["accuracy_gap"] = max(group_accuracies) - min(group_accuracies)

    equalized_odds = []
    equal_opportunity = []
    demographic_parity = []
    for task_class in task_classes:
        true_positive_rates = []
        false_positive_rates = []
        predicted_positive_rates = []
        for group in groups:
            group_mask = sensitive == group
            positives = group_mask & (labels == task_class)
            negatives = group_mask & (labels != task_class)
            tpr = (
                float((predictions[positives] == task_class).mean()) if positives.any() else np.nan
            )
            fpr = (
                float((predictions[negatives] == task_class).mean()) if negatives.any() else np.nan
            )
            positive_rate = float((predictions[group_mask] == task_class).mean())
            metrics[f"TPR_group_{int(group)}_class_{int(task_class)}"] = tpr
            metrics[f"FPR_group_{int(group)}_class_{int(task_class)}"] = fpr
            true_positive_rates.append(tpr)
            false_positive_rates.append(fpr)
            predicted_positive_rates.append(positive_rate)

        finite_tpr = np.asarray(true_positive_rates)[np.isfinite(true_positive_rates)]
        finite_fpr = np.asarray(false_positive_rates)[np.isfinite(false_positive_rates)]
        tpr_gap = float(np.ptp(finite_tpr)) if len(finite_tpr) > 1 else 0.0
        fpr_gap = float(np.ptp(finite_fpr)) if len(finite_fpr) > 1 else 0.0
        equal_opportunity.append(tpr_gap)
        equalized_odds.append(0.5 * (tpr_gap + fpr_gap))
        demographic_parity.append(float(np.ptp(predicted_positive_rates)))

    # For binary classification, report the conventional positive-class metric
    # rather than averaging the two algebraically redundant one-vs-rest views.
    positive_index = -1 if len(task_classes) == 2 else None
    metrics["EOD"] = float(
        equalized_odds[positive_index] if positive_index is not None else np.mean(equalized_odds)
    )
    metrics["EOp"] = float(
        equal_opportunity[positive_index]
        if positive_index is not None
        else np.mean(equal_opportunity)
    )
    metrics["DP"] = float(
        demographic_parity[positive_index]
        if positive_index is not None
        else np.mean(demographic_parity)
    )

    if set(groups) == {0, 1} and set(task_classes) == {0, 1}:
        metrics["TPR_majority"] = metrics["TPR_group_0_class_1"]
        metrics["TPR_minority"] = metrics["TPR_group_1_class_1"]
        metrics["FPR_majority"] = metrics["FPR_group_0_class_1"]
        metrics["FPR_minority"] = metrics["FPR_group_1_class_1"]

    return metrics


def _unpack_evaluation_batch(batch, config: FairNetConfig, device: torch.device):
    if isinstance(batch, Mapping):
        inputs = {
            key: batch[key].to(device)
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in batch
        }
        if not {"input_ids", "attention_mask"}.issubset(inputs):
            raise KeyError("Text batches require input_ids and attention_mask")
        labels = batch["labels"].to(device)
        if "attributes" in batch:
            attributes = batch["attributes"]
            sensitive = {
                attr: attributes[:, attr].to(device) for attr in config.sensitive_attributes
            }
        elif "sensitive" in batch:
            values = batch["sensitive"]
            if values.ndim == 1:
                if len(config.sensitive_attributes) != 1:
                    raise ValueError("One-dimensional sensitive labels support one attribute")
                values = values[:, None]
            if values.shape[1] != len(config.sensitive_attributes):
                raise ValueError("Sensitive-label columns must match sensitive_attributes")
            sensitive = {
                attr: values[:, position].to(device)
                for position, attr in enumerate(config.sensitive_attributes)
            }
        else:
            raise KeyError("Text batches require attributes or sensitive")
        return inputs, labels, sensitive

    inputs, attributes = batch[0].to(device), batch[1]
    labels = attributes[:, config.target_attribute].to(device)
    sensitive = {attr: attributes[:, attr].to(device) for attr in config.sensitive_attributes}
    return inputs, labels, sensitive


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    config: FairNetConfig,
    device: torch.device,
    use_lora: bool = True,
) -> dict[str, Any]:
    """Evaluate vision or text FairNet models for every sensitive attribute."""

    if len(loader) == 0:
        raise ValueError("Evaluation loader is empty")
    model.eval()
    labels_all = []
    predictions_all = []
    sensitive_all = {attr: [] for attr in config.sensitive_attributes}
    risk_all = {attr: [] for attr in config.sensitive_attributes}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            inputs, labels, sensitive = _unpack_evaluation_batch(batch, config, device)
            if use_lora:
                direct_labels = sensitive if config.attribute_mode == AttributeMode.FULL else None
                if isinstance(inputs, Mapping):
                    outputs, risk_scores = model(
                        **inputs,
                        return_risk_scores=True,
                        sensitive_labels=direct_labels,
                    )
                else:
                    outputs, risk_scores = model(
                        inputs,
                        return_risk_scores=True,
                        sensitive_labels=direct_labels,
                    )
                for attr, scores in risk_scores.items():
                    risk_all[attr].extend(scores.cpu().numpy().reshape(-1))
            else:
                model._clear_lora_activation()
                if isinstance(inputs, Mapping):
                    features = model.get_cls_features(**inputs, use_lora=False)
                else:
                    features = model.get_cls_features(inputs, use_lora=False)
                outputs = model.classifier(features)

            predictions = (
                (outputs.reshape(-1) > 0.5).long()
                if outputs.shape[-1] == 1
                else outputs.argmax(dim=-1)
            )
            labels_all.extend(labels.cpu().numpy().reshape(-1))
            predictions_all.extend(predictions.cpu().numpy().reshape(-1))
            for attr in config.sensitive_attributes:
                sensitive_all[attr].extend(sensitive[attr].cpu().numpy().reshape(-1))

    per_attribute = {
        attr: compute_fairness_metrics(labels_all, predictions_all, sensitive_all[attr])
        for attr in config.sensitive_attributes
    }
    primary = config.sensitive_attributes[0]
    metrics: dict[str, Any] = dict(per_attribute[primary])
    metrics["per_attribute"] = per_attribute

    if use_lora:
        for attr, scores in risk_all.items():
            values = np.asarray(scores)
            metrics[f"lora_activation_rate_{attr}"] = float(
                np.mean(values > config.activation_threshold)
            )
            metrics[f"risk_score_mean_{attr}"] = float(values.mean())
            metrics[f"risk_score_std_{attr}"] = float(values.std())
        metrics["lora_activation_rate"] = metrics[f"lora_activation_rate_{primary}"]

    return metrics


def print_metrics(metrics: Mapping[str, Any], title: str = "Evaluation Results") -> None:
    """Print the primary paper metrics and available group diagnostics."""

    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    print(f"Overall accuracy:      {metrics.get('accuracy', 0):.4f}")
    print(f"Worst-group accuracy:  {metrics.get('worst_group_accuracy', 0):.4f}")
    print(f"Accuracy gap:          {metrics.get('accuracy_gap', 0):.4f}")
    print(f"Equalized odds diff.:  {metrics.get('EOD', 0):.4f}")
    print(f"Equal opportunity:     {metrics.get('EOp', 0):.4f}")
    print(f"Demographic parity:    {metrics.get('DP', 0):.4f}")
    if "lora_activation_rate" in metrics:
        print(f"LoRA activation rate:  {metrics['lora_activation_rate']:.4f}")


def compute_class_weights(
    loader: DataLoader, target_attr: int, device: torch.device
) -> torch.Tensor:
    """Compute inverse-frequency class weights for vision-style batches."""

    counts: dict[int, int] = {}
    for batch in loader:
        labels = batch[1][:, target_attr].cpu().numpy()
        for label in labels:
            key = int(label)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise ValueError("Cannot compute class weights from an empty loader")
    classes = sorted(counts)
    if classes != list(range(len(classes))):
        raise ValueError("Class labels must be contiguous integers starting at zero")
    total = sum(counts.values())
    weights = [total / (len(classes) * counts[label]) for label in classes]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_group_indices(
    loader: DataLoader, target_attr: int, sensitive_attr: int
) -> dict[str, list[int]]:
    """Return iteration-order positions for each label-by-group cell."""

    groups: dict[str, list[int]] = {}
    offset = 0
    for batch in loader:
        attributes = batch[1]
        labels = attributes[:, target_attr].cpu().numpy()
        sensitive = attributes[:, sensitive_attr].cpu().numpy()
        for index, (label, group) in enumerate(zip(labels, sensitive)):
            key = f"group_{int(group)}_{int(label)}"
            groups.setdefault(key, []).append(offset + index)
        offset += len(labels)
    return groups


def save_checkpoint(
    model: nn.Module,
    config: FairNetConfig,
    metrics: Mapping[str, Any],
    path: str | Path,
) -> None:
    """Save model parameters, configuration values, and evaluation metrics."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            key: value.value if isinstance(value, Enum) else value
            for key, value in config.__dict__.items()
        },
        "metrics": dict(metrics),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint created by :func:`save_checkpoint`."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before ``weights_only`` was added.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return dict(checkpoint.get("metrics", {}))
