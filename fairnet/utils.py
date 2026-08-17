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


#: Worst-group accuracy over the ``(task label, sensitive group)`` cells. This
#: is the convention used by GroupDRO, JTT, DFR, Sebra and D3M, and it is the
#: one the paper's Table 1 numbers follow.
WGA_LABEL_BY_GROUP = "label_by_group"

#: Worst-group accuracy over the sensitive groups alone, which is the formula
#: written in Supplementary C.4.
WGA_SENSITIVE_GROUP = "sensitive_group"


def compute_fairness_metrics(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    sensitive: Sequence[int] | np.ndarray,
    wga_definition: str = WGA_LABEL_BY_GROUP,
) -> dict[str, Any]:
    """Compute the paper's ACC, WGA, EOD, and supporting diagnostics.

    **On the two WGA definitions.** Supplementary C.4 writes
    ``WGA = min(P(Y_hat = Y | S = 0), P(Y_hat = Y | S = 1))``, a minimum over the
    two sensitive groups. The numbers actually reported in Table 1 follow the
    other, far more common convention: the minimum over the
    ``(task label, sensitive group)`` cells. Two independent checks confirm it.

    * On CelebA the sensitive groups are "blond" (29,983 images, of which only
      1,749 are male) and "not blond". A model that simply predicted "not male"
      for every blond image would already score 94.2% on the blond group, so the
      reported ERM WGA of 77.9% cannot be a minimum over sensitive groups. It is
      consistent with the accuracy on the small blond-and-male cell.
    * On MultiNLI the reported ERM pair (ACC 82.6, WGA 67.3) matches the
      standard six-cell worst-group accuracy of Sagawa et al. for this dataset.

    Both quantities are always returned. ``worst_group_accuracy`` follows
    ``wga_definition`` and defaults to the Table 1 convention, so reproduction
    numbers are comparable with the paper and its baselines;
    ``worst_sensitive_group_accuracy`` and ``worst_label_group_cell_accuracy``
    are always available under their own explicit names.

    For multiclass tasks, EOD and EOp are macro averages of one-vs-rest group
    disparities.
    """

    if wga_definition not in {WGA_LABEL_BY_GROUP, WGA_SENSITIVE_GROUP}:
        raise ValueError(
            f"wga_definition must be {WGA_LABEL_BY_GROUP!r} or {WGA_SENSITIVE_GROUP!r}"
        )

    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    sensitive = np.asarray(sensitive).reshape(-1)
    if not (len(labels) == len(predictions) == len(sensitive)):
        raise ValueError("labels, predictions, and sensitive must have equal length")
    if len(labels) == 0:
        raise ValueError("Fairness metrics require at least one sample")

    metrics: dict[str, Any] = {"accuracy": float(accuracy_score(labels, predictions))}
    task_classes = np.unique(labels)
    groups = np.unique(sensitive)

    group_accuracies = []
    cell_accuracies = []
    for group in groups:
        group_mask = sensitive == group
        group_accuracy = float((predictions[group_mask] == labels[group_mask]).mean())
        metrics[f"acc_group_{int(group)}"] = group_accuracy
        group_accuracies.append(group_accuracy)

        for task_class in task_classes:
            cell = group_mask & (labels == task_class)
            if cell.any():
                cell_accuracy = float((predictions[cell] == labels[cell]).mean())
                metrics[f"acc_group_{int(group)}_{int(task_class)}"] = cell_accuracy
                metrics[f"count_group_{int(group)}_{int(task_class)}"] = int(cell.sum())
                cell_accuracies.append(cell_accuracy)

    metrics["worst_sensitive_group_accuracy"] = min(group_accuracies)
    metrics["worst_label_group_cell_accuracy"] = min(cell_accuracies)
    metrics["worst_group_accuracy"] = (
        metrics["worst_label_group_cell_accuracy"]
        if wga_definition == WGA_LABEL_BY_GROUP
        else metrics["worst_sensitive_group_accuracy"]
    )
    metrics["wga_definition"] = wga_definition
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
        attr: compute_fairness_metrics(
            labels_all,
            predictions_all,
            sensitive_all[attr],
            wga_definition=config.wga_definition,
        )
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


def set_activation_threshold(model: nn.Module, config: FairNetConfig, threshold: float) -> None:
    """Retune the conditional-LoRA activation threshold tau in place.

    Supplementary C.3.1 selects tau on a validation grid search (typical values
    0.5-0.8), so the threshold has to be adjustable after training without
    rebuilding the model. Both the config and every injected ``LoRALinear`` are
    updated so gating and reporting stay in sync.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    config.activation_threshold = threshold
    for lora in model.lora_modules.values():
        lora.threshold = threshold


def evaluate_detector(
    model: nn.Module,
    loader: DataLoader,
    config: FairNetConfig,
    device: torch.device,
    attribute: int | None = None,
) -> dict[str, float]:
    """Measure the bias detector's TPR/FPR on the minority group.

    These are the quantities Condition 8 of the paper depends on and the ones
    reported in Supplementary Tables 5, B, H, and I.
    """

    attribute = config.sensitive_attributes[0] if attribute is None else attribute
    if attribute not in config.sensitive_attributes:
        raise ValueError(f"{attribute} is not a configured sensitive attribute")

    model.eval()
    scores_all: list[float] = []
    targets_all: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring detector"):
            inputs, _, sensitive = _unpack_evaluation_batch(batch, config, device)
            if isinstance(inputs, Mapping):
                hidden = model.get_intermediate_features(**inputs)
                mask = inputs.get("attention_mask")
            else:
                hidden = model.get_intermediate_features(inputs)
                mask = None
            detector = model.bias_detectors[f"detector_{attribute}"]
            scores = detector(hidden, is_sequence=True, attention_mask=mask)
            scores_all.extend(scores.cpu().numpy().reshape(-1).tolist())
            targets_all.extend(sensitive[attribute].cpu().numpy().reshape(-1).tolist())

    scores = np.asarray(scores_all)
    targets = np.asarray(targets_all)
    fired = scores > config.activation_threshold
    minority = targets == 1
    majority = targets == 0
    tpr = float(fired[minority].mean()) if minority.any() else float("nan")
    fpr = float(fired[majority].mean()) if majority.any() else float("nan")
    return {
        "TPR": tpr,
        "FPR": fpr,
        "TPR_FPR_ratio": float(tpr / fpr) if fpr > 0 else float("inf"),
        "minority_prevalence": float(minority.mean()),
    }


def sweep_activation_threshold(
    model: nn.Module,
    loader: DataLoader,
    config: FairNetConfig,
    device: torch.device,
    thresholds: Sequence[float] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0),
) -> list[dict[str, Any]]:
    """Reproduce the Supplementary Table I threshold ablation.

    Runs :func:`evaluate_model` once per threshold and restores the original
    threshold afterwards. Detector rates are included whenever a learned
    detector gates the correction; FairNet-Full gates on ground-truth labels and
    therefore has no detector rates to report.
    """

    original = config.activation_threshold
    rows: list[dict[str, Any]] = []
    try:
        for threshold in thresholds:
            set_activation_threshold(model, config, threshold)
            metrics = evaluate_model(model, loader, config, device, use_lora=True)
            row: dict[str, Any] = {
                "threshold": float(threshold),
                "ACC": metrics["accuracy"],
                "WGA": metrics["worst_group_accuracy"],
                "EOD": metrics["EOD"],
                "lora_activation_rate": metrics.get("lora_activation_rate"),
            }
            if config.attribute_mode != AttributeMode.FULL:
                row.update(evaluate_detector(model, loader, config, device))
            rows.append(row)
    finally:
        set_activation_threshold(model, config, original)
    return rows


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


def summarize_metrics(runs: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    """Aggregate repeated runs into ``mean`` and ``std`` per metric.

    The paper reports mean +- standard deviation over repeated runs (NeurIPS
    checklist item 7), so multi-seed reproduction results are summarised the
    same way.
    """

    if not runs:
        raise ValueError("summarize_metrics requires at least one run")
    summary: dict[str, Any] = {"num_runs": len(runs)}
    for key in keys:
        values = np.asarray([float(run[key]) for run in runs], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "values": values.tolist(),
        }
    return summary


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
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        saved = checkpoint.get("config", {})
        raise RuntimeError(
            f"{checkpoint_path} does not match this model. A checkpoint stores one "
            "specific LoRA placement, so it can only be loaded into a model built "
            "with the same lora_layers and lora_target_modules. Saved config used "
            f"lora_layers={saved.get('lora_layers')} and "
            f"lora_target_modules={saved.get('lora_target_modules')}."
        ) from error
    return dict(checkpoint.get("metrics", {}))
