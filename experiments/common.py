"""Shared plumbing for the paper-reproduction experiment scripts."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from fairnet import (
    AttributeMode,
    FairNetBERT,
    FairNetConfig,
    FairNetPartialTrainer,
    FairNetTrainer,
    FairNetUnlabeledTrainer,
    FairNetViT,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

#: Metrics reported in Table 1 of the paper, in the paper's column order.
PAPER_METRICS = ("accuracy", "worst_group_accuracy", "EOD")


def _resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    for suffix in ("", ".yaml", ".yml"):
        alternative = CONFIG_DIR / f"{candidate.name}{suffix}"
        if alternative.exists():
            return alternative
    raise FileNotFoundError(f"No experiment config at {path}")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _seen: Optional[set] = None) -> Dict[str, Any]:
    """Read an experiment YAML file, resolving bare names against ``configs/``.

    A config may set ``base: <other-config>`` to inherit and override another
    file, which keeps the per-variant configs down to the handful of keys that
    actually differ between Table 1 rows.
    """

    import yaml

    candidate = _resolve_config_path(path)
    seen = _seen or set()
    if candidate.resolve() in seen:
        raise ValueError(f"Circular config inheritance through {candidate}")
    seen = seen | {candidate.resolve()}

    with candidate.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{candidate} must contain a YAML mapping")

    parent = config.pop("base", None)
    if parent is not None:
        inherited = load_config(parent, seen)
        inherited.pop("_config_path", None)
        inherited.pop("name", None)
        config = _deep_merge(inherited, config)

    config.setdefault("name", candidate.stem)
    config["_config_path"] = str(candidate)
    return config


def build_fairnet_config(spec: Dict[str, Any], seed: int, device: torch.device) -> FairNetConfig:
    """Instantiate :class:`FairNetConfig` from the ``fairnet`` block of a config."""

    known = {field.name for field in fields(FairNetConfig)}
    values = {key: value for key, value in spec.get("fairnet", {}).items() if key in known}
    unknown = set(spec.get("fairnet", {})) - known
    if unknown:
        raise ValueError(f"Unknown FairNetConfig keys: {sorted(unknown)}")
    values["seed"] = seed
    values["device"] = str(device)

    variant = spec["variant"]
    if variant in {"erm", "full"}:
        values["attribute_mode"] = AttributeMode.FULL
    elif variant == "partial":
        values["attribute_mode"] = AttributeMode.PARTIAL
    elif variant == "unlabeled":
        values["attribute_mode"] = AttributeMode.UNLABELED
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return FairNetConfig(**values)


def build_loaders(spec: Dict[str, Any], config: FairNetConfig, seed: int):
    """Construct train/validation/test loaders for the configured dataset."""

    dataset = spec["dataset"]
    options = dict(spec.get("data", {}))

    if dataset == "celeba":
        from fairnet import create_celeba_loaders

        return create_celeba_loaders(
            root=options.pop("root"),
            batch_size=config.batch_size,
            image_size=config.image_size,
            target_attr=config.target_attribute,
            sensitive_attr=config.sensitive_attributes[0],
            seed=seed,
            **options,
        )
    if dataset == "synthetic":
        from fairnet import create_synthetic_loaders

        return create_synthetic_loaders(
            batch_size=config.batch_size,
            image_size=config.image_size,
            seed=seed,
            **options,
        )
    if dataset == "multinli":
        from fairnet import create_multinli_loaders

        return create_multinli_loaders(
            model_name=spec["model"].get("model_name", "bert-base-uncased"),
            batch_size=config.batch_size,
            max_length=config.max_seq_length,
            seed=seed,
            **options,
        )
    if dataset == "hatexplain":
        from fairnet import create_hatexplain_loaders

        return create_hatexplain_loaders(
            model_name=spec["model"].get("model_name", "bert-base-uncased"),
            batch_size=config.batch_size,
            max_length=config.max_seq_length,
            seed=seed,
            **options,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def build_model(spec: Dict[str, Any], config: FairNetConfig) -> torch.nn.Module:
    """Construct the backbone described by the ``model`` block."""

    model_spec = dict(spec["model"])
    kind = model_spec.pop("kind")

    if kind == "vit":
        from transformers import ViTConfig

        backbone = ViTConfig(
            image_size=model_spec.pop("image_size", config.image_size),
            patch_size=model_spec.pop("patch_size", 16),
            num_channels=model_spec.pop("num_channels", 3),
            hidden_size=model_spec.pop("hidden_size", config.hidden_dim),
            num_hidden_layers=model_spec.pop("num_hidden_layers", 8),
            num_attention_heads=model_spec.pop("num_attention_heads", 8),
            intermediate_size=model_spec.pop("intermediate_size", 768),
            hidden_dropout_prob=model_spec.pop("hidden_dropout_prob", 0.0),
            attention_probs_dropout_prob=model_spec.pop("attention_probs_dropout_prob", 0.0),
        )
        if model_spec:
            raise ValueError(f"Unused ViT model keys: {sorted(model_spec)}")
        return FairNetViT(backbone, config)

    if kind == "bert":
        return FairNetBERT(
            config,
            num_classes=model_spec.pop("num_classes", 3),
            model_name=model_spec.pop("model_name", "bert-base-uncased"),
        )

    if kind == "distilbert":
        from fairnet import FairNetDistilBERT

        return FairNetDistilBERT(
            config,
            num_classes=model_spec.pop("num_classes", 3),
            model_name=model_spec.pop("model_name", "distilbert-base-uncased"),
        )

    raise ValueError(f"Unknown model kind: {kind}")


def build_trainer(model, config: FairNetConfig, device: torch.device) -> FairNetTrainer:
    """Return the trainer matching the configured attribute mode."""

    if config.attribute_mode == AttributeMode.FULL:
        return FairNetTrainer(model, config, device)
    if config.attribute_mode == AttributeMode.PARTIAL:
        return FairNetPartialTrainer(model, config, device)
    return FairNetUnlabeledTrainer(model, config, device)


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_report() -> Dict[str, Any]:
    """Capture the versions and hardware a result was produced on."""

    import numpy
    import sklearn
    import transformers

    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "gpus": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
    }
    try:
        report["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        report["git_commit"] = None
    return report


def serialise_config(config: FairNetConfig) -> Dict[str, Any]:
    return {
        key: value.value if isinstance(value, AttributeMode) else value
        for key, value in config.__dict__.items()
    }


def write_result(path: str | Path, payload: Dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    print(f"\nWrote {destination}")
    return destination


def format_paper_row(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return ``(ACC, WGA, EOD)`` as percentages, the units used in the paper."""

    return (
        100.0 * float(metrics["accuracy"]),
        100.0 * float(metrics["worst_group_accuracy"]),
        100.0 * float(metrics["EOD"]),
    )
