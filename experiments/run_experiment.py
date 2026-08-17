"""Run one FairNet (or ERM) configuration end to end and record the result.

Every row of the paper's Table 1 corresponds to one config in
``experiments/configs`` executed at one or more seeds::

    python experiments/run_experiment.py --config celeba_erm       --seed 0
    python experiments/run_experiment.py --config celeba_full      --seed 0
    python experiments/run_experiment.py --config celeba_partial   --seed 0
    python experiments/run_experiment.py --config celeba_unlabeled --seed 0

Results are written as JSON under ``results/<name>/seed<k>.json`` and turned into
the README tables by ``experiments/aggregate_results.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (  # noqa: E402
    PAPER_METRICS,
    build_fairnet_config,
    build_loaders,
    build_model,
    build_trainer,
    environment_report,
    format_paper_row,
    load_config,
    resolve_device,
    serialise_config,
    write_result,
)
from fairnet import (  # noqa: E402
    AttributeMode,
    evaluate_detector,
    evaluate_model,
    load_checkpoint,
    print_metrics,
    save_checkpoint,
    seed_everything,
    sweep_activation_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Config name or path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override the dataset root recorded in the config",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override the result directory name",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a fairnet config field, e.g. --set stage1_epochs=2",
    )
    parser.add_argument(
        "--sweep-threshold",
        action="store_true",
        help="Also run the Supplementary Table I activation-threshold grid",
    )
    parser.add_argument(
        "--save-checkpoint",
        default=None,
        help="Optional path to store the trained model",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        default=None,
        help=(
            "Share one ERM base model across variants. The file is written on "
            "the first run and reused (skipping Stage 1) afterwards, so the "
            "Table 1 rows differ only in Stages 2-4."
        ),
    )
    args = parser.parse_args()

    spec = load_config(args.config)
    if args.data_root:
        spec.setdefault("data", {})["root"] = args.data_root
    if args.name:
        spec["name"] = args.name
    for override in args.set:
        key, _, value = override.partition("=")
        if not _:
            raise SystemExit(f"--set expects KEY=VALUE, got {override!r}")
        spec.setdefault("fairnet", {})[key] = yaml.safe_load(value)

    device = resolve_device(args.device or spec.get("device"))
    seed_everything(args.seed, deterministic=spec.get("deterministic", False))

    config = build_fairnet_config(spec, args.seed, device)
    train_loader, val_loader, test_loader = build_loaders(spec, config, args.seed)
    model = build_model(spec, config)
    trainer = build_trainer(model, config, device)

    variant = spec["variant"]
    stage1_path = Path(args.stage1_checkpoint) if args.stage1_checkpoint else None
    reused_stage1 = False
    if stage1_path is not None and stage1_path.is_file():
        # LoRA matrices are zero-initialised in B, so a base checkpoint saved
        # before Stage 4 restores an unmodified ERM model.
        load_checkpoint(model, stage1_path, device)
        trainer.skip_stage1 = True
        reused_stage1 = True
        print(f"Loaded shared Stage 1 base model from {stage1_path}")

    started = time.time()

    if variant == "erm":
        # The ERM row of Table 1 is the base model of Stage 1 with the
        # correction mechanism never engaged.
        trainer.stage1_train_base(train_loader, val_loader)
        use_lora = False
    else:
        trainer.train_full(train_loader, val_loader)
        use_lora = True

    elapsed = time.time() - started

    if stage1_path is not None and not reused_stage1:
        if variant != "erm":
            raise SystemExit(
                "Write the shared Stage 1 checkpoint from the ERM run first; "
                f"{stage1_path} does not exist"
            )
        save_checkpoint(model, config, {}, stage1_path)
        print(f"Saved shared Stage 1 base model to {stage1_path}")

    metrics = evaluate_model(model, test_loader, config, device, use_lora=use_lora)
    print_metrics(metrics, title=f"{spec['name']} (seed {args.seed}) test metrics")
    accuracy, worst_group, equalized_odds = format_paper_row(metrics)
    print(f"\nPaper units -> ACC {accuracy:.1f}  WGA {worst_group:.1f}  EOD {equalized_odds:.1f}")

    payload = {
        "name": spec["name"],
        "variant": variant,
        "dataset": spec["dataset"],
        "seed": args.seed,
        "config_path": spec["_config_path"],
        "fairnet_config": serialise_config(config),
        "model": spec["model"],
        "data": spec.get("data", {}),
        "train_seconds": elapsed,
        "reused_stage1_checkpoint": reused_stage1,
        "test_metrics": {key: metrics[key] for key in metrics if key != "per_attribute"},
        "paper_units": {
            "ACC": accuracy,
            "WGA": worst_group,
            "EOD": equalized_odds,
        },
        "history": {key: list(values) for key, values in trainer.history.items()},
        "environment": environment_report(),
    }

    if variant not in {"erm", "full"}:
        payload["detector_test_rates"] = evaluate_detector(model, test_loader, config, device)

    if args.sweep_threshold and variant != "erm":
        payload["threshold_sweep"] = sweep_activation_threshold(
            model, test_loader, config, device
        )

    if config.attribute_mode == AttributeMode.PARTIAL:
        payload["labeled_fraction"] = config.labeled_fraction
        payload["num_labeled"] = len(getattr(trainer, "labeled_indices", []) or [])

    write_result(Path(args.out_dir) / spec["name"] / f"seed{args.seed}.json", payload)

    if args.save_checkpoint:
        save_checkpoint(model, config, metrics, args.save_checkpoint)

    missing = [key for key in PAPER_METRICS if key not in metrics]
    if missing:
        raise SystemExit(f"Evaluation did not produce {missing}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
