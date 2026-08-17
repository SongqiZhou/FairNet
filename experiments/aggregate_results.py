"""Turn ``results/*/seed*.json`` into the comparison tables used in the README.

Usage::

    python experiments/aggregate_results.py --results results
    python experiments/aggregate_results.py --results results --out REPRODUCTION.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.paper_reference import TABLE5_CELEBA, paper_row  # noqa: E402

#: Lower is better for these metrics, so "not weaker than the paper" flips sign.
LOWER_IS_BETTER = {"EOD"}

METRICS = ("ACC", "WGA", "EOD")


def load_runs(results_dir: Path) -> Dict[str, List[dict]]:
    runs: Dict[str, List[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*/seed*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        runs[payload["name"]].append(payload)
    return runs


def mean_std(values: List[float]) -> tuple[float, float]:
    count = len(values)
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, variance**0.5


def format_cell(mean: float, std: float, seeds: int) -> str:
    return f"{mean:.1f}" if seeds < 2 else f"{mean:.1f} ± {std:.1f}"


def comparison_table(runs: Dict[str, List[dict]]) -> str:
    lines = [
        "| Config | Seeds | ACC (repro / paper) | WGA (repro / paper) "
        "| EOD (repro / paper) | Meets paper |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for name in sorted(runs):
        payloads = sorted(runs[name], key=lambda item: item["seed"])
        seeds = len(payloads)
        reference = paper_row(name)
        cells = []
        verdicts = []
        for index, metric in enumerate(METRICS):
            values = [payload["paper_units"][metric] for payload in payloads]
            mean, std = mean_std(values)
            if reference is None:
                cells.append(format_cell(mean, std, seeds))
                continue
            target = reference[index]
            cells.append(f"{format_cell(mean, std, seeds)} / {target:.1f}")
            if metric in LOWER_IS_BETTER:
                verdicts.append(mean <= target)
            else:
                verdicts.append(mean >= target)
        if reference is None:
            verdict = "n/a"
        else:
            verdict = "yes" if all(verdicts) else f"{sum(verdicts)}/{len(verdicts)}"
        lines.append(f"| `{name}` | {seeds} | " + " | ".join(cells) + f" | {verdict} |")
    return "\n".join(lines)


def detector_table(runs: Dict[str, List[dict]]) -> str:
    rows = []
    for name in sorted(runs):
        payloads = [item for item in runs[name] if item.get("detector_test_rates")]
        if not payloads:
            continue
        tpr, tpr_std = mean_std([item["detector_test_rates"]["TPR"] * 100 for item in payloads])
        fpr, fpr_std = mean_std([item["detector_test_rates"]["FPR"] * 100 for item in payloads])
        ratio = tpr / fpr if fpr > 0 else float("inf")
        rows.append(
            f"| `{name}` | {format_cell(tpr, tpr_std, len(payloads))} "
            f"| {format_cell(fpr, fpr_std, len(payloads))} | {ratio:.1f} |"
        )
    if not rows:
        return ""
    header = [
        "| Config | Detector TPR (%) | Detector FPR (%) | TPR/FPR |",
        "| --- | ---: | ---: | ---: |",
    ]
    return "\n".join(header + rows)


def threshold_table(runs: Dict[str, List[dict]]) -> str:
    rows = []
    for name in sorted(runs):
        for payload in sorted(runs[name], key=lambda item: item["seed"]):
            for entry in payload.get("threshold_sweep", []):
                tpr = entry.get("TPR")
                fpr = entry.get("FPR")
                rows.append(
                    f"| `{name}` | {payload['seed']} | {entry['threshold']:.1f} "
                    f"| {'-' if tpr is None else f'{100 * tpr:.1f}'} "
                    f"| {'-' if fpr is None else f'{100 * fpr:.2f}'} "
                    f"| {100 * entry['ACC']:.1f} | {100 * entry['WGA']:.1f} "
                    f"| {100 * entry['EOD']:.1f} |"
                )
    if not rows:
        return ""
    header = [
        "| Config | Seed | tau | TPR (%) | FPR (%) | ACC (%) | WGA (%) | EOD (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    return "\n".join(header + rows)


def label_fraction_table(runs: Dict[str, List[dict]]) -> str:
    """Supplementary Table 5: FairNet-Partial against the labelled fraction."""

    rows = []
    for name in runs:
        payloads = [item for item in runs[name] if item.get("labeled_fraction") is not None]
        if not payloads or "frac" not in name:
            continue
        fraction = payloads[0]["labeled_fraction"]
        labelled = payloads[0].get("num_labeled", 0)
        detector = [
            item["detector_test_rates"] for item in payloads if item.get("detector_test_rates")
        ]
        tpr = mean_std([item["TPR"] * 100 for item in detector])[0] if detector else None
        fpr = mean_std([item["FPR"] * 100 for item in detector])[0] if detector else None
        cells = []
        for metric in METRICS:
            mean, std = mean_std([item["paper_units"][metric] for item in payloads])
            cells.append(format_cell(mean, std, len(payloads)))
        reference = TABLE5_CELEBA.get(round(fraction, 4))
        target = (
            f"{reference[2]:.1f} / {reference[3]:.1f} / {reference[4]:.1f}" if reference else "-"
        )
        rows.append(
            (
                fraction,
                f"| {fraction:.3%} | {labelled} "
                f"| {'-' if tpr is None else f'{tpr:.1f}'} "
                f"| {'-' if fpr is None else f'{fpr:.2f}'} "
                f"| {cells[0]} | {cells[1]} | {cells[2]} | {target} |",
            )
        )
    if not rows:
        return ""
    header = [
        "| Labelled fraction | Num | TPR (%) | FPR (%) | ACC (%) | WGA (%) | EOD (%) "
        "| Paper ACC/WGA/EOD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    return "\n".join(header + [row for _, row in sorted(rows)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default=None, help="Write markdown here instead of stdout")
    args = parser.parse_args()

    results_dir = Path(args.results)
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"No result files under {results_dir}")

    sections = ["## Reproduction vs. paper\n", comparison_table(runs)]
    detectors = detector_table(runs)
    if detectors:
        sections += ["\n## Bias detector rates on the test split\n", detectors]
    fractions = label_fraction_table(runs)
    if fractions:
        sections += ["\n## Labelled-fraction sweep (Supplementary Table 5)\n", fractions]
    thresholds = threshold_table(runs)
    if thresholds:
        sections += ["\n## Activation-threshold sweep (Supplementary Table I)\n", thresholds]

    markdown = "\n".join(sections) + "\n"
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
