"""The paper's published numbers, transcribed for automatic comparison.

Every entry is ``(ACC, WGA, EOD)`` in percent, exactly as printed in the NeurIPS
2025 proceedings version. ``experiments/aggregate_results.py`` diffs reproduction
runs against these so a regression is visible without re-reading the PDF.
"""

from __future__ import annotations

#: Table 1, CelebA columns.
TABLE1_CELEBA = {
    "ERM": (95.8, 77.9, 10.6),
    "Lu et al.": (95.4, 81.4, 8.3),
    "D3M": (95.2, 82.0, 8.1),
    "FairNet-Unlabeled": (95.8, 82.3, 7.3),
    "GroupDRO": (94.0, 87.4, 4.7),
    "DFR": (94.3, 86.0, 7.7),
    "Sebra": (94.8, 85.2, 8.1),
    "FairNet-Partial": (95.9, 86.5, 5.6),
    "Eq.Odds": (95.0, 83.2, 7.2),
    "GSTAR": (94.2, 85.4, 6.6),
    "FairNet-Full": (95.9, 88.2, 3.8),
}

#: Table 1, MultiNLI columns.
TABLE1_MULTINLI = {
    "ERM": (82.6, 67.3, 12.5),
    "Lu et al.": (82.0, 72.8, 8.5),
    "D3M": (81.0, 72.8, 8.3),
    "FairNet-Unlabeled": (82.5, 73.1, 8.1),
    "GroupDRO": (80.8, 78.2, 5.5),
    "DFR": (81.2, 74.1, 6.7),
    "Sebra": (81.5, 74.2, 6.5),
    "FairNet-Partial": (82.6, 76.5, 6.2),
    "Eq.Odds": (81.3, 75.3, 6.3),
    "GSTAR": (80.8, 76.6, 6.2),
    "FairNet-Full": (82.6, 78.5, 4.7),
}

#: Table 3, CelebA ablation columns.
TABLE3_CELEBA = {
    "FairNet-Partial": (95.9, 86.5, 5.6),
    "w/o detector": (94.1, 86.7, 5.3),
    "w/o contrastive loss": (95.8, 81.2, 8.5),
    "w/o both": (94.3, 82.3, 7.8),
    "ERM": (95.8, 77.9, 10.6),
}

#: Supplementary Table C, contrastive-loss ablation in the full-label setting.
TABLE_C_CELEBA = {
    "FairNet-Full": (95.9, 88.2, 3.8),
    "w/o contrastive loss": (95.9, 81.7, 8.2),
    "ERM": (95.8, 77.9, 10.6),
}

#: Supplementary Table 5: FairNet-Partial against the labelled fraction on
#: CelebA. Values are ``(TPR, FPR, ACC, WGA, EOD)`` in percent.
TABLE5_CELEBA = {
    0.001: (76.0, 6.95, 95.7, 81.9, 7.3),
    0.005: (86.7, 7.73, 95.7, 83.5, 7.0),
    0.01: (89.5, 8.03, 95.7, 84.7, 6.5),
    0.05: (94.0, 8.11, 95.8, 85.0, 6.6),
    0.10: (93.9, 7.04, 95.8, 85.5, 6.4),
    0.50: (95.0, 4.55, 95.9, 85.7, 6.3),
}

#: Supplementary Table I: activation-threshold sweep on CelebA. Values are
#: ``(TPR, FPR, ACC, WGA, EOD)`` in percent.
TABLE_I_CELEBA = {
    0.0: (100.0, 100.0, 94.1, 87.1, 4.8),
    0.2: (98.8, 18.7, 95.4, 86.9, 5.1),
    0.4: (96.7, 6.34, 95.6, 86.5, 5.4),
    0.5: (94.1, 3.45, 95.9, 86.2, 5.8),
    0.6: (72.3, 2.60, 95.9, 85.7, 6.1),
    0.8: (62.9, 1.48, 96.0, 82.1, 7.4),
    1.0: (0.0, 0.0, 95.8, 77.9, 10.6),
}

#: Supplementary Table B: FairNet-Unlabeled detector quality on CelebA.
TABLE_B_CELEBA = {
    "FairNet-Unlabeled": (74.2, 7.71, 95.8, 81.9, 7.5),
    "ERM": (None, None, 95.8, 77.9, 10.6),
}

#: Maps a reproduction config name to its paper row in Table 1 / Table 3.
CONFIG_TO_PAPER_ROW = {
    "celeba_erm": ("TABLE1_CELEBA", "ERM"),
    "celeba_erm_converged": ("TABLE1_CELEBA", "ERM"),
    "celeba_full_converged": ("TABLE1_CELEBA", "FairNet-Full"),
    "celeba_full": ("TABLE1_CELEBA", "FairNet-Full"),
    "celeba_partial": ("TABLE1_CELEBA", "FairNet-Partial"),
    "celeba_unlabeled": ("TABLE1_CELEBA", "FairNet-Unlabeled"),
    "celeba_ablate_detector": ("TABLE3_CELEBA", "w/o detector"),
    "celeba_ablate_contrastive": ("TABLE3_CELEBA", "w/o contrastive loss"),
    "celeba_ablate_both": ("TABLE3_CELEBA", "w/o both"),
    "celeba_full_ablate_contrastive": ("TABLE_C_CELEBA", "w/o contrastive loss"),
    "multinli_erm": ("TABLE1_MULTINLI", "ERM"),
    "multinli_full": ("TABLE1_MULTINLI", "FairNet-Full"),
    "multinli_partial": ("TABLE1_MULTINLI", "FairNet-Partial"),
    "multinli_unlabeled": ("TABLE1_MULTINLI", "FairNet-Unlabeled"),
}

TABLES = {
    "TABLE1_CELEBA": TABLE1_CELEBA,
    "TABLE1_MULTINLI": TABLE1_MULTINLI,
    "TABLE3_CELEBA": TABLE3_CELEBA,
    "TABLE_C_CELEBA": TABLE_C_CELEBA,
}


def paper_row(config_name: str):
    """Return the paper's ``(ACC, WGA, EOD)`` for a config, or ``None``."""

    entry = CONFIG_TO_PAPER_ROW.get(config_name)
    if entry is None:
        return None
    table, row = entry
    return TABLES[table][row]
