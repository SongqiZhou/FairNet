"""Text dataset pipelines for the paper's language benchmarks.

Supplementary C.1 specifies two language datasets:

* **MultiNLI** - three-way natural language inference, with the presence of a
  negation cue in the hypothesis as the sensitive attribute.
* **HateXplain** - three-way hate speech classification, with race (African
  American) and gender (Female) target communities as sensitive attributes.

Both loaders emit the mapping batch contract documented in the README:
``input_ids``, ``attention_mask``, ``token_type_ids``, ``labels``, and
``sensitive``. ``sensitive`` is ``[batch, num_attributes]`` and its column order
matches ``FairNetConfig.sensitive_attributes``.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# Supplementary C.1 describes the MultiNLI sensitive attribute as "presence of
# negation cues (e.g., 'not', 'n't', 'never') in the hypothesis".
PAPER_NEGATION_WORDS = ("not", "n't", "never")

# The negation list introduced by Gururangan et al. and used by Sagawa et al.'s
# GroupDRO release, kept available so results can be compared against that line
# of work on identical groups.
GROUPDRO_NEGATION_WORDS = ("nobody", "no", "never", "nothing")

_TOKEN_PATTERN = re.compile(r"[a-z']+")


def has_negation(text: str, negation_words: Sequence[str] = PAPER_NEGATION_WORDS) -> int:
    """Return 1 when any negation cue occurs in ``text``.

    Matching is done on lowercased word tokens so that ``"n't"`` fires on
    contractions such as ``"isn't"`` without also firing on unrelated words that
    merely contain the substring.
    """

    lowered = text.lower()
    tokens = set(_TOKEN_PATTERN.findall(lowered))
    for word in negation_words:
        word = word.lower()
        if word.startswith("'") or word.startswith("n'"):
            # Contraction suffixes are matched against the raw string because
            # tokenisation splits them inconsistently across corpora.
            if word in lowered:
                return 1
        elif word in tokens:
            return 1
    return 0


class TokenizedTextDataset(Dataset):
    """Pre-tokenised text dataset producing FairNet's mapping batches.

    Args:
        texts: Either a list of strings or a list of ``(first, second)`` pairs.
        labels: Integer task labels.
        sensitive: ``[num_samples, num_attributes]`` binary group labels.
        tokenizer: A Hugging Face tokenizer.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        texts: Sequence,
        labels: Sequence[int],
        sensitive: Sequence[Sequence[int]],
        tokenizer,
        max_length: int = 128,
    ):
        if not (len(texts) == len(labels) == len(sensitive)):
            raise ValueError("texts, labels, and sensitive must have equal length")
        if len(texts) == 0:
            raise ValueError("Cannot build an empty text dataset")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self.labels = torch.tensor(list(labels), dtype=torch.long)
        self.sensitive = torch.tensor(
            [list(row) for row in sensitive],
            dtype=torch.long,
        )
        if self.sensitive.ndim != 2:
            raise ValueError("sensitive must be a two-dimensional [samples, attributes] array")
        if not torch.all((self.sensitive == 0) | (self.sensitive == 1)):
            raise ValueError("sensitive attributes must be binary and encoded as 0/1")

        first = [pair[0] if isinstance(pair, (tuple, list)) else pair for pair in texts]
        second = (
            [pair[1] for pair in texts]
            if texts and isinstance(texts[0], (tuple, list))
            else None
        )
        encoded = tokenizer(
            first,
            second,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.encoded = {key: value for key, value in encoded.items()}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
        item["sensitive"] = self.sensitive[index]
        return item

    def group_counts(self) -> dict:
        """Return per ``(label, group)`` counts for the first sensitive column."""

        counts = {}
        primary = self.sensitive[:, 0]
        for label in self.labels.unique().tolist():
            for group in (0, 1):
                mask = (self.labels == label) & (primary == group)
                counts[(int(label), int(group))] = int(mask.sum().item())
        return counts


def _build_loaders(
    datasets: Sequence[Dataset],
    batch_size: int,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train, validation, test = datasets
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        validation, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, validation_loader, test_loader


def create_multinli_loaders(
    tokenizer=None,
    model_name: str = "bert-base-uncased",
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 2,
    seed: int = 42,
    negation_words: Sequence[str] = PAPER_NEGATION_WORDS,
    split_scheme: str = "matched",
    cache_dir: Optional[str] = None,
    max_train_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build MultiNLI loaders with negation as the sensitive attribute.

    Args:
        split_scheme: ``"matched"`` uses the official ``train`` split for
            training and deterministically halves ``validation_matched`` into
            validation and test. This is the closest public approximation to
            Supplementary C.1's "train, validation-matched, and test-matched"
            because MultiNLI's test-matched labels are held out by the
            benchmark. ``"groupdro"`` instead reproduces the 50/20/30 random
            split of Sagawa et al., which several baselines in Table 1 were
            originally evaluated under.
        negation_words: Cue list defining the sensitive attribute. Defaults to
            the paper's list; :data:`GROUPDRO_NEGATION_WORDS` is also provided.

    Returns:
        train_loader, validation_loader, test_loader
    """

    if split_scheme not in {"matched", "groupdro"}:
        raise ValueError("split_scheme must be 'matched' or 'groupdro'")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    raw = load_dataset("nyu-mll/multi_nli", cache_dir=cache_dir)

    def to_columns(rows):
        premises = rows["premise"]
        hypotheses = rows["hypothesis"]
        labels = rows["label"]
        texts = list(zip(premises, hypotheses))
        sensitive = [[has_negation(text, negation_words)] for text in hypotheses]
        return texts, labels, sensitive

    if split_scheme == "matched":
        train_texts, train_labels, train_sensitive = to_columns(raw["train"][:])
        held_texts, held_labels, held_sensitive = to_columns(raw["validation_matched"][:])
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(held_labels))
        midpoint = len(order) // 2
        val_index, test_index = order[:midpoint], order[midpoint:]
        splits = [
            (train_texts, train_labels, train_sensitive),
            (
                [held_texts[i] for i in val_index],
                [held_labels[i] for i in val_index],
                [held_sensitive[i] for i in val_index],
            ),
            (
                [held_texts[i] for i in test_index],
                [held_labels[i] for i in test_index],
                [held_sensitive[i] for i in test_index],
            ),
        ]
    else:
        texts, labels, sensitive = to_columns(raw["train"][:])
        extra = to_columns(raw["validation_matched"][:])
        texts = texts + extra[0]
        labels = list(labels) + list(extra[1])
        sensitive = sensitive + extra[2]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(labels))
        first = int(0.5 * len(order))
        second = int(0.7 * len(order))
        chunks = [order[:first], order[first:second], order[second:]]
        splits = [
            (
                [texts[i] for i in chunk],
                [labels[i] for i in chunk],
                [sensitive[i] for i in chunk],
            )
            for chunk in chunks
        ]

    if max_train_samples is not None:
        texts, labels, sensitive = splits[0]
        splits[0] = (
            texts[:max_train_samples],
            labels[:max_train_samples],
            sensitive[:max_train_samples],
        )

    datasets = [
        TokenizedTextDataset(texts, labels, sensitive, tokenizer, max_length)
        for texts, labels, sensitive in splits
    ]
    for name, dataset in zip(("train", "validation", "test"), datasets):
        print(f"MultiNLI {name}: {len(dataset)} examples, groups {dataset.group_counts()}")
    return _build_loaders(datasets, batch_size, num_workers, seed)


# HateXplain annotator target communities mapped onto the paper's two axes.
AFRICAN_TARGETS = ("African",)
FEMALE_TARGETS = ("Women",)

# The Hugging Face copy of HateXplain is a loading script, which `datasets` 5
# refuses to execute, so the canonical release files are read directly.
HATEXPLAIN_DATA_URL = (
    "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
)
HATEXPLAIN_SPLIT_URL = (
    "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"
)

#: Label encoding used by the Hugging Face mirror, kept for comparability.
HATEXPLAIN_LABELS = {"hatespeech": 0, "normal": 1, "offensive": 2}


def _download_json(url: str, cache_dir: Optional[str], filename: str):
    import json
    import urllib.request
    from pathlib import Path

    root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "fairnet"
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    if not target.exists():
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            target.write_bytes(response.read())
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_hatexplain_loaders(
    tokenizer=None,
    model_name: str = "bert-base-uncased",
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 2,
    seed: int = 42,
    sensitive_axes: Sequence[str] = ("race", "gender"),
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build HateXplain loaders for the intersectional experiment of Table 2.

    The task label is the majority vote over the three annotator labels
    (0 hate speech, 1 normal, 2 offensive in the upstream encoding). A sensitive
    attribute is 1 when a majority of annotators marked the corresponding target
    community, matching the "African American vs. Other" and "Female vs. Male"
    axes of Table 2.

    Args:
        sensitive_axes: Ordered subset of ``("race", "gender")``. The column
            order must match ``FairNetConfig.sensitive_attributes``.
    """

    axes = tuple(sensitive_axes)
    unknown = set(axes) - {"race", "gender"}
    if unknown:
        raise ValueError(f"Unknown sensitive axes: {sorted(unknown)}")
    if not axes:
        raise ValueError("At least one sensitive axis is required")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    from collections import Counter

    from transformers import AutoTokenizer

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    posts = _download_json(HATEXPLAIN_DATA_URL, cache_dir, "hatexplain_dataset.json")
    divisions = _download_json(HATEXPLAIN_SPLIT_URL, cache_dir, "hatexplain_divisions.json")

    target_lists = {"race": AFRICAN_TARGETS, "gender": FEMALE_TARGETS}

    def to_columns(post_ids):
        texts, labels, sensitive = [], [], []
        for post_id in post_ids:
            record = posts.get(post_id)
            if record is None:
                continue
            annotators = record["annotators"]
            annotator_labels = [entry["label"] for entry in annotators]
            if not annotator_labels:
                continue
            majority_label, _ = Counter(annotator_labels).most_common(1)[0]
            targets = [list(entry.get("target", [])) for entry in annotators]
            row = []
            for axis in axes:
                wanted = target_lists[axis]
                votes = sum(
                    1
                    for annotator_targets in targets
                    if any(target in annotator_targets for target in wanted)
                )
                row.append(int(votes * 2 > len(targets)) if targets else 0)
            texts.append(" ".join(record["post_tokens"]))
            labels.append(HATEXPLAIN_LABELS[majority_label])
            sensitive.append(row)
        return texts, labels, sensitive

    splits = [to_columns(divisions[name]) for name in ("train", "val", "test")]
    datasets = [
        TokenizedTextDataset(texts, labels, sensitive, tokenizer, max_length)
        for texts, labels, sensitive in splits
    ]
    for name, dataset in zip(("train", "validation", "test"), datasets):
        print(f"HateXplain {name}: {len(dataset)} examples, groups {dataset.group_counts()}")
    return _build_loaders(datasets, batch_size, num_workers, seed)


__all__ = [
    "AFRICAN_TARGETS",
    "FEMALE_TARGETS",
    "GROUPDRO_NEGATION_WORDS",
    "PAPER_NEGATION_WORDS",
    "TokenizedTextDataset",
    "create_hatexplain_loaders",
    "create_multinli_loaders",
    "has_negation",
]
