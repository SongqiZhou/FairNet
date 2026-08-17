"""Materialise CelebA into the directory layout FairNet's loaders expect.

The paper (Supplementary C.1) uses the aligned CelebA images with the standard
train/validation/test partition shipped with the dataset. The original download
links from the dataset authors are Google Drive URLs that are frequently rate
limited, so this script rebuilds the exact same files from a Hugging Face mirror
that stores the aligned JPEGs together with their original ``image_id`` and the
40 ``{-1, 1}`` attribute columns.

The result is byte-for-byte the layout documented in the README::

    <out>/
    ├── img_align_celeba/000001.jpg ... 202599.jpg
    ├── list_attr_celeba.txt
    └── list_eval_partition.txt

``list_eval_partition.txt`` is regenerated from the official index boundaries
(1-162770 train, 162771-182637 validation, 182638-202599 test), which is exactly
what the file distributed with CelebA contains.

Usage::

    python scripts/prepare_celeba.py --out data/celeba
"""

from __future__ import annotations

import argparse
from pathlib import Path

ATTR_NAMES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]

# Official CelebA partition boundaries (inclusive, 1-indexed image numbers).
TRAIN_END = 162770
VAL_END = 182637
NUM_IMAGES = 202599

REPO_ID = "tpremoli/CelebA-attrs"


def _partition(image_id: str) -> int:
    index = int(Path(image_id).stem)
    if index <= TRAIN_END:
        return 0
    if index <= VAL_END:
        return 1
    return 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/celeba", help="Output root directory")
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help="Hugging Face dataset repository holding the aligned images",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for the raw parquet shards",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="How many times to re-fetch shards that arrive truncated",
    )
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
    from tqdm import tqdm

    out = Path(args.out)
    image_dir = out / "img_align_celeba"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading parquet shards from {args.repo_id} ...")
    local = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            cache_dir=args.cache_dir,
            allow_patterns=["data/*.parquet"],
            max_workers=4,
        )
    )

    shards = sorted((local / "data").glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards found under {local / 'data'}")

    # Large multi-shard snapshots occasionally land truncated. Verify every
    # footer and re-fetch the shards that failed rather than crashing halfway
    # through extraction.
    remote = {name for name in list_repo_files(args.repo_id, repo_type="dataset")}
    for attempt in range(1, args.max_retries + 1):
        broken = []
        for shard in tqdm(shards, desc=f"Verifying shards (attempt {attempt})"):
            try:
                pq.ParquetFile(shard).metadata
            except Exception:
                broken.append(shard)
        if not broken:
            break
        print(f"{len(broken)} shard(s) are incomplete; re-downloading")
        for shard in broken:
            relative = f"data/{shard.name}"
            if relative not in remote:
                raise SystemExit(f"{relative} is not present in {args.repo_id}")
            shard.unlink(missing_ok=True)
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=relative,
                cache_dir=args.cache_dir,
                force_download=True,
            )
    else:
        raise SystemExit(f"Shards still incomplete after {args.max_retries} attempts")

    rows: dict[str, list[int]] = {}
    for shard in tqdm(shards, desc="Extracting shards"):
        parquet = pq.ParquetFile(shard)
        columns = parquet.schema_arrow.names
        missing = [name for name in ATTR_NAMES if name not in columns]
        if missing:
            raise SystemExit(f"Shard {shard.name} is missing attributes: {missing[:5]}")
        # Read row group by row group: a full shard of decoded JPEG bytes does
        # not need to be resident all at once.
        for group in range(parquet.num_row_groups):
            data = parquet.read_row_group(group).to_pydict()
            identifiers = (
                data["image_id"]
                if "image_id" in data
                else [record["path"] for record in data["image"]]
            )
            for position, raw_id in enumerate(identifiers):
                if raw_id is None:
                    raise SystemExit(f"Shard {shard.name} row {position} has no image identifier")
                image_id = f"{int(Path(raw_id).stem):06d}.jpg"
                target = image_dir / image_id
                if not target.exists():
                    target.write_bytes(data["image"][position]["bytes"])
                values = [int(data[name][position]) for name in ATTR_NAMES]
                if any(value not in (-1, 1) for value in values):
                    raise SystemExit(f"{image_id} has attributes outside {{-1, 1}}")
                rows[image_id] = values

    if len(rows) != NUM_IMAGES:
        raise SystemExit(f"Expected {NUM_IMAGES} images, materialised {len(rows)}")

    ordered = sorted(rows)
    attr_path = out / "list_attr_celeba.txt"
    with attr_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(ordered)}\n")
        handle.write(" ".join(ATTR_NAMES) + "\n")
        for image_id in ordered:
            values = " ".join(f"{value:>2d}" for value in rows[image_id])
            handle.write(f"{image_id} {values}\n")

    split_path = out / "list_eval_partition.txt"
    with split_path.open("w", encoding="utf-8") as handle:
        for image_id in ordered:
            handle.write(f"{image_id} {_partition(image_id)}\n")

    counts = [0, 0, 0]
    for image_id in ordered:
        counts[_partition(image_id)] += 1
    print(f"\nWrote {attr_path} and {split_path}")
    print(f"train={counts[0]} val={counts[1]} test={counts[2]} images={len(ordered)}")


if __name__ == "__main__":
    main()
