"""
FairNet Datasets

Contains vision dataset classes and data loading utilities for:
- CelebA (face attributes)
- UTKFace (age, gender, race)
- A deterministic synthetic bias benchmark
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CelebADataset(Dataset):
    """
    CelebA dataset for face attribute prediction.

    Paper experiments use:
    - Target attribute: Male (idx 20)
    - Sensitive attribute: Blond Hair (idx 9)

    Args:
        root: Root directory containing CelebA data
        split: 'train', 'val', or 'test'
        transform: Image transforms
        target_attr: Target attribute index
        sensitive_attr: Sensitive attribute index
    """

    # CelebA attribute names (40 attributes)
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

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_attr: int = 20,  # Male
        sensitive_attr: int = 9,  # Blond Hair
    ):
        if not 0 <= target_attr < len(self.ATTR_NAMES):
            raise ValueError("target_attr must be a valid CelebA attribute index")
        if not 0 <= sensitive_attr < len(self.ATTR_NAMES):
            raise ValueError("sensitive_attr must be a valid CelebA attribute index")
        if target_attr == sensitive_attr:
            raise ValueError("target_attr and sensitive_attr must differ")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_attr = target_attr
        self.sensitive_attr = sensitive_attr

        # Load split file
        split_file = self.root / "list_eval_partition.txt"
        attr_file = self.root / "list_attr_celeba.txt"

        # Parse split file
        self.images = []
        self.attributes = []

        split_map = {"train": 0, "val": 1, "test": 2}
        if split not in split_map:
            raise ValueError("split must be 'train', 'val', or 'test'")
        split_idx = split_map[split]

        # Read attributes
        with open(attr_file, "r", encoding="utf-8") as f:
            num_images = int(f.readline().strip())
            attr_names = f.readline().strip().split()
            if attr_names != self.ATTR_NAMES:
                raise ValueError("Unexpected CelebA attribute order")

            for line in f:
                parts = line.strip().split()
                if len(parts) != len(self.ATTR_NAMES) + 1:
                    raise ValueError(f"Malformed CelebA attribute row: {line!r}")
                img_name = parts[0]
                attrs = [int(x) for x in parts[1:]]
                if any(value not in {-1, 1} for value in attrs):
                    raise ValueError("CelebA attributes must be encoded as -1 or 1")
                # Convert from {-1, 1} to {0, 1}
                attrs = [(a + 1) // 2 for a in attrs]
                self.images.append(img_name)
                self.attributes.append(attrs)

        # Filter by split
        if len(self.images) != num_images:
            raise ValueError(
                f"CelebA metadata declares {num_images} images but contains {len(self.images)}"
            )

        with open(split_file, "r", encoding="utf-8") as f:
            splits = {}
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2 or int(parts[1]) not in {0, 1, 2}:
                    raise ValueError(f"Malformed CelebA split row: {line!r}")
                splits[parts[0]] = int(parts[1])
        missing_splits = set(self.images) - set(splits)
        if missing_splits:
            raise ValueError(f"CelebA split metadata is missing {len(missing_splits)} images")

        filtered_images = []
        filtered_attrs = []
        for img, attrs in zip(self.images, self.attributes):
            if splits.get(img, -1) == split_idx:
                filtered_images.append(img)
                filtered_attrs.append(attrs)

        self.images = filtered_images
        if not self.images:
            raise ValueError(f"CelebA split {split!r} contains no images")
        self.attributes = torch.tensor(filtered_attrs, dtype=torch.long)

        print(f"CelebA {split}: {len(self.images)} images")
        self._print_group_stats()

    def _print_group_stats(self):
        """Print group distribution."""
        labels = self.attributes[:, self.target_attr]
        sensitive = self.attributes[:, self.sensitive_attr]

        for s in [0, 1]:
            for y in [0, 1]:
                count = ((sensitive == s) & (labels == y)).sum().item()
                pct = 100 * count / len(self.images)
                print(f"  Group (s={s}, y={y}): {count} ({pct:.1f}%)")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.root / "img_align_celeba" / self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.attributes[idx]


class UTKFaceDataset(Dataset):
    """
    UTKFace dataset for age estimation with demographic attributes.

    Filename format: [age]_[gender]_[race]_[date&time].jpg
    - Age: 0-116
    - Gender: 0 (male), 1 (female)
    - Race: 0-4 (White, Black, Asian, Indian, Others)

    Args:
        root: Root directory containing UTKFace images
        transform: Image transforms
        age_threshold: Age threshold for binary classification
        sensitive_attr: 'gender' or 'race'
        race_reference: Race encoded as group 0; every other race is group 1
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        age_threshold: int = 30,
        sensitive_attr: str = "gender",
        race_reference: int = 0,
    ):
        if sensitive_attr not in {"gender", "race"}:
            raise ValueError("sensitive_attr must be 'gender' or 'race'")
        if not 0 <= race_reference <= 4:
            raise ValueError("race_reference must be between 0 and 4")
        self.root = Path(root)
        self.transform = transform
        self.age_threshold = age_threshold
        self.sensitive_attr = sensitive_attr

        self.images = []
        self.labels = []
        self.sensitive = []

        # Parse filenames
        for img_path in self.root.glob("*.jpg"):
            parts = img_path.stem.split("_")
            if len(parts) >= 3:
                try:
                    age = int(parts[0])
                    gender = int(parts[1])
                    race = int(parts[2])

                    self.images.append(img_path)
                    self.labels.append(1 if age > age_threshold else 0)
                    self.sensitive.append(
                        gender if sensitive_attr == "gender" else int(race != race_reference)
                    )
                except ValueError:
                    continue

        self.labels = torch.tensor(self.labels)
        self.sensitive = torch.tensor(self.sensitive)

        if not self.images:
            raise ValueError(f"No valid UTKFace images found in {self.root}")

        print(f"UTKFace: {len(self.images)} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return combined attributes tensor
        attributes = torch.zeros(40, dtype=torch.long)
        attributes[20] = self.labels[idx]  # Target at idx 20
        attributes[9] = self.sensitive[idx]  # Sensitive at idx 9

        return image, attributes


class SyntheticBiasedDataset(Dataset):
    """
    Synthetic dataset for testing FairNet.

    Creates artificial data with controllable bias:
    - Majority group: high accuracy possible
    - Minority group: spurious correlations

    Args:
        num_samples: Total number of samples
        minority_ratio: Fraction of minority samples
        bias_strength: How strong the spurious correlation is
        image_size: Size of generated images
        seed: Random seed
    """

    def __init__(
        self,
        num_samples: int = 10000,
        minority_ratio: float = 0.1,
        bias_strength: float = 0.9,
        image_size: int = 64,
        seed: int = 42,
    ):
        if not 0 < minority_ratio < 1:
            raise ValueError("minority_ratio must be in (0, 1)")
        if not 0 <= bias_strength <= 1:
            raise ValueError("bias_strength must be in [0, 1]")
        if num_samples < 2 or image_size <= 0:
            raise ValueError("num_samples must be at least two and image_size positive")

        rng = np.random.default_rng(seed)

        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed
        self.bias_strength = bias_strength

        # Generate groups
        num_minority = min(max(round(num_samples * minority_ratio), 1), num_samples - 1)
        num_majority = num_samples - num_minority

        # Majority group (s=0)
        # Clear signal: color correlates perfectly with label
        maj_labels = rng.integers(0, 2, num_majority)
        maj_sensitive = np.zeros(num_majority)

        # Minority group (s=1)
        # Biased: color is spuriously correlated but not perfectly
        min_labels = rng.integers(0, 2, num_minority)
        min_sensitive = np.ones(num_minority)

        labels = np.concatenate([maj_labels, min_labels])
        sensitive = np.concatenate([maj_sensitive, min_sensitive])
        permutation = rng.permutation(num_samples)
        self.labels = torch.tensor(labels[permutation], dtype=torch.long)
        self.sensitive = torch.tensor(sensitive[permutation], dtype=torch.long)

        # Create attributes tensor
        self.attributes = torch.zeros(num_samples, 40, dtype=torch.long)
        self.attributes[:, 20] = self.labels  # Target
        self.attributes[:, 9] = self.sensitive.long()  # Sensitive

        print(f"Synthetic Dataset: {num_samples} samples")
        print(f"  Minority ratio: {minority_ratio:.1%}")
        print(f"  Bias strength: {bias_strength:.1%}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate each image deterministically on demand to avoid retaining
        # hundreds of MB for the default synthetic training set.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, idx]))
        label = self.labels[idx].item()
        sensitive = self.sensitive[idx].item()
        base_color = np.array([0.3, 0.3, 0.7]) if label == 0 else np.array([0.7, 0.3, 0.3])
        noise_scale = 0.1 if sensitive == 0 else 0.2
        if sensitive == 1 and rng.random() < self.bias_strength:
            base_color = 1 - base_color
        image = np.clip(
            base_color + rng.normal(0, noise_scale, (self.image_size, self.image_size, 3)),
            0,
            1,
        ).astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, self.attributes[idx]


def create_celeba_loaders(
    root: str,
    batch_size: int = 128,
    image_size: int = 64,
    num_workers: int = 4,
    target_attr: int = 20,
    sensitive_attr: int = 9,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create CelebA data loaders with standard transforms.

    Args:
        root: CelebA root directory
        batch_size: Batch size
        image_size: Image resize size
        num_workers: Number of data loader workers
        seed: DataLoader shuffling seed

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size <= 0 or image_size <= 0 or num_workers < 0:
        raise ValueError("batch_size and image_size must be positive and num_workers non-negative")
    from torchvision import transforms

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = CelebADataset(
        root,
        split="train",
        transform=train_transform,
        target_attr=target_attr,
        sensitive_attr=sensitive_attr,
    )
    val_dataset = CelebADataset(
        root,
        split="val",
        transform=eval_transform,
        target_attr=target_attr,
        sensitive_attr=sensitive_attr,
    )
    test_dataset = CelebADataset(
        root,
        split="test",
        transform=eval_transform,
        target_attr=target_attr,
        sensitive_attr=sensitive_attr,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def create_synthetic_loaders(
    batch_size: int = 128,
    num_train: int = 10000,
    num_val: int = 2000,
    num_test: int = 2000,
    minority_ratio: float = 0.1,
    bias_strength: float = 0.9,
    image_size: int = 64,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create synthetic biased data loaders for testing.

    Args:
        batch_size: Batch size
        num_train: Number of training samples
        num_val: Number of validation samples
        num_test: Number of test samples
        minority_ratio: Fraction of minority samples
        bias_strength: Bias strength (0=unbiased, 1=fully biased)

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    train_dataset = SyntheticBiasedDataset(
        num_samples=num_train,
        minority_ratio=minority_ratio,
        bias_strength=bias_strength,
        image_size=image_size,
        seed=seed,
    )

    val_dataset = SyntheticBiasedDataset(
        num_samples=num_val,
        minority_ratio=minority_ratio,
        bias_strength=bias_strength,
        image_size=image_size,
        seed=seed + 1,
    )

    test_dataset = SyntheticBiasedDataset(
        num_samples=num_test,
        minority_ratio=minority_ratio,
        bias_strength=0.5,  # Test with moderate bias
        image_size=image_size,
        seed=seed + 2,
    )

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
