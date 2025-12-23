"""
FairNet Datasets

Contains dataset classes and data loading utilities for:
- CelebA (face attributes)
- UTKFace (age, gender, race)
- MultiNLI (natural language inference)
- HateXplain (hate speech detection)
- CivilComments (toxicity detection)
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset


class CelebADataset(Dataset):
    """
    CelebA dataset for face attribute prediction.
    
    Paper experiments use:
    - Target attribute: Attractive (idx 2)
    - Sensitive attribute: Male (idx 20)
    
    Args:
        root: Root directory containing CelebA data
        split: 'train', 'val', or 'test'
        transform: Image transforms
        target_attr: Target attribute index
        sensitive_attr: Sensitive attribute index
    """
    
    # CelebA attribute names (40 attributes)
    ATTR_NAMES = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
        'Wearing_Necktie', 'Young'
    ]
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_attr: int = 2,  # Attractive
        sensitive_attr: int = 20  # Male
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_attr = target_attr
        self.sensitive_attr = sensitive_attr
        
        # Load split file
        split_file = self.root / 'list_eval_partition.txt'
        attr_file = self.root / 'list_attr_celeba.txt'
        
        # Parse split file
        self.images = []
        self.attributes = []
        
        split_map = {'train': 0, 'val': 1, 'test': 2}
        split_idx = split_map[split]
        
        # Read attributes
        with open(attr_file, 'r') as f:
            num_images = int(f.readline().strip())
            attr_names = f.readline().strip().split()
            
            for line in f:
                parts = line.strip().split()
                img_name = parts[0]
                attrs = [int(x) for x in parts[1:]]
                # Convert from {-1, 1} to {0, 1}
                attrs = [(a + 1) // 2 for a in attrs]
                self.images.append(img_name)
                self.attributes.append(attrs)
        
        # Filter by split
        with open(split_file, 'r') as f:
            splits = {}
            for line in f:
                parts = line.strip().split()
                splits[parts[0]] = int(parts[1])
        
        filtered_images = []
        filtered_attrs = []
        for img, attrs in zip(self.images, self.attributes):
            if splits.get(img, -1) == split_idx:
                filtered_images.append(img)
                filtered_attrs.append(attrs)
        
        self.images = filtered_images
        self.attributes = torch.tensor(filtered_attrs)
        
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
        img_path = self.root / 'img_align_celeba' / self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
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
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        age_threshold: int = 30,
        sensitive_attr: str = 'gender'
    ):
        self.root = Path(root)
        self.transform = transform
        self.age_threshold = age_threshold
        self.sensitive_attr = sensitive_attr
        
        self.images = []
        self.labels = []
        self.sensitive = []
        
        # Parse filenames
        for img_path in self.root.glob('*.jpg'):
            parts = img_path.stem.split('_')
            if len(parts) >= 3:
                try:
                    age = int(parts[0])
                    gender = int(parts[1])
                    race = int(parts[2])
                    
                    self.images.append(img_path)
                    self.labels.append(1 if age > age_threshold else 0)
                    self.sensitive.append(gender if sensitive_attr == 'gender' else race)
                except ValueError:
                    continue
        
        self.labels = torch.tensor(self.labels)
        self.sensitive = torch.tensor(self.sensitive)
        
        print(f"UTKFace: {len(self.images)} images")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert('RGB')
        
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
        seed: int = 42
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.image_size = image_size
        
        # Generate groups
        num_minority = int(num_samples * minority_ratio)
        num_majority = num_samples - num_minority
        
        # Majority group (s=0)
        # Clear signal: color correlates perfectly with label
        maj_labels = np.random.randint(0, 2, num_majority)
        maj_sensitive = np.zeros(num_majority)
        
        # Minority group (s=1)
        # Biased: color is spuriously correlated but not perfectly
        min_labels = np.random.randint(0, 2, num_minority)
        min_sensitive = np.ones(num_minority)
        
        self.labels = torch.tensor(np.concatenate([maj_labels, min_labels]))
        self.sensitive = torch.tensor(np.concatenate([maj_sensitive, min_sensitive]))
        
        # Generate images
        self.images = []
        for i in range(num_samples):
            label = self.labels[i].item()
            sensitive = self.sensitive[i].item()
            
            # Base pattern based on true label
            if label == 0:
                base_color = np.array([0.3, 0.3, 0.7])  # Blue-ish
            else:
                base_color = np.array([0.7, 0.3, 0.3])  # Red-ish
            
            # Add spurious feature for majority
            if sensitive == 0:
                # Majority: add clear texture/pattern
                noise = np.random.randn(image_size, image_size, 3) * 0.1
            else:
                # Minority: flip colors with probability (bias_strength)
                if np.random.rand() < bias_strength:
                    base_color = 1 - base_color  # Flip
                noise = np.random.randn(image_size, image_size, 3) * 0.2
            
            image = np.clip(base_color + noise, 0, 1)
            self.images.append(image.astype(np.float32))
        
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
        image = torch.tensor(self.images[idx]).permute(2, 0, 1)
        return image, self.attributes[idx]


def create_celeba_loaders(
    root: str,
    batch_size: int = 128,
    image_size: int = 64,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create CelebA data loaders with standard transforms.
    
    Args:
        root: CelebA root directory
        batch_size: Batch size
        image_size: Image resize size
        num_workers: Number of data loader workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = CelebADataset(root, split='train', transform=train_transform)
    val_dataset = CelebADataset(root, split='val', transform=eval_transform)
    test_dataset = CelebADataset(root, split='test', transform=eval_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def create_synthetic_loaders(
    batch_size: int = 128,
    num_train: int = 10000,
    num_val: int = 2000,
    num_test: int = 2000,
    minority_ratio: float = 0.1,
    bias_strength: float = 0.9
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create synthetic biased data loaders for testing.
    
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
    train_dataset = SyntheticBiasedDataset(
        num_samples=num_train,
        minority_ratio=minority_ratio,
        bias_strength=bias_strength,
        seed=42
    )
    
    val_dataset = SyntheticBiasedDataset(
        num_samples=num_val,
        minority_ratio=minority_ratio,
        bias_strength=bias_strength,
        seed=43
    )
    
    test_dataset = SyntheticBiasedDataset(
        num_samples=num_test,
        minority_ratio=minority_ratio,
        bias_strength=0.5,  # Test with moderate bias
        seed=44
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader
