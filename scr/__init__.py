"""
FairNet: Dynamic Fairness Correction without Performance Loss via Contrastive Conditional LoRA

Paper: arXiv:2510.19421v1 [cs.LG] 22 Oct 2025

This package implements FairNet with support for:
- Full: All sensitive attribute labels available
- Partial: Only k% of sensitive attribute labels available  
- Unlabeled: No sensitive attribute labels (uses unsupervised detection)
"""

from .config import FairNetConfig, AttributeMode, ViTFairNetConfig, BERTFairNetConfig
from .modules import (
    AttentionPooling,
    BiasDetector,
    LoRALinear,
    LoRAInjector,
    TripletContrastiveLoss,
    StaticPrototypeBank,
    UnsupervisedBiasDetector,
)
from .models import FairNetViT, FairNetBERT
from .trainers import (
    FairNetTrainer,
    FairNetPartialTrainer,
    FairNetUnlabeledTrainer,
)
from .utils import (
    seed_everything, 
    evaluate_model, 
    print_metrics,
    save_checkpoint,
    load_checkpoint,
    compute_class_weights,
    get_group_indices,
)
from .datasets import (
    CelebADataset,
    UTKFaceDataset,
    SyntheticBiasedDataset,
    create_celeba_loaders,
    create_synthetic_loaders,
)

__version__ = "1.0.0"
__author__ = "FairNet Implementation"

__all__ = [
    # Config
    "FairNetConfig",
    "AttributeMode",
    "ViTFairNetConfig",
    "BERTFairNetConfig",
    # Modules
    "AttentionPooling",
    "BiasDetector", 
    "LoRALinear",
    "LoRAInjector",
    "TripletContrastiveLoss",
    "StaticPrototypeBank",
    "UnsupervisedBiasDetector",
    # Models
    "FairNetViT",
    "FairNetBERT",
    # Trainers
    "FairNetTrainer",
    "FairNetPartialTrainer",
    "FairNetUnlabeledTrainer",
    # Utils
    "seed_everything",
    "evaluate_model",
    "print_metrics",
    "save_checkpoint",
    "load_checkpoint",
    "compute_class_weights",
    "get_group_indices",
    # Datasets
    "CelebADataset",
    "UTKFaceDataset",
    "SyntheticBiasedDataset",
    "create_celeba_loaders",
    "create_synthetic_loaders",
]
