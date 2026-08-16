"""Public API for the FairNet research implementation."""

from .config import AttributeMode, BERTFairNetConfig, FairNetConfig, ViTFairNetConfig
from .datasets import (
    CelebADataset,
    SyntheticBiasedDataset,
    UTKFaceDataset,
    create_celeba_loaders,
    create_synthetic_loaders,
)
from .models import FairNetBERT, FairNetViT
from .modules import (
    AttentionPooling,
    BiasDetector,
    LoRAInjector,
    LoRALinear,
    StaticPrototypeBank,
    TripletContrastiveLoss,
    UnsupervisedBiasDetector,
)
from .trainers import FairNetPartialTrainer, FairNetTrainer, FairNetUnlabeledTrainer
from .utils import (
    compute_class_weights,
    compute_fairness_metrics,
    evaluate_model,
    get_group_indices,
    load_checkpoint,
    print_metrics,
    save_checkpoint,
    seed_everything,
)

__version__ = "1.1.0"
__author__ = "Songqi Zhou, Zeyuan Liu, and Benben Jiang"

__all__ = [
    "AttentionPooling",
    "AttributeMode",
    "BERTFairNetConfig",
    "BiasDetector",
    "CelebADataset",
    "FairNetBERT",
    "FairNetConfig",
    "FairNetPartialTrainer",
    "FairNetTrainer",
    "FairNetUnlabeledTrainer",
    "FairNetViT",
    "LoRAInjector",
    "LoRALinear",
    "StaticPrototypeBank",
    "SyntheticBiasedDataset",
    "TripletContrastiveLoss",
    "UTKFaceDataset",
    "UnsupervisedBiasDetector",
    "ViTFairNetConfig",
    "compute_class_weights",
    "compute_fairness_metrics",
    "create_celeba_loaders",
    "create_synthetic_loaders",
    "evaluate_model",
    "get_group_indices",
    "load_checkpoint",
    "print_metrics",
    "save_checkpoint",
    "seed_everything",
]
