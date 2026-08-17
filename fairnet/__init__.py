"""Public API for the FairNet research implementation."""

from .config import AttributeMode, BERTFairNetConfig, FairNetConfig, ViTFairNetConfig
from .datasets import (
    CelebADataset,
    SyntheticBiasedDataset,
    UTKFaceDataset,
    create_celeba_loaders,
    create_synthetic_loaders,
)
from .models import FairNetBERT, FairNetDistilBERT, FairNetViT
from .modules import (
    AttentionPooling,
    BiasDetector,
    LoRAInjector,
    LoRALinear,
    StaticPrototypeBank,
    TripletContrastiveLoss,
    UnsupervisedBiasDetector,
)
from .text_datasets import (
    GROUPDRO_NEGATION_WORDS,
    PAPER_NEGATION_WORDS,
    TokenizedTextDataset,
    create_hatexplain_loaders,
    create_multinli_loaders,
    has_negation,
)
from .trainers import FairNetPartialTrainer, FairNetTrainer, FairNetUnlabeledTrainer
from .utils import (
    WGA_LABEL_BY_GROUP,
    WGA_SENSITIVE_GROUP,
    compute_class_weights,
    compute_fairness_metrics,
    evaluate_detector,
    evaluate_model,
    get_group_indices,
    load_checkpoint,
    print_metrics,
    save_checkpoint,
    seed_everything,
    set_activation_threshold,
    summarize_metrics,
    sweep_activation_threshold,
)

__version__ = "1.1.0"
__author__ = "Songqi Zhou, Zeyuan Liu, and Benben Jiang"

__all__ = [
    "AttentionPooling",
    "WGA_LABEL_BY_GROUP",
    "WGA_SENSITIVE_GROUP",
    "AttributeMode",
    "BERTFairNetConfig",
    "BiasDetector",
    "CelebADataset",
    "GROUPDRO_NEGATION_WORDS",
    "PAPER_NEGATION_WORDS",
    "TokenizedTextDataset",
    "FairNetBERT",
    "FairNetDistilBERT",
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
    "create_hatexplain_loaders",
    "create_multinli_loaders",
    "create_synthetic_loaders",
    "evaluate_detector",
    "evaluate_model",
    "get_group_indices",
    "has_negation",
    "load_checkpoint",
    "print_metrics",
    "save_checkpoint",
    "seed_everything",
    "set_activation_threshold",
    "summarize_metrics",
    "sweep_activation_threshold",
]
