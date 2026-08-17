"""
FairNet Configuration Module

Contains all configuration classes for FairNet training and inference.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AttributeMode(Enum):
    """Training mode based on sensitive attribute availability."""

    FULL = "full"  # All sensitive labels available
    PARTIAL = "partial"  # Only k% of sensitive labels available
    UNLABELED = "unlabeled"  # No sensitive labels (unsupervised detection)


@dataclass
class FairNetConfig:
    """
    Configuration for FairNet model and training.

    Based on paper Section 3 and Supplementary Material C.

    Attributes:
        hidden_dim: Hidden dimension of the backbone model
        lora_rank: Rank r for LoRA decomposition (paper uses r=8)
        lora_alpha: Scaling factor α for LoRA (paper uses α=16)
        lora_dropout: Dropout rate for LoRA layers
        activation_threshold: Threshold τ for conditional LoRA activation (paper uses τ=0.5)
        lora_target_modules: Which attention modules to apply LoRA to
        lora_layers: Which layers to apply LoRA (None = last half)
        detector_hidden: Hidden dimension for bias detector MLP
        detector_num_layers: Number of layers in bias detector MLP
        stage1_epochs: Epochs for Stage 1 (base model training)
        stage2_epochs: Epochs for Stage 2 (bias detector training)
        stage4_epochs: Epochs for Stage 4 (LoRA training)
        stage1_lr: Learning rate for Stage 1
        stage2_lr: Learning rate for Stage 2
        stage4_lr: Learning rate for Stage 4
        weight_decay: Weight decay for AdamW optimizer
        contrastive_margin: Margin m for triplet contrastive loss (paper uses m=0.5)
        lambda_D: Weight for detector loss
        lambda_C: Weight for contrastive loss
        gradient_clip: Maximum gradient norm for clipping
        warmup_steps: Number of warmup steps for learning rate scheduler
        batch_size: Batch size for training
        image_size: Input image size (for vision models)
        attribute_mode: Training mode (FULL/PARTIAL/UNLABELED)
        labeled_fraction: Fraction of samples with sensitive labels (for PARTIAL mode)
        sensitive_attributes: List of sensitive attribute indices
        target_attribute: Target attribute index for classification
        device: Device to use for training
    """

    # Model architecture
    hidden_dim: int = 768

    # LoRA configuration (Section 3.3)
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    activation_threshold: float = 0.5
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    lora_layers: Optional[List[int]] = None

    # Bias detector configuration (Section 3.2)
    detector_hidden: int = 128
    detector_num_layers: int = 2
    detector_layer: int = 4  # Which layer to extract features for detector

    # Training configuration
    stage1_epochs: int = 10
    stage2_epochs: int = 5
    stage4_epochs: int = 10
    stage1_lr: float = 1e-5
    stage2_lr: float = 1e-4
    stage4_lr: float = 1e-4
    weight_decay: float = 0.01

    # Loss configuration (Equation 2)
    contrastive_margin: float = 0.5
    # Plain (not squared) Euclidean distance over L2-normalised representations
    # is the only reading under which Supplementary C.3.2's "Euclidean distance"
    # and its margin range [0.1, 1.0] are simultaneously true. Alternatives:
    # "squared_euclidean" and "cosine". See ``TripletContrastiveLoss``.
    contrastive_distance: str = "euclidean"
    contrastive_normalize: bool = True
    lambda_D: float = 1.0
    lambda_C: float = 1.0

    # Weight of Equation 3's L_task inside Stage 4. Stage 4 only ever sees
    # triggered (minority) samples, and on CelebA that subset is 94% one class,
    # so an unweighted task loss there pushes the correction further toward the
    # majority class of the minority group and makes the worst group worse. 0.0
    # follows Section 3.4's description of Stage 4 as "fine-tuning of LoRA
    # modules using a contrastive loss formulation"; set it above zero to carry
    # Equation 3's task term, in which case class_balanced_stage4_task is
    # strongly recommended.
    stage4_task_weight: float = 0.0
    class_balanced_stage4_task: bool = True

    # Weight each contrastive anchor by the inverse frequency of its task class
    # within the batch. A sensitive attribute has one shared (A, B) LoRA pair,
    # so anchors from the common class of the minority group would otherwise
    # dominate the update. On CelebA the minority group is 94% one class, and
    # without this the correction moves every triggered sample toward that
    # class and lowers worst-group accuracy below ERM.
    class_balanced_contrastive: bool = True

    # Stage 4 objective. "contrastive" is FairNet (Equation 2). "task" is the
    # "w/o contrastive loss" ablation of Section 5.4 and Supplementary D.3,
    # which trains the same LoRA matrices with the ordinary cross-entropy /
    # binary cross-entropy task loss on flagged instances instead.
    stage4_objective: str = "contrastive"

    # Optimization
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    batch_size: int = 128
    lr_schedule: str = "warmup_constant"  # or "warmup_cosine"
    drop_last: bool = False

    # Checkpoint selection. ``None`` follows the paper: FairNet-Full may select
    # on validation WGA because Supplementary C.3.1 grants it validation
    # sensitive labels, while Partial and Unlabeled must not (Section 5.1) and
    # therefore select on overall validation accuracy.
    model_selection_metric: Optional[str] = None

    # Data configuration
    image_size: int = 64
    max_seq_length: int = 128  # For NLP models

    # Attribute mode configuration (Section 5.1)
    attribute_mode: AttributeMode = AttributeMode.FULL
    labeled_fraction: float = 1.0  # For PARTIAL mode

    # Where Partial and Unlabeled take their Stage 3/4 contrastive anchors from.
    # Section 3.3 defines an anchor as a sample identified as minority "either
    # via ground-truth label s = 1 or predicted as such by the bias detector",
    # so "detector" scores the whole training set with the trained detector and
    # anchors on every flagged sample. "given" uses only what each setting
    # started from: the labelled subset for Partial, the raw LOF pseudo-labels
    # for Unlabeled. Anchoring on the labelled subset alone fits the correction
    # to far fewer samples and measurably weakens it.
    anchor_source: str = "detector"

    # Unsupervised detection configuration (Supplementary D.2)
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.1
    lof_n_jobs: Optional[int] = None  # Throughput only; None is scikit-learn's default
    # Which representation LOF sees. Supplementary C.3.1 allows either the
    # intermediate representation or "the pooled h_pooled"; "cls" takes the CLS
    # token at detector_layer, "mean" averages the sequence.
    lof_feature: str = "cls"

    # Attribute indices
    sensitive_attributes: List[int] = field(default_factory=lambda: [9])
    target_attribute: int = 20

    # Which worst-group accuracy to report. "label_by_group" is the convention
    # behind the paper's Table 1 and its baselines; "sensitive_group" is the
    # literal formula printed in Supplementary C.4. See
    # ``fairnet.utils.compute_fairness_metrics`` for why they differ.
    wga_definition: str = "label_by_group"

    # Device
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        if isinstance(self.attribute_mode, str):
            self.attribute_mode = AttributeMode(self.attribute_mode)
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.hidden_dim <= 0 or self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("hidden_dim, lora_rank, and lora_alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not 0 <= self.activation_threshold <= 1:
            raise ValueError("activation_threshold must be in [0, 1]")
        if not 0 < self.labeled_fraction <= 1:
            raise ValueError("labeled_fraction must be in (0, 1]")
        if self.attribute_mode == AttributeMode.PARTIAL and self.labeled_fraction == 1.0:
            raise ValueError("labeled_fraction should be < 1.0 for PARTIAL mode")
        if not self.sensitive_attributes:
            raise ValueError("sensitive_attributes must contain at least one index")
        if len(set(self.sensitive_attributes)) != len(self.sensitive_attributes):
            raise ValueError("sensitive_attributes cannot contain duplicates")
        if any(index < 0 for index in self.sensitive_attributes):
            raise ValueError("sensitive attribute indices cannot be negative")
        if self.target_attribute < 0:
            raise ValueError("target_attribute cannot be negative")
        if self.target_attribute in self.sensitive_attributes:
            raise ValueError("target_attribute cannot also be a sensitive attribute")
        if self.detector_layer < 0:
            raise ValueError("detector_layer cannot be negative")
        if self.detector_num_layers <= 0 or self.detector_hidden < 2:
            raise ValueError("detector dimensions must be positive")
        if any(epoch < 0 for epoch in (self.stage1_epochs, self.stage2_epochs, self.stage4_epochs)):
            raise ValueError("training epochs cannot be negative")
        if any(rate <= 0 for rate in (self.stage1_lr, self.stage2_lr, self.stage4_lr)):
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0 or self.contrastive_margin < 0:
            raise ValueError("weight_decay and contrastive_margin cannot be negative")
        if self.lambda_D < 0 or self.lambda_C < 0:
            raise ValueError("loss weights cannot be negative")
        if self.gradient_clip <= 0 or self.warmup_steps < 0:
            raise ValueError("gradient_clip must be positive and warmup_steps non-negative")
        if self.batch_size <= 0 or self.image_size <= 0 or self.max_seq_length <= 0:
            raise ValueError("batch_size and input sizes must be positive")
        if self.lof_feature not in {"cls", "mean"}:
            raise ValueError("lof_feature must be 'cls' or 'mean'")
        if self.anchor_source not in {"detector", "given"}:
            raise ValueError("anchor_source must be 'detector' or 'given'")
        allowed_distances = {"euclidean", "squared_euclidean", "cosine"}
        if self.contrastive_distance not in allowed_distances:
            raise ValueError(f"contrastive_distance must be one of {sorted(allowed_distances)}")
        if self.stage4_objective not in {"contrastive", "task"}:
            raise ValueError("stage4_objective must be 'contrastive' or 'task'")
        if self.wga_definition not in {"label_by_group", "sensitive_group"}:
            raise ValueError("wga_definition must be 'label_by_group' or 'sensitive_group'")
        if self.lr_schedule not in {"warmup_constant", "warmup_cosine"}:
            raise ValueError("lr_schedule must be 'warmup_constant' or 'warmup_cosine'")
        allowed_selection = {"worst_group_accuracy", "accuracy"}
        if self.model_selection_metric is not None:
            if self.model_selection_metric not in allowed_selection:
                raise ValueError(
                    f"model_selection_metric must be one of {sorted(allowed_selection)}"
                )
            if (
                self.model_selection_metric == "worst_group_accuracy"
                and self.attribute_mode != AttributeMode.FULL
            ):
                raise ValueError(
                    "Only FairNet-Full may select checkpoints on validation WGA; "
                    "Partial and Unlabeled have no validation sensitive labels"
                )
        if self.lof_n_neighbors <= 0:
            raise ValueError("lof_n_neighbors must be positive")
        if not 0 < self.lof_contamination <= 0.5:
            raise ValueError("lof_contamination must be in (0, 0.5]")
        allowed_targets = {"query", "key", "value", "dense"}
        if not self.lora_target_modules:
            raise ValueError("lora_target_modules cannot be empty")
        unknown_targets = set(self.lora_target_modules) - allowed_targets
        if unknown_targets:
            raise ValueError(f"Unknown LoRA target modules: {sorted(unknown_targets)}")
        if self.lora_layers is not None:
            if not self.lora_layers:
                raise ValueError("lora_layers cannot be empty")
            if len(set(self.lora_layers)) != len(self.lora_layers):
                raise ValueError("lora_layers cannot contain duplicates")
            if any(layer < 0 for layer in self.lora_layers):
                raise ValueError("lora_layers cannot contain negative indices")


@dataclass
class ViTFairNetConfig(FairNetConfig):
    """Configuration specific to ViT-based FairNet."""

    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 768
    patch_size: int = 16


@dataclass
class BERTFairNetConfig(FairNetConfig):
    """Configuration specific to BERT-based FairNet."""

    model_name: str = "bert-base-uncased"
    num_classes: int = 3  # For MultiNLI: entailment, neutral, contradiction
