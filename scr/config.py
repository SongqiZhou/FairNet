"""
FairNet Configuration Module

Contains all configuration classes for FairNet training and inference.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AttributeMode(Enum):
    """Training mode based on sensitive attribute availability."""
    FULL = "full"           # All sensitive labels available
    PARTIAL = "partial"     # Only k% of sensitive labels available
    UNLABELED = "unlabeled" # No sensitive labels (unsupervised detection)


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
    lambda_D: float = 1.0
    lambda_C: float = 1.0
    
    # Optimization
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    batch_size: int = 128
    
    # Data configuration
    image_size: int = 64
    max_seq_length: int = 128  # For NLP models
    
    # Attribute mode configuration (Section 5.1)
    attribute_mode: AttributeMode = AttributeMode.FULL
    labeled_fraction: float = 1.0  # For PARTIAL mode
    
    # Unsupervised detection configuration (Supplementary D.2)
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.1
    
    # Attribute indices
    sensitive_attributes: List[int] = field(default_factory=lambda: [9])
    target_attribute: int = 20
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.activation_threshold < 0 or self.activation_threshold > 1:
            raise ValueError("activation_threshold must be in [0, 1]")
        if self.labeled_fraction <= 0 or self.labeled_fraction > 1:
            raise ValueError("labeled_fraction must be in (0, 1]")
        if self.attribute_mode == AttributeMode.PARTIAL and self.labeled_fraction == 1.0:
            raise ValueError("labeled_fraction should be < 1.0 for PARTIAL mode")


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
