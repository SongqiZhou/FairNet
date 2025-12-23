"""
FairNet Core Modules

Contains all the building blocks for FairNet:
- AttentionPooling: Attention-based sequence pooling
- BiasDetector: Detects minority group membership
- LoRALinear: LoRA-enhanced linear layer
- LoRAInjector: Injects LoRA into transformer models
- TripletContrastiveLoss: Contrastive loss for representation alignment
- StaticPrototypeBank: Stores class/group prototypes
- UnsupervisedBiasDetector: LOF-based detector for unlabeled setting
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data import DataLoader
from tqdm import tqdm


# =============================================================================
# Attention Pooling (Supplementary C.3.1)
# =============================================================================

class AttentionPooling(nn.Module):
    """
    Attention Pooling from Supplementary C.3.1.
    
    Computes weighted sum of sequence elements:
        s_i = v^T tanh(W h_i + b)
        α_i = softmax(s_i)
        h_pooled = Σ α_i h_i
    
    Args:
        hidden_dim: Dimension of hidden states
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.v.weight)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
        
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        # Compute attention scores: [batch, seq, 1]
        scores = self.v(torch.tanh(self.W(hidden_states)))
        
        # Normalize to get attention weights: [batch, seq, 1]
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum: [batch, hidden_dim]
        return torch.sum(hidden_states * weights, dim=1)


# =============================================================================
# Bias Detector (Section 3.2)
# =============================================================================

class BiasDetector(nn.Module):
    """
    Bias Detector D_φ^(l) from Section 3.2.
    
    Detects minority group membership from intermediate representations.
    Outputs risk score p_s^(l)(x) ∈ [0,1] indicating minority group likelihood.
    
    Architecture:
        AttentionPooling → MLP → Sigmoid
    
    Args:
        hidden_dim: Input hidden dimension
        mlp_hidden: Hidden dimension for MLP layers
        num_layers: Number of MLP layers
        dropout: Dropout rate
        use_attention_pooling: Whether to use attention pooling for sequences
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        mlp_hidden: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.1, 
        use_attention_pooling: bool = True
    ):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(hidden_dim)
        
        # Build MLP
        layers = []
        in_dim = hidden_dim
        for i in range(num_layers):
            out_dim = mlp_hidden if i < num_layers - 1 else mlp_hidden // 2
            layers.extend([
                nn.Linear(in_dim, out_dim), 
                nn.ReLU(), 
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                # Smaller gain for output layer to start near 0.5
                gain = 0.1 if m.out_features == 1 else 1.0
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        is_sequence: bool = True
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, dim] or [batch, dim]
            is_sequence: Whether input is a sequence
        
        Returns:
            risk_score: [batch, 1] in range [0, 1]
        """
        if is_sequence and self.use_attention_pooling:
            features = self.attention_pool(hidden_states)
        else:
            features = hidden_states
        
        return self.mlp(features)


# =============================================================================
# Unsupervised Bias Detector (Supplementary D.2)
# =============================================================================

class UnsupervisedBiasDetector(nn.Module):
    """
    Unsupervised Bias Detector using Local Outlier Factor (LOF).
    
    Used in FairNet-Unlabeled setting (Section 5.1, Supplementary D.2).
    Identifies minority samples via outlier detection on representations.
    
    The intuition is that minority group samples are outliers in the
    representation space learned by the biased base model.
    
    Args:
        hidden_dim: Dimension of input features
        n_neighbors: Number of neighbors for LOF
        contamination: Expected proportion of outliers (minority rate)
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        n_neighbors: int = 20, 
        contamination: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.lof: Optional[LocalOutlierFactor] = None
        self.fitted = False
        
        # Learnable threshold bias for fine-tuning
        self.threshold_bias = nn.Parameter(torch.zeros(1))
    
    def fit(self, features: torch.Tensor):
        """
        Fit LOF on feature representations.
        
        Args:
            features: [num_samples, hidden_dim] tensor of features
        """
        features_np = features.detach().cpu().numpy()
        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True  # Enable prediction on new data
        )
        self.lof.fit(features_np)
        self.fitted = True
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute outlier scores as risk scores.
        
        Args:
            hidden_states: [batch, hidden_dim] features
        
        Returns:
            risk_scores: [batch, 1] normalized outlier scores
        """
        if not self.fitted:
            raise RuntimeError("UnsupervisedBiasDetector must be fitted first")
        
        features_np = hidden_states.detach().cpu().numpy()
        
        # Get negative LOF scores (higher = more outlier-like)
        scores = -self.lof.score_samples(features_np)
        
        # Convert to tensor and normalize
        scores_tensor = torch.tensor(
            scores, 
            device=hidden_states.device, 
            dtype=hidden_states.dtype
        )
        
        # Z-score normalization + sigmoid for [0, 1] range
        scores_normalized = (scores_tensor - scores_tensor.mean()) / (scores_tensor.std() + 1e-8)
        
        return torch.sigmoid(scores_normalized + self.threshold_bias).unsqueeze(-1)


# =============================================================================
# LoRA Linear Layer (Section 3.3)
# =============================================================================

class LoRALinear(nn.Module):
    """
    LoRA-enhanced Linear layer with conditional activation.
    
    Paper Section 3.3: ΔW_j = B_j A_j where:
        - A ∈ R^{r×k} (down-projection)
        - B ∈ R^{d×r} (up-projection)
        - r is the rank (typically 8)
    
    Conditional activation based on risk score:
        - If risk_score > threshold: use W + BA (corrected)
        - Otherwise: use W only (original)
    
    Args:
        original_linear: The original nn.Linear to enhance
        rank: LoRA rank r
        alpha: Scaling factor α (effective scaling = α/r)
        threshold: Activation threshold τ
        dropout: Dropout rate for LoRA path
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        threshold: float = 0.5,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.threshold = threshold
        
        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize: A with Kaiming, B with zeros (so initial ΔW = 0)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Runtime state for conditional activation
        self._risk_score: Optional[torch.Tensor] = None
        self._force_lora: bool = False
    
    def set_risk_score(
        self, 
        risk_score: Optional[torch.Tensor], 
        force_lora: bool = False
    ):
        """Set risk score for conditional activation."""
        self._risk_score = risk_score
        self._force_lora = force_lora
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditional LoRA.
        
        Args:
            x: Input tensor [batch, ..., in_features]
        
        Returns:
            Output tensor [batch, ..., out_features]
        """
        # Always compute original output
        original_output = self.original_linear(x)
        
        # If no activation, return original
        if not self._force_lora and self._risk_score is None:
            return original_output
        
        # Compute LoRA delta: x @ A^T @ B^T * scaling
        x_dropped = self.lora_dropout(x)
        lora_delta = (x_dropped @ self.lora_A.T) @ self.lora_B.T * self.scaling
        
        # Force LoRA for all samples (training mode)
        if self._force_lora:
            return original_output + lora_delta
        
        # Conditional activation based on risk score
        gate = (self._risk_score > self.threshold).float()
        
        # Handle dimension alignment for broadcasting
        while gate.dim() < lora_delta.dim():
            gate = gate.unsqueeze(-1)
        
        return original_output + gate * lora_delta
    
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get trainable LoRA parameters."""
        return [self.lora_A, self.lora_B]
    
    def get_lora_weight(self) -> torch.Tensor:
        """Get effective LoRA weight matrix ΔW = B @ A * scaling."""
        return (self.lora_B @ self.lora_A) * self.scaling


# =============================================================================
# LoRA Injector
# =============================================================================

class LoRAInjector:
    """
    Utility class to inject LoRA modules into transformer models.
    
    Supports injecting into:
    - ViT attention layers (query, key, value, dense)
    - BERT attention layers (query, key, value, dense)
    """
    
    @staticmethod
    def inject_lora_into_vit(
        vit_model,
        rank: int = 8,
        alpha: float = 16.0,
        threshold: float = 0.5,
        dropout: float = 0.0,
        target_modules: List[str] = ["query", "value"],
        target_layers: Optional[List[int]] = None
    ) -> Dict[str, LoRALinear]:
        """
        Inject LoRA into ViT attention layers.
        
        Args:
            vit_model: HuggingFace ViTModel
            rank: LoRA rank
            alpha: LoRA scaling factor
            threshold: Activation threshold
            dropout: LoRA dropout
            target_modules: Which projections to apply LoRA to
            target_layers: Which layers (None = all)
        
        Returns:
            Dictionary of injected LoRALinear modules
        """
        lora_modules = {}
        num_layers = len(vit_model.encoder.layer)
        
        if target_layers is None:
            target_layers = list(range(num_layers))
        
        for layer_idx in target_layers:
            if layer_idx >= num_layers:
                continue
            
            layer = vit_model.encoder.layer[layer_idx]
            attention = layer.attention.attention
            
            for module_name in target_modules:
                if module_name == "query":
                    original = attention.query
                    lora = LoRALinear(original, rank, alpha, threshold, dropout)
                    attention.query = lora
                    lora_modules[f"layer_{layer_idx}_query"] = lora
                    
                elif module_name == "key":
                    original = attention.key
                    lora = LoRALinear(original, rank, alpha, threshold, dropout)
                    attention.key = lora
                    lora_modules[f"layer_{layer_idx}_key"] = lora
                    
                elif module_name == "value":
                    original = attention.value
                    lora = LoRALinear(original, rank, alpha, threshold, dropout)
                    attention.value = lora
                    lora_modules[f"layer_{layer_idx}_value"] = lora
                    
                elif module_name == "dense":
                    original = layer.attention.output.dense
                    lora = LoRALinear(original, rank, alpha, threshold, dropout)
                    layer.attention.output.dense = lora
                    lora_modules[f"layer_{layer_idx}_dense"] = lora
        
        return lora_modules
    
    @staticmethod
    def inject_lora_into_bert(
        bert_model,
        rank: int = 8,
        alpha: float = 16.0,
        threshold: float = 0.5,
        dropout: float = 0.0,
        target_modules: List[str] = ["query", "value"],
        target_layers: Optional[List[int]] = None
    ) -> Dict[str, LoRALinear]:
        """
        Inject LoRA into BERT attention layers.
        
        Args:
            bert_model: HuggingFace BertModel
            rank: LoRA rank
            alpha: LoRA scaling factor  
            threshold: Activation threshold
            dropout: LoRA dropout
            target_modules: Which projections to apply LoRA to
            target_layers: Which layers (None = all)
        
        Returns:
            Dictionary of injected LoRALinear modules
        """
        lora_modules = {}
        num_layers = len(bert_model.encoder.layer)
        
        if target_layers is None:
            target_layers = list(range(num_layers))
        
        for layer_idx in target_layers:
            if layer_idx >= num_layers:
                continue
            
            layer = bert_model.encoder.layer[layer_idx]
            attention = layer.attention.self
            
            for module_name in target_modules:
                if module_name == "query":
                    original = attention.query
                    lora = LoRALinear(original, rank, alpha, threshold, dropout)
                    attention.query = lora
                    lora_modules[f"layer_{layer_idx}_query"] = lora
                    
                elif module_name == "key":
                    original = attention.key
                    lora = LoRALinear(original, rank, alpha, threshold, dropout)
                    attention.key = lora
                    lora_modules[f"layer_{layer_idx}_key"] = lora
                    
                elif module_name == "value":
                    original = attention.value
                    lora = LoRALinear(original, rank, alpha, threshold, dropout)
                    attention.value = lora
                    lora_modules[f"layer_{layer_idx}_value"] = lora
                    
                elif module_name == "dense":
                    original = layer.attention.output.dense
                    lora = LoRALinear(original, rank, alpha, threshold, dropout)
                    layer.attention.output.dense = lora
                    lora_modules[f"layer_{layer_idx}_dense"] = lora
        
        return lora_modules


# =============================================================================
# Triplet Contrastive Loss (Equation 2)
# =============================================================================

class TripletContrastiveLoss(nn.Module):
    """
    Triplet Contrastive Loss from Equation 2.
    
    L_contrastive(x_a, x_p, x_n) = [D(z_a, z_p) - D(z_a, z_n) + margin]_+
    
    where:
        - x_a: anchor (minority sample)
        - x_p: positive (same-class majority prototype)
        - x_n: negative (different-class majority prototype)
        - D: distance function (Euclidean or cosine)
        - margin: minimum separation margin
    
    Args:
        margin: Margin m for triplet loss (paper uses m=0.5)
        distance_type: "euclidean" or "cosine"
    """
    
    def __init__(self, margin: float = 0.5, distance_type: str = "euclidean"):
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type
    
    def _compute_distance(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise distance."""
        if self.distance_type == "euclidean":
            return torch.sum((x1 - x2) ** 2, dim=-1)
        elif self.distance_type == "cosine":
            x1_norm = F.normalize(x1, p=2, dim=-1)
            x2_norm = F.normalize(x2, p=2, dim=-1)
            return 1 - torch.sum(x1_norm * x2_norm, dim=-1)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
    
    def forward(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet contrastive loss.
        
        Args:
            anchor: [batch, dim] minority sample representations
            positive: [batch, dim] same-class majority prototypes
            negative: [batch, dim] different-class majority prototypes
        
        Returns:
            Scalar loss value
        """
        d_pos = self._compute_distance(anchor, positive)
        d_neg = self._compute_distance(anchor, negative)
        
        # Hinge loss: [d_pos - d_neg + margin]_+
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        
        return loss.mean()


# =============================================================================
# Static Prototype Bank (Stage 3)
# =============================================================================

class StaticPrototypeBank:
    """
    Prototype Bank for Stage 3.
    
    Stores average embeddings per (class, group) computed from the frozen
    base model. Used as targets for contrastive learning in Stage 4.
    
    For each (class y, group s) combination, stores:
        prototype[y][s] = mean(features of samples with label y and group s)
    
    Args:
        feature_dim: Dimension of feature vectors
        device: Device to store prototypes on
    """
    
    def __init__(self, feature_dim: int, device: torch.device):
        self.feature_dim = feature_dim
        self.device = device
        self.prototypes: Dict[int, Dict[int, torch.Tensor]] = {}
        self.counts: Dict[int, Dict[int, int]] = {}
    
    def compute_from_loader(
        self,
        model: nn.Module,
        loader: DataLoader,
        get_features_fn,
        target_attr: int,
        sensitive_attr: int,
        desc: str = "Building prototypes"
    ):
        """
        Compute prototypes from a data loader.
        
        Args:
            model: Model to extract features from
            loader: DataLoader providing (images, attributes)
            get_features_fn: Function(model, images) -> features
            target_attr: Index of target attribute
            sensitive_attr: Index of sensitive attribute
            desc: Description for progress bar
        """
        print("Stage 3: Computing Static Prototypes...")
        model.eval()
        
        sums: Dict[int, Dict[int, torch.Tensor]] = {}
        counts: Dict[int, Dict[int, int]] = {}
        
        with torch.no_grad():
            for images, attributes in tqdm(loader, desc=desc):
                images = images.to(self.device)
                labels = attributes[:, target_attr].to(self.device)
                sensitive = attributes[:, sensitive_attr].to(self.device)
                features = get_features_fn(model, images)
                
                for i in range(len(labels)):
                    y = labels[i].item()
                    s = sensitive[i].item()
                    
                    if y not in sums:
                        sums[y] = {}
                        counts[y] = {}
                    if s not in sums[y]:
                        sums[y][s] = torch.zeros(self.feature_dim, device=self.device)
                        counts[y][s] = 0
                    
                    sums[y][s] += features[i]
                    counts[y][s] += 1
        
        # Compute averages
        for y in sums:
            self.prototypes[y] = {}
            self.counts[y] = {}
            for s in sums[y]:
                self.prototypes[y][s] = sums[y][s] / counts[y][s]
                self.counts[y][s] = counts[y][s]
        
        # Print statistics
        print("\nPrototype Statistics:")
        for y in sorted(self.prototypes.keys()):
            for s in sorted(self.prototypes[y].keys()):
                print(f"  Class {y}, Group {s}: {self.counts[y][s]} samples")
    
    def compute_from_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        sensitive: torch.Tensor
    ):
        """
        Compute prototypes directly from feature tensors.
        
        Args:
            features: [N, dim] feature vectors
            labels: [N] target labels
            sensitive: [N] sensitive attribute values
        """
        sums: Dict[int, Dict[int, torch.Tensor]] = {}
        counts: Dict[int, Dict[int, int]] = {}
        
        for i in range(len(labels)):
            y = labels[i].item()
            s = sensitive[i].item()
            
            if y not in sums:
                sums[y] = {}
                counts[y] = {}
            if s not in sums[y]:
                sums[y][s] = torch.zeros(self.feature_dim, device=self.device)
                counts[y][s] = 0
            
            sums[y][s] += features[i]
            counts[y][s] += 1
        
        for y in sums:
            self.prototypes[y] = {}
            self.counts[y] = {}
            for s in sums[y]:
                self.prototypes[y][s] = sums[y][s] / counts[y][s]
                self.counts[y][s] = counts[y][s]
    
    def get_targets(
        self, 
        anchor_labels: torch.Tensor, 
        majority_group: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get positive and negative prototype targets for anchors.
        
        For each anchor with label y:
            - positive: prototype[y][majority_group] (same class, majority)
            - negative: prototype[1-y][majority_group] (different class, majority)
        
        Args:
            anchor_labels: [batch] labels of anchor samples
            majority_group: Index of majority group (default 0)
        
        Returns:
            pos_targets: [batch, dim] positive prototypes
            neg_targets: [batch, dim] negative prototypes
        """
        pos_targets = []
        neg_targets = []
        
        for y in anchor_labels:
            y = y.item()
            
            # Positive: same class, majority group
            if y in self.prototypes and majority_group in self.prototypes[y]:
                pos_targets.append(self.prototypes[y][majority_group])
            elif y in self.prototypes:
                pos_targets.append(list(self.prototypes[y].values())[0])
            else:
                pos_targets.append(torch.zeros(self.feature_dim, device=self.device))
            
            # Negative: different class, majority group
            neg_y = 1 - y  # Assumes binary classification
            if neg_y in self.prototypes:
                if majority_group in self.prototypes[neg_y]:
                    neg_targets.append(self.prototypes[neg_y][majority_group])
                else:
                    neg_targets.append(list(self.prototypes[neg_y].values())[0])
            else:
                neg_targets.append(torch.zeros(self.feature_dim, device=self.device))
        
        return torch.stack(pos_targets), torch.stack(neg_targets)
