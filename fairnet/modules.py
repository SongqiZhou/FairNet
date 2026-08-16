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
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        # Xavier initialization
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]

        Returns:
            pooled: [batch_size, hidden_dim]
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden_dim]")
        # Compute attention scores: [batch, seq, 1]
        scores = self.v(torch.tanh(self.W(hidden_states)))
        if attention_mask is not None:
            if attention_mask.shape != hidden_states.shape[:2]:
                raise ValueError("attention_mask must have shape [batch, sequence]")
            if not attention_mask.to(dtype=torch.bool).any(dim=1).all():
                raise ValueError("each attention_mask row must contain at least one token")
            scores = scores.masked_fill(
                ~attention_mask.to(device=scores.device, dtype=torch.bool).unsqueeze(-1),
                torch.finfo(scores.dtype).min,
            )

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
        use_attention_pooling: bool = True,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if mlp_hidden < 2:
            raise ValueError("mlp_hidden must be at least 2")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.use_attention_pooling = use_attention_pooling

        if use_attention_pooling:
            self.attention_pool = AttentionPooling(hidden_dim)

        # Build MLP
        layers = []
        in_dim = hidden_dim
        for i in range(num_layers):
            out_dim = mlp_hidden if i < num_layers - 1 else mlp_hidden // 2
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
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
        is_sequence: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, dim] or [batch, dim]
            is_sequence: Whether input is a sequence

        Returns:
            risk_score: [batch, 1] in range [0, 1]
        """
        if is_sequence and self.use_attention_pooling:
            features = self.attention_pool(hidden_states, attention_mask=attention_mask)
        else:
            features = hidden_states

        return torch.sigmoid(self.mlp(features))

    def forward_logits(
        self,
        hidden_states: torch.Tensor,
        is_sequence: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return numerically stable logits for ``BCEWithLogitsLoss``."""
        if is_sequence and self.use_attention_pooling:
            features = self.attention_pool(hidden_states, attention_mask=attention_mask)
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

    def __init__(self, hidden_dim: int, n_neighbors: int = 20, contamination: float = 0.1):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        self.hidden_dim = hidden_dim
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.lof: Optional[LocalOutlierFactor] = None
        self.fitted = False
        self.score_mean = 0.0
        self.score_std = 1.0

    def fit(self, features: torch.Tensor):
        """
        Fit LOF on feature representations.

        Args:
            features: [num_samples, hidden_dim] tensor of features
        """
        if features.ndim != 2 or features.shape[1] != self.hidden_dim:
            raise ValueError(f"features must have shape [samples, {self.hidden_dim}]")
        features_np = features.detach().cpu().numpy()
        if len(features_np) < 3:
            raise ValueError("LOF requires at least three samples")
        self.lof = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(features_np) - 1),
            contamination=self.contamination,
            novelty=True,  # Enable prediction on new data
        )
        self.lof.fit(features_np)
        training_scores = -self.lof.score_samples(features_np)
        # LOF predicts an outlier when score_samples < offset_. Centering the
        # transformed risk at -offset_ makes a risk of 0.5 match that boundary.
        self.score_mean = -float(self.lof.offset_)
        self.score_std = max(float(training_scores.std()), 1e-8)
        self.fitted = True

    def fit_predict(self, features: torch.Tensor) -> torch.Tensor:
        """Fit LOF and return its training-set outlier assignments as 0/1."""

        self.fit(features)
        labels = self.lof.negative_outlier_factor_ < self.lof.offset_
        return torch.as_tensor(labels, dtype=torch.long, device=features.device)

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
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_dim:
            raise ValueError(f"hidden_states must have shape [samples, {self.hidden_dim}]")

        features_np = hidden_states.detach().cpu().numpy()

        # Get negative LOF scores (higher = more outlier-like)
        scores = -self.lof.score_samples(features_np)

        # Convert to tensor and normalize
        scores_tensor = torch.tensor(scores, device=hidden_states.device, dtype=hidden_states.dtype)

        # Use training-set calibration so scores do not change with batch size.
        scores_normalized = (scores_tensor - self.score_mean) / self.score_std
        return torch.sigmoid(scores_normalized).unsqueeze(-1)


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
        dropout: float = 0.0,
        adapter_names: Optional[List[str]] = None,
    ):
        super().__init__()

        if rank <= 0 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.original_linear = original_linear
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.threshold = threshold

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.adapter_names = list(adapter_names or ["default"])
        if not self.adapter_names or len(set(self.adapter_names)) != len(self.adapter_names):
            raise ValueError("adapter_names must be non-empty and unique")
        if any(not name or "." in name for name in self.adapter_names):
            raise ValueError("adapter names must be non-empty and cannot contain dots")
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name in self.adapter_names:
            matrix_a = nn.Parameter(torch.empty(rank, self.in_features))
            matrix_b = nn.Parameter(torch.empty(self.out_features, rank))
            nn.init.kaiming_uniform_(matrix_a, a=math.sqrt(5))
            nn.init.zeros_(matrix_b)
            self.lora_A[name] = matrix_a
            self.lora_B[name] = matrix_b
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Runtime state for conditional activation
        self._risk_scores: Dict[str, torch.Tensor] = {}
        self._forced_adapters: set[str] = set()

    def set_risk_score(
        self,
        risk_score: Optional[torch.Tensor],
        force_lora: bool = False,
    ):
        """Backward-compatible activation setter for the first adapter."""
        name = self.adapter_names[0]
        self.set_activation(
            {} if risk_score is None else {name: risk_score},
            {name} if force_lora else set(),
        )

    def set_activation(
        self,
        risk_scores: Optional[Dict[str, torch.Tensor]] = None,
        forced_adapters: Optional[set[str]] = None,
    ) -> None:
        risk_scores = risk_scores or {}
        forced_adapters = forced_adapters or set()
        unknown = (set(risk_scores) | set(forced_adapters)) - set(self.adapter_names)
        if unknown:
            raise KeyError(f"Unknown LoRA adapters: {sorted(unknown)}")
        self._risk_scores = risk_scores
        self._forced_adapters = forced_adapters

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

        if not self._forced_adapters and not self._risk_scores:
            return original_output

        x_dropped = self.lora_dropout(x)
        output = original_output
        for name in self.adapter_names:
            if name not in self._forced_adapters and name not in self._risk_scores:
                continue
            delta = ((x_dropped @ self.lora_A[name].T) @ self.lora_B[name].T) * self.scaling
            if name in self._forced_adapters:
                output = output + delta
                continue

            gate = (self._risk_scores[name] > self.threshold).to(
                device=delta.device, dtype=delta.dtype
            )
            if gate.ndim not in (1, 2) or (gate.ndim == 2 and gate.shape[1] != 1):
                raise ValueError("risk scores must have shape [batch] or [batch, 1]")
            if delta.ndim < 2 or gate.shape[0] != delta.shape[0]:
                raise ValueError("risk-score batch size must match the input batch size")
            while gate.dim() < delta.dim():
                gate = gate.unsqueeze(-1)
            output = output + gate * delta
        return output

    def get_lora_parameters(self, adapter_name: Optional[str] = None) -> List[nn.Parameter]:
        """Get trainable LoRA parameters."""
        names = self.adapter_names if adapter_name is None else [adapter_name]
        if any(name not in self.adapter_names for name in names):
            raise KeyError(f"Unknown LoRA adapter: {adapter_name}")
        return [parameter for name in names for parameter in (self.lora_A[name], self.lora_B[name])]

    def get_lora_weight(self, adapter_name: Optional[str] = None) -> torch.Tensor:
        """Get effective LoRA weight matrix ΔW = B @ A * scaling."""
        names = self.adapter_names if adapter_name is None else [adapter_name]
        if any(name not in self.adapter_names for name in names):
            raise KeyError(f"Unknown LoRA adapter: {adapter_name}")
        weights = [(self.lora_B[name] @ self.lora_A[name]) * self.scaling for name in names]
        return torch.stack(weights).sum(dim=0)


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
        target_modules: Optional[List[str]] = None,
        target_layers: Optional[List[int]] = None,
        adapter_names: Optional[List[str]] = None,
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
        target_modules = target_modules or ["query", "value"]
        unknown_modules = set(target_modules) - {"query", "key", "value", "dense"}
        if unknown_modules:
            raise ValueError(f"Unknown ViT LoRA targets: {sorted(unknown_modules)}")
        if target_layers is None:
            target_layers = list(range(num_layers))

        for layer_idx in target_layers:
            if not 0 <= layer_idx < num_layers:
                raise IndexError(f"ViT layer index out of range: {layer_idx}")

            layer = vit_model.encoder.layer[layer_idx]
            attention = layer.attention.attention

            for module_name in target_modules:
                if module_name == "query":
                    original = attention.query
                    lora = LoRALinear(original, rank, alpha, threshold, dropout, adapter_names)
                    attention.query = lora
                    lora_modules[f"layer_{layer_idx}_query"] = lora

                elif module_name == "key":
                    original = attention.key
                    lora = LoRALinear(original, rank, alpha, threshold, dropout, adapter_names)
                    attention.key = lora
                    lora_modules[f"layer_{layer_idx}_key"] = lora

                elif module_name == "value":
                    original = attention.value
                    lora = LoRALinear(original, rank, alpha, threshold, dropout, adapter_names)
                    attention.value = lora
                    lora_modules[f"layer_{layer_idx}_value"] = lora

                elif module_name == "dense":
                    original = layer.attention.output.dense
                    lora = LoRALinear(original, rank, alpha, threshold, dropout, adapter_names)
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
        target_modules: Optional[List[str]] = None,
        target_layers: Optional[List[int]] = None,
        adapter_names: Optional[List[str]] = None,
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
        target_modules = target_modules or ["query", "value"]
        unknown_modules = set(target_modules) - {"query", "key", "value", "dense"}
        if unknown_modules:
            raise ValueError(f"Unknown BERT LoRA targets: {sorted(unknown_modules)}")
        if target_layers is None:
            target_layers = list(range(num_layers))

        for layer_idx in target_layers:
            if not 0 <= layer_idx < num_layers:
                raise IndexError(f"BERT layer index out of range: {layer_idx}")

            layer = bert_model.encoder.layer[layer_idx]
            attention = layer.attention.self

            for module_name in target_modules:
                if module_name == "query":
                    original = attention.query
                    lora = LoRALinear(original, rank, alpha, threshold, dropout, adapter_names)
                    attention.query = lora
                    lora_modules[f"layer_{layer_idx}_query"] = lora

                elif module_name == "key":
                    original = attention.key
                    lora = LoRALinear(original, rank, alpha, threshold, dropout, adapter_names)
                    attention.key = lora
                    lora_modules[f"layer_{layer_idx}_key"] = lora

                elif module_name == "value":
                    original = attention.value
                    lora = LoRALinear(original, rank, alpha, threshold, dropout, adapter_names)
                    attention.value = lora
                    lora_modules[f"layer_{layer_idx}_value"] = lora

                elif module_name == "dense":
                    original = layer.attention.output.dense
                    lora = LoRALinear(original, rank, alpha, threshold, dropout, adapter_names)
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
        if margin < 0:
            raise ValueError("margin cannot be negative")
        if distance_type not in {"euclidean", "cosine"}:
            raise ValueError("distance_type must be 'euclidean' or 'cosine'")
        self.margin = margin
        self.distance_type = distance_type

    def _compute_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
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
        desc: str = "Building prototypes",
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

                    sums[y][s] += features[i].detach()
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
        self, features: torch.Tensor, labels: torch.Tensor, sensitive: torch.Tensor
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

            sums[y][s] += features[i].detach()
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
        majority_group: int = 0,
        anchor_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get positive and negative prototype targets for anchors.

        For each anchor with label y:
            - positive: prototype[y][majority_group] (same class, majority)
            - negative: the nearest different-class majority prototype

        Args:
            anchor_labels: [batch] labels of anchor samples
            majority_group: Index of majority group (default 0)

        Returns:
            pos_targets: [batch, dim] positive prototypes
            neg_targets: [batch, dim] negative prototypes
        """
        pos_targets = []
        neg_targets = []

        if anchor_features is not None and len(anchor_features) != len(anchor_labels):
            raise ValueError("anchor_features and anchor_labels must have equal length")

        available_classes = sorted(self.prototypes)
        if len(available_classes) < 2:
            raise RuntimeError("At least two task classes are required for triplet targets")

        for index, y in enumerate(anchor_labels):
            y = int(y.item())

            # Positive: same class, majority group
            if y in self.prototypes and majority_group in self.prototypes[y]:
                pos_targets.append(self.prototypes[y][majority_group])
            elif y in self.prototypes:
                pos_targets.append(list(self.prototypes[y].values())[0])
            else:
                raise KeyError(f"No prototype is available for task class {y}")

            negative_candidates = []
            for negative_class in available_classes:
                if negative_class == y:
                    continue
                groups = self.prototypes[negative_class]
                negative_candidates.append(groups.get(majority_group, next(iter(groups.values()))))
            if anchor_features is None or len(negative_candidates) == 1:
                neg_targets.append(negative_candidates[0])
            else:
                candidates = torch.stack(negative_candidates)
                distances = torch.sum((candidates - anchor_features[index].detach()) ** 2, dim=-1)
                neg_targets.append(candidates[torch.argmin(distances)])

        return torch.stack(pos_targets), torch.stack(neg_targets)
