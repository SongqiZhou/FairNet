"""
FairNet Models

Contains the main model architectures:
- FairNetViT: Vision Transformer based FairNet
- FairNetBERT: BERT based FairNet for NLP tasks
"""

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, ViTConfig, ViTModel

from .config import AttributeMode, FairNetConfig
from .modules import (
    BiasDetector,
    LoRAInjector,
)


def _resolve_sensitive_labels(sensitive_labels, attr, position):
    """Return a [batch, 1] direct gate for a sensitive attribute."""
    if sensitive_labels is None:
        return None
    if isinstance(sensitive_labels, Mapping):
        if attr not in sensitive_labels and str(attr) not in sensitive_labels:
            raise KeyError(f"Missing sensitive labels for attribute {attr}")
        values = sensitive_labels.get(attr, sensitive_labels.get(str(attr)))
    else:
        values = sensitive_labels
        if not isinstance(values, torch.Tensor):
            raise TypeError("sensitive_labels must be a tensor or mapping of tensors")
        if values.ndim == 2 and values.shape[1] > 1:
            values = values[:, position]
        elif values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
    if not isinstance(values, torch.Tensor):
        raise TypeError("sensitive_labels must be a tensor or mapping of tensors")
    values = values.float().reshape(-1, 1)
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("FairNet currently requires binary sensitive labels")
    return values


class FairNetViT(nn.Module):
    """
    FairNet with Vision Transformer backbone.

    Architecture follows Figure 1 in the paper:
    1. ViT encoder with LoRA-enhanced attention layers
    2. Bias Detector on intermediate representations
    3. Conditional LoRA activation based on risk score
    4. Classification head

    Key features:
    - LoRA is injected into transformer attention layers (not final features)
    - Two-pass forward: detect bias, then apply conditional correction
    - Supports multiple sensitive attributes

    Args:
        vit_config: HuggingFace ViTConfig
        config: FairNetConfig
        detector_layer: Which layer to extract features for bias detection
    """

    def __init__(
        self,
        vit_config: ViTConfig,
        config: FairNetConfig,
        detector_layer: Optional[int] = None,
        vit_model: Optional[ViTModel] = None,
    ):
        super().__init__()
        self.config = config
        self.detector_layer = config.detector_layer if detector_layer is None else detector_layer
        self.hidden_dim = vit_config.hidden_size
        if config.hidden_dim != self.hidden_dim:
            raise ValueError(
                "config.hidden_dim must match the ViT hidden_size "
                f"({config.hidden_dim} != {self.hidden_dim})"
            )

        # Create ViT backbone
        self.vit = ViTModel(vit_config) if vit_model is None else vit_model
        if self.vit.config.hidden_size != vit_config.hidden_size:
            raise ValueError("vit_model and vit_config hidden sizes must match")
        if self.vit.config.num_hidden_layers != vit_config.num_hidden_layers:
            raise ValueError("vit_model and vit_config layer counts must match")

        # Determine which layers to apply LoRA
        num_layers = vit_config.num_hidden_layers
        if not 0 <= self.detector_layer < num_layers:
            raise ValueError(f"detector_layer must be between 0 and {num_layers - 1}")
        lora_layers = config.lora_layers
        if lora_layers is None:
            # Default: apply to last half of layers
            lora_layers = list(range(num_layers // 2, num_layers))

        # Inject LoRA into attention layers
        self.lora_modules = LoRAInjector.inject_lora_into_vit(
            self.vit,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            threshold=config.activation_threshold,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            target_layers=lora_layers,
            adapter_names=[str(attr) for attr in config.sensitive_attributes],
        )

        print(f"Injected LoRA into {len(self.lora_modules)} modules:")
        for name in self.lora_modules:
            print(f"  - {name}")

        # Bias detectors (one per sensitive attribute)
        self.bias_detectors = nn.ModuleDict(
            {
                f"detector_{attr}": BiasDetector(
                    hidden_dim=self.hidden_dim,
                    mlp_hidden=config.detector_hidden,
                    num_layers=config.detector_num_layers,
                    dropout=0.1,
                    use_attention_pooling=True,
                )
                for attr in config.sensitive_attributes
            }
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def freeze_base(self):
        """Freeze base ViT parameters (excluding LoRA)."""
        for name, p in self.vit.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False

    def unfreeze_base(self):
        """Unfreeze base ViT parameters."""
        for name, p in self.vit.named_parameters():
            if "lora_" not in name:
                p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def freeze_detectors(self):
        """Freeze all bias detectors."""
        for d in self.bias_detectors.values():
            for p in d.parameters():
                p.requires_grad = False

    def unfreeze_detectors(self):
        """Unfreeze all bias detectors."""
        for d in self.bias_detectors.values():
            for p in d.parameters():
                p.requires_grad = True

    def freeze_lora(self):
        """Freeze all LoRA parameters."""
        for lora in self.lora_modules.values():
            for p in lora.get_lora_parameters():
                p.requires_grad = False

    def unfreeze_lora(self):
        """Unfreeze all LoRA parameters."""
        for lora in self.lora_modules.values():
            for p in lora.get_lora_parameters():
                p.requires_grad = True

    def get_lora_parameters(self, attribute: Optional[int] = None) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimization."""
        params = []
        for lora in self.lora_modules.values():
            params.extend(lora.get_lora_parameters(None if attribute is None else str(attribute)))
        return params

    def _set_lora_activation(
        self,
        risk_scores: Optional[Dict[int, torch.Tensor]] = None,
        forced_attributes: Optional[List[int]] = None,
    ):
        """Set attribute-specific activation for every injected layer."""
        scores = {str(attribute): score for attribute, score in (risk_scores or {}).items()}
        forced = {str(attribute) for attribute in (forced_attributes or [])}
        for lora in self.lora_modules.values():
            lora.set_activation(scores, forced)

    def _clear_lora_activation(self):
        """Clear risk score from all LoRA modules."""
        for lora in self.lora_modules.values():
            lora.set_activation()

    def get_intermediate_features(
        self, x: torch.Tensor, layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get intermediate layer representations for bias detection.

        Args:
            x: Input images [batch, channels, height, width]
            layer_idx: Which layer to extract from

        Returns:
            Hidden states [batch, seq_len, hidden_dim]
        """
        if layer_idx is None:
            layer_idx = self.detector_layer

        # Disable LoRA for feature extraction
        self._clear_lora_activation()

        outputs = self.vit(x, output_hidden_states=True, return_dict=True)
        return outputs.hidden_states[layer_idx + 1]

    def get_cls_features(self, x: torch.Tensor, use_lora: bool = True) -> torch.Tensor:
        """
        Get CLS token features.

        Args:
            x: Input images
            use_lora: Whether LoRA is currently active

        Returns:
            CLS features [batch, hidden_dim]
        """
        if not use_lora:
            self._clear_lora_activation()
        return self.vit(x).last_hidden_state[:, 0]

    def forward(
        self,
        x: torch.Tensor,
        return_risk_scores: bool = False,
        return_features: bool = False,
        force_lora: bool = False,
        target_attribute: Optional[int] = None,
        sensitive_labels: Optional[Any] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass with conditional LoRA activation.

        Process:
        1. First pass: Get intermediate features (LoRA disabled)
        2. Compute risk scores using bias detector
        3. Second pass: Get features with conditional LoRA
        4. Classify

        Args:
            x: Input images [batch, channels, height, width]
            return_risk_scores: Whether to return risk scores
            return_features: Whether to return corrected features
            force_lora: Force LoRA for all samples (training mode)
            target_attribute: Only use detector for this attribute

        Returns:
            output: Classification logits [batch, 1]
            risk_scores: (optional) Dict of risk scores per attribute
            features: (optional) Corrected features [batch, hidden_dim]
        """
        if (
            self.config.attribute_mode == AttributeMode.FULL
            and sensitive_labels is None
            and not force_lora
        ):
            raise ValueError("FairNet-Full requires sensitive_labels for conditional inference")

        # Step 1: Get intermediate features only when a learned detector is needed.
        self._clear_lora_activation()
        intermediate_features = (
            self.get_intermediate_features(x)
            if sensitive_labels is None and not force_lora
            else None
        )

        # Step 2: Compute risk scores
        risk_scores = {}
        if not force_lora:
            for position, attr in enumerate(self.config.sensitive_attributes):
                if target_attribute is not None and attr != target_attribute:
                    continue

                risk_score = _resolve_sensitive_labels(sensitive_labels, attr, position)
                if risk_score is None:
                    risk_score = self.bias_detectors[f"detector_{attr}"](
                        intermediate_features, is_sequence=True
                    )
                risk_score = risk_score.to(x.device)
                risk_scores[attr] = risk_score

        # Step 3: Second pass with conditional LoRA
        if force_lora:
            forced = (
                [target_attribute]
                if target_attribute is not None
                else list(self.config.sensitive_attributes)
            )
            self._set_lora_activation(forced_attributes=forced)
        elif risk_scores:
            self._set_lora_activation(risk_scores=risk_scores)

        corrected_features = self.get_cls_features(x, use_lora=True)

        # Clear LoRA state
        self._clear_lora_activation()

        # Step 4: Classification
        output = self.classifier(corrected_features)

        # Return based on flags
        if return_risk_scores and return_features:
            return output, risk_scores, corrected_features
        elif return_risk_scores:
            return output, risk_scores
        elif return_features:
            return output, corrected_features
        return output


class FairNetBERT(nn.Module):
    """
    FairNet with BERT backbone for NLP tasks.

    Used for tasks like:
    - MultiNLI (Natural Language Inference)
    - HateXplain (Hate Speech Detection)
    - CivilComments (Toxicity Detection)

    Args:
        config: FairNetConfig
        num_classes: Number of output classes
        model_name: Pretrained BERT model name
        detector_layer: Which layer for bias detection
    """

    def __init__(
        self,
        config: FairNetConfig,
        num_classes: int = 3,
        model_name: str = "bert-base-uncased",
        detector_layer: Optional[int] = None,
        bert_config: Optional[BertConfig] = None,
        bert_model: Optional[BertModel] = None,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.accepts_token_type_ids = self._accepts_token_type_ids()
        self.config = config
        self.num_classes = num_classes

        if bert_model is not None and bert_config is not None:
            raise ValueError("Pass either bert_model or bert_config, not both")
        if bert_model is not None:
            self.bert = bert_model
        elif bert_config is not None:
            self.bert = self._backbone_class()(bert_config)
        else:
            self.bert = self._backbone_class().from_pretrained(model_name)
        self.hidden_dim = self.bert.config.hidden_size
        if config.hidden_dim != self.hidden_dim:
            raise ValueError(
                "config.hidden_dim must match the BERT hidden_size "
                f"({config.hidden_dim} != {self.hidden_dim})"
            )
        self.detector_layer = config.detector_layer if detector_layer is None else detector_layer

        # Determine LoRA layers
        num_layers = self.bert.config.num_hidden_layers
        if not 0 <= self.detector_layer < num_layers:
            raise ValueError(f"detector_layer must be between 0 and {num_layers - 1}")
        lora_layers = config.lora_layers
        if lora_layers is None:
            lora_layers = list(range(num_layers // 2, num_layers))

        # Inject LoRA
        self.lora_modules = LoRAInjector.inject_lora_into_bert(
            self.bert,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            threshold=config.activation_threshold,
            dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            target_layers=lora_layers,
            adapter_names=[str(attr) for attr in config.sensitive_attributes],
        )

        print(
            f"Injected LoRA into {len(self.lora_modules)} "
            f"{type(self.bert).__name__} modules"
        )

        # Bias detectors
        self.bias_detectors = nn.ModuleDict(
            {
                f"detector_{attr}": BiasDetector(
                    hidden_dim=self.hidden_dim,
                    mlp_hidden=config.detector_hidden,
                    num_layers=config.detector_num_layers,
                    dropout=0.1,
                    use_attention_pooling=True,
                )
                for attr in config.sensitive_attributes
            }
        )

        # Classification head
        classifier_layers = [
            nn.Linear(self.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        ]
        if num_classes == 1:
            classifier_layers.append(nn.Sigmoid())
        self.classifier = nn.Sequential(*classifier_layers)

    @staticmethod
    def _backbone_class():
        return BertModel

    @staticmethod
    def _accepts_token_type_ids() -> bool:
        return True

    def _encoder_kwargs(self, token_type_ids: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Extra keyword arguments for the backbone's forward pass."""

        if self.accepts_token_type_ids:
            return {"token_type_ids": token_type_ids}
        return {}

    def freeze_base(self):
        for name, p in self.bert.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False

    def unfreeze_base(self):
        for name, p in self.bert.named_parameters():
            if "lora_" not in name:
                p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    def freeze_detectors(self):
        for d in self.bias_detectors.values():
            for p in d.parameters():
                p.requires_grad = False

    def unfreeze_detectors(self):
        for d in self.bias_detectors.values():
            for p in d.parameters():
                p.requires_grad = True

    def freeze_lora(self):
        for lora in self.lora_modules.values():
            for p in lora.get_lora_parameters():
                p.requires_grad = False

    def unfreeze_lora(self):
        for lora in self.lora_modules.values():
            for p in lora.get_lora_parameters():
                p.requires_grad = True

    def get_lora_parameters(self, attribute: Optional[int] = None) -> List[nn.Parameter]:
        params = []
        for lora in self.lora_modules.values():
            params.extend(lora.get_lora_parameters(None if attribute is None else str(attribute)))
        return params

    def _set_lora_activation(
        self,
        risk_scores: Optional[Dict[int, torch.Tensor]] = None,
        forced_attributes: Optional[List[int]] = None,
    ):
        scores = {str(attribute): score for attribute, score in (risk_scores or {}).items()}
        forced = {str(attribute) for attribute in (forced_attributes or [])}
        for lora in self.lora_modules.values():
            lora.set_activation(scores, forced)

    def _clear_lora_activation(self):
        for lora in self.lora_modules.values():
            lora.set_activation()

    def get_intermediate_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: Optional[int] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if layer_idx is None:
            layer_idx = self.detector_layer

        self._clear_lora_activation()
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **self._encoder_kwargs(token_type_ids),
        )
        return outputs.hidden_states[layer_idx + 1]

    def get_cls_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_lora: bool = True,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not use_lora:
            self._clear_lora_activation()
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            **self._encoder_kwargs(token_type_ids),
        )
        return outputs.last_hidden_state[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_risk_scores: bool = False,
        return_features: bool = False,
        force_lora: bool = False,
        target_attribute: Optional[int] = None,
        sensitive_labels: Optional[Any] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for BERT-based FairNet.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_risk_scores: Return risk scores
            return_features: Return corrected features
            force_lora: Force LoRA activation
            target_attribute: Only use specific attribute detector

        Returns:
            logits: Classification logits [batch, num_classes]
            risk_scores: (optional) Dict of risk scores
            features: (optional) Corrected features
        """
        if (
            self.config.attribute_mode == AttributeMode.FULL
            and sensitive_labels is None
            and not force_lora
        ):
            raise ValueError("FairNet-Full requires sensitive_labels for conditional inference")

        # Step 1: Get intermediate features only when a learned detector is needed.
        self._clear_lora_activation()
        intermediate_features = (
            self.get_intermediate_features(
                input_ids,
                attention_mask,
                token_type_ids=token_type_ids,
            )
            if sensitive_labels is None and not force_lora
            else None
        )

        # Step 2: Compute risk scores
        risk_scores = {}
        if not force_lora:
            for position, attr in enumerate(self.config.sensitive_attributes):
                if target_attribute is not None and attr != target_attribute:
                    continue

                risk_score = _resolve_sensitive_labels(sensitive_labels, attr, position)
                if risk_score is None:
                    risk_score = self.bias_detectors[f"detector_{attr}"](
                        intermediate_features,
                        is_sequence=True,
                        attention_mask=attention_mask,
                    )
                risk_score = risk_score.to(input_ids.device)
                risk_scores[attr] = risk_score

        # Step 3: Second pass with LoRA
        if force_lora:
            forced = (
                [target_attribute]
                if target_attribute is not None
                else list(self.config.sensitive_attributes)
            )
            self._set_lora_activation(forced_attributes=forced)
        elif risk_scores:
            self._set_lora_activation(risk_scores=risk_scores)

        corrected_features = self.get_cls_features(
            input_ids,
            attention_mask,
            use_lora=True,
            token_type_ids=token_type_ids,
        )
        self._clear_lora_activation()

        # Step 4: Classification
        logits = self.classifier(corrected_features)

        if return_risk_scores and return_features:
            return logits, risk_scores, corrected_features
        elif return_risk_scores:
            return logits, risk_scores
        elif return_features:
            return logits, corrected_features
        return logits


class FairNetDistilBERT(FairNetBERT):
    """FairNet with a DistilBERT backbone.

    Supplementary C.2 evaluates the HateXplain experiments of Table 2 on both
    DistilBERT-base and BERT-base. DistilBERT stores its blocks under
    ``transformer.layer``, names its attention projections ``q_lin``/``v_lin``,
    and takes no ``token_type_ids``; everything else is shared with
    :class:`FairNetBERT`.
    """

    def __init__(
        self,
        config: FairNetConfig,
        num_classes: int = 3,
        model_name: str = "distilbert-base-uncased",
        detector_layer: Optional[int] = None,
        bert_config=None,
        bert_model=None,
    ):
        super().__init__(
            config,
            num_classes=num_classes,
            model_name=model_name,
            detector_layer=detector_layer,
            bert_config=bert_config,
            bert_model=bert_model,
        )

    @staticmethod
    def _backbone_class():
        from transformers import DistilBertModel

        return DistilBertModel

    @staticmethod
    def _accepts_token_type_ids() -> bool:
        return False
