"""Training pipelines for the three FairNet label-availability settings."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .config import AttributeMode, FairNetConfig
from .modules import StaticPrototypeBank, TripletContrastiveLoss, UnsupervisedBiasDetector
from .utils import compute_fairness_metrics


def _require_binary_sensitive(sensitive):
    for attr, values in sensitive.items():
        if not torch.all((values == 0) | (values == 1)):
            raise ValueError(f"Sensitive attribute {attr} must be binary and encoded as 0/1")
    return sensitive


class _RelabeledDataset(Dataset):
    """Overlay one sensitive-attribute column without changing the base dataset."""

    def __init__(self, dataset: Dataset, labels: torch.Tensor, attribute: int):
        if len(dataset) != len(labels):
            raise ValueError("Pseudo labels must align one-to-one with the dataset")
        self.dataset = dataset
        self.labels = labels.long().cpu()
        self.attribute = attribute

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        pseudo_label = self.labels[index]
        if isinstance(sample, Mapping):
            result = dict(sample)
            if "attributes" in result:
                attributes = result["attributes"].clone()
                attributes[self.attribute] = pseudo_label
                result["attributes"] = attributes
            else:
                result["sensitive"] = pseudo_label
            return result

        inputs, attributes, *rest = sample
        attributes = attributes.clone()
        attributes[self.attribute] = pseudo_label
        return (inputs, attributes, *rest)


class FairNetTrainer:
    """Trainer for FairNet-Full and shared stages used by other settings.

    Vision loaders may yield ``(images, attributes)``. NLP loaders may yield a
    dictionary containing ``input_ids``, ``attention_mask``, ``labels``, and
    either ``attributes`` or ``sensitive``. In the full-label setting, known
    sensitive labels directly gate the matching LoRA adapter, as specified in
    Supplementary C.3.1 of the paper.
    """

    def __init__(self, model: nn.Module, config: FairNetConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.prototype_banks = {
            attr: StaticPrototypeBank(model.hidden_dim, device)
            for attr in config.sensitive_attributes
        }
        # Compatibility alias used by early examples.
        self.prototype_bank = self.prototype_banks[config.sensitive_attributes[0]]
        self.history = defaultdict(list)

    def _get_warmup_scheduler(self, optimizer, warmup_steps: int):
        def lr_lambda(step):
            if warmup_steps <= 0:
                return 1.0
            return min(1.0, float(step + 1) / float(warmup_steps))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _unpack_batch(self, batch):
        if isinstance(batch, Mapping):
            if "labels" not in batch:
                raise KeyError("NLP batches must contain 'labels'")
            inputs = {
                key: batch[key].to(self.device)
                for key in ("input_ids", "attention_mask", "token_type_ids")
                if key in batch
            }
            if not {"input_ids", "attention_mask"}.issubset(inputs):
                raise KeyError("NLP batches require input_ids and attention_mask")
            labels = batch["labels"].to(self.device)
            if "attributes" in batch:
                attributes = batch["attributes"]
                sensitive = {
                    attr: attributes[:, attr].to(self.device)
                    for attr in self.config.sensitive_attributes
                }
            elif "sensitive" in batch:
                values = batch["sensitive"]
                if values.ndim == 1:
                    if len(self.config.sensitive_attributes) != 1:
                        raise ValueError("One-dimensional sensitive labels support one attribute")
                    sensitive = {self.config.sensitive_attributes[0]: values.to(self.device)}
                else:
                    if values.shape[1] != len(self.config.sensitive_attributes):
                        raise ValueError("Sensitive-label columns must match sensitive_attributes")
                    sensitive = {
                        attr: values[:, position].to(self.device)
                        for position, attr in enumerate(self.config.sensitive_attributes)
                    }
            else:
                raise KeyError("NLP batches require 'attributes' or 'sensitive'")
            return inputs, labels, _require_binary_sensitive(sensitive)

        inputs, attributes = batch[0], batch[1]
        inputs = inputs.to(self.device)
        labels = attributes[:, self.config.target_attribute].to(self.device)
        sensitive = {
            attr: attributes[:, attr].to(self.device) for attr in self.config.sensitive_attributes
        }
        return inputs, labels, _require_binary_sensitive(sensitive)

    @staticmethod
    def _is_text_inputs(inputs) -> bool:
        return isinstance(inputs, Mapping)

    def _get_cls_features(self, inputs, use_lora=False):
        if self._is_text_inputs(inputs):
            return self.model.get_cls_features(**inputs, use_lora=use_lora)
        return self.model.get_cls_features(inputs, use_lora=use_lora)

    def _get_intermediate_features(self, inputs):
        if self._is_text_inputs(inputs):
            return self.model.get_intermediate_features(**inputs)
        return self.model.get_intermediate_features(inputs)

    def _attention_mask(self, inputs):
        return inputs.get("attention_mask") if self._is_text_inputs(inputs) else None

    def _forward(self, inputs, **kwargs):
        if self._is_text_inputs(inputs):
            return self.model(**inputs, **kwargs)
        return self.model(inputs, **kwargs)

    @staticmethod
    def _task_loss(outputs, labels):
        if outputs.shape[-1] == 1:
            return nn.functional.binary_cross_entropy(outputs, labels.float().reshape(-1, 1))
        return nn.functional.cross_entropy(outputs, labels.long())

    @staticmethod
    def _predictions(outputs):
        if outputs.shape[-1] == 1:
            return (outputs.reshape(-1) > 0.5).long()
        return outputs.argmax(dim=-1)

    def _base_outputs(self, inputs):
        self.model._clear_lora_activation()
        return self.model.classifier(self._get_cls_features(inputs, use_lora=False))

    def stage1_train_base(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> Dict:
        print("\n" + "=" * 60)
        print("Stage 1: Training Base Model (ERM)")
        print("=" * 60)
        if len(train_loader) == 0:
            raise ValueError("train_loader is empty")

        self.model.unfreeze_base()
        self.model.freeze_detectors()
        self.model.freeze_lora()
        self.model._clear_lora_activation()
        parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        optimizer = optim.AdamW(
            parameters,
            lr=self.config.stage1_lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = self._get_warmup_scheduler(optimizer, self.config.warmup_steps)
        best_wga = float("-inf")
        best_state = None

        for epoch in range(self.config.stage1_epochs):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(
                train_loader,
                desc=f"Stage 1 epoch {epoch + 1}/{self.config.stage1_epochs}",
            ):
                inputs, labels, _ = self._unpack_batch(batch)
                optimizer.zero_grad()
                outputs = self._base_outputs(inputs)
                loss = self._task_loss(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            average_loss = total_loss / len(train_loader)
            self.history["stage1_loss"].append(average_loss)
            if val_loader is not None:
                metrics = self._evaluate(val_loader, use_lora=False)
                self.history["stage1_wga"].append(metrics["worst_group_accuracy"])
                print(
                    f"Epoch {epoch + 1}: loss={average_loss:.4f}, "
                    f"ACC={metrics['accuracy']:.4f}, "
                    f"WGA={metrics['worst_group_accuracy']:.4f}"
                )
                if metrics["worst_group_accuracy"] > best_wga:
                    best_wga = metrics["worst_group_accuracy"]
                    best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.freeze_base()
        return {"best_wga": None if best_wga == float("-inf") else best_wga}

    def _train_detectors(self, train_loader: DataLoader) -> Dict:
        if len(train_loader) == 0:
            raise ValueError("Detector training loader is empty")
        self.model.freeze_base()
        self.model.unfreeze_detectors()
        self.model.freeze_lora()
        self.model._clear_lora_activation()
        metrics = {}

        for attr in self.config.sensitive_attributes:
            detector = self.model.bias_detectors[f"detector_{attr}"]
            optimizer = optim.Adam(detector.parameters(), lr=self.config.stage2_lr)
            positives = 0
            total_samples = 0
            for batch in train_loader:
                _, _, sensitive = self._unpack_batch(batch)
                positives += int(sensitive[attr].sum().item())
                total_samples += len(sensitive[attr])
            negatives = total_samples - positives
            if positives == 0 or negatives == 0:
                raise ValueError(f"Detector {attr} requires both binary sensitive groups")
            positive_weight = torch.tensor(
                negatives / positives, device=self.device, dtype=torch.float32
            )
            for epoch in range(self.config.stage2_epochs):
                self.model.train()
                correct = 0
                total = 0
                for batch in tqdm(
                    train_loader,
                    desc=f"Detector {attr} epoch {epoch + 1}/{self.config.stage2_epochs}",
                ):
                    inputs, _, sensitive = self._unpack_batch(batch)
                    targets = sensitive[attr].float().reshape(-1, 1)
                    optimizer.zero_grad()
                    with torch.no_grad():
                        hidden_states = self._get_intermediate_features(inputs)
                    logits = detector.forward_logits(
                        hidden_states,
                        is_sequence=True,
                        attention_mask=self._attention_mask(inputs),
                    )
                    loss = self.config.lambda_D * nn.functional.binary_cross_entropy_with_logits(
                        logits, targets, pos_weight=positive_weight
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(detector.parameters(), self.config.gradient_clip)
                    optimizer.step()
                    correct += ((logits.sigmoid() > 0.5) == targets.bool()).sum().item()
                    total += len(targets)
                print(f"Detector {attr} epoch {epoch + 1}: ACC={correct / total:.4f}")
            metrics.update(self._evaluate_detector(train_loader, attr))

        self.model.freeze_detectors()
        return metrics

    def stage2_train_detector(self, train_loader: DataLoader) -> Dict:
        print("\n" + "=" * 60)
        if self.config.attribute_mode == AttributeMode.FULL:
            print("Stage 2: Direct sensitive-label gating (FairNet-Full)")
            print("=" * 60)
            self.model.freeze_detectors()
            return {"direct_sensitive_gating": True}
        print("Stage 2: Training Bias Detector(s)")
        print("=" * 60)
        return self._train_detectors(train_loader)

    def _evaluate_detector(self, loader: DataLoader, attr: int) -> Dict:
        self.model.eval()
        detector = self.model.bias_detectors[f"detector_{attr}"]
        predictions = []
        targets = []
        with torch.no_grad():
            for batch in loader:
                inputs, _, sensitive = self._unpack_batch(batch)
                hidden_states = self._get_intermediate_features(inputs)
                scores = detector(
                    hidden_states,
                    is_sequence=True,
                    attention_mask=self._attention_mask(inputs),
                )
                predictions.extend(
                    (scores > self.config.activation_threshold).cpu().numpy().ravel()
                )
                targets.extend(sensitive[attr].cpu().numpy().ravel())

        predictions = np.asarray(predictions, dtype=bool)
        targets = np.asarray(targets)
        minority = targets == 1
        majority = targets == 0
        tpr = predictions[minority].mean() if minority.any() else 0.0
        fpr = predictions[majority].mean() if majority.any() else 0.0
        return {
            f"tpr_{attr}": float(tpr),
            f"fpr_{attr}": float(fpr),
            f"tpr_fpr_ratio_{attr}": float(tpr / (fpr + 1e-8)),
        }

    def stage3_build_prototypes(self, train_loader: DataLoader):
        print("\n" + "=" * 60)
        print("Stage 3: Building Static Contrastive Prototypes")
        print("=" * 60)
        self.model.eval()
        self.model._clear_lora_activation()

        sums = {attr: {} for attr in self.config.sensitive_attributes}
        counts = {attr: {} for attr in self.config.sensitive_attributes}
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Building prototypes"):
                inputs, labels, sensitive = self._unpack_batch(batch)
                features = self._get_cls_features(inputs, use_lora=False)
                for attr in self.config.sensitive_attributes:
                    for index in range(len(labels)):
                        task_class = int(labels[index].item())
                        group = int(sensitive[attr][index].item())
                        sums[attr].setdefault(task_class, {})
                        counts[attr].setdefault(task_class, {})
                        if group not in sums[attr][task_class]:
                            sums[attr][task_class][group] = torch.zeros(
                                self.model.hidden_dim, device=self.device
                            )
                            counts[attr][task_class][group] = 0
                        sums[attr][task_class][group] += features[index]
                        counts[attr][task_class][group] += 1

        for attr, bank in self.prototype_banks.items():
            bank.prototypes = {
                task_class: {
                    group: vector / counts[attr][task_class][group]
                    for group, vector in groups.items()
                }
                for task_class, groups in sums[attr].items()
            }
            bank.counts = counts[attr]
            if len(bank.prototypes) < 2:
                raise RuntimeError(
                    f"Attribute {attr} has fewer than two task classes in prototypes"
                )

    def stage4_train_lora(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> Dict:
        print("\n" + "=" * 60)
        print("Stage 4: Training Attribute-Specific Conditional LoRA")
        print("=" * 60)
        if len(train_loader) == 0:
            raise ValueError("LoRA training loader is empty")

        self.model.freeze_base()
        self.model.freeze_detectors()
        self.model.unfreeze_lora()
        lora_parameters = self.model.get_lora_parameters()
        if not lora_parameters:
            raise RuntimeError("No LoRA parameters were injected")
        optimizer = optim.Adam(lora_parameters, lr=self.config.stage4_lr)
        scheduler = self._get_warmup_scheduler(optimizer, min(50, self.config.warmup_steps))
        contrastive = TripletContrastiveLoss(margin=self.config.contrastive_margin)
        best_wga = float("-inf")
        best_state = None

        for epoch in range(self.config.stage4_epochs):
            self.model.train()
            total_loss = 0.0
            updated_batches = 0
            for batch in tqdm(
                train_loader,
                desc=f"Stage 4 epoch {epoch + 1}/{self.config.stage4_epochs}",
            ):
                inputs, labels, sensitive = self._unpack_batch(batch)
                attribute_losses = []
                for attr in self.config.sensitive_attributes:
                    minority = sensitive[attr] == 1
                    if not minority.any():
                        continue
                    if self._is_text_inputs(inputs):
                        minority_inputs = {key: value[minority] for key, value in inputs.items()}
                    else:
                        minority_inputs = inputs[minority]
                    _, corrected = self._forward(
                        minority_inputs,
                        return_features=True,
                        force_lora=True,
                        target_attribute=attr,
                    )
                    positive, negative = self.prototype_banks[attr].get_targets(
                        labels[minority], anchor_features=corrected
                    )
                    attribute_losses.append(contrastive(corrected, positive, negative))

                if not attribute_losses:
                    continue
                optimizer.zero_grad()
                loss = self.config.lambda_C * torch.stack(attribute_losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_parameters, self.config.gradient_clip)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                updated_batches += 1

            if updated_batches == 0:
                raise RuntimeError("No minority samples reached Stage 4; check labels and batching")
            average_loss = total_loss / updated_batches
            self.history["stage4_loss"].append(average_loss)
            if val_loader is not None:
                metrics = self._evaluate(val_loader, use_lora=True)
                self.history["stage4_wga"].append(metrics["worst_group_accuracy"])
                print(
                    f"Epoch {epoch + 1}: loss={average_loss:.4f}, "
                    f"ACC={metrics['accuracy']:.4f}, "
                    f"WGA={metrics['worst_group_accuracy']:.4f}"
                )
                if metrics["worst_group_accuracy"] > best_wga:
                    best_wga = metrics["worst_group_accuracy"]
                    best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self._evaluate(val_loader, use_lora=True) if val_loader is not None else {}

    def train_full(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        if self.config.attribute_mode != AttributeMode.FULL:
            raise ValueError(
                "FairNetTrainer requires attribute_mode=FULL; use the partial "
                "or unlabeled trainer for other modes"
            )
        self.stage1_train_base(train_loader, val_loader)
        self.stage2_train_detector(train_loader)
        self.stage3_build_prototypes(train_loader)
        return self.stage4_train_lora(train_loader, val_loader)

    @staticmethod
    def _fairness_metrics(labels, predictions, sensitive):
        return compute_fairness_metrics(labels, predictions, sensitive)

    def _evaluate(self, loader: DataLoader, use_lora: bool = True) -> Dict:
        if loader is None or len(loader) == 0:
            raise ValueError("Evaluation loader is empty")
        self.model.eval()
        labels_all = []
        predictions_all = []
        sensitive_all = {attr: [] for attr in self.config.sensitive_attributes}
        risk_all = {attr: [] for attr in self.config.sensitive_attributes}

        with torch.no_grad():
            for batch in loader:
                inputs, labels, sensitive = self._unpack_batch(batch)
                if use_lora:
                    direct_labels = (
                        sensitive if self.config.attribute_mode == AttributeMode.FULL else None
                    )
                    outputs, risk_scores = self._forward(
                        inputs,
                        return_risk_scores=True,
                        sensitive_labels=direct_labels,
                    )
                    for attr, scores in risk_scores.items():
                        risk_all[attr].extend(scores.cpu().numpy().ravel())
                else:
                    outputs = self._base_outputs(inputs)
                predictions = self._predictions(outputs)
                labels_all.extend(labels.cpu().numpy().ravel())
                predictions_all.extend(predictions.cpu().numpy().ravel())
                for attr in self.config.sensitive_attributes:
                    sensitive_all[attr].extend(sensitive[attr].cpu().numpy().ravel())

        per_attribute = {}
        for attr in self.config.sensitive_attributes:
            attr_metrics = self._fairness_metrics(labels_all, predictions_all, sensitive_all[attr])
            per_attribute[attr] = attr_metrics

        primary = self.config.sensitive_attributes[0]
        metrics = dict(per_attribute[primary])
        metrics["per_attribute"] = per_attribute
        if use_lora:
            for attr, scores in risk_all.items():
                metrics[f"lora_activation_rate_{attr}"] = float(
                    np.mean(np.asarray(scores) > self.config.activation_threshold)
                )
            metrics["lora_activation_rate"] = metrics[f"lora_activation_rate_{primary}"]
        return metrics


class FairNetPartialTrainer(FairNetTrainer):
    """FairNet-Partial: use all task labels and only a fraction of group labels."""

    def __init__(
        self,
        model: nn.Module,
        config: FairNetConfig,
        device: torch.device,
        labeled_fraction: Optional[float] = None,
    ):
        super().__init__(model, config, device)
        if config.attribute_mode != AttributeMode.PARTIAL:
            raise ValueError("FairNetPartialTrainer requires attribute_mode=PARTIAL")
        self.labeled_fraction = (
            config.labeled_fraction if labeled_fraction is None else labeled_fraction
        )
        if not 0 < self.labeled_fraction < 1:
            raise ValueError("labeled_fraction must be in (0, 1) for partial training")
        self.labeled_indices: Optional[List[int]] = None

    def _get_labeled_indices(self, dataset_size: int) -> List[int]:
        if dataset_size == 0:
            raise ValueError("Cannot sample labels from an empty dataset")
        generator = np.random.default_rng(self.config.seed)
        count = max(1, round(dataset_size * self.labeled_fraction))
        return generator.choice(dataset_size, count, replace=False).tolist()

    def _create_labeled_loader(self, train_loader: DataLoader) -> DataLoader:
        if self.labeled_indices is None:
            self.labeled_indices = self._get_labeled_indices(len(train_loader.dataset))
        return DataLoader(
            Subset(train_loader.dataset, self.labeled_indices),
            batch_size=train_loader.batch_size or self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=train_loader.num_workers,
            collate_fn=train_loader.collate_fn,
            pin_memory=train_loader.pin_memory,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def train_full(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        self.stage1_train_base(train_loader, val_loader)
        labeled_loader = self._create_labeled_loader(train_loader)
        print(
            f"Using {len(self.labeled_indices)} sensitive-labeled samples "
            f"({self.labeled_fraction:.1%})"
        )
        self.stage2_train_detector(labeled_loader)
        self.stage3_build_prototypes(labeled_loader)
        return self.stage4_train_lora(labeled_loader, val_loader)


class FairNetUnlabeledTrainer(FairNetTrainer):
    """FairNet-Unlabeled using LOF pseudo labels followed by detector training."""

    def __init__(self, model: nn.Module, config: FairNetConfig, device: torch.device):
        super().__init__(model, config, device)
        if config.attribute_mode != AttributeMode.UNLABELED:
            raise ValueError("FairNetUnlabeledTrainer requires attribute_mode=UNLABELED")
        if len(config.sensitive_attributes) != 1:
            raise ValueError("Unlabeled training currently requires one sensitive attribute")
        self.unsupervised_detector = UnsupervisedBiasDetector(
            hidden_dim=model.hidden_dim,
            n_neighbors=config.lof_n_neighbors,
            contamination=config.lof_contamination,
        )
        self.pseudo_labels: Optional[torch.Tensor] = None
        self._pseudo_loader: Optional[DataLoader] = None

    def _stable_loader(self, train_loader: DataLoader) -> DataLoader:
        return DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size or self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=train_loader.num_workers,
            collate_fn=train_loader.collate_fn,
            pin_memory=train_loader.pin_memory,
        )

    def stage2_generate_pseudo_labels(self, train_loader: DataLoader) -> Dict:
        print("\n" + "=" * 60)
        print("Stage 2: LOF pseudo labels and detector training")
        print("=" * 60)
        stable_loader = self._stable_loader(train_loader)
        self.model.eval()
        features = []
        with torch.no_grad():
            for batch in tqdm(stable_loader, desc="Extracting stable-order features"):
                inputs, _, _ = self._unpack_batch(batch)
                hidden_states = self._get_intermediate_features(inputs)
                features.append(hidden_states[:, 0].cpu())
        all_features = torch.cat(features)
        self.pseudo_labels = self.unsupervised_detector.fit_predict(all_features).cpu()

        attr = self.config.sensitive_attributes[0]
        relabeled = _RelabeledDataset(train_loader.dataset, self.pseudo_labels, attribute=attr)
        self._pseudo_loader = DataLoader(
            relabeled,
            batch_size=train_loader.batch_size or self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=train_loader.num_workers,
            collate_fn=train_loader.collate_fn,
            pin_memory=train_loader.pin_memory,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        detector_metrics = self._train_detectors(self._pseudo_loader)
        minority_count = int(self.pseudo_labels.sum().item())
        detector_metrics.update(
            {
                "pseudo_minority_count": minority_count,
                "pseudo_minority_rate": minority_count / len(self.pseudo_labels),
            }
        )
        return detector_metrics

    def train_full(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict:
        self.stage1_train_base(train_loader, val_loader)
        self.stage2_generate_pseudo_labels(train_loader)
        self.stage3_build_prototypes(self._pseudo_loader)
        return self.stage4_train_lora(self._pseudo_loader, val_loader)
