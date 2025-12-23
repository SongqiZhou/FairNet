"""
FairNet Trainers

Contains trainers for different settings:
- FairNetTrainer: Full setting (all sensitive labels available)
- FairNetPartialTrainer: Partial setting (k% labels available)
- FairNetUnlabeledTrainer: Unlabeled setting (no sensitive labels)

All trainers implement the four-stage training pipeline from Figure 1:
1. Train base model with ERM
2. Train bias detector
3. Build contrastive prototypes
4. Train LoRA with contrastive loss
"""

import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .config import FairNetConfig
from .modules import (
    StaticPrototypeBank,
    TripletContrastiveLoss,
    UnsupervisedBiasDetector,
)


class FairNetTrainer:
    """
    Trainer for FairNet-Full setting.
    
    All sensitive attribute labels are available during training.
    This is the standard training procedure from Section 3.4.
    
    Four-stage pipeline:
    1. Stage 1: Train base model with ERM
    2. Stage 2: Train bias detector with class-balanced BCE
    3. Stage 3: Build static prototypes from frozen base model
    4. Stage 4: Train LoRA with contrastive loss on minority samples
    
    Args:
        model: FairNetViT or FairNetBERT model
        config: FairNetConfig
        device: torch.device
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: FairNetConfig,
        device: torch.device
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.prototype_bank = StaticPrototypeBank(model.hidden_dim, device)
        self.history = defaultdict(list)
    
    def _get_warmup_scheduler(self, optimizer, num_steps: int, warmup_steps: int):
        """Create linear warmup scheduler."""
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def stage1_train_base(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        Stage 1: Train base model with ERM.
        
        - LoRA is disabled
        - Only base model and classifier are trained
        - Uses standard BCE loss
        """
        print("\n" + "=" * 60)
        print("Stage 1: Training Base Model (ERM)")
        print("=" * 60)
        
        self.model.unfreeze_base()
        self.model.freeze_detectors()
        self.model.freeze_lora()
        self.model._clear_lora_activation()
        
        # Get non-LoRA parameters
        base_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and 'lora_' not in n
        ]
        
        optimizer = optim.AdamW(
            base_params,
            lr=self.config.stage1_lr,
            weight_decay=self.config.weight_decay
        )
        num_steps = len(train_loader) * self.config.stage1_epochs
        scheduler = self._get_warmup_scheduler(optimizer, num_steps, self.config.warmup_steps)
        criterion = nn.BCELoss()
        
        best_wga = 0
        best_state = None
        
        for epoch in range(self.config.stage1_epochs):
            self.model.train()
            total_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.stage1_epochs}")
            for batch in pbar:
                images, attributes = batch[0], batch[1]
                images = images.to(self.device)
                targets = attributes[:, self.config.target_attribute].float().unsqueeze(1).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward without LoRA
                self.model._clear_lora_activation()
                cls_features = self.model.get_cls_features(images, use_lora=False)
                outputs = self.model.classifier(cls_features)
                
                loss = criterion(outputs, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            self.history['stage1_loss'].append(avg_loss)
            
            if val_loader:
                metrics = self._evaluate(val_loader, use_lora=False)
                self.history['stage1_wga'].append(metrics['worst_group_accuracy'])
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={metrics['accuracy']:.4f}, WGA={metrics['worst_group_accuracy']:.4f}")
                
                if metrics['worst_group_accuracy'] > best_wga:
                    best_wga = metrics['worst_group_accuracy']
                    best_state = copy.deepcopy(self.model.state_dict())
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        self.model.freeze_base()
        return {'best_wga': best_wga}
    
    def stage2_train_detector(
        self,
        train_loader: DataLoader
    ) -> Dict:
        """
        Stage 2: Train bias detector(s).
        
        - Base model is frozen
        - Train detector to predict sensitive attribute
        - Uses class-balanced BCE loss (Equation 1)
        """
        print("\n" + "=" * 60)
        print("Stage 2: Training Bias Detector(s)")
        print("=" * 60)
        
        self.model.freeze_base()
        self.model.unfreeze_detectors()
        self.model.freeze_lora()
        self.model._clear_lora_activation()
        
        metrics = {}
        
        for attr in self.config.sensitive_attributes:
            print(f"\nTraining detector for attribute {attr}...")
            detector = self.model.bias_detectors[f"detector_{attr}"]
            optimizer = optim.Adam(detector.parameters(), lr=self.config.stage2_lr)
            
            for epoch in range(self.config.stage2_epochs):
                self.model.train()
                correct = 0
                total = 0
                
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.stage2_epochs}")
                for batch in pbar:
                    images, attributes = batch[0], batch[1]
                    images = images.to(self.device)
                    sensitive = attributes[:, attr].float().unsqueeze(1).to(self.device)
                    
                    optimizer.zero_grad()
                    
                    with torch.no_grad():
                        hidden_states = self.model.get_intermediate_features(images)
                    
                    risk_score = detector(hidden_states, is_sequence=True)
                    
                    # Class-balanced BCE (Equation 1)
                    pos_count = (sensitive == 1).sum().clamp(min=1)
                    neg_count = (sensitive == 0).sum().clamp(min=1)
                    pos_weight = neg_count / pos_count
                    
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    logits = torch.log(risk_score / (1 - risk_score + 1e-8) + 1e-8)
                    loss = criterion(logits, sensitive)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(detector.parameters(), self.config.gradient_clip)
                    optimizer.step()
                    
                    pred = (risk_score > 0.5).float()
                    correct += (pred == sensitive).sum().item()
                    total += sensitive.size(0)
                    
                    pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
                
                print(f"Epoch {epoch+1}: Detector Accuracy = {correct/total:.4f}")
            
            # Evaluate detector
            detector_metrics = self._evaluate_detector(train_loader, attr)
            metrics.update(detector_metrics)
        
        self.model.freeze_detectors()
        return metrics
    
    def _evaluate_detector(self, loader: DataLoader, attr: int) -> Dict:
        """Evaluate detector TPR and FPR."""
        self.model.eval()
        detector = self.model.bias_detectors[f"detector_{attr}"]
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                images, attributes = batch[0], batch[1]
                images = images.to(self.device)
                hidden_states = self.model.get_intermediate_features(images)
                risk_score = detector(hidden_states, is_sequence=True)
                
                preds = (risk_score > self.config.activation_threshold).cpu().numpy().flatten()
                targets = attributes[:, attr].numpy()
                
                all_preds.extend(preds)
                all_targets.extend(targets)
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Compute TPR and FPR
        minority_mask = all_targets == 1
        majority_mask = all_targets == 0
        
        tpr = all_preds[minority_mask].mean() if minority_mask.sum() > 0 else 0
        fpr = all_preds[majority_mask].mean() if majority_mask.sum() > 0 else 0
        
        print(f"\nDetector {attr} - TPR: {tpr:.4f}, FPR: {fpr:.4f}, TPR/FPR: {tpr/(fpr+1e-8):.2f}")
        
        return {
            f'tpr_{attr}': tpr,
            f'fpr_{attr}': fpr,
            f'tpr_fpr_ratio_{attr}': tpr / (fpr + 1e-8)
        }
    
    def stage3_build_prototypes(self, train_loader: DataLoader):
        """
        Stage 3: Build contrastive pair prototypes.
        
        - Uses frozen base model
        - Computes average embeddings per (class, group)
        """
        print("\n" + "=" * 60)
        print("Stage 3: Building Contrastive Pair Prototypes")
        print("=" * 60)
        
        self.model._clear_lora_activation()
        
        def get_features(model, images):
            return model.get_cls_features(images, use_lora=False)
        
        self.prototype_bank.compute_from_loader(
            self.model,
            train_loader,
            get_features,
            self.config.target_attribute,
            self.config.sensitive_attributes[0]
        )
    
    def stage4_train_lora(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        Stage 4: Train LoRA with contrastive loss.
        
        - Base model and detector are frozen
        - Only LoRA parameters are trained
        - Uses triplet contrastive loss (Equation 2)
        - Only trains on minority samples
        """
        print("\n" + "=" * 60)
        print("Stage 4: Training LoRA with Contrastive Loss")
        print("=" * 60)
        
        self.model.freeze_base()
        self.model.freeze_detectors()
        self.model.unfreeze_lora()
        
        lora_params = self.model.get_lora_parameters()
        print(f"Training {len(lora_params)} LoRA parameter tensors")
        
        optimizer = optim.Adam(lora_params, lr=self.config.stage4_lr)
        num_steps = len(train_loader) * self.config.stage4_epochs
        scheduler = self._get_warmup_scheduler(optimizer, num_steps, 50)
        contrastive_loss_fn = TripletContrastiveLoss(margin=self.config.contrastive_margin)
        
        best_wga = 0
        best_state = None
        
        for epoch in range(self.config.stage4_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.stage4_epochs}")
            for batch in pbar:
                images, attributes = batch[0], batch[1]
                images = images.to(self.device)
                labels = attributes[:, self.config.target_attribute].to(self.device)
                sensitive = attributes[:, self.config.sensitive_attributes[0]].to(self.device)
                
                # Only train on minority samples
                minority_mask = (sensitive == 1)
                if minority_mask.sum() < 2:
                    continue
                
                optimizer.zero_grad()
                
                # Forward with forced LoRA
                _, corrected_features = self.model(
                    images[minority_mask],
                    return_features=True,
                    force_lora=True
                )
                
                # Get prototype targets
                pos_targets, neg_targets = self.prototype_bank.get_targets(labels[minority_mask])
                
                # Contrastive loss
                c_loss = contrastive_loss_fn(corrected_features, pos_targets, neg_targets)
                loss = self.config.lambda_C * c_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, self.config.gradient_clip)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                self.history['stage4_loss'].append(avg_loss)
            
            if val_loader and num_batches > 0:
                metrics = self._evaluate(val_loader, use_lora=True)
                self.history['stage4_wga'].append(metrics['worst_group_accuracy'])
                print(f"Epoch {epoch+1}: Acc={metrics['accuracy']:.4f}, WGA={metrics['worst_group_accuracy']:.4f}")
                
                if metrics['worst_group_accuracy'] > best_wga:
                    best_wga = metrics['worst_group_accuracy']
                    best_state = copy.deepcopy(self.model.state_dict())
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        return self._evaluate(val_loader, use_lora=True) if val_loader else {}
    
    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Execute full four-stage training pipeline."""
        self.stage1_train_base(train_loader, val_loader)
        self.stage2_train_detector(train_loader)
        self.stage3_build_prototypes(train_loader)
        return self.stage4_train_lora(train_loader, val_loader)
    
    def _evaluate(self, loader: DataLoader, use_lora: bool = True) -> Dict:
        """Evaluate model with group-wise metrics."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_sensitive = []
        all_risk_scores = []
        
        groups = {
            'group_0_0': {'preds': [], 'labels': []},
            'group_0_1': {'preds': [], 'labels': []},
            'group_1_0': {'preds': [], 'labels': []},
            'group_1_1': {'preds': [], 'labels': []},
        }
        
        with torch.no_grad():
            for batch in loader:
                images, attributes = batch[0], batch[1]
                images = images.to(self.device)
                sensitive = attributes[:, self.config.sensitive_attributes[0]].numpy()
                labels = attributes[:, self.config.target_attribute].numpy()
                
                if use_lora:
                    outputs, risk_scores = self.model(images, return_risk_scores=True)
                    rs = risk_scores[self.config.sensitive_attributes[0]].cpu().numpy().flatten()
                    all_risk_scores.extend(rs)
                else:
                    self.model._clear_lora_activation()
                    cls_features = self.model.get_cls_features(images, use_lora=False)
                    outputs = self.model.classifier(cls_features)
                
                preds = (outputs > 0.5).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_sensitive.extend(sensitive)
                
                for i in range(len(labels)):
                    key = f"group_{int(sensitive[i])}_{int(labels[i])}"
                    groups[key]['preds'].append(preds[i])
                    groups[key]['labels'].append(labels[i])
        
        metrics = {'accuracy': accuracy_score(all_labels, all_preds)}
        
        # Group accuracies
        group_accs = []
        for name, data in groups.items():
            if data['labels']:
                acc = accuracy_score(data['labels'], data['preds'])
                metrics[f'acc_{name}'] = acc
                group_accs.append(acc)
        
        metrics['worst_group_accuracy'] = min(group_accs) if group_accs else 0
        metrics['accuracy_gap'] = max(group_accs) - min(group_accs) if group_accs else 0
        
        # Fairness metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_sensitive = np.array(all_sensitive)
        
        # TPR/FPR per group
        tpr_maj = np.mean(all_preds[(all_labels == 1) & (all_sensitive == 0)]) if ((all_labels == 1) & (all_sensitive == 0)).sum() > 0 else 0
        tpr_min = np.mean(all_preds[(all_labels == 1) & (all_sensitive == 1)]) if ((all_labels == 1) & (all_sensitive == 1)).sum() > 0 else 0
        fpr_maj = np.mean(all_preds[(all_labels == 0) & (all_sensitive == 0)]) if ((all_labels == 0) & (all_sensitive == 0)).sum() > 0 else 0
        fpr_min = np.mean(all_preds[(all_labels == 0) & (all_sensitive == 1)]) if ((all_labels == 0) & (all_sensitive == 1)).sum() > 0 else 0
        
        metrics['EOD'] = 0.5 * (abs(tpr_maj - tpr_min) + abs(fpr_maj - fpr_min))
        metrics['EOp'] = abs(tpr_maj - tpr_min)
        
        if all_risk_scores:
            metrics['lora_activation_rate'] = np.mean(np.array(all_risk_scores) > self.config.activation_threshold)
        
        return metrics


class FairNetPartialTrainer(FairNetTrainer):
    """
    Trainer for FairNet-Partial setting (Section 5.1).
    
    Only k% of samples have sensitive attribute labels.
    - Stage 1: Uses all data (target labels available)
    - Stage 2: Uses only labeled subset for detector training
    - Stage 3: Uses only labeled subset for prototype building
    - Stage 4: Uses only labeled minority samples for LoRA training
    
    Args:
        model: FairNetViT or FairNetBERT model
        config: FairNetConfig (must have attribute_mode=PARTIAL)
        device: torch.device
        labeled_fraction: Fraction of samples with sensitive labels
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: FairNetConfig,
        device: torch.device,
        labeled_fraction: Optional[float] = None
    ):
        super().__init__(model, config, device)
        self.labeled_fraction = labeled_fraction or config.labeled_fraction
        self.labeled_indices: Optional[List[int]] = None
    
    def _get_labeled_indices(self, dataset_size: int) -> List[int]:
        """Randomly select indices for labeled samples."""
        np.random.seed(42)  # For reproducibility
        num_labeled = int(dataset_size * self.labeled_fraction)
        indices = np.random.choice(dataset_size, num_labeled, replace=False)
        return indices.tolist()
    
    def _create_labeled_loader(
        self,
        train_loader: DataLoader,
        labeled_indices: List[int]
    ) -> DataLoader:
        """Create a DataLoader with only labeled samples."""
        labeled_subset = Subset(train_loader.dataset, labeled_indices)
        return DataLoader(
            labeled_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=getattr(train_loader, 'num_workers', 0)
        )
    
    def stage2_train_detector(self, train_loader: DataLoader) -> Dict:
        """
        Stage 2 for Partial: Train detector using ONLY labeled samples.
        """
        print("\n" + "=" * 60)
        print(f"Stage 2: Training Bias Detector (Partial: {self.labeled_fraction*100:.1f}% labeled)")
        print("=" * 60)
        
        # Get labeled indices if not already set
        if self.labeled_indices is None:
            self.labeled_indices = self._get_labeled_indices(len(train_loader.dataset))
        
        print(f"Using {len(self.labeled_indices)} labeled samples ({self.labeled_fraction*100:.1f}%)")
        
        # Create labeled subset loader
        labeled_loader = self._create_labeled_loader(train_loader, self.labeled_indices)
        
        # Use parent's detector training on labeled subset
        self.model.freeze_base()
        self.model.unfreeze_detectors()
        self.model.freeze_lora()
        self.model._clear_lora_activation()
        
        metrics = {}
        
        for attr in self.config.sensitive_attributes:
            print(f"\nTraining detector for attribute {attr}...")
            detector = self.model.bias_detectors[f"detector_{attr}"]
            optimizer = optim.Adam(detector.parameters(), lr=self.config.stage2_lr)
            
            for epoch in range(self.config.stage2_epochs):
                self.model.train()
                correct = 0
                total = 0
                
                pbar = tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{self.config.stage2_epochs}")
                for batch in pbar:
                    images, attributes = batch[0], batch[1]
                    images = images.to(self.device)
                    sensitive = attributes[:, attr].float().unsqueeze(1).to(self.device)
                    
                    optimizer.zero_grad()
                    
                    with torch.no_grad():
                        hidden_states = self.model.get_intermediate_features(images)
                    
                    risk_score = detector(hidden_states, is_sequence=True)
                    
                    pos_count = (sensitive == 1).sum().clamp(min=1)
                    neg_count = (sensitive == 0).sum().clamp(min=1)
                    pos_weight = neg_count / pos_count
                    
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    logits = torch.log(risk_score / (1 - risk_score + 1e-8) + 1e-8)
                    loss = criterion(logits, sensitive)
                    
                    loss.backward()
                    optimizer.step()
                    
                    pred = (risk_score > 0.5).float()
                    correct += (pred == sensitive).sum().item()
                    total += sensitive.size(0)
                    pbar.set_postfix({'acc': f'{correct/total:.4f}'})
                
                print(f"Epoch {epoch+1}: Detector Accuracy = {correct/total:.4f}")
            
            # Evaluate on full data
            detector_metrics = self._evaluate_detector(train_loader, attr)
            metrics.update(detector_metrics)
        
        self.model.freeze_detectors()
        return metrics
    
    def stage3_build_prototypes(self, train_loader: DataLoader):
        """Stage 3 for Partial: Build prototypes using labeled samples only."""
        print("\n" + "=" * 60)
        print("Stage 3: Building Prototypes (using labeled samples)")
        print("=" * 60)
        
        if self.labeled_indices is None:
            self.labeled_indices = self._get_labeled_indices(len(train_loader.dataset))
        
        labeled_loader = self._create_labeled_loader(train_loader, self.labeled_indices)
        
        self.model._clear_lora_activation()
        
        def get_features(model, images):
            return model.get_cls_features(images, use_lora=False)
        
        self.prototype_bank.compute_from_loader(
            self.model,
            labeled_loader,
            get_features,
            self.config.target_attribute,
            self.config.sensitive_attributes[0]
        )
    
    def stage4_train_lora(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Stage 4 for Partial: Train LoRA using labeled minority samples."""
        print("\n" + "=" * 60)
        print("Stage 4: Training LoRA (using labeled samples)")
        print("=" * 60)
        
        if self.labeled_indices is None:
            self.labeled_indices = self._get_labeled_indices(len(train_loader.dataset))
        
        labeled_loader = self._create_labeled_loader(train_loader, self.labeled_indices)
        
        # Use parent's LoRA training on labeled subset
        self.model.freeze_base()
        self.model.freeze_detectors()
        self.model.unfreeze_lora()
        
        lora_params = self.model.get_lora_parameters()
        optimizer = optim.Adam(lora_params, lr=self.config.stage4_lr)
        scheduler = self._get_warmup_scheduler(
            optimizer,
            len(labeled_loader) * self.config.stage4_epochs,
            50
        )
        contrastive_loss_fn = TripletContrastiveLoss(margin=self.config.contrastive_margin)
        
        best_wga = 0
        best_state = None
        
        for epoch in range(self.config.stage4_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(labeled_loader, desc=f"Epoch {epoch+1}/{self.config.stage4_epochs}")
            for batch in pbar:
                images, attributes = batch[0], batch[1]
                images = images.to(self.device)
                labels = attributes[:, self.config.target_attribute].to(self.device)
                sensitive = attributes[:, self.config.sensitive_attributes[0]].to(self.device)
                
                minority_mask = (sensitive == 1)
                if minority_mask.sum() < 2:
                    continue
                
                optimizer.zero_grad()
                
                _, corrected_features = self.model(
                    images[minority_mask],
                    return_features=True,
                    force_lora=True
                )
                
                pos_targets, neg_targets = self.prototype_bank.get_targets(labels[minority_mask])
                c_loss = contrastive_loss_fn(corrected_features, pos_targets, neg_targets)
                loss = self.config.lambda_C * c_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, self.config.gradient_clip)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if val_loader and num_batches > 0:
                metrics = self._evaluate(val_loader, use_lora=True)
                print(f"Epoch {epoch+1}: Acc={metrics['accuracy']:.4f}, WGA={metrics['worst_group_accuracy']:.4f}")
                
                if metrics['worst_group_accuracy'] > best_wga:
                    best_wga = metrics['worst_group_accuracy']
                    best_state = copy.deepcopy(self.model.state_dict())
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        return self._evaluate(val_loader, use_lora=True) if val_loader else {}
    
    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Execute full training with partial labels."""
        # Initialize labeled indices
        self.labeled_indices = self._get_labeled_indices(len(train_loader.dataset))
        print(f"\nUsing {len(self.labeled_indices)} labeled samples ({self.labeled_fraction*100:.1f}%)")
        
        self.stage1_train_base(train_loader, val_loader)
        self.stage2_train_detector(train_loader)
        self.stage3_build_prototypes(train_loader)
        return self.stage4_train_lora(train_loader, val_loader)


class FairNetUnlabeledTrainer(FairNetTrainer):
    """
    Trainer for FairNet-Unlabeled setting (Section 5.1, Supplementary D.2).
    
    No sensitive attribute labels are available.
    Uses unsupervised outlier detection (LOF) to identify minority samples.
    
    Modified pipeline:
    1. Stage 1: Train base model with ERM (same as Full)
    2. Stage 2: Generate pseudo-sensitive labels using LOF
    3. Stage 3: Build prototypes using pseudo labels
    4. Stage 4: Train LoRA using pseudo-labeled minority samples
    
    Args:
        model: FairNetViT or FairNetBERT model
        config: FairNetConfig (must have attribute_mode=UNLABELED)
        device: torch.device
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: FairNetConfig,
        device: torch.device
    ):
        super().__init__(model, config, device)
        
        # Unsupervised detector using LOF
        self.unsupervised_detector = UnsupervisedBiasDetector(
            hidden_dim=model.hidden_dim,
            n_neighbors=config.lof_n_neighbors,
            contamination=config.lof_contamination
        )
        
        # Pseudo labels: sample_idx -> pseudo_sensitive
        self.pseudo_labels: Dict[int, int] = {}
    
    def stage2_generate_pseudo_labels(self, train_loader: DataLoader) -> Dict:
        """
        Stage 2 for Unlabeled: Generate pseudo-sensitive labels using LOF.
        
        Outliers in the representation space are treated as minority samples.
        """
        print("\n" + "=" * 60)
        print("Stage 2: Generating Pseudo-Sensitive Labels (LOF)")
        print("=" * 60)
        
        self.model.eval()
        self.model._clear_lora_activation()
        
        # Collect all features
        all_features = []
        all_indices = []
        idx = 0
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Extracting features"):
                images = batch[0].to(self.device)
                cls_features = self.model.get_cls_features(images, use_lora=False)
                all_features.append(cls_features.cpu())
                all_indices.extend(range(idx, idx + len(images)))
                idx += len(images)
        
        all_features = torch.cat(all_features, dim=0)
        
        # Fit LOF
        print("Fitting LOF...")
        self.unsupervised_detector.fit(all_features)
        
        # Get outlier scores
        with torch.no_grad():
            scores = self.unsupervised_detector(all_features)
        
        # Assign pseudo labels (outliers = minority)
        threshold = self.config.activation_threshold
        pseudo_labels = (scores > threshold).cpu().numpy().flatten()
        
        for i, sample_idx in enumerate(all_indices):
            self.pseudo_labels[sample_idx] = int(pseudo_labels[i])
        
        minority_count = sum(self.pseudo_labels.values())
        total = len(self.pseudo_labels)
        print(f"Pseudo-minority samples: {minority_count}/{total} ({100*minority_count/total:.1f}%)")
        
        return {
            'pseudo_minority_count': minority_count,
            'pseudo_minority_rate': minority_count / total
        }
    
    def stage3_build_prototypes(self, train_loader: DataLoader):
        """Stage 3 for Unlabeled: Build prototypes using pseudo labels."""
        print("\n" + "=" * 60)
        print("Stage 3: Building Prototypes (using pseudo labels)")
        print("=" * 60)
        
        self.model.eval()
        self.model._clear_lora_activation()
        
        sums: Dict[int, Dict[int, torch.Tensor]] = {}
        counts: Dict[int, Dict[int, int]] = {}
        
        with torch.no_grad():
            idx = 0
            for batch in tqdm(train_loader, desc="Building prototypes"):
                images, attributes = batch[0], batch[1]
                images = images.to(self.device)
                labels = attributes[:, self.config.target_attribute]
                cls_features = self.model.get_cls_features(images, use_lora=False)
                
                for i in range(len(labels)):
                    y = labels[i].item()
                    s = self.pseudo_labels.get(idx + i, 0)  # Use pseudo label
                    
                    if y not in sums:
                        sums[y] = {}
                        counts[y] = {}
                    if s not in sums[y]:
                        sums[y][s] = torch.zeros(self.model.hidden_dim, device=self.device)
                        counts[y][s] = 0
                    
                    sums[y][s] += cls_features[i]
                    counts[y][s] += 1
                
                idx += len(images)
        
        # Store prototypes
        for y in sums:
            self.prototype_bank.prototypes[y] = {}
            self.prototype_bank.counts[y] = {}
            for s in sums[y]:
                self.prototype_bank.prototypes[y][s] = sums[y][s] / counts[y][s]
                self.prototype_bank.counts[y][s] = counts[y][s]
        
        print("\nPrototype Statistics (Pseudo-labeled):")
        for y in sorted(self.prototype_bank.prototypes.keys()):
            for s in sorted(self.prototype_bank.prototypes[y].keys()):
                label = "Pseudo-Minority" if s == 1 else "Pseudo-Majority"
                print(f"  Class {y} + {label}: {self.prototype_bank.counts[y][s]} samples")
    
    def stage4_train_lora(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Stage 4 for Unlabeled: Train LoRA using pseudo-labeled minority samples."""
        print("\n" + "=" * 60)
        print("Stage 4: Training LoRA (using pseudo labels)")
        print("=" * 60)
        
        self.model.freeze_base()
        self.model.freeze_detectors()
        self.model.unfreeze_lora()
        
        lora_params = self.model.get_lora_parameters()
        optimizer = optim.Adam(lora_params, lr=self.config.stage4_lr)
        scheduler = self._get_warmup_scheduler(
            optimizer,
            len(train_loader) * self.config.stage4_epochs,
            50
        )
        contrastive_loss_fn = TripletContrastiveLoss(margin=self.config.contrastive_margin)
        
        best_wga = 0
        best_state = None
        
        for epoch in range(self.config.stage4_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            idx = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.stage4_epochs}")
            for batch in pbar:
                images, attributes = batch[0], batch[1]
                images = images.to(self.device)
                labels = attributes[:, self.config.target_attribute].to(self.device)
                
                # Get pseudo-sensitive labels for this batch
                batch_pseudo = torch.tensor([
                    self.pseudo_labels.get(idx + i, 0) 
                    for i in range(len(images))
                ], device=self.device)
                idx += len(images)
                
                minority_mask = (batch_pseudo == 1)
                if minority_mask.sum() < 2:
                    continue
                
                optimizer.zero_grad()
                
                _, corrected_features = self.model(
                    images[minority_mask],
                    return_features=True,
                    force_lora=True
                )
                
                pos_targets, neg_targets = self.prototype_bank.get_targets(labels[minority_mask])
                c_loss = contrastive_loss_fn(corrected_features, pos_targets, neg_targets)
                loss = self.config.lambda_C * c_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, self.config.gradient_clip)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if val_loader and num_batches > 0:
                metrics = self._evaluate(val_loader, use_lora=True)
                print(f"Epoch {epoch+1}: Acc={metrics['accuracy']:.4f}, WGA={metrics['worst_group_accuracy']:.4f}")
                
                if metrics['worst_group_accuracy'] > best_wga:
                    best_wga = metrics['worst_group_accuracy']
                    best_state = copy.deepcopy(self.model.state_dict())
        
        if best_state:
            self.model.load_state_dict(best_state)
        
        return self._evaluate(val_loader, use_lora=True) if val_loader else {}
    
    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """Execute full training without sensitive labels."""
        self.stage1_train_base(train_loader, val_loader)
        self.stage2_generate_pseudo_labels(train_loader)
        self.stage3_build_prototypes(train_loader)
        return self.stage4_train_lora(train_loader, val_loader)
