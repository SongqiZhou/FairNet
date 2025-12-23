"""
FairNet Utilities

Contains helper functions for:
- Random seed setting
- Model evaluation
- Metrics printing
- Data loading utilities
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    config,
    device: torch.device,
    use_lora: bool = True
) -> Dict:
    """
    Comprehensive model evaluation with fairness metrics.
    
    Computes:
    - Overall accuracy
    - Per-group accuracy
    - Worst Group Accuracy (WGA)
    - Equalized Odds Difference (EOD)
    - Equal Opportunity (EOp)
    - Demographic Parity (DP)
    - LoRA activation rate
    
    Args:
        model: FairNet model
        loader: DataLoader for evaluation
        config: FairNetConfig
        device: torch.device
        use_lora: Whether to use LoRA during evaluation
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_sensitive = []
    all_risk_scores = []
    
    # Group-wise results
    groups = {
        'group_0_0': {'preds': [], 'labels': []},  # Majority, Negative
        'group_0_1': {'preds': [], 'labels': []},  # Majority, Positive
        'group_1_0': {'preds': [], 'labels': []},  # Minority, Negative
        'group_1_1': {'preds': [], 'labels': []},  # Minority, Positive
    }
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images, attributes = batch[0], batch[1]
            images = images.to(device)
            sensitive = attributes[:, config.sensitive_attributes[0]].numpy()
            labels = attributes[:, config.target_attribute].numpy()
            
            if use_lora:
                outputs, risk_scores = model(images, return_risk_scores=True)
                rs = risk_scores[config.sensitive_attributes[0]].cpu().numpy().flatten()
                all_risk_scores.extend(rs)
            else:
                model._clear_lora_activation()
                cls_features = model.get_cls_features(images, use_lora=False)
                outputs = model.classifier(cls_features)
            
            preds = (outputs > 0.5).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_sensitive.extend(sensitive)
            
            # Assign to groups
            for i in range(len(labels)):
                key = f"group_{int(sensitive[i])}_{int(labels[i])}"
                groups[key]['preds'].append(preds[i])
                groups[key]['labels'].append(labels[i])
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_sensitive = np.array(all_sensitive)
    
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    
    # Group accuracies
    group_accs = []
    for name, data in groups.items():
        if data['labels']:
            acc = accuracy_score(data['labels'], data['preds'])
            metrics[f'acc_{name}'] = acc
            group_accs.append(acc)
    
    # Worst Group Accuracy
    metrics['worst_group_accuracy'] = min(group_accs) if group_accs else 0
    metrics['accuracy_gap'] = max(group_accs) - min(group_accs) if group_accs else 0
    
    # TPR and FPR per group
    # Majority (s=0)
    maj_pos = (all_labels == 1) & (all_sensitive == 0)
    maj_neg = (all_labels == 0) & (all_sensitive == 0)
    tpr_maj = all_preds[maj_pos].mean() if maj_pos.sum() > 0 else 0
    fpr_maj = all_preds[maj_neg].mean() if maj_neg.sum() > 0 else 0
    
    # Minority (s=1)
    min_pos = (all_labels == 1) & (all_sensitive == 1)
    min_neg = (all_labels == 0) & (all_sensitive == 1)
    tpr_min = all_preds[min_pos].mean() if min_pos.sum() > 0 else 0
    fpr_min = all_preds[min_neg].mean() if min_neg.sum() > 0 else 0
    
    # Fairness metrics
    metrics['TPR_majority'] = tpr_maj
    metrics['TPR_minority'] = tpr_min
    metrics['FPR_majority'] = fpr_maj
    metrics['FPR_minority'] = fpr_min
    
    # Equalized Odds Difference
    metrics['EOD'] = 0.5 * (abs(tpr_maj - tpr_min) + abs(fpr_maj - fpr_min))
    
    # Equal Opportunity (TPR difference)
    metrics['EOp'] = abs(tpr_maj - tpr_min)
    
    # Demographic Parity
    pos_rate_maj = all_preds[all_sensitive == 0].mean() if (all_sensitive == 0).sum() > 0 else 0
    pos_rate_min = all_preds[all_sensitive == 1].mean() if (all_sensitive == 1).sum() > 0 else 0
    metrics['DP'] = abs(pos_rate_maj - pos_rate_min)
    
    # LoRA activation rate
    if all_risk_scores:
        all_risk_scores = np.array(all_risk_scores)
        metrics['lora_activation_rate'] = np.mean(all_risk_scores > config.activation_threshold)
        metrics['risk_score_mean'] = np.mean(all_risk_scores)
        metrics['risk_score_std'] = np.std(all_risk_scores)
        
        # Risk score per group
        metrics['risk_score_majority'] = np.mean(all_risk_scores[all_sensitive == 0])
        metrics['risk_score_minority'] = np.mean(all_risk_scores[all_sensitive == 1])
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Results"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the output
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    # Performance metrics
    print("\n📊 Performance Metrics:")
    print(f"  Overall Accuracy:      {metrics.get('accuracy', 0):.4f}")
    print(f"  Worst Group Accuracy:  {metrics.get('worst_group_accuracy', 0):.4f}")
    print(f"  Accuracy Gap:          {metrics.get('accuracy_gap', 0):.4f}")
    
    # Group-wise accuracy
    print("\n📈 Group-wise Accuracy:")
    print(f"  Majority + Negative (0,0): {metrics.get('acc_group_0_0', 0):.4f}")
    print(f"  Majority + Positive (0,1): {metrics.get('acc_group_0_1', 0):.4f}")
    print(f"  Minority + Negative (1,0): {metrics.get('acc_group_1_0', 0):.4f}")
    print(f"  Minority + Positive (1,1): {metrics.get('acc_group_1_1', 0):.4f}")
    
    # Fairness metrics
    print("\n⚖️  Fairness Metrics:")
    print(f"  Equalized Odds Diff:   {metrics.get('EOD', 0):.4f}")
    print(f"  Equal Opportunity:     {metrics.get('EOp', 0):.4f}")
    print(f"  Demographic Parity:    {metrics.get('DP', 0):.4f}")
    
    # TPR/FPR
    print("\n🎯 TPR/FPR per Group:")
    print(f"  TPR (Majority): {metrics.get('TPR_majority', 0):.4f}")
    print(f"  TPR (Minority): {metrics.get('TPR_minority', 0):.4f}")
    print(f"  FPR (Majority): {metrics.get('FPR_majority', 0):.4f}")
    print(f"  FPR (Minority): {metrics.get('FPR_minority', 0):.4f}")
    
    # LoRA metrics
    if 'lora_activation_rate' in metrics:
        print("\n🔧 LoRA Activation:")
        print(f"  Activation Rate:       {metrics.get('lora_activation_rate', 0):.4f}")
        print(f"  Risk Score Mean:       {metrics.get('risk_score_mean', 0):.4f}")
        print(f"  Risk Score Std:        {metrics.get('risk_score_std', 0):.4f}")
        print(f"  Risk (Majority):       {metrics.get('risk_score_majority', 0):.4f}")
        print(f"  Risk (Minority):       {metrics.get('risk_score_minority', 0):.4f}")
    
    print("\n" + "=" * 60)


def compute_class_weights(
    loader: DataLoader,
    target_attr: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        loader: DataLoader
        target_attr: Index of target attribute
        device: torch.device
    
    Returns:
        Class weights tensor
    """
    counts = {}
    for batch in loader:
        labels = batch[1][:, target_attr].numpy()
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    
    total = sum(counts.values())
    num_classes = len(counts)
    weights = {k: total / (num_classes * v) for k, v in counts.items()}
    
    weight_tensor = torch.tensor([weights[i] for i in sorted(weights.keys())], device=device)
    return weight_tensor


def get_group_indices(
    loader: DataLoader,
    target_attr: int,
    sensitive_attr: int
) -> Dict[str, List[int]]:
    """
    Get sample indices for each group.
    
    Args:
        loader: DataLoader
        target_attr: Index of target attribute
        sensitive_attr: Index of sensitive attribute
    
    Returns:
        Dictionary mapping group names to sample indices
    """
    groups = {
        'group_0_0': [],
        'group_0_1': [],
        'group_1_0': [],
        'group_1_1': [],
    }
    
    idx = 0
    for batch in loader:
        attributes = batch[1]
        labels = attributes[:, target_attr].numpy()
        sensitive = attributes[:, sensitive_attr].numpy()
        
        for i in range(len(labels)):
            key = f"group_{int(sensitive[i])}_{int(labels[i])}"
            groups[key].append(idx + i)
        
        idx += len(labels)
    
    return groups


def save_checkpoint(
    model: nn.Module,
    config,
    metrics: Dict,
    path: str
):
    """
    Save model checkpoint.
    
    Args:
        model: FairNet model
        config: FairNetConfig
        metrics: Evaluation metrics
        path: Save path
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'metrics': metrics
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: FairNet model
        path: Checkpoint path
        device: torch.device
    
    Returns:
        Loaded metrics
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded from {path}")
    return checkpoint.get('metrics', {})
