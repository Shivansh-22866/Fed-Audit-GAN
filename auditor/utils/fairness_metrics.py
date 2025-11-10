"""
Fairness Auditing Metrics
==========================
Phase 2 of Fed-AuditGAN: Generative Fairness Auditing

This module handles:
1. Training the Generator to find fairness vulnerabilities
2. Calculating bias using counterfactual probes
3. Generating high-quality fairness auditing datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# New DCGAN-compatible FairnessAuditor (used when initializing with num_classes)
class FairnessAuditor:
    """
    Handles fairness auditing using generated probes.
    
    Supports two modes:
    1. DCGAN mode: Initialize with num_classes and device for synthetic probe auditing
    2. Legacy mode: Initialize with generator and global_model for counterfactual auditing
    
    Args:
        generator (nn.Module, optional): Fairness probe generator (legacy mode)
        global_model (nn.Module, optional): The frozen global model to audit (legacy mode)
        num_classes (int, optional): Number of classes (DCGAN mode)
        device (str): Device for computation ('cpu' or 'cuda')
    """
    
    def __init__(self, generator=None, global_model=None, num_classes=None, device='cpu'):
        self.device = device
        self.num_classes = num_classes
        self._global_model_ref = None  # For uncertainty-based sensitive attributes
        
        # Legacy mode (with generator)
        if generator is not None:
            self.generator = generator.to(device)
            self.global_model = global_model.to(device)
            self.mode = 'legacy'
            
            # Freeze global model
            self.global_model.eval()
            for param in self.global_model.parameters():
                param.requires_grad = False
        
        # DCGAN mode (without generator)
        else:
            self.generator = None
            self.global_model = None
            self.mode = 'dcgan'
            if num_classes is None:
                raise ValueError("num_classes must be provided in DCGAN mode")
    
    def set_global_model(self, model: nn.Module):
        """
        Set reference to global model for uncertainty-based sensitive attribute generation.
        
        Args:
            model: Current global model to use for uncertainty calculations
        """
        self._global_model_ref = model
    
    def create_sensitive_attributes_from_heterogeneity(
        self,
        dataloader: torch.utils.data.DataLoader,
        strategy: str = 'class_imbalance'
    ) -> torch.Tensor:
        """
        Create synthetic sensitive attributes from data heterogeneity.
        
        This addresses the fundamental problem: using class labels (0-9) as sensitive 
        attributes doesn't measure demographic fairness. Instead, we create binary
        demographic groups based on data characteristics.
        
        Strategies:
        1. 'class_imbalance': Split by class representation
           - Disadvantaged group (1): samples from underrepresented classes
           - Advantaged group (0): samples from well-represented classes
        
        2. 'uncertainty': Split by model uncertainty (requires set_global_model())
           - Disadvantaged group (1): high-uncertainty samples (model struggles)
           - Advantaged group (0): low-uncertainty samples (model confident)
        
        3. 'mixed': Combine class_imbalance and uncertainty (50/50 weight)
        
        Args:
            dataloader: Data loader with samples
            strategy: Strategy for creating sensitive attributes
                     ('class_imbalance', 'uncertainty', or 'mixed')
        
        Returns:
            Tensor of binary sensitive attributes (0=advantaged, 1=disadvantaged)
        
        Example:
            >>> auditor = FairnessAuditor(num_classes=10, device='cuda')
            >>> auditor.set_global_model(global_model)
            >>> sensitive_attr = auditor.create_sensitive_attributes_from_heterogeneity(
            ...     probe_loader, strategy='class_imbalance'
            ... )
            >>> # Returns tensor like [0, 1, 1, 0, 1, ...] (binary groups)
        """
        all_targets = []
        all_data = []
        
        # Collect all data and targets
        for data, target in dataloader:
            all_data.append(data)
            all_targets.append(target)
        
        all_data = torch.cat(all_data, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        n_samples = len(all_targets)
        
        if strategy == 'class_imbalance':
            # Strategy 1: ENHANCED - Split by class representation with percentile thresholds
            class_counts = torch.bincount(all_targets, minlength=self.num_classes)
            
            # Use 40th percentile instead of median for more aggressive disadvantaged group
            # This captures more underrepresented samples
            sorted_counts = torch.sort(class_counts.float())[0]
            threshold_idx = int(0.4 * len(sorted_counts))
            threshold_count = sorted_counts[threshold_idx]
            
            # Samples from underrepresented classes (bottom 40%) → disadvantaged (1)
            # Samples from well-represented classes (top 60%) → advantaged (0)
            sensitive_attrs = (class_counts[all_targets] <= threshold_count).long()
            
            # Log distribution
            class_dist = torch.bincount(all_targets[sensitive_attrs == 0])
            disadv_dist = torch.bincount(all_targets[sensitive_attrs == 1])
            
            logger.info(f"ENHANCED Class imbalance sensitive attributes:")
            logger.info(f"  Disadvantaged samples: {sensitive_attrs.sum().item()}/{n_samples} "
                       f"({100*sensitive_attrs.sum().item()/n_samples:.1f}%)")
            logger.info(f"  Threshold: {threshold_count:.0f} samples (40th percentile)")
            logger.info(f"  Advantaged classes: {(class_counts > threshold_count).sum().item()} classes")
            logger.info(f"  Disadvantaged classes: {(class_counts <= threshold_count).sum().item()} classes")
        
        elif strategy == 'uncertainty':
            # Strategy 2: Split by model uncertainty
            if self._global_model_ref is None:
                raise ValueError("Global model not set. Call set_global_model() first.")
            
            self._global_model_ref.eval()
            uncertainties = []
            
            with torch.no_grad():
                for data in torch.split(all_data, 256):  # Process in batches
                    data = data.to(self.device)
                    output = self._global_model_ref(data)
                    probs = torch.softmax(output, dim=1)
                    
                    # Uncertainty = 1 - max(prob)
                    max_probs, _ = probs.max(dim=1)
                    uncertainty = 1.0 - max_probs
                    uncertainties.append(uncertainty.cpu())
            
            uncertainties = torch.cat(uncertainties)
            median_uncertainty = torch.median(uncertainties)
            
            # High uncertainty → disadvantaged (1)
            # Low uncertainty → advantaged (0)
            sensitive_attrs = (uncertainties > median_uncertainty).long()
            
            logger.info(f"Uncertainty sensitive attributes: "
                       f"{sensitive_attrs.sum().item()}/{n_samples} disadvantaged")
        
        elif strategy == 'mixed':
            # Strategy 3: Combine class_imbalance and uncertainty (50/50)
            if self._global_model_ref is None:
                raise ValueError("Global model not set. Call set_global_model() first.")
            
            # Get class imbalance scores
            class_counts = torch.bincount(all_targets, minlength=self.num_classes)
            median_count = torch.median(class_counts.float())
            imbalance_scores = (class_counts[all_targets] < median_count).float()
            
            # Get uncertainty scores
            self._global_model_ref.eval()
            uncertainties = []
            
            with torch.no_grad():
                for data in torch.split(all_data, 256):
                    data = data.to(self.device)
                    output = self._global_model_ref(data)
                    probs = torch.softmax(output, dim=1)
                    max_probs, _ = probs.max(dim=1)
                    uncertainty = 1.0 - max_probs
                    uncertainties.append(uncertainty.cpu())
            
            uncertainties = torch.cat(uncertainties)
            
            # Normalize both to [0, 1]
            uncertainty_norm = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min() + 1e-8)
            
            # Combine: 50% imbalance + 50% uncertainty
            combined_scores = 0.5 * imbalance_scores + 0.5 * uncertainty_norm
            median_combined = torch.median(combined_scores)
            
            sensitive_attrs = (combined_scores > median_combined).long()
            
            logger.info(f"Mixed sensitive attributes: "
                       f"{sensitive_attrs.sum().item()}/{n_samples} disadvantaged")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'class_imbalance', 'uncertainty', or 'mixed'.")
        
        return sensitive_attrs
    
    def compute_demographic_parity(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        sensitive_attribute: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute demographic parity violation (DCGAN mode).
        
        Measures if positive prediction rates are equal across demographic groups.
        Now uses REAL sensitive attributes (0=advantaged, 1=disadvantaged) instead
        of class labels.
        
        Args:
            model: Model to audit
            dataloader: Data loader with samples
            sensitive_attribute: Tensor of binary sensitive attributes (0/1)
                                If None, falls back to old class-based splitting
            
        Returns:
            Demographic parity violation score (0.0 = perfect fairness)
        """
        model.eval()
        
        # Collect all predictions
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        if len(all_predictions) == 0:
            return 0.0
        
        # Use provided sensitive attributes (NEW!)
        if sensitive_attribute is not None:
            if len(sensitive_attribute) != len(all_predictions):
                logger.warning(f"Sensitive attribute length mismatch: "
                             f"{len(sensitive_attribute)} vs {len(all_predictions)}")
                return 0.0
            
            sensitive_attr_np = sensitive_attribute.cpu().numpy() if torch.is_tensor(sensitive_attribute) else sensitive_attribute
            
            # Group 0 = advantaged, Group 1 = disadvantaged
            group_advantaged_mask = sensitive_attr_np == 0
            group_disadvantaged_mask = sensitive_attr_np == 1
            
            if group_advantaged_mask.sum() == 0 or group_disadvantaged_mask.sum() == 0:
                return 0.0
            
            # "Positive" outcome = correct prediction
            # Calculate accuracy for each group
            advantaged_correct = (all_predictions[group_advantaged_mask] == all_targets[group_advantaged_mask])
            disadvantaged_correct = (all_predictions[group_disadvantaged_mask] == all_targets[group_disadvantaged_mask])
            
            advantaged_accuracy = advantaged_correct.mean()
            disadvantaged_accuracy = disadvantaged_correct.mean()
            
            # Demographic parity violation = difference in accuracies
            dp_violation = abs(float(advantaged_accuracy) - float(disadvantaged_accuracy))
            
        else:
            # Fallback to old class-based splitting (for backward compatibility)
            group_a_mask = all_targets < 5
            group_b_mask = all_targets >= 5
            
            if group_a_mask.sum() == 0 or group_b_mask.sum() == 0:
                return 0.0
            
            group_a_positive_rate = (all_predictions[group_a_mask] >= 5).mean()
            group_b_positive_rate = (all_predictions[group_b_mask] >= 5).mean()
            
            dp_violation = abs(float(group_a_positive_rate) - float(group_b_positive_rate))
        
        return dp_violation
    
    def compute_equalized_odds(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        sensitive_attribute: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute equalized odds violation (DCGAN mode).
        
        Measures if TPR and FPR are equal across demographic groups.
        Now uses REAL sensitive attributes (0=advantaged, 1=disadvantaged).
        
        Args:
            model: Model to audit
            dataloader: Data loader with samples
            sensitive_attribute: Tensor of binary sensitive attributes (0/1)
                                If None, falls back to old class-based splitting
            
        Returns:
            Equalized odds violation score (0.0 = perfect fairness)
        """
        model.eval()
        
        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        if len(all_predictions) == 0:
            return 0.0
        
        # Use provided sensitive attributes (NEW!)
        if sensitive_attribute is not None:
            if len(sensitive_attribute) != len(all_predictions):
                logger.warning(f"Sensitive attribute length mismatch: "
                             f"{len(sensitive_attribute)} vs {len(all_predictions)}")
                return 0.0
            
            sensitive_attr_np = sensitive_attribute.cpu().numpy() if torch.is_tensor(sensitive_attribute) else sensitive_attribute
            
            # Group 0 = advantaged, Group 1 = disadvantaged
            group_advantaged_mask = sensitive_attr_np == 0
            group_disadvantaged_mask = sensitive_attr_np == 1
            
            if group_advantaged_mask.sum() == 0 or group_disadvantaged_mask.sum() == 0:
                return 0.0
            
            # For each group, compute TPR and FPR per class
            # TPR: P(correct | true class c), averaged over all classes
            # FPR: P(pred class c | true class != c), averaged over all classes
            
            def compute_tpr_fpr_multiclass(preds, targets):
                """Compute average TPR and FPR across all classes"""
                tprs = []
                fprs = []
                
                for c in range(self.num_classes):
                    true_positive = ((preds == c) & (targets == c)).sum()
                    false_negative = ((preds != c) & (targets == c)).sum()
                    false_positive = ((preds == c) & (targets != c)).sum()
                    true_negative = ((preds != c) & (targets != c)).sum()
                    
                    # TPR = TP / (TP + FN)
                    if (true_positive + false_negative) > 0:
                        tpr = true_positive / (true_positive + false_negative)
                        tprs.append(tpr)
                    
                    # FPR = FP / (FP + TN)
                    if (false_positive + true_negative) > 0:
                        fpr = false_positive / (false_positive + true_negative)
                        fprs.append(fpr)
                
                avg_tpr = np.mean(tprs) if tprs else 0.0
                avg_fpr = np.mean(fprs) if fprs else 0.0
                
                return avg_tpr, avg_fpr
            
            # Compute for advantaged group
            adv_tpr, adv_fpr = compute_tpr_fpr_multiclass(
                all_predictions[group_advantaged_mask],
                all_targets[group_advantaged_mask]
            )
            
            # Compute for disadvantaged group
            dis_tpr, dis_fpr = compute_tpr_fpr_multiclass(
                all_predictions[group_disadvantaged_mask],
                all_targets[group_disadvantaged_mask]
            )
            
            # Equalized Odds violation = average of TPR and FPR gaps
            tpr_gap = abs(float(adv_tpr) - float(dis_tpr))
            fpr_gap = abs(float(adv_fpr) - float(dis_fpr))
            
            eo_violation = (tpr_gap + fpr_gap) / 2.0
            
        else:
            # Fallback to old class-based splitting
            group_a_mask = all_targets < 5
            group_b_mask = all_targets >= 5
            
            if group_a_mask.sum() == 0 or group_b_mask.sum() == 0:
                return 0.0
            
            # Group A statistics
            group_a_preds = all_predictions[group_a_mask]
            group_a_true = all_targets[group_a_mask]
            
            a_positive_mask = group_a_true >= 5
            if a_positive_mask.sum() > 0:
                group_a_tpr = (group_a_preds[a_positive_mask] >= 5).mean()
            else:
                group_a_tpr = 0.0
            
            a_negative_mask = group_a_true < 5
            if a_negative_mask.sum() > 0:
                group_a_fpr = (group_a_preds[a_negative_mask] >= 5).mean()
            else:
                group_a_fpr = 0.0
            
            # Group B statistics
            group_b_preds = all_predictions[group_b_mask]
            group_b_true = all_targets[group_b_mask]
            
            b_positive_mask = group_b_true >= 5
            if b_positive_mask.sum() > 0:
                group_b_tpr = (group_b_preds[b_positive_mask] >= 5).mean()
            else:
                group_b_tpr = 0.0
            
            b_negative_mask = group_b_true < 5
            if b_negative_mask.sum() > 0:
                group_b_fpr = (group_b_preds[b_negative_mask] >= 5).mean()
            else:
                group_b_fpr = 0.0
            
            tpr_gap = abs(float(group_a_tpr) - float(group_b_tpr))
            fpr_gap = abs(float(group_a_fpr) - float(group_b_fpr))
            
            eo_violation = (tpr_gap + fpr_gap) / 2.0
        
        return eo_violation
    
    def compute_class_balance(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> float:
        """
        Compute class balance metric (DCGAN mode).
        
        Measures how balanced the model's predictions are across all classes.
        Lower is better (more uniform).
        
        Args:
            model: Model to audit
            dataloader: Data loader with samples
            
        Returns:
            Class balance score (0.0 = perfectly uniform)
        """
        model.eval()
        
        class_counts = np.zeros(self.num_classes)
        total_samples = 0
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                
                for c in range(self.num_classes):
                    class_counts[c] += (pred == c).sum().item()
                
                total_samples += data.size(0)
        
        if total_samples == 0:
            return 0.0
        
        # Compute class distribution
        class_probs = class_counts / total_samples
        
        # Ideal uniform distribution
        ideal_prob = 1.0 / self.num_classes
        
        # Measure deviation from uniform (L1 distance / 2)
        imbalance = float(np.sum(np.abs(class_probs - ideal_prob)) / 2.0)
        
        return imbalance
    
    def audit_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        sensitive_attribute: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform comprehensive fairness audit (DCGAN mode).
        
        Args:
            model: Model to audit
            dataloader: Data loader with samples
            sensitive_attribute: Sensitive attribute for each sample
            
        Returns:
            Dictionary of fairness metrics
        """
        logger.info("Performing fairness audit...")
        
        metrics = {
            'demographic_parity': self.compute_demographic_parity(model, dataloader, sensitive_attribute),
            'equalized_odds': self.compute_equalized_odds(model, dataloader, sensitive_attribute),
            'class_balance': self.compute_class_balance(model, dataloader)
        }
        
        logger.info(f"Fairness Audit Results:")
        logger.info(f"  Demographic Parity: {metrics['demographic_parity']:.4f}")
        logger.info(f"  Equalized Odds: {metrics['equalized_odds']:.4f}")
        logger.info(f"  Class Balance: {metrics['class_balance']:.4f}")
        
        return metrics
    
    def calculate_bias(self, model, probe_loader):
        """
        Calculate bias using counterfactual fairness probes.
        
        Measures average prediction difference between original samples
        and their counterfactual modifications.
        
        Args:
            model (nn.Module): The model to audit
            probe_loader (DataLoader): DataLoader of (x, x') pairs
            
        Returns:
            float: Average prediction difference (bias score)
        """
        model.eval()
        total_diff = 0.0
        count = 0
        
        with torch.no_grad():
            for x, x_prime in probe_loader:
                x = x.to(self.device)
                x_prime = x_prime.to(self.device)
                
                # Get predictions
                pred_original = model(x)
                pred_counterfactual = model(x_prime)
                
                # Calculate prediction difference
                diff = torch.abs(pred_original - pred_counterfactual).mean()
                total_diff += diff.item()
                count += 1
        
        avg_bias = total_diff / count if count > 0 else 0.0
        return avg_bias
    
    def train_generator(self, seed_data_loader, n_steps=100, alpha=1.0, beta=0.5, lr=0.001):
        """
        Train generator to find fairness vulnerabilities (Phase 2).
        
        The generator is trained with two objectives:
        1. Realism: Generated samples should be realistic (minimize ||x - x'||)
        2. Adversarial: Maximize prediction difference on global model
        
        Loss: L_G = α * L_realism - β * L_adversarial
        
        Args:
            seed_data_loader (DataLoader): Representative dataset for generating probes
            n_steps (int): Number of training steps
            alpha (float): Weight for realism loss
            beta (float): Weight for adversarial loss
            lr (float): Learning rate for generator
            
        Returns:
            dict: Training statistics
        """
        self.generator.train()
        self.global_model.eval()
        
        optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        
        stats = {
            'losses': [],
            'realism_losses': [],
            'adversarial_losses': []
        }
        
        print(f"\n{'='*60}")
        print(f"Training Generator for Fairness Auditing")
        print(f"{'='*60}")
        print(f"Steps: {n_steps} | Alpha: {alpha} | Beta: {beta} | LR: {lr}")
        
        pbar = tqdm(range(n_steps), desc="Generator Training")
        
        for step in pbar:
            epoch_loss = 0.0
            epoch_realism = 0.0
            epoch_adversarial = 0.0
            batch_count = 0
            
            for batch_data in seed_data_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0].to(self.device)
                else:
                    x = batch_data.to(self.device)
                
                # Flatten if needed (for fully connected generator)
                if len(x.shape) > 2 and hasattr(self.generator, 'input_dim'):
                    x = x.view(x.size(0), -1)
                
                # Generate counterfactual
                x_prime = self.generator(x)
                
                # Get predictions from frozen global model
                with torch.no_grad():
                    pred_original = self.global_model(x.view(x.size(0), *self.global_model_input_shape) 
                                                     if hasattr(self, 'global_model_input_shape') else x)
                
                pred_counterfactual = self.global_model(x_prime.view(x_prime.size(0), *self.global_model_input_shape)
                                                       if hasattr(self, 'global_model_input_shape') else x_prime)
                
                # Loss 1: Realism (minimize difference in non-sensitive attributes)
                loss_realism = torch.norm(x - x_prime, p=2) / x.size(0)
                
                # Loss 2: Adversarial (maximize prediction difference)
                # Note: We negate this to maximize when minimizing total loss
                loss_adversarial = -torch.abs(pred_original - pred_counterfactual).mean()
                
                # Combined loss
                total_loss = alpha * loss_realism + beta * loss_adversarial
                
                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += total_loss.item()
                epoch_realism += loss_realism.item()
                epoch_adversarial += (-loss_adversarial.item())  # Show actual adversarial value
                batch_count += 1
                
                if batch_count >= 10:  # Limit batches per step
                    break
            
            # Record averages
            if batch_count > 0:
                stats['losses'].append(epoch_loss / batch_count)
                stats['realism_losses'].append(epoch_realism / batch_count)
                stats['adversarial_losses'].append(epoch_adversarial / batch_count)
                
                pbar.set_postfix({
                    'Loss': f"{stats['losses'][-1]:.4f}",
                    'Realism': f"{stats['realism_losses'][-1]:.4f}",
                    'Adversarial': f"{stats['adversarial_losses'][-1]:.4f}"
                })
        
        print(f"\n✓ Generator training complete!")
        print(f"  Final Loss: {stats['losses'][-1]:.4f}")
        print(f"  Final Realism: {stats['realism_losses'][-1]:.4f}")
        print(f"  Final Adversarial: {stats['adversarial_losses'][-1]:.4f}")
        
        return stats
    
    def generate_probes(self, seed_data_loader, n_probes=1000):
        """
        Generate fairness auditing probes after training generator.
        
        Creates a dataset of (x, x') counterfactual pairs that can be used
        to audit model fairness.
        
        Args:
            seed_data_loader (DataLoader): Source data for generating probes
            n_probes (int): Number of probe pairs to generate
            
        Returns:
            list: List of (x, x') probe pairs as tuples
        """
        self.generator.eval()
        probes = []
        
        print(f"\nGenerating {n_probes} fairness probes...")
        
        with torch.no_grad():
            for batch_data in seed_data_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0].to(self.device)
                else:
                    x = batch_data.to(self.device)
                
                # Flatten if needed
                if len(x.shape) > 2 and hasattr(self.generator, 'input_dim'):
                    x = x.view(x.size(0), -1)
                
                # Generate counterfactuals
                x_prime = self.generator(x)
                
                # Store as CPU tensors
                for i in range(x.size(0)):
                    probes.append((x[i].cpu(), x_prime[i].cpu()))
                    
                    if len(probes) >= n_probes:
                        break
                
                if len(probes) >= n_probes:
                    break
        
        print(f"✓ Generated {len(probes)} probe pairs")
        return probes[:n_probes]


class CounterfactualFairnessMetric:
    """
    Implements various fairness metrics using counterfactual reasoning.
    """
    
    @staticmethod
    def demographic_parity(model, data_loader, sensitive_attr_idx, device='cpu'):
        """
        Measure demographic parity: P(Y=1|A=0) ≈ P(Y=1|A=1)
        
        Args:
            model: Trained model
            data_loader: DataLoader with (x, y, sensitive_attr) tuples
            sensitive_attr_idx: Index of sensitive attribute
            device: Computation device
            
        Returns:
            float: Demographic parity difference
        """
        model.eval()
        
        preds_sensitive_0 = []
        preds_sensitive_1 = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:
                    x, _, sensitive = batch
                else:
                    x, _ = batch
                    sensitive = x[:, sensitive_attr_idx]
                
                x = x.to(device)
                sensitive = sensitive.to(device)
                
                pred = model(x)
                pred_class = (pred > 0.5).float() if pred.size(1) == 1 else pred.argmax(dim=1)
                
                # Separate by sensitive attribute
                mask_0 = (sensitive == 0)
                mask_1 = (sensitive == 1)
                
                if mask_0.any():
                    preds_sensitive_0.extend(pred_class[mask_0].cpu().numpy())
                if mask_1.any():
                    preds_sensitive_1.extend(pred_class[mask_1].cpu().numpy())
        
        # Calculate positive rates
        rate_0 = np.mean(preds_sensitive_0) if len(preds_sensitive_0) > 0 else 0
        rate_1 = np.mean(preds_sensitive_1) if len(preds_sensitive_1) > 0 else 0
        
        return abs(rate_0 - rate_1)
    
    @staticmethod
    def equalized_odds(model, data_loader, sensitive_attr_idx, device='cpu'):
        """
        Measure equalized odds: TPR and FPR should be equal across sensitive groups.
        
        Args:
            model: Trained model
            data_loader: DataLoader with (x, y, sensitive_attr) tuples
            sensitive_attr_idx: Index of sensitive attribute
            device: Computation device
            
        Returns:
            dict: {'tpr_diff': float, 'fpr_diff': float}
        """
        model.eval()
        
        tp_0, fp_0, tn_0, fn_0 = 0, 0, 0, 0
        tp_1, fp_1, tn_1, fn_1 = 0, 0, 0, 0
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:
                    x, y, sensitive = batch
                else:
                    x, y = batch
                    sensitive = x[:, sensitive_attr_idx]
                
                x, y = x.to(device), y.to(device)
                sensitive = sensitive.to(device)
                
                pred = model(x)
                pred_class = (pred > 0.5).float() if pred.size(1) == 1 else pred.argmax(dim=1)
                
                # Calculate confusion matrix for each group
                for s_val, (tp, fp, tn, fn) in [(0, (tp_0, fp_0, tn_0, fn_0)), 
                                                 (1, (tp_1, fp_1, tn_1, fn_1))]:
                    mask = (sensitive == s_val)
                    if mask.any():
                        y_masked = y[mask]
                        pred_masked = pred_class[mask]
                        
                        if s_val == 0:
                            tp_0 += ((pred_masked == 1) & (y_masked == 1)).sum().item()
                            fp_0 += ((pred_masked == 1) & (y_masked == 0)).sum().item()
                            tn_0 += ((pred_masked == 0) & (y_masked == 0)).sum().item()
                            fn_0 += ((pred_masked == 0) & (y_masked == 1)).sum().item()
                        else:
                            tp_1 += ((pred_masked == 1) & (y_masked == 1)).sum().item()
                            fp_1 += ((pred_masked == 1) & (y_masked == 0)).sum().item()
                            tn_1 += ((pred_masked == 0) & (y_masked == 0)).sum().item()
                            fn_1 += ((pred_masked == 0) & (y_masked == 1)).sum().item()
        
        # Calculate TPR and FPR for each group
        tpr_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0
        tpr_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0
        
        fpr_0 = fp_0 / (fp_0 + tn_0) if (fp_0 + tn_0) > 0 else 0
        fpr_1 = fp_1 / (fp_1 + tn_1) if (fp_1 + tn_1) > 0 else 0
        
        return {
            'tpr_diff': abs(tpr_0 - tpr_1),
            'fpr_diff': abs(fpr_0 - fpr_1)
        }
