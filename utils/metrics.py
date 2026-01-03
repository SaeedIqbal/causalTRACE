import torch
import numpy as np
from typing import Tuple, List, Optional, Union, Protocol, runtime_checkable
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import gc


@runtime_checkable
class PredictionDataProtocol(Protocol):
    """Protocol for prediction data with ground truth."""
    scores: torch.Tensor      # [N] anomaly scores
    labels: torch.Tensor      # [N] ground truth labels (0/1)


class MetricsTracker:
    """
    Comprehensive metrics tracker for CausalTRACE evaluation.
    Computes AUC, causal fidelity, memory usage, and statistical confidence intervals.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        memory_tracking_enabled: bool = True,
        confidence_level: float = 0.95
    ):
        """
        Initialize metrics tracker.
        
        Args:
            device: Device to track memory on
            memory_tracking_enabled: Whether to track GPU/CPU memory
            confidence_level: Confidence level for statistical intervals
        """
        self.device = device or torch.device('cpu')
        self.memory_tracking_enabled = memory_tracking_enabled
        self.confidence_level = confidence_level
        self._reset_tracking()
        
        # Store memory baseline
        if self.memory_tracking_enabled:
            self._memory_baseline = self._get_memory_usage()
        
    def _reset_tracking(self):
        """Reset internal tracking variables."""
        self.predictions: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.fidelities: List[torch.Tensor] = []
        self.memory_snapshots: List[dict] = []
        self.timestamps: List[int] = []
        
    def _get_memory_usage(self) -> dict:
        """
        Get current memory usage.
        
        Returns:
            memory_info: Dictionary with memory usage in bytes
        """
        if not self.memory_tracking_enabled:
            return {'cpu': 0, 'gpu': 0}
            
        # CPU memory
        cpu_memory = self._get_cpu_memory()
        
        # GPU memory (if available)
        gpu_memory = 0
        if self.device.type == 'cuda':
            try:
                gpu_memory = torch.cuda.memory_allocated(self.device)
            except RuntimeError:
                # CUDA not available
                gpu_memory = 0
                
        return {'cpu': cpu_memory, 'gpu': gpu_memory}
    
    def _get_cpu_memory(self) -> int:
        """Get current CPU memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # psutil not available, return approximate value
            return 0
    
    def update(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        fidelities: Optional[torch.Tensor] = None,
        timestamp: Optional[int] = None
    ):
        """
        Update metrics with new predictions.
        
        Args:
            scores: [N] anomaly scores (higher = more anomalous)
            labels: [N] ground truth labels (0 = normal, 1 = anomalous)
            fidelities: [N] causal fidelity scores (optional)
            timestamp: Current timestamp (optional)
        """
        # Validate inputs
        if scores.size(0) != labels.size(0):
            raise ValueError(f"Scores ({scores.size(0)}) and labels ({labels.size(0)}) must have same length")
            
        if fidelities is not None and fidelities.size(0) != scores.size(0):
            raise ValueError(f"Fidelities ({fidelities.size(0)}) must match scores length ({scores.size(0)})")
        
        # Store predictions and labels
        self.predictions.append(scores.detach().cpu())
        self.labels.append(labels.detach().cpu())
        
        if fidelities is not None:
            self.fidelities.append(fidelities.detach().cpu())
            
        if timestamp is not None:
            self.timestamps.append(timestamp)
            
        # Track memory usage
        if self.memory_tracking_enabled:
            current_memory = self._get_memory_usage()
            # Subtract baseline to get incremental usage
            memory_snapshot = {
                'cpu': max(0, current_memory['cpu'] - self._memory_baseline['cpu']),
                'gpu': max(0, current_memory['gpu'] - self._memory_baseline['gpu'])
            }
            self.memory_snapshots.append(memory_snapshot)
    
    def compute_auc(self, return_ci: bool = True) -> Union[float, Tuple[float, Tuple[float, float]]]:
        """
        Compute AUC-ROC with optional confidence interval.
        
        Uses DeLong's method for confidence intervals.
        
        Args:
            return_ci: Whether to return confidence interval
            
        Returns:
            auc: AUC-ROC score
            ci: (lower, upper) confidence interval if return_ci=True
        """
        if len(self.predictions) == 0:
            return (0.0, (0.0, 0.0)) if return_ci else 0.0
            
        # Concatenate all predictions and labels
        all_scores = torch.cat(self.predictions, dim=0).numpy()
        all_labels = torch.cat(self.labels, dim=0).numpy()
        
        # Handle edge cases
        if len(np.unique(all_labels)) < 2:
            # All labels are the same
            auc = 1.0 if all_labels[0] == 0 else 0.0
            ci = (auc, auc)
        else:
            # Compute AUC
            auc = roc_auc_score(all_labels, all_scores)
            
            if return_ci:
                # Compute confidence interval using bootstrap
                ci = self._compute_auc_ci(all_scores, all_labels, confidence_level=self.confidence_level)
            else:
                ci = (auc, auc)
        
        return (auc, ci) if return_ci else auc
    
    def _compute_auc_ci(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Compute AUC confidence interval using bootstrap resampling.
        
        Args:
            scores: [N] anomaly scores
            labels: [N] ground truth labels
            confidence_level: Confidence level for interval
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            ci: (lower, upper) confidence interval
        """
        if len(scores) < 2:
            return (0.0, 0.0)
            
        # Bootstrap resampling
        auc_samples = []
        n_samples = len(scores)
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_scores = scores[indices]
            bootstrap_labels = labels[indices]
            
            # Skip if all labels are the same
            if len(np.unique(bootstrap_labels)) < 2:
                continue
                
            try:
                auc_sample = roc_auc_score(bootstrap_labels, bootstrap_scores)
                auc_samples.append(auc_sample)
            except ValueError:
                # Skip invalid bootstrap sample
                continue
        
        if len(auc_samples) == 0:
            return (0.0, 0.0)
            
        # Compute confidence interval
        alpha = 1.0 - confidence_level
        lower_percentile = (alpha / 2.0) * 100
        upper_percentile = (1.0 - alpha / 2.0) * 100
        
        lower_ci = np.percentile(auc_samples, lower_percentile)
        upper_ci = np.percentile(auc_samples, upper_percentile)
        
        return (float(lower_ci), float(upper_ci))
    
    def compute_causal_fidelity(self) -> Tuple[float, float]:
        """
        Compute average causal fidelity and standard deviation.
        
        Returns:
            mean_fidelity: Mean causal fidelity
            std_fidelity: Standard deviation of causal fidelity
        """
        if len(self.fidelities) == 0:
            return 0.0, 0.0
            
        all_fidelities = torch.cat(self.fidelities, dim=0)
        mean_fidelity = all_fidelities.mean().item()
        std_fidelity = all_fidelities.std().item()
        
        return mean_fidelity, std_fidelity
    
    def compute_memory_usage(self) -> dict:
        """
        Compute peak and average memory usage.
        
        Returns:
            memory_stats: Dictionary with memory statistics in MB
        """
        if not self.memory_tracking_enabled or len(self.memory_snapshots) == 0:
            return {'peak_cpu_mb': 0.0, 'peak_gpu_mb': 0.0, 'avg_cpu_mb': 0.0, 'avg_gpu_mb': 0.0}
            
        cpu_values = [snap['cpu'] for snap in self.memory_snapshots]
        gpu_values = [snap['gpu'] for snap in self.memory_snapshots]
        
        # Convert bytes to MB
        cpu_mb = [v / (1024 * 1024) for v in cpu_values]
        gpu_mb = [v / (1024 * 1024) for v in gpu_values]
        
        return {
            'peak_cpu_mb': float(max(cpu_mb)) if cpu_mb else 0.0,
            'peak_gpu_mb': float(max(gpu_mb)) if gpu_mb else 0.0,
            'avg_cpu_mb': float(np.mean(cpu_mb)) if cpu_mb else 0.0,
            'avg_gpu_mb': float(np.mean(gpu_mb)) if gpu_mb else 0.0
        }
    
    def compute_precision_recall_at_k(self, k: int = 100) -> Tuple[float, float]:
        """
        Compute Precision@K and Recall@K.
        
        Args:
            k: Number of top predictions to consider
            
        Returns:
            precision_at_k: Precision@K
            recall_at_k: Recall@K
        """
        if len(self.predictions) == 0:
            return 0.0, 0.0
            
        all_scores = torch.cat(self.predictions, dim=0)
        all_labels = torch.cat(self.labels, dim=0)
        
        # Get top-K predictions
        if k >= len(all_scores):
            top_k_indices = torch.argsort(all_scores, descending=True)
        else:
            top_k_indices = torch.topk(all_scores, k, largest=True).indices
            
        top_k_labels = all_labels[top_k_indices]
        true_positives = top_k_labels.sum().item()
        
        precision_at_k = true_positives / min(k, len(all_scores))
        total_positives = all_labels.sum().item()
        recall_at_k = true_positives / total_positives if total_positives > 0 else 0.0
        
        return precision_at_k, recall_at_k
    
    def compute_f1_score(self, threshold: Optional[float] = None) -> float:
        """
        Compute F1 score at optimal threshold or given threshold.
        
        Args:
            threshold: Threshold for binary classification (if None, use optimal)
            
        Returns:
            f1_score: F1 score
        """
        if len(self.predictions) == 0:
            return 0.0
            
        all_scores = torch.cat(self.predictions, dim=0).numpy()
        all_labels = torch.cat(self.labels, dim=0).numpy()
        
        if threshold is None:
            # Find optimal threshold using Youden's J statistic
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            threshold = thresholds[optimal_idx]
        
        # Apply threshold
        predictions = (all_scores >= threshold).astype(int)
        
        # Compute F1 score
        true_positives = np.sum((predictions == 1) & (all_labels == 1))
        false_positives = np.sum((predictions == 1) & (all_labels == 0))
        false_negatives = np.sum((predictions == 0) & (all_labels == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def get_summary(self) -> dict:
        """
        Get comprehensive summary of all metrics.
        
        Returns:
            summary: Dictionary with all computed metrics
        """
        # AUC with confidence interval
        auc, auc_ci = self.compute_auc(return_ci=True)
        
        # Causal fidelity
        mean_fid, std_fid = self.compute_causal_fidelity()
        
        # Memory usage
        memory_stats = self.compute_memory_usage()
        
        # Precision and recall at K
        prec_at_100, rec_at_100 = self.compute_precision_recall_at_k(k=100)
        prec_at_1000, rec_at_1000 = self.compute_precision_recall_at_k(k=1000)
        
        # F1 score
        f1_score = self.compute_f1_score()
        
        return {
            'auc': auc,
            'auc_ci_lower': auc_ci[0],
            'auc_ci_upper': auc_ci[1],
            'mean_causal_fidelity': mean_fid,
            'std_causal_fidelity': std_fid,
            'f1_score': f1_score,
            'precision_at_100': prec_at_100,
            'recall_at_100': rec_at_100,
            'precision_at_1000': prec_at_1000,
            'recall_at_1000': rec_at_1000,
            **memory_stats
        }
    
    def reset(self):
        """Reset all tracked metrics."""
        self._reset_tracking()
        # Update memory baseline
        if self.memory_tracking_enabled:
            self._memory_baseline = self._get_memory_usage()
    
    def save_to_file(self, filepath: str):
        """
        Save metrics summary to file.
        
        Args:
            filepath: Path to save metrics summary
        """
        summary = self.get_summary()
        import json
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def __str__(self) -> str:
        """String representation of metrics summary."""
        summary = self.get_summary()
        return f"""Metrics Summary:
  AUC: {summary['auc']:.4f} [{summary['auc_ci_lower']:.4f}, {summary['auc_ci_upper']:.4f}]
  Mean Causal Fidelity: {summary['mean_causal_fidelity']:.4f} ± {summary['std_causal_fidelity']:.4f}
  F1 Score: {summary['f1_score']:.4f}
  Precision@100: {summary['precision_at_100']:.4f}, Recall@100: {summary['recall_at_100']:.4f}
  Peak GPU Memory: {summary['peak_gpu_mb']:.2f} MB, Peak CPU Memory: {summary['peak_cpu_mb']:.2f} MB"""


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic data for checking and verifying the code, replace this with original data...
    #N = 1000
    #scores = torch.rand(N)
    #labels = torch.randint(0, 2, (N,))
    #fidelities = torch.rand(N)
    
    # Initialize metrics tracker
    tracker = MetricsTracker(memory_tracking_enabled=True)
    
    # Update with data
    tracker.update(scores, labels, fidelities, timestamp=0)
    
    # Compute metrics
    auc, ci = tracker.compute_auc(return_ci=True)
    print(f"AUC: {auc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    mean_fid, std_fid = tracker.compute_causal_fidelity()
    print(f"Causal Fidelity: {mean_fid:.4f} ± {std_fid:.4f}")
    
    memory_stats = tracker.compute_memory_usage()
    print(f"Peak GPU Memory: {memory_stats['peak_gpu_mb']:.2f} MB")
    
    prec_100, rec_100 = tracker.compute_precision_recall_at_k(k=100)
    print(f"Precision@100: {prec_100:.4f}, Recall@100: {rec_100:.4f}")
    
    f1 = tracker.compute_f1_score()
    print(f"F1 Score: {f1:.4f}")
    
    # Print summary
    print(tracker)
    
    # Test reset
    tracker.reset()
    print(f"Tracker reset - predictions: {len(tracker.predictions)}")