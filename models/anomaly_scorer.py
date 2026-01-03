import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Protocol, runtime_checkable
from torch_geometric.nn import knn


@runtime_checkable
class EmbeddingDataProtocol(Protocol):
    """Protocol for embedding data with causal information."""
    z: torch.Tensor                # [N, d_z] node embeddings
    parents: torch.Tensor          # [N, max_parents] parent indices (-1 for padding)
    parent_embeddings: torch.Tensor # [N, max_parents, d_z] parent embeddings
    timestamp: Optional[int]       # Current time step


class CausalMemoryBank:
    """
    Memory-efficient storage for reference causal distributions.
    Stores Gaussian parameters (mean, variance) for K prototypes.
    """
    
    def __init__(self, K: int, emb_dim: int, device: torch.device = None):
        """
        Initialize causal memory bank.
        
        Args:
            K: Number of prototypes
            emb_dim: Embedding dimension
            device: Device to store prototypes on
        """
        self.K = K
        self.emb_dim = emb_dim
        self.device = device or torch.device('cpu')
        
        # Prototype parameters
        self.prototypes = torch.zeros(K, emb_dim, device=self.device)      # μ_k
        self.variances = torch.ones(K, emb_dim, device=self.device)        # σ_k^2
        self.weights = torch.zeros(K, device=self.device)                  # w_k (usage counts)
        self.initialized = False
        
    def add_prototype(self, mean: torch.Tensor, variance: torch.Tensor, weight: float = 1.0):
        """
        Add or update a prototype in the memory bank.
        
        Args:
            mean: [emb_dim] prototype mean
            variance: [emb_dim] prototype variance
            weight: Usage weight for this prototype
        """
        if not self.initialized:
            # Initialize with first prototype
            self.prototypes[0] = mean
            self.variances[0] = variance
            self.weights[0] = weight
            self.initialized = True
            return 0  # Return prototype index
            
        # Find nearest existing prototype
        distances = torch.sum((self.prototypes - mean.unsqueeze(0)) ** 2, dim=1)
        nearest_idx = torch.argmin(distances).item()
        
        # Update nearest prototype with exponential moving average
        lr = 0.1  # Learning rate for prototype updates
        self.prototypes[nearest_idx] = (1 - lr) * self.prototypes[nearest_idx] + lr * mean
        self.variances[nearest_idx] = (1 - lr) * self.variances[nearest_idx] + lr * variance
        self.weights[nearest_idx] += weight
        
        return nearest_idx

    def find_nearest_prototype(self, query: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Find nearest prototype to query embedding.
        
        Args:
            query: [emb_dim] query embedding
            
        Returns:
            idx: Prototype index
            distance: Distance to nearest prototype
        """
        if not self.initialized:
            return 0, torch.tensor(float('inf'), device=self.device)
            
        distances = torch.sum((self.prototypes - query.unsqueeze(0)) ** 2, dim=1)
        nearest_idx = torch.argmin(distances).item()
        min_distance = distances[nearest_idx]
        
        return nearest_idx, min_distance

    def get_prototype_params(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get Gaussian parameters for prototype.
        
        Args:
            idx: Prototype index
            
        Returns:
            mean: [emb_dim] prototype mean
            variance: [emb_dim] prototype variance
        """
        if not self.initialized or idx >= self.K:
            # Return default parameters
            mean = torch.zeros(self.emb_dim, device=self.device)
            variance = torch.ones(self.emb_dim, device=self.device)
            return mean, variance
            
        return self.prototypes[idx].clone(), self.variances[idx].clone()

    def get_all_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all prototype parameters.
        
        Returns:
            prototypes: [K, emb_dim] prototype means
            variances: [K, emb_dim] prototype variances
            weights: [K] prototype weights
        """
        return self.prototypes.clone(), self.variances.clone(), self.weights.clone()


class CausalFidelityScorer:
    """
    Causal Fidelity Scoring (Section 5.4).
    Computes anomaly scores based on KL divergence from reference causal distributions.
    """
    
    def __init__(
        self,
        memory_bank: CausalMemoryBank,
        kl_threshold: float = 0.5,
        min_variance: float = 1e-6,
        calibration_alpha: float = 0.05
    ):
        """
        Initialize causal fidelity scorer.
        
        Args:
            memory_bank: Reference distribution storage
            kl_threshold: Threshold ε for anomaly detection (Section 5.4)
            min_variance: Minimum variance to prevent numerical issues
            calibration_alpha: Significance level for statistical calibration
        """
        self.memory_bank = memory_bank
        self.kl_threshold = kl_threshold
        self.min_variance = min_variance
        self.calibration_alpha = calibration_alpha
        
    def _compute_kl_divergence(
        self,
        mu1: torch.Tensor,
        var1: torch.Tensor,
        mu2: torch.Tensor,
        var2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between two diagonal Gaussian distributions.
        
        KL(N1 || N2) = 0.5 * [log(|Σ2|/|Σ1|) - d + Tr(Σ2^{-1} Σ1) + (μ2-μ1)^T Σ2^{-1} (μ2-μ1)]
        
        For diagonal Gaussians:
        KL = 0.5 * sum[ log(σ2_i^2/σ1_i^2) - 1 + σ1_i^2/σ2_i^2 + (μ2_i-μ1_i)^2/σ2_i^2 ]
        
        Args:
            mu1: [d] mean of distribution 1
            var1: [d] variance of distribution 1
            mu2: [d] mean of distribution 2
            var2: [d] variance of distribution 2
            
        Returns:
            kl: KL divergence value
        """
        # Ensure minimum variance for numerical stability
        var1 = torch.clamp(var1, min=self.min_variance)
        var2 = torch.clamp(var2, min=self.min_variance)
        
        # Compute KL divergence components
        log_ratio = torch.log(var2 / var1)
        inv_var2 = 1.0 / var2
        diff_sq = (mu2 - mu1) ** 2
        
        kl = 0.5 * (
            torch.sum(log_ratio) +
            torch.sum(var1 * inv_var2) +
            torch.sum(diff_sq * inv_var2) -
            mu1.size(0)
        )
        
        return kl

    def _estimate_current_distribution(
        self,
        z: torch.Tensor,
        parent_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate current causal distribution parameters.
        
        Args:
            z: [d_z] current node embedding
            parent_embeddings: [P, d_z] parent embeddings (P = number of parents)
            
        Returns:
            mu_current: [d_z] estimated mean
            var_current: [d_z] estimated variance
        """
        if parent_embeddings.size(0) == 0:
            # No parents case - use node embedding as mean, unit variance
            mu_current = z
            var_current = torch.ones_like(z) * 0.1  # Small variance
            return mu_current, var_current
        
        # Estimate distribution from parent context
        # Simple approach: use mean of parents as context, add noise
        parent_mean = parent_embeddings.mean(dim=0)
        parent_var = parent_embeddings.var(dim=0, unbiased=False)
        
        # Current distribution: centered around node embedding with parent variance
        mu_current = z
        var_current = torch.clamp(parent_var, min=self.min_variance)
        
        return mu_current, var_current

    def compute_causal_fidelity(
        self,
        z: torch.Tensor,
        parent_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute causal fidelity F_i^{(t)} = exp(-KL(P_do^{(t)} || P_hat)).
        
        Args:
            z: [d_z] node embedding
            parent_embeddings: [P, d_z] parent embeddings
            
        Returns:
            fidelity: Causal fidelity score ∈ (0, 1]
            prototype_idx: Index of nearest prototype used
        """
        # Estimate current distribution
        mu_current, var_current = self._estimate_current_distribution(z, parent_embeddings)
        
        # Find nearest prototype in memory bank
        prototype_idx, _ = self.memory_bank.find_nearest_prototype(parent_embeddings.mean(dim=0))
        mu_ref, var_ref = self.memory_bank.get_prototype_params(prototype_idx)
        
        # Compute KL divergence
        kl_div = self._compute_kl_divergence(mu_current, var_current, mu_ref, var_ref)
        
        # Compute causal fidelity
        fidelity = torch.exp(-kl_div)
        fidelity = torch.clamp(fidelity, min=1e-8, max=1.0)  # Numerical stability
        
        return fidelity, prototype_idx

    def compute_anomaly_score(
        self,
        z: torch.Tensor,
        parent_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute final anomaly score s_i^{(t)} = 1 - F_i^{(t)}.
        
        Args:
            z: [d_z] node embedding
            parent_embeddings: [P, d_z] parent embeddings
            
        Returns:
            anomaly_score: Anomaly score ∈ [0, 1)
        """
        fidelity, _ = self.compute_causal_fidelity(z, parent_embeddings)
        anomaly_score = 1.0 - fidelity
        return anomaly_score

    def is_anomalous(
        self,
        z: torch.Tensor,
        parent_embeddings: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Tuple[bool, torch.Tensor]:
        """
        Determine if node is anomalous based on KL threshold.
        
        Args:
            z: [d_z] node embedding
            parent_embeddings: [P, d_z] parent embeddings
            threshold: Custom KL threshold (overrides self.kl_threshold)
            
        Returns:
            is_anomalous: Boolean indicating anomaly status
            kl_div: KL divergence value for debugging
        """
        # Estimate current distribution
        mu_current, var_current = self._estimate_current_distribution(z, parent_embeddings)
        
        # Find nearest prototype
        prototype_idx, _ = self.memory_bank.find_nearest_prototype(parent_embeddings.mean(dim=0))
        mu_ref, var_ref = self.memory_bank.get_prototype_params(prototype_idx)
        
        # Compute KL divergence
        kl_div = self._compute_kl_divergence(mu_current, var_current, mu_ref, var_ref)
        
        # Determine anomaly status
        threshold = threshold or self.kl_threshold
        is_anomalous = kl_div > threshold
        
        return is_anomalous, kl_div

    def calibrate_threshold(
        self,
        normal_kl_values: torch.Tensor,
        alpha: Optional[float] = None
    ) -> float:
        """
        Calibrate KL threshold using normal data (Section 5.4).
        
        Uses (1-α)-quantile of KL distribution under normal data.
        
        Args:
            normal_kl_values: [N] KL values from normal nodes
            alpha: Significance level (overrides self.calibration_alpha)
            
        Returns:
            calibrated_threshold: Calibrated KL threshold
        """
        alpha = alpha or self.calibration_alpha
        # Compute (1-alpha)-quantile
        calibrated_threshold = torch.quantile(normal_kl_values, 1.0 - alpha)
        return calibrated_threshold.item()

    def compute_fidelity_for_batch(
        self,
        z_batch: torch.Tensor,
        parent_embeddings_batch: torch.Tensor,
        parent_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute causal fidelity for a batch of nodes.
        
        Args:
            z_batch: [B, d_z] batch of node embeddings
            parent_embeddings_batch: [B, max_parents, d_z] batch of parent embeddings
            parent_lengths: [B] number of actual parents per node
            
        Returns:
            fidelities: [B] causal fidelity scores
        """
        B = z_batch.size(0)
        fidelities = torch.zeros(B, device=z_batch.device)
        
        for i in range(B):
            # Extract actual parents (remove padding)
            num_parents = parent_lengths[i].item()
            if num_parents == 0:
                parents_i = torch.empty(0, self.memory_bank.emb_dim, device=z_batch.device)
            else:
                parents_i = parent_embeddings_batch[i, :num_parents]
            
            fidelity, _ = self.compute_causal_fidelity(z_batch[i], parents_i)
            fidelities[i] = fidelity
            
        return fidelities

    def update_memory_bank(
        self,
        parent_embeddings: torch.Tensor,
        z: torch.Tensor,
        weight: float = 1.0
    ):
        """
        Update memory bank with new causal experience.
        
        Args:
            parent_embeddings: [P, d_z] parent embeddings
            z: [d_z] node embedding
            weight: Usage weight for this experience
        """
        if parent_embeddings.size(0) == 0:
            return
            
        # Use parent mean as prototype key
        parent_mean = parent_embeddings.mean(dim=0)
        
        # Estimate variance from parents
        if parent_embeddings.size(0) > 1:
            parent_var = parent_embeddings.var(dim=0, unbiased=False)
        else:
            parent_var = torch.ones_like(parent_mean) * 0.1
            
        # Add to memory bank
        self.memory_bank.add_prototype(parent_mean, parent_var, weight)


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic data
    d_z, P = 64, 5
    z = torch.randn(d_z)
    parent_embeddings = torch.randn(P, d_z)
    
    # Initialize memory bank and scorer
    memory_bank = CausalMemoryBank(K=10, emb_dim=d_z)
    scorer = CausalFidelityScorer(memory_bank, kl_threshold=0.5)
    
    # Update memory bank with reference distribution
    scorer.update_memory_bank(parent_embeddings, z, weight=1.0)
    
    # Compute causal fidelity
    fidelity, proto_idx = scorer.compute_causal_fidelity(z, parent_embeddings)
    print(f"Causal fidelity: {fidelity:.4f}, Prototype: {proto_idx}")
    
    # Compute anomaly score
    anomaly_score = scorer.compute_anomaly_score(z, parent_embeddings)
    print(f"Anomaly score: {anomaly_score:.4f}")
    
    # Check if anomalous
    is_anomalous, kl_div = scorer.is_anomalous(z, parent_embeddings)
    print(f"Is anomalous: {is_anomalous}, KL divergence: {kl_div:.4f}")
    
    # Test batch computation
    B = 10
    z_batch = torch.randn(B, d_z)
    parent_batch = torch.randn(B, 8, d_z)  # max 8 parents
    parent_lengths = torch.randint(1, 6, (B,))  # 1-5 actual parents
    
    fidelities = scorer.compute_fidelity_for_batch(z_batch, parent_batch, parent_lengths)
    print(f"Batch fidelities shape: {fidelities.shape}")
    print(f"Batch fidelities range: [{fidelities.min():.4f}, {fidelities.max():.4f}]")
    
    # Test calibration
    normal_kl = torch.abs(torch.randn(100))  # Simulated normal KL values
    calibrated_threshold = scorer.calibrate_threshold(normal_kl, alpha=0.05)
    print(f"Calibrated threshold: {calibrated_threshold:.4f}")