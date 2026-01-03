import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Protocol, runtime_checkable
import math


@runtime_checkable
class CausalMechanismProtocol(Protocol):
    """Protocol for causal mechanism representations."""
    parent_embeddings: torch.Tensor  # [P, d_z] parent embeddings
    node_embedding: torch.Tensor     # [d_z] node embedding
    timestamp: Optional[int]         # Time of observation


class CausalMemoryBank:
    """
    Compressed Causal Memory (Section 5.3).
    Stores K Gaussian prototypes representing invariant causal mechanisms.
    Supports online updates, nearest neighbor search, and memory compression.
    """
    
    def __init__(
        self,
        K: int,
        emb_dim: int,
        device: torch.device = None,
        learning_rate: float = 0.01,
        covariance_regularization: float = 1e-6,
        min_cluster_weight: float = 1e-8,
        max_distance_threshold: float = 10.0
    ):
        """
        Initialize compressed causal memory bank.
        
        Args:
            K: Number of prototypes (clusters)
            emb_dim: Embedding dimension d_z
            device: Device to store prototypes on
            learning_rate: Learning rate for online prototype updates
            covariance_regularization: Regularization for covariance matrices
            min_cluster_weight: Minimum weight to prevent numerical issues
            max_distance_threshold: Maximum distance for prototype assignment
        """
        self.K = K
        self.emb_dim = emb_dim
        self.device = device or torch.device('cpu')
        self.learning_rate = learning_rate
        self.cov_regularization = covariance_regularization
        self.min_weight = min_cluster_weight
        self.max_distance = max_distance_threshold
        
        # Initialize prototype parameters
        self._initialize_prototypes()
        
    def _initialize_prototypes(self):
        """Initialize K prototypes with random parameters."""
        # Prototype means: [K, emb_dim]
        self.prototypes = torch.randn(self.K, self.emb_dim, device=self.device) * 0.1
        
        # Prototype covariances: [K, emb_dim, emb_dim] (diagonal for efficiency)
        self.covariances = torch.eye(self.emb_dim, device=self.device).unsqueeze(0).repeat(self.K, 1, 1)
        self.covariances *= 0.1  # Small initial variance
        
        # Cluster weights (usage counts): [K]
        self.weights = torch.zeros(self.K, device=self.device)
        
        # Total number of updates
        self.total_updates = 0
        self.initialized = False
        
    def _flatten_parent_embeddings(self, parent_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Flatten parent embeddings into a single vector for clustering.
        
        Args:
            parent_embeddings: [P, d_z] parent embeddings
            
        Returns:
            flattened: [P * d_z] flattened vector
        """
        if parent_embeddings.size(0) == 0:
            # Return zero vector for no parents
            return torch.zeros(self.emb_dim, device=parent_embeddings.device)
            
        # Use mean of parent embeddings as representation
        return parent_embeddings.mean(dim=0)
    
    def _compute_mahalanobis_distance(
        self,
        query: torch.Tensor,
        prototype_idx: int
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distance between query and prototype.
        
        d(x, μ) = (x - μ)^T Σ^{-1} (x - μ)
        
        Args:
            query: [d] query vector
            prototype_idx: Prototype index
            
        Returns:
            distance: Mahalanobis distance
        """
        mu = self.prototypes[prototype_idx]
        cov = self.covariances[prototype_idx]
        
        # Regularize covariance matrix for numerical stability
        cov_reg = cov + self.cov_regularization * torch.eye(self.emb_dim, device=self.device)
        
        # Compute inverse covariance using Cholesky decomposition for stability
        try:
            L = torch.cholesky(cov_reg)
            diff = query - mu
            # Solve L y = diff, then L^T z = y, distance = z^T z
            y = torch.triangular_solve(diff.unsqueeze(-1), L, upper=False).solution.squeeze(-1)
            z = torch.triangular_solve(y.unsqueeze(-1), L.t(), upper=True).solution.squeeze(-1)
            distance = torch.sum(z * z)
        except RuntimeError:
            # Fall back to pseudo-inverse if Cholesky fails
            cov_inv = torch.pinverse(cov_reg)
            diff = query - mu
            distance = torch.sum(diff * (cov_inv @ diff))
            
        return distance
    
    def _find_nearest_prototype(self, query: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Find nearest prototype using Mahalanobis distance.
        
        Args:
            query: [d] query vector
            
        Returns:
            nearest_idx: Index of nearest prototype
            min_distance: Minimum distance
        """
        if not self.initialized:
            return 0, torch.tensor(float('inf'), device=self.device)
            
        distances = torch.zeros(self.K, device=self.device)
        for k in range(self.K):
            distances[k] = self._compute_mahalanobis_distance(query, k)
            
        nearest_idx = torch.argmin(distances).item()
        min_distance = distances[nearest_idx]
        
        # If distance exceeds threshold, return invalid assignment
        if min_distance > self.max_distance:
            return -1, min_distance
            
        return nearest_idx, min_distance
    
    def _update_prototype(
        self,
        prototype_idx: int,
        new_mean: torch.Tensor,
        new_covariance: torch.Tensor,
        weight: float = 1.0
    ):
        """
        Update prototype parameters using exponential moving average.
        
        Args:
            prototype_idx: Prototype index to update
            new_mean: [d] new mean vector
            new_covariance: [d, d] new covariance matrix
            weight: Weight for this update
        """
        lr = self.learning_rate
        
        # Update mean with EMA
        self.prototypes[prototype_idx] = (
            (1 - lr) * self.prototypes[prototype_idx] + lr * new_mean
        )
        
        # Update covariance with EMA
        self.covariances[prototype_idx] = (
            (1 - lr) * self.covariances[prototype_idx] + lr * new_covariance
        )
        
        # Update weight
        self.weights[prototype_idx] += weight
        self.total_updates += 1
        self.initialized = True
        
    def add_mechanism(
        self,
        parent_embeddings: torch.Tensor,
        node_embedding: torch.Tensor,
        weight: float = 1.0
    ) -> int:
        """
        Add a causal mechanism to the memory bank.
        
        Args:
            parent_embeddings: [P, d_z] parent embeddings
            node_embedding: [d_z] node embedding
            weight: Weight for this mechanism
            
        Returns:
            assigned_prototype: Prototype index assigned to this mechanism
        """
        # Flatten parent embeddings to create mechanism representation
        mechanism_repr = self._flatten_parent_embeddings(parent_embeddings)
        
        # Find nearest prototype
        nearest_idx, distance = self._find_nearest_prototype(mechanism_repr)
        
        if nearest_idx == -1:
            # No suitable prototype found, assign to least used prototype
            if self.total_updates < self.K:
                # Use next available prototype
                nearest_idx = self.total_updates
            else:
                # Assign to prototype with minimum weight
                nearest_idx = torch.argmin(self.weights).item()
        
        # Estimate covariance from the mechanism
        if parent_embeddings.size(0) > 1:
            # Compute sample covariance from parents
            parent_mean = parent_embeddings.mean(dim=0)
            centered = parent_embeddings - parent_mean
            covariance = (centered.t() @ centered) / (parent_embeddings.size(0) - 1)
        else:
            # Single parent or no parents: use isotropic covariance
            covariance = torch.eye(self.emb_dim, device=self.device) * 0.1
        
        # Update the assigned prototype
        self._update_prototype(nearest_idx, mechanism_repr, covariance, weight)
        
        return nearest_idx
    
    def get_prototype_params(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get parameters for a specific prototype.
        
        Args:
            idx: Prototype index
            
        Returns:
            mean: [d_z] prototype mean
            covariance: [d_z, d_z] prototype covariance
            weight: Prototype weight
        """
        if idx < 0 or idx >= self.K:
            # Return default parameters for invalid index
            mean = torch.zeros(self.emb_dim, device=self.device)
            covariance = torch.eye(self.emb_dim, device=self.device) * 0.1
            weight = self.min_weight
            return mean, covariance, weight
            
        mean = self.prototypes[idx].clone()
        covariance = self.covariances[idx].clone()
        weight = max(self.weights[idx].item(), self.min_weight)
        
        return mean, covariance, weight
    
    def get_all_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all prototype parameters.
        
        Returns:
            prototypes: [K, d_z] prototype means
            covariances: [K, d_z, d_z] prototype covariances
            weights: [K] prototype weights
        """
        weights_safe = torch.clamp(self.weights, min=self.min_weight)
        return self.prototypes.clone(), self.covariances.clone(), weights_safe.clone()
    
    def compute_prototype_assignments(
        self,
        mechanism_representations: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prototype assignments for a batch of mechanism representations.
        
        Args:
            mechanism_representations: [B, d_z] batch of mechanism representations
            
        Returns:
            assignments: [B] prototype indices
        """
        B = mechanism_representations.size(0)
        assignments = torch.zeros(B, dtype=torch.long, device=self.device)
        
        for i in range(B):
            nearest_idx, _ = self._find_nearest_prototype(mechanism_representations[i])
            if nearest_idx == -1:
                # Assign to most used prototype if no good match
                nearest_idx = torch.argmax(self.weights).item()
            assignments[i] = nearest_idx
            
        return assignments
    
    def get_memory_usage_bytes(self) -> int:
        """
        Get current memory usage in bytes.
        
        Returns:
            memory_bytes: Memory usage in bytes
        """
        # Prototypes: K * d_z * 4 bytes (float32)
        # Covariances: K * d_z * d_z * 4 bytes
        # Weights: K * 4 bytes
        prototypes_bytes = self.K * self.emb_dim * 4
        covariances_bytes = self.K * self.emb_dim * self.emb_dim * 4
        weights_bytes = self.K * 4
        
        return prototypes_bytes + covariances_bytes + weights_bytes
    
    def get_compression_ratio(self, num_mechanisms: int) -> float:
        """
        Compute compression ratio compared to storing all mechanisms.
        
        Args:
            num_mechanisms: Number of mechanisms that would be stored without compression
            
        Returns:
            compression_ratio: Ratio of compressed to uncompressed storage
        """
        if num_mechanisms == 0:
            return 0.0
            
        # Uncompressed: num_mechanisms * d_z * 4 bytes
        uncompressed_bytes = num_mechanisms * self.emb_dim * 4
        
        # Compressed: our current memory usage
        compressed_bytes = self.get_memory_usage_bytes()
        
        return compressed_bytes / uncompressed_bytes if uncompressed_bytes > 0 else float('inf')
    
    def reset(self):
        """Reset the memory bank to initial state."""
        self._initialize_prototypes()
    
    def __len__(self) -> int:
        """Return number of prototypes."""
        return self.K
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get prototype parameters by index."""
        return self.get_prototype_params(idx)


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic data for verification and checking the code, replace it
    #K, d_z, P = 10, 64, 5
    #parent_embeddings = torch.randn(P, d_z)
    #node_embedding = torch.randn(d_z)
    
    # Initialize memory bank
    memory_bank = CausalMemoryBank(K=K, emb_dim=d_z)
    
    # Add mechanism to memory bank
    prototype_idx = memory_bank.add_mechanism(parent_embeddings, node_embedding, weight=1.0)
    print(f"Assigned to prototype: {prototype_idx}")
    
    # Get prototype parameters
    mean, cov, weight = memory_bank.get_prototype_params(prototype_idx)
    print(f"Prototype {prototype_idx}: mean shape={mean.shape}, cov shape={cov.shape}, weight={weight:.4f}")
    
    # Test batch assignment
    B = 5
    batch_representations = torch.randn(B, d_z)
    assignments = memory_bank.compute_prototype_assignments(batch_representations)
    print(f"Batch assignments: {assignments.tolist()}")
    
    # Test memory usage
    memory_bytes = memory_bank.get_memory_usage_bytes()
    compression_ratio = memory_bank.get_compression_ratio(num_mechanisms=1000)
    print(f"Memory usage: {memory_bytes} bytes")
    print(f"Compression ratio (vs 1000 mechanisms): {compression_ratio:.4f}")
    
    # Test reset
    memory_bank.reset()
    print(f"Memory bank reset - initialized: {memory_bank.initialized}")