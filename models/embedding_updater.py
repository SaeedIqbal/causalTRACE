import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Protocol, runtime_checkable
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


@runtime_checkable
class GraphDataProtocol(Protocol):
    """Protocol for graph data with temporal information."""
    x: torch.Tensor          # [N, D] node features
    edge_index: torch.Tensor # [2, E] edge list
    timestamp: Optional[int] # Current time step


class CausalMessagePassing(MessagePassing):
    """
    Custom message passing layer for causal embedding computation.
    Aggregates information from causal parents only.
    """
    
    def __init__(self, emb_dim: int, aggr: str = 'mean'):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.message_proj = nn.Linear(emb_dim * 2, emb_dim)
        self.update_proj = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, parent_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute causal message passing.
        
        Args:
            x: [N, emb_dim] node embeddings
            edge_index: [2, E] edge list
            parent_mask: [E] boolean mask indicating causal parent edges
            
        Returns:
            out: [N, emb_dim] updated embeddings
        """
        # Filter edge_index to only causal parent edges
        causal_edge_index = edge_index[:, parent_mask]
        if causal_edge_index.size(1) == 0:
            return x
            
        return self.propagate(causal_edge_index, x=x)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute message from neighbor j to node i."""
        # x_i: [E, emb_dim] - target node features
        # x_j: [E, emb_dim] - source node features (parents)
        msg_input = torch.cat([x_i, x_j], dim=-1)
        return self.message_proj(msg_input)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node embedding with aggregated messages."""
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_proj(update_input)


class CausalEmbeddingUpdater(nn.Module):
    """
    Continual Causal Embedding Engine (Section 5.3).
    Maintains online embeddings with causal uncertainty-guided sampling.
    """
    
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        mc_samples: int = 10,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        min_uncertainty: float = 1e-6
    ):
        """
        Initialize continual causal embedding updater.
        
        Args:
            input_dim: Input feature dimension
            emb_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            mc_samples: Number of Monte Carlo samples for uncertainty (S in Eq. 5.3)
            learning_rate: Learning rate for online updates
            momentum: Momentum for exponential moving averages
            min_uncertainty: Minimum uncertainty to prevent numerical issues
        """
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.mc_samples = mc_samples
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.min_uncertainty = min_uncertainty
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers for embedding computation
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            self.gnn_layers.append(CausalMessagePassing(in_dim))
        
        # Final projection to embedding space
        self.final_proj = nn.Linear(hidden_dim, emb_dim)
        
        # Online statistics trackers (exponential moving averages)
        self.register_buffer('embedding_ema', torch.zeros(0))
        self.register_buffer('embedding_var', torch.zeros(0))
        self.register_buffer('node_counts', torch.zeros(0))
        self.register_buffer('initialized', torch.tensor(False))
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_base_embedding(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute base embedding without causal attention."""
        h = F.relu(self.input_proj(x))
        
        for layer in self.gnn_layers:
            # For base embedding, use all neighbors
            h = F.relu(layer(h, edge_index, torch.ones(edge_index.size(1), dtype=torch.bool, device=x.device)))
        
        z = self.final_proj(h)
        return z

    def _compute_causal_embedding(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        parent_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute embedding using only causal parents."""
        h = F.relu(self.input_proj(x))
        
        for layer in self.gnn_layers:
            h = F.relu(layer(h, edge_index, parent_mask))
        
        z = self.final_proj(h)
        return z

    def _estimate_causal_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        base_embedding: torch.Tensor,
        parent_mask: torch.Tensor,
        mc_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Estimate causal uncertainty using Monte Carlo sampling (Eq. 5.3).
        
        u_i^{(t)} = Tr(Cov_do[z_i^{(t)}])
        
        Args:
            x: [N, input_dim] input features
            edge_index: [2, E] edge list
            base_embedding: [N, emb_dim] base embeddings
            parent_mask: [E] causal parent mask
            mc_samples: Number of Monte Carlo samples (overrides self.mc_samples)
            
        Returns:
            uncertainty: [N] causal uncertainty scores
        """
        if mc_samples is None:
            mc_samples = self.mc_samples
            
        N = x.size(0)
        if mc_samples == 0:
            return torch.zeros(N, device=x.device)
            
        # Monte Carlo sampling of parent embeddings
        embeddings_mc = []
        
        for _ in range(mc_samples):
            # Add small noise to parent embeddings to simulate do-interventions
            noise_scale = 0.1
            x_perturbed = x + torch.randn_like(x) * noise_scale
            
            # Recompute embedding with perturbed inputs
            z_perturbed = self._compute_causal_embedding(x_perturbed, edge_index, parent_mask)
            embeddings_mc.append(z_perturbed)
        
        # Stack Monte Carlo samples [mc_samples, N, emb_dim]
        embeddings_mc = torch.stack(embeddings_mc, dim=0)
        
        # Compute covariance for each node
        # embeddings_mc: [mc_samples, N, emb_dim]
        # mean_mc: [N, emb_dim]
        mean_mc = embeddings_mc.mean(dim=0)
        
        # Center the samples
        centered = embeddings_mc - mean_mc.unsqueeze(0)  # [mc_samples, N, emb_dim]
        
        # Compute covariance trace for each node
        # For each node i, compute Tr(Cov(z_i))
        # Cov(z_i) = (1/(mc_samples-1)) * sum_j (z_ij - mean_i)(z_ij - mean_i)^T
        # Tr(Cov(z_i)) = (1/(mc_samples-1)) * sum_j ||z_ij - mean_i||^2
        if mc_samples > 1:
            squared_diffs = torch.sum(centered * centered, dim=2)  # [mc_samples, N]
            uncertainty = squared_diffs.sum(dim=0) / (mc_samples - 1)  # [N]
        else:
            uncertainty = torch.zeros(N, device=x.device)
        
        # Ensure minimum uncertainty for numerical stability
        uncertainty = torch.clamp(uncertainty, min=self.min_uncertainty)
        
        return uncertainty

    def forward(
        self,
        data: GraphDataProtocol,
        parent_mask: Optional[torch.Tensor] = None,
        update_stats: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute embeddings and causal uncertainty for current snapshot.
        
        Args:
            data: Graph data for current time step
            parent_mask: [E] causal parent mask (if None, computed internally)
            update_stats: Whether to update online statistics
            
        Returns:
            z: [N, emb_dim] embeddings
            u: [N] causal uncertainty scores
        """
        x, edge_index = data.x, data.edge_index
        N = x.size(0)
        
        # Initialize buffers if needed
        if not self.initialized or self.embedding_ema.size(0) != N:
            self.embedding_ema = torch.zeros(N, self.emb_dim, device=x.device)
            self.embedding_var = torch.zeros(N, self.emb_dim, device=x.device)
            self.node_counts = torch.zeros(N, device=x.device)
            self.initialized = torch.tensor(True)
        
        # Compute base embedding (using all neighbors)
        base_z = self._compute_base_embedding(x, edge_index)
        
        # If parent_mask not provided, create default (all edges are parents)
        if parent_mask is None:
            parent_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=x.device)
        
        # Compute causal embedding (using only parents)
        causal_z = self._compute_causal_embedding(x, edge_index, parent_mask)
        
        # Estimate causal uncertainty
        u = self._estimate_causal_uncertainty(
            x, edge_index, base_z, parent_mask, self.mc_samples
        )
        
        # Update online statistics if requested
        if update_stats:
            self._update_online_stats(causal_z, u)
        
        return causal_z, u

    def _update_online_stats(self, z: torch.Tensor, u: torch.Tensor):
        """Update exponential moving averages for online learning."""
        N = z.size(0)
        
        # Update node counts
        self.node_counts[:N] += 1
        
        # Update embedding EMA: EMA_t = (1 - lr) * EMA_{t-1} + lr * z_t
        lr = self.learning_rate
        self.embedding_ema[:N] = (1 - lr) * self.embedding_ema[:N] + lr * z
        
        # Update embedding variance using Welford's online algorithm
        delta = z - self.embedding_ema[:N]
        self.embedding_var[:N] = (1 - lr) * self.embedding_var[:N] + lr * (delta * delta)

    def get_online_stats(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get current online statistics.
        
        Returns:
            embedding_ema: [N, emb_dim] exponential moving average of embeddings
            embedding_var: [N, emb_dim] variance of embeddings
            node_counts: [N] number of updates per node
        """
        return self.embedding_ema, self.embedding_var, self.node_counts

    def sample_edges_by_uncertainty(
        self,
        edge_index: torch.Tensor,
        uncertainty: torch.Tensor,
        num_samples: int,
        replacement: bool = True
    ) -> torch.Tensor:
        """
        Sample edges based on causal uncertainty (Section 5.3).
        
        P(e = (v_i, v_j)) ∝ u_i + u_j
        
        Args:
            edge_index: [2, E] edge list
            uncertainty: [N] uncertainty scores for all nodes
            num_samples: Number of edges to sample
            replacement: Whether to sample with replacement
            
        Returns:
            sampled_indices: [num_samples] indices of sampled edges
        """
        if edge_index.size(1) == 0:
            return torch.empty(0, dtype=torch.long, device=edge_index.device)
            
        # Compute edge weights as sum of endpoint uncertainties
        src_uncertainty = uncertainty[edge_index[0]]  # [E]
        dst_uncertainty = uncertainty[edge_index[1]]  # [E]
        edge_weights = src_uncertainty + dst_uncertainty  # [E]
        
        # Handle zero weights
        edge_weights = torch.clamp(edge_weights, min=1e-8)
        
        # Sample edges
        sampled_indices = torch.multinomial(
            edge_weights, num_samples, replacement=replacement
        )
        
        return sampled_indices

    def reset_stats(self):
        """Reset online statistics (for new datasets or evaluation)."""
        self.initialized = torch.tensor(False)
        self.embedding_ema = torch.zeros(0)
        self.embedding_var = torch.zeros(0)
        self.node_counts = torch.zeros(0)


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic data for checking the code and verifying the model
    N, D, E = 100, 16, 500
    x = torch.randn(N, D)
    edge_index = torch.randint(0, N, (2, E))
    data = type('GraphData', (), {'x': x, 'edge_index': edge_index, 'timestamp': 0})
    
    # Initialize updater
    updater = CausalEmbeddingUpdater(input_dim=D, emb_dim=64, mc_samples=5)
    
    # Compute embeddings and uncertainty
    z, u = updater(data)
    print(f"Embeddings shape: {z.shape}")
    print(f"Uncertainty shape: {u.shape}")
    print(f"Uncertainty range: [{u.min():.4f}, {u.max():.4f}]")
    
    # Sample edges by uncertainty
    sampled_indices = updater.sample_edges_by_uncertainty(edge_index, u, num_samples=100)
    print(f"Sampled {sampled_indices.size(0)} edges")
    
    # Test online statistics
    ema, var, counts = updater.get_online_stats()
    print(f"EMA shape: {ema.shape}, Var shape: {var.shape}, Counts shape: {counts.shape}")
    
    # Test with parent mask
    parent_mask = torch.rand(E) > 0.5
    z_causal, u_causal = updater(data, parent_mask=parent_mask)
    print(f"Causal embeddings computed with {parent_mask.sum().item()} parent edges")