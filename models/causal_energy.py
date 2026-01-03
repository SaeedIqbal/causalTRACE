import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Protocol, runtime_checkable
from torch_geometric.utils import scatter


@runtime_checkable
class EmbeddingData(Protocol):
    """Protocol for embedding data with parent sets."""
    z: torch.Tensor                # [N, d_z] node embeddings
    parents: torch.Tensor          # [N, max_parents] parent indices (-1 for padding)
    parent_embeddings: torch.Tensor # [N, max_parents, d_z] parent embeddings


class CausalEnergyModel(nn.Module):
    """
    Causal-Temporal Energy Model (Section 5.2).
    Learns energy function E_φ(z_i, Pa(z_i)) = -log p_φ(z_i | do(Pa(z_i))).
    Uses contrastive divergence for training and temporal attention for parent inference.
    """
    
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        temporal_bandwidth: float = 1.0,
        sparsity_threshold: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize causal energy model.
        
        Args:
            emb_dim: Embedding dimension d_z
            hidden_dim: Hidden layer dimension for energy network
            depth: Number of layers in energy MLP
            temporal_bandwidth: σ_t for temporal attention (Eq. 5.2)
            sparsity_threshold: τ for parent selection (Eq. 5.2)
            temperature: Temperature for attention softmax
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.temporal_bandwidth = temporal_bandwidth
        self.sparsity_threshold = sparsity_threshold
        self.temperature = temperature
        
        # Energy network: MLP for E_φ(z_i, Pa(z_i))
        layers = []
        input_dim = emb_dim + emb_dim  # z_i + mean(Pa(z_i))
        for i in range(depth):
            output_dim = hidden_dim if i < depth - 1 else 1
            layers.append(nn.Linear(input_dim, output_dim))
            if i < depth - 1:
                layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.energy_net = nn.Sequential(*layers)
        
        # Parent attention network
        self.attention_query = nn.Linear(emb_dim, emb_dim)
        self.attention_key = nn.Linear(emb_dim, emb_dim)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def infer_parents(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Infer causal parent set using temporal attention (Section 5.2).
        
        Args:
            z: [N, d_z] node embeddings
            edge_index: [2, E] edge list for current snapshot
            batch: [N] batch indices (optional, for multi-graph batches)
            
        Returns:
            parents: [N, max_degree] parent indices (-1 for padding)
            parent_embeddings: [N, max_degree, d_z] parent embeddings
        """
        if edge_index.size(1) == 0:
            # No edges case
            N = z.size(0)
            return torch.full((N, 1), -1, device=z.device), torch.zeros(N, 1, self.emb_dim, device=z.device)
        
        # Compute attention scores using temporal bandwidth
        # Eq. 5.2: a_ij = exp(-||z_i - z_j||^2 / σ_t^2) / sum_k exp(-||z_i - z_k||^2 / σ_t^2)
        row, col = edge_index
        diff = z[row] - z[col]  # [E, d_z]
        squared_distances = torch.sum(diff * diff, dim=1)  # [E]
        attention_weights = torch.exp(-squared_distances / (self.temporal_bandwidth ** 2 + 1e-8))  # [E]
        
        # Normalize attention per node
        if batch is not None:
            # Handle batched graphs
            num_nodes = scatter(torch.ones_like(row), batch[row], reduce='sum')
            attention_weights = attention_weights / (scatter(attention_weights, batch[row], reduce='sum')[batch[row]] + 1e-8)
        else:
            # Single graph
            degree = scatter(torch.ones_like(row), row, dim=0, reduce='sum')
            attention_weights = attention_weights / (scatter(attention_weights, row, dim=0, reduce='sum')[row] + 1e-8)
        
        # Select parents above sparsity threshold
        parent_mask = attention_weights > self.sparsity_threshold  # [E]
        selected_edges = edge_index[:, parent_mask]  # [2, E_selected]
        selected_weights = attention_weights[parent_mask]  # [E_selected]
        
        if selected_edges.size(1) == 0:
            # No parents above threshold
            N = z.size(0)
            return torch.full((N, 1), -1, device=z.device), torch.zeros(N, 1, self.emb_dim, device=z.device)
        
        # Group parents by node
        row_selected, col_selected = selected_edges
        unique_nodes, inverse_indices = torch.unique(row_selected, return_inverse=True)
        
        # Find maximum degree for padding
        degrees = scatter(torch.ones_like(inverse_indices), inverse_indices, reduce='sum')
        max_degree = degrees.max().item()
        
        # Initialize parent tensors with padding
        N = z.size(0)
        parents = torch.full((N, max_degree), -1, dtype=torch.long, device=z.device)
        parent_embeddings = torch.zeros(N, max_degree, self.emb_dim, device=z.device)
        
        # Fill in parent information
        for i, node in enumerate(unique_nodes):
            node_mask = row_selected == node
            node_parents = col_selected[node_mask]
            node_parent_embs = z[node_parents]
            
            # Sort by attention weight (highest first)
            node_weights = selected_weights[node_mask]
            sorted_indices = torch.argsort(node_weights, descending=True)
            node_parents = node_parents[sorted_indices]
            node_parent_embs = node_parent_embs[sorted_indices]
            
            # Pad or truncate to max_degree
            actual_degree = node_parents.size(0)
            if actual_degree > max_degree:
                node_parents = node_parents[:max_degree]
                node_parent_embs = node_parent_embs[:max_degree]
                actual_degree = max_degree
            
            parents[node, :actual_degree] = node_parents
            parent_embeddings[node, :actual_degree] = node_parent_embs
        
        return parents, parent_embeddings

    def forward(
        self,
        z_i: torch.Tensor,
        parent_embeddings: torch.Tensor,
        reduce: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute energy E_φ(z_i, Pa(z_i)) for given node and parents.
        
        Args:
            z_i: [N, d_z] or [B, d_z] node embeddings
            parent_embeddings: [N, P, d_z] or [B, P, d_z] parent embeddings (P = max parents)
            reduce: How to aggregate parent information ('mean', 'max', 'attention')
            
        Returns:
            energy: [N] or [B] energy values
        """
        N = z_i.size(0)
        P = parent_embeddings.size(1)
        
        if P == 0:
            # No parents case - use zero vector as parent representation
            parent_repr = torch.zeros(N, self.emb_dim, device=z_i.device)
        elif reduce == 'mean':
            # Mean of parent embeddings (handles padding with -1 indices)
            parent_mask = (parent_embeddings.abs().sum(dim=-1, keepdim=True) > 1e-8).float()  # [N, P, 1]
            parent_repr = (parent_embeddings * parent_mask).sum(dim=1) / (parent_mask.sum(dim=1) + 1e-8)
        elif reduce == 'max':
            parent_repr, _ = parent_embeddings.max(dim=1)
        elif reduce == 'attention':
            # Learnable attention over parents
            query = self.attention_query(z_i).unsqueeze(1)  # [N, 1, d_z]
            keys = self.attention_key(parent_embeddings)    # [N, P, d_z]
            attn_scores = torch.sum(query * keys, dim=-1) / (self.temperature + 1e-8)  # [N, P]
            attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [N, P, 1]
            parent_repr = (parent_embeddings * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown reduce method: {reduce}")
        
        # Concatenate z_i and parent representation
        input_tensor = torch.cat([z_i, parent_repr], dim=-1)  # [N, 2*d_z]
        
        # Compute energy
        energy = self.energy_net(input_tensor).squeeze(-1)  # [N]
        return energy

    def contrastive_divergence_loss(
        self,
        pos_energy: torch.Tensor,
        neg_energy: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute contrastive divergence loss (Section 5.2).
        
        L_CD(φ) = E_data[E_φ] - E_model[E_φ]
        
        Args:
            pos_energy: [N] energy for positive (observed) samples
            neg_energy: [N] energy for negative (reconstructed) samples
            reduction: Reduction method ('mean', 'sum', 'none')
            
        Returns:
            loss: Scalar loss value
        """
        if reduction == 'mean':
            return pos_energy.mean() - neg_energy.mean()
        elif reduction == 'sum':
            return pos_energy.sum() - neg_energy.sum()
        elif reduction == 'none':
            return pos_energy - neg_energy
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def sample_negative(
        self,
        z_i: torch.Tensor,
        parent_embeddings: torch.Tensor,
        num_steps: int = 5,
        step_size: float = 0.1
    ) -> torch.Tensor:
        """
        Sample from model distribution using Langevin dynamics (Section 5.2).
        
        Args:
            z_i: [N, d_z] initial embeddings
            parent_embeddings: [N, P, d_z] parent embeddings
            num_steps: Number of Langevin steps
            step_size: Step size for Langevin dynamics
            
        Returns:
            z_sampled: [N, d_z] sampled embeddings from model distribution
        """
        z_sampled = z_i.clone().requires_grad_(True)
        
        for _ in range(num_steps):
            energy = self.forward(z_sampled, parent_embeddings)
            energy_sum = energy.sum()
            
            # Compute gradient
            grad = torch.autograd.grad(energy_sum, z_sampled, retain_graph=True)[0]
            
            # Langevin step: z_{t+1} = z_t - (η/2) ∇E(z_t) + √η * ε
            noise = torch.randn_like(z_sampled) * (step_size ** 0.5)
            z_sampled = z_sampled - (step_size / 2) * grad + noise
            
            # Detach to avoid gradient accumulation
            z_sampled = z_sampled.detach().requires_grad_(True)
        
        return z_sampled.detach()

    def compute_kl_divergence(
        self,
        z_i: torch.Tensor,
        parent_embeddings: torch.Tensor,
        reference_params: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute KL divergence between current distribution and reference (Section 5.4).
        
        Args:
            z_i: [N, d_z] current embeddings
            parent_embeddings: [N, P, d_z] parent embeddings
            reference_params: Tuple of (mu_ref, sigma_ref) for reference distribution
            
        Returns:
            kl_div: [N] KL divergence values
        """
        mu_ref, sigma_ref = reference_params
        
        # Estimate current distribution parameters
        with torch.no_grad():
            if parent_embeddings.size(1) > 0:
                parent_mean = parent_embeddings.mean(dim=1)
                # Simple covariance estimation (diagonal)
                current_var = torch.var(z_i, dim=0, keepdim=True).expand_as(z_i)
            else:
                parent_mean = torch.zeros_like(z_i)
                current_var = torch.ones_like(z_i)
        
        # KL divergence for diagonal Gaussians
        sigma_ref_inv = 1.0 / (sigma_ref + 1e-8)
        kl_div = 0.5 * (
            torch.sum(torch.log(sigma_ref / (current_var + 1e-8)), dim=1) +
            torch.sum(sigma_ref_inv * current_var, dim=1) +
            torch.sum((mu_ref - z_i) ** 2 * sigma_ref_inv, dim=1) -
            self.emb_dim
        )
        
        return kl_div


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic data for checking and verifying the code. Replace it with original datasets!!
    N, d_z, E = 100, 64, 500
    z = torch.randn(N, d_z)
    edge_index = torch.randint(0, N, (2, E))
    
    # Initialize model
    model = CausalEnergyModel(emb_dim=d_z, hidden_dim=128, depth=3)
    
    # Infer parents
    parents, parent_embs = model.infer_parents(z, edge_index)
    print(f"Inferred parents shape: {parents.shape}")
    print(f"Parent embeddings shape: {parent_embs.shape}")
    
    # Compute energy
    energy = model(z, parent_embs)
    print(f"Energy shape: {energy.shape}")
    print(f"Energy range: [{energy.min():.4f}, {energy.max():.4f}]")
    
    # Sample negative examples
    z_neg = model.sample_negative(z, parent_embs, num_steps=3)
    print(f"Negative sample shape: {z_neg.shape}")
    
    # Compute contrastive loss
    neg_energy = model(z_neg, parent_embs)
    loss = model.contrastive_divergence_loss(energy, neg_energy)
    print(f"Contrastive loss: {loss:.4f}")
    
    # Compute KL divergence
    mu_ref = torch.zeros(d_z)
    sigma_ref = torch.ones(d_z)
    kl_div = model.compute_kl_divergence(z, parent_embs, (mu_ref, sigma_ref))
    print(f"KL divergence shape: {kl_div.shape}")
    print(f"KL divergence range: [{kl_div.min():.4f}, {kl_div.max():.4f}]")