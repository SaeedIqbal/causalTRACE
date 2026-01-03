import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Protocol, runtime_checkable, Type
import yaml
import os
from pathlib import Path
import logging

# Import local modules
from .data.dataset_loader import load_dataset
from .models.causal_energy import CausalEnergyModel
from .models.embedding_updater import CausalEmbeddingUpdater
from .models.anomaly_scorer import CausalFidelityScorer, CausalMemoryBank
from .utils.memory_bank import CausalMemoryBank as UtilsMemoryBank
from .utils.metrics import MetricsTracker


@runtime_checkable
class TrainableModelProtocol(Protocol):
    """Protocol for trainable models."""
    def train_step(self, data: Any, optimizer: Optimizer) -> Dict[str, float]:
        """Perform a single training step."""
        pass
    
    def eval_step(self, data: Any) -> Dict[str, Any]:
        """Perform a single evaluation step."""
        pass


class BaseTrainer:
    """
    Base trainer class implementing common training functionality.
    Uses strategy pattern for model-specific training logic.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_name: str,
        device: torch.device = None
    ):
        """
        Initialize base trainer.
        
        Args:
            config: Configuration dictionary
            model_name: Name of the model to train
            device: Device to train on
        """
        self.config = config
        self.model_name = model_name
        self.device = device or torch.device(config['experiment']['device'])
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Setup directories
        self.output_dir = Path(config['experiment']['output_dir']) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and optimizer
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Initialize metrics tracker
        self.train_metrics = MetricsTracker(device=self.device, memory_tracking_enabled=True)
        self.val_metrics = MetricsTracker(device=self.device, memory_tracking_enabled=True)
        
        # Setup reproducibility
        self._setup_reproducibility()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.model_name}")
        logger.setLevel(getattr(logging, self.config['experiment']['log_level'].upper()))
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
            
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_reproducibility(self):
        """Setup reproducibility settings."""
        seed = self.config['reproducibility']['seed']
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.config['reproducibility']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = self.config['reproducibility']['cudnn_benchmark']
            
        torch.set_num_threads(self.config['reproducibility']['torch_threads'])
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        if not self.config['experiment']['save_checkpoints']:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(self, dataset_name: str):
        """Main training loop."""
        raise NotImplementedError("Subclasses must implement train method")


class CausalTracetrainer(BaseTrainer):
    """
    Trainer for CausalTRACE model (Sections 5.2-5.4).
    Implements memory-efficient streaming training with causal uncertainty-guided sampling.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device = None):
        super().__init__(config, "CausalTRACE", device)
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize CausalTRACE components."""
        model_config = self.config['models']['CausalTRACE']
        
        # Initialize embedding updater
        self.embedding_updater = CausalEmbeddingUpdater(
            input_dim=16,  # Will be set during first forward pass
            emb_dim=model_config['embedding_updater']['emb_dim'],
            hidden_dim=model_config['embedding_updater']['hidden_dim'],
            num_layers=model_config['embedding_updater']['num_layers'],
            mc_samples=model_config['embedding_updater']['mc_samples'],
            learning_rate=model_config['embedding_updater']['learning_rate'],
            momentum=model_config['embedding_updater']['momentum'],
            min_uncertainty=model_config['embedding_updater']['min_uncertainty']
        ).to(self.device)
        
        # Initialize causal energy model
        self.causal_energy = CausalEnergyModel(
            emb_dim=model_config['causal_energy']['emb_dim'],
            hidden_dim=model_config['causal_energy']['hidden_dim'],
            depth=model_config['causal_energy']['depth'],
            temporal_bandwidth=model_config['causal_energy']['temporal_bandwidth'],
            sparsity_threshold=model_config['causal_energy']['sparsity_threshold'],
            temperature=model_config['causal_energy']['temperature']
        ).to(self.device)
        
        # Initialize memory bank (using utils version for consistency)
        memory_config = model_config['memory_bank']
        self.memory_bank = UtilsMemoryBank(
            K=memory_config['K'],
            emb_dim=model_config['causal_energy']['emb_dim'],
            device=self.device,
            learning_rate=memory_config['learning_rate'],
            covariance_regularization=memory_config['covariance_regularization'],
            min_cluster_weight=memory_config['min_cluster_weight'],
            max_distance_threshold=memory_config['max_distance_threshold']
        )
        
        # Initialize anomaly scorer
        scorer_config = model_config['anomaly_scorer']
        self.anomaly_scorer = CausalFidelityScorer(
            memory_bank=self.memory_bank,
            kl_threshold=scorer_config['kl_threshold'],
            min_variance=scorer_config['min_variance'],
            calibration_alpha=scorer_config['calibration_alpha']
        )
        
        # Combine parameters for optimizer
        params = list(self.embedding_updater.parameters()) + list(self.causal_energy.parameters())
        train_config = model_config['training']
        
        if train_config['optimizer'] == "Adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
        elif train_config['optimizer'] == "SGD":
            self.optimizer = torch.optim.SGD(
                params,
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay'],
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {train_config['optimizer']}")
        
        # Scheduler (optional)
        self.scheduler = None
        
        self.model = nn.ModuleDict({
            'embedding_updater': self.embedding_updater,
            'causal_energy': self.causal_energy
        })
    
    def train_step(self, snapshot: Any) -> Dict[str, float]:
        """Perform a single training step on a temporal snapshot."""
        self.model.train()
        
        # Move data to device
        snapshot = snapshot.to(self.device)
        
        # Get base embeddings (using all neighbors)
        base_z, _ = self.embedding_updater(snapshot)
        
        # Infer causal parents
        parents, parent_embs = self.causal_energy.infer_parents(
            base_z, snapshot.edge_index
        )
        
        # Create parent mask for causal embedding computation
        parent_mask = torch.zeros(snapshot.edge_index.size(1), dtype=torch.bool, device=self.device)
        for i in range(snapshot.edge_index.size(1)):
            src, dst = snapshot.edge_index[:, i]
            if dst in parents[src]:
                parent_mask[i] = True
        
        # Compute causal embeddings (using only parents)
        causal_z, causal_uncertainty = self.embedding_updater(snapshot, parent_mask=parent_mask)
        
        # Sample edges based on causal uncertainty
        train_config = self.config['models']['CausalTRACE']['training']
        batch_size = min(
            train_config.get('batch_size', 1024),
            snapshot.edge_index.size(1)
        )
        
        if batch_size > 0:
            sampled_edges = self.embedding_updater.sample_edges_by_uncertainty(
                snapshot.edge_index, causal_uncertainty, batch_size
            )
            
            # Compute energy for positive samples
            pos_energies = []
            neg_energies = []
            
            for edge_idx in sampled_edges:
                src, dst = snapshot.edge_index[:, edge_idx]
                
                # Get parent embeddings for source node
                src_parents = parent_embs[src]
                valid_parents = src_parents[src_parents.abs().sum(dim=-1) > 0]
                
                if valid_parents.size(0) > 0:
                    # Positive energy
                    pos_energy = self.causal_energy(causal_z[src:src+1], valid_parents.unsqueeze(0))
                    pos_energies.append(pos_energy)
                    
                    # Negative sampling using Langevin dynamics
                    neg_z = self.causal_energy.sample_negative(
                        causal_z[src:src+1],
                        valid_parents.unsqueeze(0),
                        num_steps=self.config['models']['CausalTRACE']['causal_energy']['contrastive_steps'],
                        step_size=self.config['models']['CausalTRACE']['causal_energy']['step_size']
                    )
                    neg_energy = self.causal_energy(neg_z, valid_parents.unsqueeze(0))
                    neg_energies.append(neg_energy)
            
            if pos_energies and neg_energies:
                pos_energy_tensor = torch.cat(pos_energies)
                neg_energy_tensor = torch.cat(neg_energies)
                
                # Compute contrastive divergence loss
                loss = self.causal_energy.contrastive_divergence_loss(
                    pos_energy_tensor, neg_energy_tensor, reduction='mean'
                )
                
                # Update memory bank with normal mechanisms (assuming no anomalies in training)
                for i, edge_idx in enumerate(sampled_edges):
                    src, dst = snapshot.edge_index[:, edge_idx]
                    src_parents = parent_embs[src]
                    valid_parents = src_parents[src_parents.abs().sum(dim=-1) > 0]
                    if valid_parents.size(0) > 0:
                        self.memory_bank.add_mechanism(valid_parents, causal_z[src], weight=1.0)
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config['models']['CausalTRACE']['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['models']['CausalTRACE']['training']['gradient_clip']
                    )
                
                self.optimizer.step()
                
                return {'loss': loss.item(), 'pos_energy': pos_energy_tensor.mean().item(), 'neg_energy': neg_energy_tensor.mean().item()}
        
        return {'loss': 0.0, 'pos_energy': 0.0, 'neg_energy': 0.0}
    
    def eval_step(self, snapshot: Any) -> Dict[str, Any]:
        """Perform evaluation step on a temporal snapshot."""
        self.model.eval()
        
        with torch.no_grad():
            snapshot = snapshot.to(self.device)
            
            # Get embeddings and uncertainty
            base_z, _ = self.embedding_updater(snapshot)
            parents, parent_embs = self.causal_energy.infer_parents(base_z, snapshot.edge_index)
            
            # Compute causal embeddings
            causal_z, causal_uncertainty = self.embedding_updater(snapshot)
            
            # Compute anomaly scores and fidelities
            anomaly_scores = []
            fidelities = []
            
            for i in range(causal_z.size(0)):
                valid_parents = parent_embs[i][parent_embs[i].abs().sum(dim=-1) > 0]
                if valid_parents.size(0) > 0:
                    anomaly_score = self.anomaly_scorer.compute_anomaly_score(causal_z[i], valid_friends)
                    fidelity, _ = self.anomaly_scorer.compute_causal_fidelity(causal_z[i], valid_friends)
                    anomaly_scores.append(anomaly_score)
                    fidelities.append(fidelity)
                else:
                    anomaly_scores.append(torch.tensor(0.0, device=self.device))
                    fidelities.append(torch.tensor(1.0, device=self.device))
            
            anomaly_scores = torch.stack(anomaly_scores)
            fidelities = torch.stack(fidelities)
            
            return {
                'scores': anomaly_scores.cpu(),
                'fidelities': fidelities.cpu(),
                'labels': snapshot.y.cpu() if hasattr(snapshot, 'y') else torch.zeros(causal_z.size(0))
            }
    
    def train(self, dataset_name: str):
        """Main training loop for CausalTRACE."""
        self.logger.info(f"Starting training for CausalTRACE on {dataset_name}")
        
        # Load dataset
        dataset_config = self.config['datasets']['defaults'].copy()
        if dataset_name in self.config['datasets']:
            dataset_config.update(self.config['datasets'][dataset_name])
        
        temporal_data = load_dataset(
            dataset_name,
            temporal=dataset_config['temporal'],
            num_snapshots=dataset_config['num_snapshots'],
            inject_anomalies=dataset_config['inject_anomalies']
        )
        
        # Set input dimension based on dataset
        if len(temporal_data) > 0:
            input_dim = temporal_data[0].x.size(1)
            self.embedding_updater.input_proj = nn.Linear(input_dim, self.embedding_updater.input_proj.out_features)
            self.embedding_updater.to(self.device)
            self.logger.info(f"Set input dimension to {input_torch}")
        
        # Training configuration
        train_config = self.config['models']['CausalTRACE']['training']
        epochs = train_config['epochs']
        eval_every = train_config['eval_every']
        
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training phase
            train_losses = []
            for snapshot in temporal_data:
                loss_dict = self.train_step(snapshot)
                train_losses.append(loss_dict['loss'])
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            self.logger.info(f"Average training loss: {avg_train_loss:.6f}")
            
            # Validation phase
            if (epoch + 1) % eval_every == 0:
                self.val_metrics.reset()
                for snapshot in temporal_data:
                    eval_result = self.eval_step(snapshot)
                    self.val_metrics.update(
                        scores=eval_result['scores'],
                        labels=eval_result['labels'],
                        fidelities=eval_result['fidelities']
                    )
                
                val_auc, val_ci = self.val_metrics.compute_auc(return_ci=True)
                mean_fid, std_fid = self.val_metrics.compute_causal_fidelity()
                memory_stats = self.val_metrics.compute_memory_usage()
                
                self.logger.info(f"Validation AUC: {val_auc:.4f} [{val_ci[0]:.4f}, {val_ci[1]:.4f}]")
                self.logger.info(f"Mean Causal Fidelity: {mean_fid:.4f} ± {std_fid:.4f}")
                self.logger.info(f"Peak GPU Memory: {memory_stats['peak_gpu_mb']:.2f} MB")
                
                # Save checkpoint
                metrics = {
                    'val_auc': val_auc,
                    'val_auc_ci': val_ci,
                    'mean_fidelity': mean_fid,
                    'std_fidelity': std_fid,
                    'peak_gpu_mb': memory_stats['peak_gpu_mb']
                }
                self.save_checkpoint(epoch + 1, metrics)
        
        self.logger.info("Training completed successfully")


# Factory function for trainers
def create_trainer(model_name: str, config: Dict[str, Any], device: torch.device = None) -> BaseTrainer:
    """
    Factory function to create appropriate trainer based on model name.
    
    Args:
        model_name: Name of the model
        config: Configuration dictionary
        device: Device to train on
        
    Returns:
        Initialized trainer instance
    """
    if model_name == "CausalTRACE":
        return CausalTracetrainer(config, device)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Only CausalTRACE is implemented in this version.")


# Main training function
def train_model(
    model_name: str,
    dataset_name: str,
    config_path: str = "config/hyperparams.yaml",
    device: Optional[torch.device] = None
):
    """
    Main training function.
    
    Args:
        model_name: Name of the model to train
        dataset_name: Name of the dataset to train on
        config_path: Path to configuration file
        device: Device to train on (optional)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = create_trainer(model_name, config, device)
    
    # Start training
    trainer.train(dataset_name)


# Example usage
if __name__ == "__main__":
    # Train CausalTRACE on DGraph
    train_model("CausalTRACE", "DGraph")