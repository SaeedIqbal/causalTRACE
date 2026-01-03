import torch
import yaml
from typing import Dict, Any, Optional, Protocol, runtime_checkable, List
from pathlib import Path
import logging
import json

# Import local modules
from .data.dataset_loader import load_dataset
from .models.causal_energy import CausalEnergyModel
from .models.embedding_updater import CausalEmbeddingUpdater
from .models.anomaly_scorer import CausalFidelityScorer, CausalMemoryBank
from .utils.memory_bank import CausalMemoryBank as UtilsMemoryBank
from .utils.metrics import MetricsTracker
from .train import create_trainer, CausalTracetrainer


@runtime_checkable
class EvaluableModelProtocol(Protocol):
    """Protocol for evaluable models."""
    def eval_step(self,  Any) -> Dict[str, Any]:
        """Perform a single evaluation step."""
        pass


class BaseEvaluator:
    """
    Base evaluator class implementing common evaluation functionality.
    Uses strategy pattern for model-specific evaluation logic.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_name: str,
        device: torch.device = None
    ):
        """
        Initialize base evaluator.
        
        Args:
            config: Configuration dictionary
            model_name: Name of the model to evaluate
            device: Device to evaluate on
        """
        self.config = config
        self.model_name = model_name
        self.device = device or torch.device(config['experiment']['device'])
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Setup output directory
        self.output_dir = Path(config['experiment']['output_dir']) / model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            device=self.device,
            memory_tracking_enabled=True,
            confidence_level=config['evaluation']['confidence_level']
        )
        
        # Initialize model
        self.model = None
        
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
        log_file = self.output_dir / "evaluation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load trained model from checkpoint."""
        raise NotImplementedError("Subclasses must implement load_model method")
    
    def evaluate(self, dataset_name: str, split: str = "test") -> Dict[str, Any]:
        """Main evaluation loop."""
        raise NotImplementedError("Subclasses must implement evaluate method")


class CausalTraceEvaluator(BaseEvaluator):
    """
    Evaluator for CausalTRACE model (Sections 5.2-5.4).
    Implements memory-efficient streaming evaluation with causal fidelity computation.
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device = None):
        super().__init__(config, "CausalTRACE", device)
        self.embedding_updater = None
        self.causal_energy = None
        self.memory_bank = None
        self.anomaly_scorer = None
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load trained CausalTRACE model from checkpoint."""
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoint_files = list(self.output_dir.glob("checkpoint_epoch_*.pt"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoints found in {self.output_dir}")
            checkpoint_path = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        
        self.logger.info(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recreate model architecture
        model_config = self.config['models']['CausalTRACE']
        input_dim = checkpoint['config']['models']['CausalTRACE'].get('input_dim', 16)
        
        # Initialize components
        self.embedding_updater = CausalEmbeddingUpdater(
            input_dim=input_dim,
            emb_dim=model_config['embedding_updater']['emb_dim'],
            hidden_dim=model_config['embedding_updater']['hidden_dim'],
            num_layers=model_config['embedding_updater']['num_layers'],
            mc_samples=model_config['embedding_updater']['mc_samples'],
            learning_rate=model_config['embedding_updater']['learning_rate'],
            momentum=model_config['embedding_updater']['momentum'],
            min_uncertainty=model_config['embedding_updater']['min_uncertainty']
        ).to(self.device)
        
        self.causal_energy = CausalEnergyModel(
            emb_dim=model_config['causal_energy']['emb_dim'],
            hidden_dim=model_config['causal_energy']['hidden_dim'],
            depth=model_config['causal_energy']['depth'],
            temporal_bandwidth=model_config['causal_energy']['temporal_bandwidth'],
            sparsity_threshold=model_config['causal_energy']['sparsity_threshold'],
            temperature=model_config['causal_energy']['temperature']
        ).to(self.device)
        
        # Initialize memory bank
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
        
        # Load model state
        self.embedding_updater.load_state_dict(checkpoint['model_state_dict']['embedding_updater'])
        self.causal_energy.load_state_dict(checkpoint['model_state_dict']['causal_energy'])
        
        # Load memory bank state (if available)
        if 'memory_bank' in checkpoint:
            self.memory_bank.prototypes = checkpoint['memory_bank']['prototypes'].to(self.device)
            self.memory_bank.covariances = checkpoint['memory_bank']['covariances'].to(self.device)
            self.memory_bank.weights = checkpoint['memory_bank']['weights'].to(self.device)
            self.memory_bank.initialized = checkpoint['memory_bank']['initialized']
        
        self.logger.info("Model loaded successfully")
    
    def eval_step(self, snapshot: Any) -> Dict[str, Any]:
        """Perform evaluation step on a temporal snapshot."""
        self.embedding_updater.eval()
        self.causal_energy.eval()
        
        with torch.no_grad():
            snapshot = snapshot.to(self.device)
            
            # Get embeddings
            base_z, _ = self.embedding_updater(snapshot)
            
            # Infer causal parents
            parents, parent_embs = self.causal_energy.infer_parents(base_z, snapshot.edge_index)
            
            # Compute causal embeddings
            causal_z, causal_uncertainty = self.embedding_updater(snapshot)
            
            # Compute anomaly scores and fidelities
            anomaly_scores = []
            fidelities = []
            
            for i in range(causal_z.size(0)):
                # Get valid parents (remove padding)
                valid_mask = parent_embs[i].abs().sum(dim=-1) > 0
                valid_parents = parent_embs[i][valid_mask]
                
                if valid_parents.size(0) > 0:
                    # Compute anomaly score
                    anomaly_score = self.anomaly_scorer.compute_anomaly_score(causal_z[i], valid_parents)
                    fidelity, _ = self.anomaly_scorer.compute_causal_fidelity(causal_z[i], valid_parents)
                    anomaly_scores.append(anomaly_score)
                    fidelities.append(fidelity)
                else:
                    # No parents - treat as normal (high fidelity, low anomaly score)
                    anomaly_scores.append(torch.tensor(0.0, device=self.device))
                    fidelities.append(torch.tensor(1.0, device=self.device))
            
            anomaly_scores = torch.stack(anomaly_scores)
            fidelities = torch.stack(fidelities)
            
            # Get ground truth labels (if available)
            labels = snapshot.y if hasattr(snapshot, 'y') else torch.zeros(causal_z.size(0), device=self.device)
            
            return {
                'scores': anomaly_scores.cpu(),
                'fidelities': fidelities.cpu(),
                'labels': labels.cpu(),
                'embeddings': causal_z.cpu(),
                'uncertainty': causal_uncertainty.cpu()
            }
    
    def evaluate(self, dataset_name: str, split: str = "test") -> Dict[str, Any]:
        """Main evaluation loop for CausalTRACE."""
        self.logger.info(f"Starting evaluation for CausalTRACE on {dataset_name} ({split} split)")
        
        # Load dataset
        dataset_config = self.config['datasets']['defaults'].copy()
        if dataset_name in self.config['datasets']:
            dataset_config.update(self.config['datasets'][dataset_name])
        
        # Load temporal data
        temporal_data = load_dataset(
            dataset_name,
            temporal=dataset_config['temporal'],
            num_snapshots=dataset_config['num_snapshots'],
            inject_anomalies=dataset_config['inject_anomalies']
        )
        
        # Load model if not already loaded
        if self.embedding_updater is None:
            self.load_model()
        
        # Reset metrics tracker
        self.metrics_tracker.reset()
        
        # Evaluate on all snapshots
        for i, snapshot in enumerate(temporal_data):
            self.logger.info(f"Evaluating snapshot {i+1}/{len(temporal_data)}")
            
            # Perform evaluation step
            eval_result = self.eval_step(snapshot)
            
            # Update metrics tracker
            self.metrics_tracker.update(
                scores=eval_result['scores'],
                labels=eval_result['labels'],
                fidelities=eval_result['fidelities'],
                timestamp=i
            )
            
            # Save predictions if configured
            if self.config['experiment']['save_predictions']:
                pred_file = self.output_dir / f"predictions_snapshot_{i:03d}.pt"
                torch.save({
                    'scores': eval_result['scores'],
                    'fidelities': eval_result['fidelities'],
                    'labels': eval_result['labels'],
                    'embeddings': eval_result['embeddings'],
                    'uncertainty': eval_result['uncertainty'],
                    'timestamp': i
                }, pred_file)
        
        # Compute final metrics
        results = self.metrics_tracker.get_summary()
        
        # Save results
        results_file = self.output_dir / f"evaluation_results_{dataset_name}_{split}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evaluation completed. Results saved to {results_file}")
        self.logger.info(f"Final AUC: {results['auc']:.4f} [{results['auc_ci_lower']:.4f}, {results['auc_ci_upper']:.4f}]")
        self.logger.info(f"Mean Causal Fidelity: {results['mean_causal_fidelity']:.4f}")
        self.logger.info(f"Peak GPU Memory: {results['peak_gpu_mb']:.2f} MB")
        
        return results


# Factory function for evaluators
def create_evaluator(model_name: str, config: Dict[str, Any], device: torch.device = None) -> BaseEvaluator:
    """
    Factory function to create appropriate evaluator based on model name.
    
    Args:
        model_name: Name of the model
        config: Configuration dictionary
        device: Device to evaluate on
        
    Returns:
        Initialized evaluator instance
    """
    if model_name == "CausalTRACE":
        return CausalTraceEvaluator(config, device)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Only CausalTRACE is implemented in this version.")


# Main evaluation function
def evaluate_model(
    model_name: str,
    dataset_name: str,
    config_path: str = "config/hyperparams.yaml",
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    split: str = "test"
):
    """
    Main evaluation function.
    
    Args:
        model_name: Name of the model to evaluate
        dataset_name: Name of the dataset to evaluate on
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint (optional)
        device: Device to evaluate on (optional)
        split: Dataset split to evaluate on
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = create_evaluator(model_name, config, device)
    
    # Load model
    evaluator.load_model(checkpoint_path)
    
    # Start evaluation
    results = evaluator.evaluate(dataset_name, split)
    return results


# Example usage and testing
if __name__ == "__main__":
    # Evaluate CausalTRACE on DGraph
    results = evaluate_model("CausalTRACE", "DGraph")
    print(f"Evaluation AUC: {results['auc']:.4f}")
    print(f"Peak GPU Memory: {results['peak_gpu_mb']:.2f} MB")