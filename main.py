#!/usr/bin/env python3
"""
CausalTRACE: Causal Temporal Representation and Anomaly Consistency Engine
Entry point for training, evaluation, and ablation studies.

Usage:
    python main.py train --model CausalTRACE --dataset DGraph
    python main.py evaluate --model CausalTRACE --dataset DGraph
    python main.py ablate --model CausalTRACE --dataset DGraph --param energy_depth
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import yaml
import logging
from typing import Dict, Any, List, Optional

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# Import local modules
from causaltrace.train import train_model, create_trainer
from causaltrace.evaluate import evaluate_model, create_evaluator
from causaltrace.config.hyperparams import DEFAULT_CONFIG


class CausalTRACEPipeline:
    """
    Main pipeline orchestrator for CausalTRACE experiments.
    Handles configuration, device setup, and experiment execution.
    """
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize pipeline with command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.config = self._load_and_merge_config()
        self.device = self._setup_device()
        self.logger = self._setup_logger()
        
    def _load_and_merge_config(self) -> Dict[str, Any]:
        """
        Load base configuration and merge with command-line overrides.
        
        Returns:
            Merged configuration dictionary
        """
        # Load base configuration
        if self.args.config:
            with open(self.args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = DEFAULT_CONFIG.copy()
        
        # Override with command-line arguments
        if self.args.seed is not None:
            config['reproducibility']['seed'] = self.args.seed
        
        if self.args.device:
            config['experiment']['device'] = self.args.device
            
        if self.args.output_dir:
            config['experiment']['output_dir'] = self.args.output_dir
            
        if self.args.epochs is not None:
            if self.args.model in config['models']:
                config['models'][self.args.model]['training']['epochs'] = self.args.epochs
        
        return config
    
    def _setup_device(self) -> torch.device:
        """
        Setup computation device (CPU/GPU).
        
        Returns:
            Configured torch device
        """
        device_str = self.config['experiment']['device']
        
        if device_str == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        elif device_str.startswith('cuda:') and not torch.cuda.is_available():
            logging.warning(f"{device_str} requested but CUDA not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device(device_str)
            
        return device
    
    def _setup_logger(self) -> logging.Logger:
        """
        Setup root logger for the entire pipeline.
        
        Returns:
            Configured logger instance
        """
        log_level = getattr(logging, self.config['experiment']['log_level'].upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(Path(self.config['experiment']['output_dir']) / 'pipeline.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def run(self):
        """Execute the requested operation."""
        operation = self.args.operation
        
        if operation == 'train':
            self._run_training()
        elif operation == 'evaluate':
            self._run_evaluation()
        elif operation == 'ablate':
            self._run_ablation()
        elif operation == 'benchmark':
            self._run_benchmark()
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _run_training(self):
        """Execute training operation."""
        self.logger.info(f"Starting training: {self.args.model} on {self.args.dataset}")
        
        try:
            train_model(
                model_name=self.args.model,
                dataset_name=self.args.dataset,
                config_path=self.args.config,
                device=self.device
            )
            self.logger.info("Training completed successfully")
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def _run_evaluation(self):
        """Execute evaluation operation."""
        self.logger.info(f"Starting evaluation: {self.args.model} on {self.args.dataset}")
        
        try:
            results = evaluate_model(
                model_name=self.args.model,
                dataset_name=self.args.dataset,
                config_path=self.args.config,
                checkpoint_path=self.args.checkpoint,
                device=self.device,
                split=self.args.split
            )
            
            # Print summary
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"Model: {self.args.model}")
            print(f"Dataset: {self.args.dataset}")
            print(f"AUC: {results['auc']:.4f} [{results['auc_ci_lower']:.4f}, {results['auc_ci_upper']:.4f}]")
            print(f"Mean Causal Fidelity: {results['mean_causal_fidelity']:.4f}")
            print(f"Peak GPU Memory: {results['peak_gpu_mb']:.2f} MB")
            print(f"F1 Score: {results['f1_score']:.4f}")
            print("="*60)
            
            self.logger.info("Evaluation completed successfully")
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def _run_ablation(self):
        """Execute ablation study operation."""
        self.logger.info(f"Starting ablation study: {self.args.model} on {self.args.dataset}")
        
        if not self.args.param:
            raise ValueError("Ablation parameter (--param) must be specified")
        
        if self.args.param not in self.config['ablation']:
            available_params = list(self.config['ablation'].keys())
            raise ValueError(f"Unknown ablation parameter: {self.args.param}. Available: {available_params}")
        
        # Get parameter values
        param_values = self.config['ablation'][self.args.param]['values']
        results = {}
        
        self.logger.info(f"Ablation parameter: {self.args.param}, values: {param_values}")
        
        for value in param_values:
            self.logger.info(f"Testing {self.args.param} = {value}")
            
            # Create modified config
            modified_config = self._create_modified_config(self.args.param, value)
            config_path = self._save_temp_config(modified_config, f"ablation_{self.args.param}_{value}.yaml")
            
            try:
                # Train model with modified config
                train_model(
                    model_name=self.args.model,
                    dataset_name=self.args.dataset,
                    config_path=config_path,
                    device=self.device
                )
                
                # Evaluate model
                eval_results = evaluate_model(
                    model_name=self.args.model,
                    dataset_name=self.args.dataset,
                    config_path=config_path,
                    device=self.device
                )
                
                results[str(value)] = {
                    'auc': eval_results['auc'],
                    'auc_ci': [eval_results['auc_ci_lower'], eval_results['auc_ci_upper']],
                    'fidelity': eval_results['mean_causal_fidelity'],
                    'memory': eval_results['peak_gpu_mb']
                }
                
                self.logger.info(f"Completed {self.args.param} = {value}: AUC = {eval_results['auc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed for {self.args.param} = {value}: {str(e)}")
                results[str(value)] = {'error': str(e)}
            
            # Cleanup temp config
            os.remove(config_path)
        
        # Save ablation results
        ablation_file = Path(self.config['experiment']['output_dir']) / f"ablation_{self.args.param}_{self.args.dataset}.json"
        import json
        with open(ablation_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Ablation results saved to {ablation_file}")
    
    def _create_modified_config(self, param_name: str, param_value: Any) -> Dict[str, Any]:
        """Create a modified configuration with the specified parameter value."""
        import copy
        config = copy.deepcopy(self.config)
        
        # Map parameter names to config paths
        param_mapping = {
            'energy_depth': ['models', 'CausalTRACE', 'causal_energy', 'depth'],
            'mc_samples': ['models', 'CausalTRACE', 'embedding_updater', 'mc_samples'],
            'prototypes_K': ['models', 'CausalTRACE', 'memory_bank', 'K'],
            'temporal_bandwidth': ['models', 'CausalTRACE', 'causal_energy', 'temporal_bandwidth'],
            'kl_threshold': ['models', 'CausalTRACE', 'anomaly_scorer', 'kl_threshold'],
            'learning_rate': ['models', 'CausalTRACE', 'embedding_updater', 'learning_rate'],
            'sparsity_threshold': ['models', 'CausalTRACE', 'causal_energy', 'sparsity_threshold']
        }
        
        if param_name not in param_mapping:
            raise ValueError(f"Cannot map parameter {param_name} to config path")
        
        # Navigate to the parameter location and update it
        current = config
        path = param_mapping[param_name]
        for key in path[:-1]:
            current = current[key]
        current[path[-1]] = param_value
        
        return config
    
    def _save_temp_config(self, config: Dict[str, Any], filename: str) -> str:
        """Save configuration to temporary file and return path."""
        temp_dir = Path(self.config['experiment']['output_dir']) / 'temp_configs'
        temp_dir.mkdir(parents=True, exist_ok=True)
        config_path = temp_dir / filename
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    def _run_benchmark(self):
        """Execute benchmark against all SOTA methods."""
        self.logger.info(f"Starting benchmark: {self.args.dataset}")
        
        # Get all available models
        models = list(self.config['models'].keys())
        results = {}
        
        for model in models:
            self.logger.info(f"Benchmarking {model} on {self.args.dataset}")
            
            try:
                # Train model
                train_model(
                    model_name=model,
                    dataset_name=self.args.dataset,
                    config_path=self.args.config,
                    device=self.device
                )
                
                # Evaluate model
                eval_results = evaluate_model(
                    model_name=model,
                    dataset_name=self.args.dataset,
                    config_path=self.args.config,
                    device=self.device
                )
                
                results[model] = {
                    'auc': eval_results['auc'],
                    'auc_ci': [eval_results['auc_ci_lower'], eval_results['auc_ci_upper']],
                    'fidelity': eval_results.get('mean_causal_fidelity', 0.0),
                    'memory': eval_results['peak_gpu_mb'],
                    'f1_score': eval_results['f1_score']
                }
                
                self.logger.info(f"Completed {model}: AUC = {eval_results['auc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed for {model}: {str(e)}")
                results[model] = {'error': str(e)}
        
        # Save benchmark results
        benchmark_file = Path(self.config['experiment']['output_dir']) / f"benchmark_{self.args.dataset}.json"
        import json
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print comparison table
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Model':<20} {'AUC':<10} {'Fidelity':<12} {'Memory (MB)':<15} {'F1 Score':<10}")
        print("-"*80)
        
        for model, result in results.items():
            if 'error' in result:
                print(f"{model:<20} {'ERROR':<10} {'-':<12} {'-':<15} {'-':<10}")
            else:
                print(f"{model:<20} {result['auc']:<10.4f} {result['fidelity']:<12.4f} {result['memory']:<15.2f} {result['f1_score']:<10.4f}")
        
        print("="*80)
        self.logger.info(f"Benchmark results saved to {benchmark_file}")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration if config file is not provided."""
    return {
        'experiment': {
            'name': 'causaltrace_default',
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'output_dir': 'results',
            'save_checkpoints': True,
            'save_predictions': True,
            'log_level': 'INFO'
        },
        'reproducibility': {
            'seed': 42,
            'deterministic': False,
            'cudnn_benchmark': True,
            'torch_threads': 4
        },
        'datasets': {
            'defaults': {
                'temporal': True,
                'num_snapshots': 10,
                'inject_anomalies': True,
                'anomaly_ratio': 0.05,
                'train_split': 0.7,
                'val_split': 0.1,
                'test_split': 0.2,
                'batch_size': 1024,
                'num_workers': 4
            }
        },
        'models': {
            'CausalTRACE': {
                'embedding_updater': {
                    'input_dim': 16,
                    'emb_dim': 64,
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'mc_samples': 10,
                    'learning_rate': 0.01,
                    'momentum': 0.9,
                    'min_uncertainty': 1e-6
                }
            }
        },
        'evaluation': {
            'confidence_level': 0.95,
            'bootstrap_samples': 1000
        }
    }


# Global default configuration
DEFAULT_CONFIG = create_default_config()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CausalTRACE: Causal Temporal Representation and Anomaly Consistency Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Operation subcommands
    subparsers = parser.add_subparsers(dest='operation', help='Operation to perform')
    subparsers.required = True
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', required=True, help='Model name (e.g., CausalTRACE, MEGAD)')
    train_parser.add_argument('--dataset', required=True, help='Dataset name (e.g., DGraph, Weibo)')
    train_parser.add_argument('--config', help='Path to configuration file')
    train_parser.add_argument('--seed', type=int, help='Random seed')
    train_parser.add_argument('--device', help='Device (cuda, cpu, cuda:0, etc.)')
    train_parser.add_argument('--output-dir', help='Output directory')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    
    # Evaluation parser
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', required=True, help='Model name')
    eval_parser.add_argument('--dataset', required=True, help='Dataset name')
    eval_parser.add_argument('--config', help='Path to configuration file')
    eval_parser.add_argument('--checkpoint', help='Path to model checkpoint')
    eval_parser.add_argument('--split', default='test', help='Dataset split (train, val, test)')
    eval_parser.add_argument('--device', help='Device (cuda, cpu, cuda:0, etc.)')
    eval_parser.add_argument('--output-dir', help='Output directory')
    
    # Ablation parser
    ablate_parser = subparsers.add_parser('ablate', help='Run ablation study')
    ablate_parser.add_argument('--model', required=True, help='Model name')
    ablate_parser.add_argument('--dataset', required=True, help='Dataset name')
    ablate_parser.add_argument('--param', required=True, help='Ablation parameter (energy_depth, mc_samples, etc.)')
    ablate_parser.add_argument('--config', help='Path to configuration file')
    ablate_parser.add_argument('--device', help='Device (cuda, cpu, cuda:0, etc.)')
    ablate_parser.add_argument('--output-dir', help='Output directory')
    
    # Benchmark parser
    bench_parser = subparsers.add_parser('benchmark', help='Benchmark against SOTA methods')
    bench_parser.add_argument('--dataset', required=True, help='Dataset name')
    bench_parser.add_argument('--config', help='Path to configuration file')
    bench_parser.add_argument('--device', help='Device (cuda, cpu, cuda:0, etc.)')
    bench_parser.add_argument('--output-dir', help='Output directory')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create and run pipeline
        pipeline = CausalTRACEPipeline(args)
        pipeline.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()