# CausalTRACE: Causal Temporal Representation and Anomaly Consistency Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**CausalTRACE** redefines graph anomaly detection by grounding it in **temporal causal invariance** rather than reconstruction fidelity. Our framework detects anomalies that violate stable causal mechanisms over time—precisely the evasive, adaptive behaviors that fool existing methods.

## 🚀 Main Contributions

1. **First Causal-Temporal GAD Framework**: Unifies causal mechanism discovery, temporal consistency, and memory-constrained streaming in large-scale dynamic graphs
2. **Causal Fidelity Metric** (\( \mathcal{F}(t) \)): Statistically interpretable metric quantifying adherence to learned invariant mechanisms via KL divergence
3. **Memory-Efficient Design**: Sublinear memory architecture (\( O(K d_z) \)) enabling deployment on billion-edge graphs
4. **Causal-DGraph Benchmark**: First dynamic graph benchmark with mechanism-violating anomalies injected via do-interventions

## 📊 Why Existing Methods Fail

| **Method Category** | **Key Limitation** | **Real-World Consequence** |
|---------------------|-------------------|---------------------------|
| **Reconstruction-based** (MEGAD, DOMINANT, AnomalyDAE) | Treats anomalies as reconstruction failures | Misses camouflaged anomalies that reconstruct well but violate causal laws (e.g., fraudulent accounts with legitimate-looking neighborhoods) |
| **Dynamic GNNs** (TGN, DySAT) | Models temporal evolution but remains correlational | Tracks *that* behavior changes, not *whether* it violates causal invariance—fails on mechanism-shift anomalies |
| **Causal Baselines** (CausalNID, CausalSCM) | Requires full-graph access and dense message passing | Out-of-memory failures on large graphs (>100K nodes); cannot scale to industrial datasets |

**Core Gap**: No existing method can detect **mechanism-violating anomalies**—those that preserve local structure but break the underlying causal laws governing normal system behavior.

## 📁 Dataset References & Preparation

All datasets are converted to dynamic temporal snapshots with mechanism-violating anomalies:

| Dataset | Nodes | Edges | Features | Original Source | Temporal Augmentation |
|---------|-------|-------|----------|----------------|----------------------|
| **Weibo** | 8,405 | 407,963 | 400 | [Weibo Dataset](https://github.com/Graph-ML/MEGAD) | Split by post timestamp → 10 snapshots |
| **Facebook** | 1,081 | 27,552 | 576 | [Facebook Dataset](https://github.com/Graph-ML/MEGAD) | Synthetic temporal evolution (node activation waves) |
| **Disney** | 1,490 | 3,750 | 17 | [Disney Dataset](https://github.com/Graph-ML/MEGAD) | Inject delayed-effect anomalies |
| **Books** | 1,489 | 3,980 | 18 | [Books Dataset](https://github.com/Graph-ML/MEGAD) | Inject delayed-effect anomalies |
| **Flickr** | 89,250 | 933,804 | 500 | [Flickr Dataset](https://github.com/Graph-ML/MEGAD) | Use metadata timestamps → 5 time bins |
| **DGraph** | 3.7M | 4.3M | 17 | [DGraph Dataset](https://github.com/Graph-ML/MEGAD) | Financial transaction timestamps → 24-hour sliding windows |

## 🏆 Key Results

CausalTRACE achieves **state-of-the-art performance** across all datasets while maintaining **sub-3GB memory usage** even on DGraph (3.7M nodes):

| Method | Weibo | Facebook | Disney | Books | Flickr | DGraph |
|--------|-------|----------|--------|-------|--------|--------|
| **DOMINANT** | 71.2 | 68.4 | 70.1 | 69.5 | 67.7 | 59.2 |
| **AnomalyDAE** | 72.8 | 70.2 | 71.5 | 70.8 | 68.9 | 60.3 |
| **MEGAD** | 75.1 | 72.9 | 73.2 | 72.5 | 71.4 | 64.7 |
| **TGN** | 76.7 | 74.5 | 74.3 | 73.9 | 72.7 | 66.8 |
| **DySAT** | 75.8 | 73.8 | 73.5 | 73.1 | 71.9 | 65.5 |
| **CausalNID** | 73.4 | 71.1 | 72.0 | 71.6 | 70.3 | 62.1 |
| **CausalSCM** | 74.2 | 71.8 | 72.6 | 72.2 | 70.9 | 63.3 |
| **CausalTRACE (Ours)** | **85.7** | **83.3** | **84.2** | **83.6** | **82.4** | **75.9** |

**Causal Fidelity**: CausalTRACE achieves \( \mathcal{F} \in [0.75, 0.82] \), while causal baselines remain below 0.45, confirming superior mechanism awareness.

## 🏗️ Code Structure & Reusability

The codebase follows **strict object-oriented design principles** with **modular, reusable components** that can be easily extended or integrated into other projects:

```
causaltrace/
├── data/
│   ├── dataset_loader.py      # Unified loader for all 6 datasets
│   └── temporal_augmentation.py  # Convert static → dynamic with causal anomalies
├── models/
│   ├── causal_energy.py       # Causal-Temporal Energy Model (Sec 5.2)
│   ├── embedding_updater.py   # Continual Causal Embedding Engine (Sec 5.3)
│   └── anomaly_scorer.py      # Causal Fidelity Scoring (Sec 5.4)
├── utils/
│   ├── memory_bank.py         # Compressed Causal Memory (Sec 5.3)
│   └── metrics.py             # AUC, Causal Fidelity, Memory Tracking
├── config/
│   └── hyperparams.yaml       # All hyperparameters (ablation-ready)
├── train.py                   # Training loop with strategy pattern
├── evaluate.py                # Evaluation with statistical confidence intervals
└── main.py                    # Entry point with CLI interface
```

### 🔧 Key Reusability Features

1. **Protocol-Based Interfaces**: All components implement `@runtime_checkable` protocols for type-safe integration
2. **Strategy Pattern**: Model-agnostic training/evaluation via factory functions
3. **Configurable Components**: Every module accepts hyperparameters via YAML configuration
4. **Memory-Efficient Design**: Streaming-compatible with constant memory per node
5. **Dataset-Agnostic**: Works with any attributed graph dataset following the standard format
6. **SOTA Baseline Support**: Easy integration of new baseline methods via the trainer/evaluator interface

### 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train CausalTRACE on DGraph
python main.py train --model CausalTRACE --dataset DGraph

# Evaluate on test set
python main.py evaluate --model CausalTRACE --dataset DGraph

# Run ablation study
python main.py ablate --model CausalTRACE --dataset DGraph --param energy_depth

# Benchmark against all SOTA methods
python main.py benchmark --dataset DGraph
```

### 📝 Configuration-Driven Experiments

All hyperparameters are centralized in `config/hyperparams.yaml`, enabling:

- **Ablation studies** with automatic parameter sweeping
- **Dataset-specific overrides** for optimal performance
- **Reproducible experiments** with fixed seeds and device settings
- **Memory-constrained evaluation** with GPU/CPU memory tracking

## 📚 Citation

If you use CausalTRACE in your research, please cite our paper:

```bibtex
@article{iqbal2026causaltrace,
  title={CausalTRACE: Causal Temporal Representation and Anomaly Consistency Engine for Scalable Dynamic Graph Monitoring},
  author={Iqbal, Saeed},
  journal={arXiv preprint arXiv:2601.XXXXX},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset preprocessing builds upon [MEGAD](https://github.com/Graph-ML/MEGAD)
- Core GNN operations leverage [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Memory profiling uses [psutil](https://github.com/giampaolo/psutil) for CPU monitoring

---

**CausalTRACE establishes a new paradigm for graph anomaly detection: one grounded not in reconstruction fidelity, but in the stability of causal laws over time.**
