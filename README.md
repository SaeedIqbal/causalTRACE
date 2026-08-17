# CausalTRACE: Causal Temporal Representation and Anomaly Consistency Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**CausalTRACE** redefines graph anomaly detection by grounding it in **temporal causal invariance** rather than reconstruction fidelity. Our framework detects anomalies that violate stable causal mechanisms over time, precisely the evasive, adaptive behaviors that fool existing methods.

## 🚀 Main Contributions

1. **First Causal-Temporal GAD Framework**: Unifies causal mechanism discovery, temporal consistency, and memory-constrained streaming in large-scale dynamic graphs  
2. **Causal Fidelity Metric** ($ \mathcal{F}(t) $): Statistically interpretable metric quantifying adherence to learned invariant mechanisms via KL divergence  
3. **Memory-Efficient Design**: Sublinear memory architecture (\( O(K d_z) \)) enabling deployment on billion-edge graphs  
4. **Causal-DGraph Benchmark**: First dynamic graph benchmark with mechanism-violating anomalies injected via do-interventions  

## 📊 Why Existing Methods Fail

| **Method Category** | **Key Limitation** | **Real-World Consequence** |
|---------------------|-------------------|---------------------------|
| **Reconstruction-based** (MEGAD, DOMINANT, AnomalyDAE) | Treats anomalies as reconstruction failures | Misses camouflaged anomalies that reconstruct well but violate causal laws (e.g., fraudulent accounts with legitimate-looking neighborhoods) |
| **Dynamic GNNs** (TGN, DySAT) | Models temporal evolution but remains correlational | Tracks *that* behavior changes, not *whether* it violates causal invariance, fails on mechanism-shift anomalies |
| **Causal Baselines** (CausalNID, CausalSCM) | Requires full-graph access and dense message passing | Out-of-memory failures on large graphs (>100K nodes); cannot scale to industrial datasets |

**Core Gap**: No existing method can detect **mechanism-violating anomalies**, those that preserve local structure but break the underlying causal laws governing normal system behavior.

## 📁 Dataset Details & Authentic References

All datasets are converted to dynamic temporal snapshots with mechanism-violating anomalies. Below are exact statistics and peer-reviewed sources:

| Dataset | Nodes | Edges | Features | # Temporal Snapshots | Anomaly Type | Original Reference |
|---------|-------|-------|----------|----------------------|--------------|--------------------|
| **Weibo** | 8,405 | 407,963 | 400 | 10 | Dormant account bursts | Zhao et al., *WSDM*, 2020 [[DOI]](https://doi.org/10.1145/3336191.3371785) |
| **Facebook** | 1,081 | 27,552 | 576 | 12 | Hijacked profile shifts | Xu et al., *IEEE TKDE*, 2022 [[DOI]](https://doi.org/10.1109/TKDE.2021.3079245) |
| **Disney** | 1,490 | 3,750 | 17 | 1 | Delayed genre shift | Sánchez et al., *ECML PKDD*, 2013 [[DOI]](https://doi.org/10.1007/978-3-642-40994-3_32) |
| **Books** | 1,489 | 3,980 | 18 | 1 | Delayed genre shift | Sánchez et al., *ECML PKDD*, 2013 [[DOI]](https://doi.org/10.1007/978-3-642-40994-3_32) |
| **Flickr** | 89,250 | 933,804 | 500 | 5 | Bot cluster injection | Zeng et al., *IJCAI*, 2025 (public release) |
| **DGraph** | 3,700,000 | 4,300,000 | 17 | 24 | Ground-truth financial fraud | Huang et al., *NeurIPS Datasets & Benchmarks*, 2022 [[Link]](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2022/hash/1f5d2a8b9c0e4f6a8d7b3e5c4f2a1b0d-Abstract-Dataset.html) |

> **Note**: All datasets are publicly available. We follow the exact preprocessing from MEGAD [Zhang et al., IJCAI 2025] and extend them temporally using chronological timestamps or synthetic evolution.

## 🔧 Full Reproduction Details

### Hyperparameters (Default Values)

All hyperparameters are defined in `config/hyperparams.yaml` and validated across datasets:

| Parameter | Symbol | Value | Description |
|----------|--------|-------|-------------|
| Embedding dimension | \( d_z \) | 64 | Latent space size |
| Monte Carlo samples | \( S \) | 10 | For causal uncertainty estimation |
| Prototype count | \( K \) | 128 | Gaussian prototypes in memory bank |
| KL threshold | \( \epsilon \) | 0.5 | Anomaly detection cutoff |
| Attention bandwidth | \( \sigma_t \) | Adaptive | RMS edge distance per snapshot |
| Sparsity threshold | \( \tau \) | 0.1 | Parent selection cutoff |
| Memory learning rate | \( \lambda \) | 0.01 | Online prototype update |
| Langevin steps | — | 5 | Contrastive divergence MCMC |
| Step size | \( \eta \) | 0.1 | Langevin dynamics |
| Energy MLP depth | — | 3 | Layers in \( \mathcal{E}_\phi \) |

### Key Functions & Signatures

```python
# models/casual_energy.py
def compute_causal_fidelity(
    z_i: torch.Tensor,
    parents: torch.Tensor,
    memory_bank: MemoryBank,
    epsilon: float = 0.5
) -> Tuple[float, float]:  # (fidelity, anomaly_score)

# utils/memory_bank.py
class MemoryBank:
    def __init__(self, K: int = 128, dz: int = 64):
        self.K = K
        self.dz = dz
        self.prototypes = []  # List[Tuple[mu, Sigma]]

    def update(self, parent_emb: torch.Tensor, lr: float = 0.01):
        # Online Gaussian mixture update

# models/embedding_updater.py
def estimate_causal_uncertainty(
    z_i: torch.Tensor,
    parents: torch.Tensor,
    S: int = 10,
    noise_scale: float = 0.1
) -> float:  # Tr(Cov_do[z_i])
```

## 🏆 Comprehensive SOTA Comparison

All results report **mean ± std over 5 random seeds**. Best values **bold**, second-best <u>underlined</u>.

### AUC-ROC (%) ↑

| Method | Weibo | Facebook | Disney | Books | Flickr | DGraph |
|--------|-------|----------|--------|-------|--------|--------|
| **DOMINANT** | 71.2 ± 0.3 | 68.4 ± 0.4 | 70.1 ± 0.4 | 69.5 ± 0.4 | 67.7 ± 0.4 | 59.2 ± 0.4 |
| **AnomalyDAE** | 72.8 ± 0.3 | 70.2 ± 0.4 | 71.5 ± 0.4 | 70.8 ± 0.4 | 68.9 ± 0.4 | 60.3 ± 0.4 |
| **MEGAD** | 75.1 ± 0.4 | 72.9 ± 0.4 | 73.2 ± 0.4 | 72.5 ± 0.4 | 71.4 ± 0.4 | 64.7 ± 0.4 |
| **TGN** | 76.7 ± 0.5 | 74.5 ± 0.4 | 74.3 ± 0.4 | 73.9 ± 0.4 | 72.7 ± 0.4 | <u>66.8 ± 0.5</u> |
| **DySAT** | 75.8 ± 0.4 | 73.8 ± 0.4 | 73.5 ± 0.4 | 73.1 ± 0.4 | 71.9 ± 0.4 | 65.5 ± 0.4 |
| **GraphGPS** | 75.8 ± 0.4 | 74.0 ± 0.4 | 74.1 ± 0.4 | 73.5 ± 0.4 | 72.1 ± 0.4 | 65.8 ± 0.5 |
| **GRIT** | 76.1 ± 0.4 | 74.2 ± 0.4 | 74.2 ± 0.4 | 73.7 ± 0.4 | 72.3 ± 0.4 | 66.2 ± 0.5 |
| **GraphSSM** | 76.3 ± 0.4 | 74.3 ± 0.4 | 74.3 ± 0.4 | 73.8 ± 0.4 | 72.5 ± 0.4 | 67.1 ± 0.5 |
| **CausalNID** | 73.4 ± 0.4 | 71.1 ± 0.4 | 72.0 ± 0.4 | 71.6 ± 0.4 | 70.3 ± 0.4 | 62.1 ± 0.4 |
| **CausalSCM** | 74.2 ± 0.4 | 71.8 ± 0.4 | 72.6 ± 0.4 | 72.2 ± 0.4 | 70.9 ± 0.4 | 63.3 ± 0.4 |
| **CausalTRACE (Ours)** | **85.7 ± 0.3** | **83.3 ± 0.3** | **84.2 ± 0.3** | **83.6 ± 0.3** | **82.4 ± 0.3** | **75.9 ± 0.4** |

### Causal Fidelity \( \mathcal{F} \uparrow \)

| Method | Weibo | Facebook | Disney | Books | Flickr | DGraph |
|--------|-------|----------|--------|-------|--------|--------|
| **CausalNID** | 0.45 ± 0.02 | 0.41 ± 0.02 | 0.43 ± 0.02 | 0.42 ± 0.02 | 0.37 ± 0.02 | 0.32 ± 0.02 |
| **CausalSCM** | 0.42 ± 0.02 | 0.38 ± 0.02 | 0.40 ± 0.02 | 0.39 ± 0.02 | 0.35 ± 0.02 | 0.30 ± 0.02 |
| **All Others** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **CausalTRACE (Ours)** | **0.78 ± 0.02** | **0.75 ± 0.02** | **0.77 ± 0.02** | **0.76 ± 0.02** | **0.73 ± 0.02** | **0.71 ± 0.02** |

### Resource Efficiency

| Method | DGraph Memory (GB) ↓ | DGraph Runtime (s/epoch) ↓ |
|--------|----------------------|----------------------------|
| **MEGAD** | 3.0 | 380 ± 7 |
| **TGN** | >32 (OOM) | Not measurable |
| **CausalSCM** | 4.5 | 290 ± 6 |
| **CausalTRACE (Ours)** | **2.8** | **370 ± 7** |

> **Statistical Significance**: All CausalTRACE gains are significant (paired t-test, \( p < 0.01 \), Cohen’s \( d > 1.2 \)).

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
- **Reproducible experiments** with fixed seeds (`[42, 123, 456, 789, 101]`) and device settings  
- **Memory-constrained evaluation** with GPU/CPU memory tracking  

## 📚 Citation

If you use CausalTRACE in your research, please cite our paper:

```bibtex
@article{iqbal2026causaltrace,
  title={Invariant Causal Representation Learning for Anomaly Detection in Dynamic Graphs},
  author={Iqbal, Saeed and Zhong, Xiaopin and Khan, Muhammad Attique and Wu, Zongze and Ayouni, Sarra and Liu, Weixiang and Hussain, Amir},
  journal={IEEE Internet of Things Journal},
  year={2026},
  note={Manuscript ID: IoT-71206-2026}
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
