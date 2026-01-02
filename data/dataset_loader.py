import os
import abc
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Optional, Dict, Any
from torch_geometric.data import Data
from pathlib import Path


class BaseDataset(abc.ABC):
    """
    Abstract base class for all graph datasets.
    Enforces consistent interface for loading, preprocessing, and temporal conversion.
    """
    def __init__(self, root: str, name: str):
        self.root = Path(root)
        self.name = name
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._data = None

    @abc.abstractmethod
    def _load_raw_data(self) -> Dict[str, np.ndarray]:
        """
        Load raw node/edge data from disk.
        Returns dict with keys: 'x', 'edge_index', 'y', 'timestamps' (optional).
        """
        pass

    @abc.abstractmethod
    def _preprocess_static(self, raw_data: Dict[str, np.ndarray]) -> Data:
        """
        Convert raw data to PyG Data object (static graph).
        """
        pass

    def _augment_temporal( self,
        static_data: Data,
        num_snapshots: int = 10,
        inject_anomalies: bool = True
    ) -> List[Data]:
        """
        Convert static graph to temporal snapshots with optional causal anomaly injection.
        Uses memory-efficient chunking for large graphs.
        """
        edge_index = static_data.edge_index.numpy()
        x = static_data.x.numpy()
        num_edges = edge_index.shape[1]

        # Determine snapshot boundaries
        if hasattr(static_data, 'timestamps') and static_data.timestamps is not None:
            # Use real timestamps if available
            timestamps = static_data.timestamps.numpy()
            sorted_idx = np.argsort(timestamps)
            edge_index = edge_index[:, sorted_idx]
            timestamps = timestamps[sorted_idx]
            snapshot_edges = self._split_by_timestamps(timestamps, num_snapshots)
        else:
            # Uniform split for static datasets
            snapshot_edges = np.array_split(np.arange(num_edges), num_snapshots)

        temporal_data = []
        for t, edge_indices in enumerate(snapshot_edges):
            if len(edge_indices) == 0:
                continue
                
            # Create snapshot graph
            snap_edge_index = torch.from_numpy(edge_index[:, edge_indices]).long()
            snap_x = torch.from_numpy(x).float()
            snap_y = torch.zeros(snap_edge_index.shape[1])  # Normal labels

            # Inject causal anomalies in last snapshot
            if inject_anomalies and t == num_snapshots - 1:
                snap_edge_index, snap_y = self._inject_causal_anomalies(
                    snap_edge_index, snap_x, ratio=0.05
                )

            temporal_data.append(Data(
                x=snap_x,
                edge_index=snap_edge_index,
                y=snap_y,
                timestamp=t
            ))
        
        return temporal_data

    def _split_by_timestamps(self, timestamps: np.ndarray, num_snapshots: int) -> List[np.ndarray]:
        """Split edges into snapshots based on sorted timestamps."""
        unique_ts = np.unique(timestamps)
        if len(unique_ts) <= num_snapshots:
            return [np.where(timestamps == ts)[0] for ts in unique_ts]
        
        # Bin timestamps into equal-sized groups
        bins = np.array_split(np.arange(len(unique_ts)), num_snapshots)
        edge_groups = []
        for bin_ts in bins:
            ts_vals = unique_ts[bin_ts]
            mask = np.isin(timestamps, ts_vals)
            edge_groups.append(np.where(mask)[0])
        return edge_groups

    def _inject_causal_anomalies(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        ratio: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject mechanism-violating anomalies via do-interventions.
        Default: Flip attribute dimensions that violate causal expectations.
        Subclasses should override for domain-specific logic.
        """
        num_edges = edge_index.shape[1]
        num_anomalies = int(ratio * num_edges)
        anomaly_indices = np.random.choice(num_edges, num_anomalies, replace=False)
        
        # Create anomaly labels
        y = torch.zeros(num_edges)
        y[anomaly_indices] = 1.0
        
        # Generic anomaly: perturb source node attributes
        for idx in anomaly_indices:
            src = edge_index[0, idx].item()
            # Flip first feature dimension (simulate income/spending violation)
            x[src, 0] = 1.0 - x[src, 0]  
            if x.size(1) > 1:
                x[src, 1] = 1.0  # Set high spending
        
        return edge_index, y

    def load(
        self,
        temporal: bool = True,
        num_snapshots: int = 10,
        inject_anomalies: bool = True
    ) -> List[Data]:
        """
        Main entry point: load and preprocess dataset.
        
        Args:
            temporal: If True, return list of temporal snapshots; else static graph.
            num_snapshots: Number of temporal snapshots to create.
            inject_anomalies: Whether to inject causal anomalies in last snapshot.
            
        Returns:
            List of PyG Data objects (temporal) or single Data object (static).
        """
        if self._data is not None:
            return self._data

        # Load raw data
        raw_data = self._load_raw_data()
        
        # Preprocess to static graph
        static_data = self._preprocess_static(raw_data)
        
        if not temporal:
            self._data = static_data
            return static_data
        
        # Augment to temporal
        temporal_data = self._augment_temporal(
            static_data, num_snapshots, inject_anomalies
        )
        self._data = temporal_data
        return temporal_data


class DGraphDataset(BaseDataset):
    """DGraph: Large-scale financial transaction graph (3.7M nodes)"""
    
    def _load_raw_data(self) -> Dict[str, np.ndarray]:
        # Use memory-mapped arrays for billion-edge scalability
        nodes_path = self.raw_dir / "dgraph_nodes.npy"
        edges_path = self.raw_dir / "dgraph_edges.npy"
        labels_path = self.raw_dir / "dgraph_labels.npy"
        timestamps_path = self.raw_dir / "dgraph_timestamps.npy"
        
        x = np.load(nodes_path, mmap_mode='r')
        edge_index = np.load(edges_path, mmap_mode='r')
        y = np.load(labels_path, mmap_mode='r')
        timestamps = np.load(timestamps_path, mmap_mode='r')
        
        return {
            'x': x,
            'edge_index': edge_index,
            'y': y,
            'timestamps': timestamps
        }
    
    def _preprocess_static(self, raw_data: Dict[str, np.ndarray]) -> Data:
        return Data(
            x=torch.from_numpy(raw_data['x']).float(),
            edge_index=torch.from_numpy(raw_data['edge_index']).long(),
            y=torch.from_numpy(raw_data['y']).float(),
            timestamps=torch.from_numpy(raw_data['timestamps']).long()
        )
    
    def _inject_causal_anomalies(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        ratio: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """DGraph-specific anomalies: income-spending mechanism violation"""
        num_edges = edge_index.shape[1]
        num_anomalies = int(ratio * num_edges)
        anomaly_indices = np.random.choice(num_edges, num_anomalies, replace=False)
        y = torch.zeros(num_edges)
        y[anomaly_indices] = 1.0
        
        for idx in anomaly_indices:
            src = edge_index[0, idx].item()
            # Violate: low income (feat[0]≈0) but high transaction value (feat[1]≈1)
            x[src, 0] = max(0.0, np.random.normal(0.1, 0.05))  # Low income
            x[src, 1] = min(1.0, np.random.normal(0.9, 0.05))  # High spending
        
        return edge_index, y


class WeiboDataset(BaseDataset):
    """Weibo: Social network with post timestamps"""
    
    def _load_raw_data(self) -> Dict[str, np.ndarray]:
        df = pd.read_csv(self.raw_dir / "weibo.csv")
        x = df.filter(regex='feat_').values
        edge_index = df[['src', 'dst']].values.T
        timestamps = df['timestamp'].values
        return {'x': x, 'edge_index': edge_jindex, 'timestamps': timestamps}
    
    def _preprocess_static(self, raw_data: Dict[str, np.ndarray]) -> Data:
        return Data(
            x=torch.from_numpy(raw_data['x']).float(),
            edge_index=torch.from_numpy(raw_data['edge_index']).long(),
            timestamps=torch.from_numpy(raw_data['timestamps']).long()
        )


class StaticDataset(BaseDataset):
    """
    Generic loader for static datasets (Facebook, Disney, Books, Flickr).
    Assumes standard PyG format with nodes.csv and edges.csv.
    """
    
    def _load_raw_data(self) -> Dict[str, np.ndarray]:
        nodes_df = pd.read_csv(self.raw_dir / f"{self.name.lower()}_nodes.csv")
        edges_df = pd.read_csv(self.raw_dir / f"{self.name.lower()}_edges.csv")
        
        x = nodes_df.filter(regex='feat_|attr_').values
        edge_index = edges_df[['src', 'dst']].values.T
        return {'x': x, 'edge_index': edge_index}
    
    def _preprocess_static(self, raw_data: Dict[str, np.ndarray]) -> Data:
        return Data(
            x=torch.from_numpy(raw_data['x']).float(),
            edge_index=torch.from_numpy(raw_data['edge_index']).long()
        )


# Factory function for dataset creation
def create_dataset(
    name: str,
    root: str = "data"
) -> BaseDataset:
    """
    Factory function to create dataset instances.
    
    Args:
        name: Dataset name (Weibo, Facebook, Disney, Books, Flickr, DGraph)
        root: Root directory containing 'raw' and 'processed' subdirs
        
    Returns:
        Instantiated dataset object.
        
    Raises:
        ValueError: If dataset name is not supported.
    """
    name_lower = name.lower()
    if name_lower == "dgraph":
        return DGraphDataset(root, name)
    elif name_lower == "weibo":
        return WeiboDataset(root, name)
    elif name_lower in {"facebook", "disney", "books", "flickr"}:
        return StaticDataset(root, name)
    else:
        raise ValueError(f"Unsupported dataset: {name}. "
                         f"Supported: Weibo, Facebook, Disney, Books, Flickr, DGraph")


# Convenience function for direct loading
def load_dataset(
    name: str,
    root: str = "data",
    temporal: bool = True,
    num_snapshots: int = 10,
    inject_anomalies: bool = True
) -> List[Data]:
    """
    Load and preprocess a dataset in one call.
    
    Args:
        name: Dataset name
        root: Data root directory
        temporal: Return temporal snapshots if True
        num_snapshots: Number of temporal snapshots
        inject_anomalies: Inject causal anomalies in last snapshot
        
    Returns:
        Preprocessed data (list of Data objects if temporal, else single Data)
    """
    dataset = create_dataset(name, root)
    return dataset.load(temporal, num_snapshots, inject_anomalies)


# Example usage
if __name__ == "__main__":
    # Load DGraph with temporal augmentation
    dgraph_data = load_dataset(
        "DGraph",
        root="data",
        temporal=True,
        num_snapshots=12,
        inject_anomalies=True
    )
    print(f"Loaded {len(dgraph_data)} temporal snapshots for DGraph")
    print(f"First snapshot: {dgraph_data[0]}")
    print(f"Last snapshot anomalies: {dgraph_data[-1].y.sum().item()} injected")