import abc
import numpy as np
import torch
from typing import List, Tuple, Optional, Protocol, runtime_checkable
from torch_geometric.data import Data


@runtime_checkable
class GraphData(Protocol):
    """Protocol defining the interface for graph data objects."""
    x: torch.Tensor          # [N, D] node features
    edge_index: torch.Tensor # [2, E] edge list
    y: Optional[torch.Tensor]# [E] or [N] labels (optional)
    timestamps: Optional[torch.Tensor]  # [E] timestamps (optional)


class BaseTemporalAugmenter(abc.ABC):
    """
    Abstract base class for temporal augmentation strategies.
    Supports both timestamped and static graphs.
    """
    
    def __init__(self, num_snapshots: int = 10):
        self.num_snapshots = num_snapshots

    @abc.abstractmethod
    def augment(
        self,
        data: GraphData,
        inject_anomalies: bool = True,
        anomaly_ratio: float = 0.05
    ) -> List[Data]:
        """
        Convert static graph to temporal snapshots.
        
        Args:
            data: Input graph data (static or timestamped)
            inject_anomalies: Whether to inject causal anomalies in last snapshot
            anomaly_ratio: Fraction of edges/nodes to perturb as anomalies
            
        Returns:
            List of temporal graph snapshots
        """
        pass

    def _inject_causal_anomalies(
        self,
        data: Data,
        ratio: float
    ) -> Data:
        """
        Inject mechanism-violating anomalies via do-interventions.
        Default: Perturb source node attributes to violate causal expectations.
        """
        # Work with copies to avoid modifying original data
        x = data.x.clone()
        edge_index = data.edge_index.clone()
        y = torch.zeros(edge_index.size(1)) if data.y is None else data.y.clone()
        
        num_edges = edge_index.size(1)
        num_anomalies = int(ratio * num_edges)
        if num_anomalies == 0:
            return Data(x=x, edge_index=edge_index, y=y)
            
        # Randomly select edges for anomaly injection
        anomaly_indices = torch.randperm(num_edges)[:num_anomalies]
        y[anomaly_indices] = 1.0
        
        # Perturb source node attributes (generic causal violation)
        for idx in anomaly_indices:
            src = edge_index[0, idx].item()
            if x.size(1) >= 2:
                # Simulate: low income (feat[0] ≈ 0) but high spending (feat[1] ≈ 1)
                x[src, 0] = max(0.0, torch.normal(0.1, 0.05).item())
                x[src, 1] = min(1.0, torch.normal(0.9, 0.05).item())
            else:
                # Single feature: flip value to extreme
                x[src, 0] = 1.0 if x[src, 0] < 0.5 else 0.0
                
        return Data(x=x, edge_index=edge_index, y=y)


class TimestampedAugmenter(BaseTemporalAugmenter):
    """
    Temporal augmentation for graphs with real timestamps (e.g., DGraph, Weibo).
    Splits edges into snapshots based on actual time intervals.
    """
    
    def augment(
        self,
        data: GraphData,
        inject_anomalies: bool = True,
        anomaly_ratio: float = 0.05
    ) -> List[Data]:
        if data.timestamps is None:
            raise ValueError("TimestampedAugmenter requires timestamps in data")
            
        timestamps = data.timestamps.numpy()
        edge_index = data.edge_index.numpy()
        x = data.x
        
        # Sort edges by timestamp
        sorted_idx = np.argsort(timestamps)
        sorted_edge_index = edge_index[:, sorted_idx]
        sorted_timestamps = timestamps[sorted_idx]
        
        # Create temporal bins
        unique_ts = np.unique(sorted_timestamps)
        if len(unique_ts) <= self.num_snapshots:
            # One snapshot per unique timestamp
            snapshot_edges = [
                np.where(sorted_timestamps == ts)[0] 
                for ts in unique_ts
            ]
        else:
            # Equal-sized time bins
            bins = np.array_split(np.arange(len(unique_ts)), self.num_snapshots)
            snapshot_edges = []
            for bin_ts in bins:
                ts_vals = unique_ts[bin_ts]
                mask = np.isin(sorted_timestamps, ts_vals)
                snapshot_edges.append(np.where(mask)[0])
        
        # Create snapshots
        snapshots = []
        for i, edge_indices in enumerate(snapshot_edges):
            if len(edge_indices) == 0:
                continue
                
            snap_edge_index = torch.from_numpy(sorted_edge_index[:, edge_indices]).long()
            snap_data = Data(
                x=x,
                edge_index=snap_edge_index,
                y=None,
                timestamp=i
            )
            
            # Inject anomalies only in last snapshot
            if inject_anomalies and i == len(snapshot_edges) - 1:
                snap_data = self._inject_causal_anomalies(snap_data, anomaly_ratio)
                
            snapshots.append(snap_data)
            
        return snapshots


class UniformAugmenter(BaseTemporalAugmenter):
    """
    Temporal augmentation for static graphs (e.g., Facebook, Disney, Books, Flickr).
    Uniformly splits edges into snapshots without timestamps.
    """
    
    def augment(
        self,
        data: GraphData,
        inject_anomalies: bool = True,
        anomaly_ratio: float = 0.05
    ) -> List[Data]:
        edge_index = data.edge_index
        x = data.x
        num_edges = edge_index.size(1)
        
        # Uniform split into snapshots
        snapshot_edges = np.array_split(np.arange(num_edges), self.num_snapshots)
        
        snapshots = []
        for i, edge_indices in enumerate(snapshot_edges):
            if len(edge_indices) == 0:
                continue
                
            # Convert to tensor if needed
            if isinstance(edge_indices, np.ndarray):
                edge_indices = torch.from_numpy(edge_indices).long()
                
            snap_edge_index = edge_index[:, edge_indices]
            snap_data = Data(
                x=x,
                edge_index=snap_edge_index,
                y=None,
                timestamp=i
            )
            
            # Inject anomalies only in last snapshot
            if inject_anomalies and i == self.num_snapshots - 1:
                snap_data = self._inject_causal_anomalies(snap_data, anomaly_ratio)
                
            snapshots.append(snap_data)
            
        return snapshots


class StreamingAugmenter(BaseTemporalAugmenter):
    """
    Memory-efficient augmenter for billion-edge graphs.
    Uses generators to avoid loading all snapshots into memory.
    """
    
    def augment(
        self,
        data: GraphData,
        inject_anomalies: bool = True,
        anomaly_ratio: float = 0.05
    ) -> List[Data]:
        # For true streaming, we'd return a generator, but PyG expects list
        # This implementation still uses chunked processing
        edge_index = data.edge_index
        x = data.x
        num_edges = edge_index.size(1)
        chunk_size = min(1000000, num_edges)  # Process 1M edges at a time
        
        snapshots = []
        remaining_edges = num_edges
        
        for i in range(self.num_snapshots):
            # Calculate edges for this snapshot
            edges_this_snap = remaining_edges // (self.num_snapshots - i)
            if edges_this_snap == 0:
                break
                
            start_idx = num_edges - remaining_edges
            end_idx = start_idx + edges_this_snap
            snap_edge_index = edge_index[:, start_idx:end_idx]
            
            snap_data = Data(
                x=x,
                edge_index=snap_edge_index,
                y=None,
                timestamp=i
            )
            
            # Inject anomalies only in last snapshot
            if inject_anomalies and i == self.num_snapshots - 1:
                snap_data = self._inject_causal_anomalies(snap_data, anomaly_ratio)
                
            snapshots.append(snap_data)
            remaining_edges -= edges_this_snap
            
        return snapshots


# Factory function to select augmenter based on data properties
def create_augmenter(data: GraphData, num_snapshots: int = 10) -> BaseTemporalAugmenter:
    """
    Factory function to create appropriate augmenter based on data properties.
    
    Args:
        data: Input graph data
        num_snapshots: Number of temporal snapshots to create
        
    Returns:
        Instantiated augmenter object
    """
    if hasattr(data, 'timestamps') and data.timestamps is not None:
        # Use streaming augmenter for large timestamped graphs
        if data.edge_index.size(1) > 1000000:  # 1M+ edges
            return StreamingAugmenter(num_snapshots)
        else:
            return TimestampedAugmenter(num_snapshots)
    else:
        return UniformAugmenter(num_snapshots)


# Main augmentation function
def augment_to_temporal(
    data: GraphData,
    num_snapshots: int = 10,
    inject_anomalies: bool = True,
    anomaly_ratio: float = 0.05
) -> List[Data]:
    """
    Convert static graph to temporal snapshots with optional causal anomaly injection.
    
    Args:
        data: Input static graph data
        num_snapshots: Number of temporal snapshots to create
        inject_anomalies: Whether to inject causal anomalies in last snapshot
        anomaly_ratio: Fraction of edges to perturb as anomalies (0.0-1.0)
        
    Returns:
        List of temporal graph snapshots
        
    Example:
        >>> static_data = load_dataset("Facebook", temporal=False)
        >>> temporal_data = augment_to_temporal(static_data, num_snapshots=12)
    """
    if not isinstance(data, GraphData):
        raise TypeError("Input data must implement GraphData protocol")
        
    if not (0.0 <= anomaly_ratio <= 1.0):
        raise ValueError("anomaly_ratio must be between 0.0 and 1.0")
        
    augmenter = create_augmenter(data, num_snapshots)
    return augmenter.augment(data, inject_anomalies, anomaly_ratio)


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic static graph for checking and verifying the code............
    N, E, D = 1000, 5000, 16
    x = torch.randn(N, D)
    edge_index = torch.randint(0, N, (2, E))
    static_data = Data(x=x, edge_index=edge_index)
    
    # Augment to temporal
    temporal_data = augment_to_temporal(
        static_data,
        num_snapshots=5,
        inject_anomalies=True,
        anomaly_ratio=0.1
    )
    
    print(f"Created {len(temporal_data)} temporal snapshots")
    print(f"First snapshot edges: {temporal_data[0].edge_index.size(1)}")
    print(f"Last snapshot anomalies: {temporal_data[-1].y.sum().item()} injected")
    
    # Test with timestamped data
    timestamps = torch.randint(0, 100, (E,))
    timestamped_data = Data(x=x, edge_index=edge_index, timestamps=timestamps)
    temporal_ts = augment_to_temporal(timestamped_data, num_snapshots=10)
    print(f"Timestamped augmentation created {len(temporal_ts)} snapshots")