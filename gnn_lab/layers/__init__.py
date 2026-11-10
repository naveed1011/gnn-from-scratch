"""Hand-written GNN layers. Every one of them is plain PyTorch: no PyG, no DGL."""

from .gat import GATConvScratch, GATScratch
from .gcn import GCNConvScratch, GCNScratch
from .gin import GINConvScratch, GINScratch
from .pooling import BatchedGraph, GlobalPooling, collate_graphs, global_pool, graph_batches
from .sage import AGGREGATORS, GraphSAGEScratch, SAGEConvScratch

__all__ = [
    "GCNConvScratch", "GCNScratch",
    "SAGEConvScratch", "GraphSAGEScratch", "AGGREGATORS",
    "GATConvScratch", "GATScratch",
    "GINConvScratch", "GINScratch",
    "global_pool", "GlobalPooling", "collate_graphs", "graph_batches", "BatchedGraph",
]

# A single registry so scripts and notebooks can pick a model by name.
NODE_MODELS = {
    "gcn": GCNScratch,
    "sage": GraphSAGEScratch,
    "gat": GATScratch,
}
