"""
gnn_lab -- Graph Neural Networks written from scratch in plain PyTorch.

The point of this package is that there is no magic anywhere. Every layer is
built from ``index_select`` (gather from the sender) and ``index_add_``
(scatter onto the receiver). If you can read those two operations you can read
every model here.

Quick start
-----------
>>> from gnn_lab import load_cora, GCNScratch, train_node, set_seed
>>> set_seed(0)
>>> data = load_cora()                     # downloads ~1 MB once
>>> model = GCNScratch(data.num_features, 16, data.num_classes)
>>> result = train_node(model, data, model_name="GCN", epochs=200)
>>> print(result.summary_line())

Check that the from-scratch code is actually correct
----------------------------------------------------
>>> from gnn_lab import equivalence
>>> equivalence.run_all()

That compares every layer against ``torch_geometric`` numerically, with weights
copied across verbatim, in float64. See :mod:`gnn_lab.equivalence`.

Layout
------
``ops``          scatter_add / mean / max / softmax, self-loops, gcn_norm
``layers``       GCNConvScratch, SAGEConvScratch, GATConvScratch, GINConvScratch, pooling
``data``         Cora (Planetoid + LINQS), MUTAG/PROTEINS (TU format), synthetic
``train``        node-classification and graph-classification trainers
``metrics``      accuracy, macro-F1, confusion matrix, over-smoothing diagnostics
``equivalence``  numerical proof that the hand-written layers match PyG
``viz``          plots: graphs, attention, t-SNE, over-smoothing curves
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import equivalence, metrics, ops, train, utils, viz
from .data import (GraphDataset, NodeDataset, karate_club, load_cora, load_mutag,
                   load_proteins, path_graph, toy_graph, triangle_square_dataset,
                   two_communities)
from .layers import (GATConvScratch, GATScratch, GCNConvScratch, GCNScratch,
                     GINConvScratch, GINScratch, SAGEConvScratch, GraphSAGEScratch,
                     collate_graphs, global_pool)
from .metrics import (accuracy, dirichlet_energy, macro_f1, majority_baseline,
                      mean_pairwise_cosine)
from .train import evaluate_node, train_graph_classification, train_node, train_node_multi_seed
from .utils import Timer, count_parameters, get_device, set_seed

NODE_MODELS = {"gcn": GCNScratch, "sage": GraphSAGEScratch, "gat": GATScratch}

__all__ = [
    "__version__",
    # submodules
    "ops", "layers", "data", "train", "metrics", "equivalence", "utils", "viz",
    # layers
    "GCNConvScratch", "GCNScratch",
    "SAGEConvScratch", "GraphSAGEScratch",
    "GATConvScratch", "GATScratch",
    "GINConvScratch", "GINScratch",
    "global_pool", "collate_graphs",
    # data
    "NodeDataset", "GraphDataset", "load_cora", "load_mutag", "load_proteins",
    "karate_club", "path_graph", "toy_graph", "two_communities", "triangle_square_dataset",
    # training + metrics
    "train_node", "train_node_multi_seed", "evaluate_node", "train_graph_classification",
    "accuracy", "macro_f1", "majority_baseline", "mean_pairwise_cosine", "dirichlet_energy",
    # utils
    "set_seed", "get_device", "count_parameters", "Timer",
    "NODE_MODELS",
]


def self_test(root: str = "data", quick: bool = True) -> bool:
    """Run the from-scratch-vs-PyG equivalence checks. Returns True if none failed.

    ``quick=True`` skips the dataset comparison, which is the only check that
    needs to download Cora.
    """
    results = equivalence.run_all(root=root, include_data=not quick)
    return equivalence.all_passed(results)
