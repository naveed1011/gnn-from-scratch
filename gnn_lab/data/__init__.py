"""Datasets: Cora (Planetoid + LINQS), MUTAG/PROTEINS (TU format), and small synthetic graphs."""

from .cora import (CORA_CLASS_NAMES, NodeDataset, load_cora, load_cora_linqs,
                   load_cora_planetoid)
from .synthetic import (KARATE_CLUB_LABELS, KARATE_EDGES, karate_club, path_graph,
                        toy_graph, triangle_square_dataset, triangle_vs_square,
                        two_communities)
from .tu import TU_SPECS, GraphDataset, load_mutag, load_proteins, load_tu

__all__ = [
    "NodeDataset", "GraphDataset",
    "load_cora", "load_cora_planetoid", "load_cora_linqs", "CORA_CLASS_NAMES",
    "load_tu", "load_mutag", "load_proteins", "TU_SPECS",
    "karate_club", "path_graph", "toy_graph", "two_communities",
    "triangle_vs_square", "triangle_square_dataset",
    "KARATE_EDGES", "KARATE_CLUB_LABELS",
]
