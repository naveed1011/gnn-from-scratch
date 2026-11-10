"""
GraphSAGE layer -- Hamilton, Ying & Leskovec, "Inductive Representation
Learning on Large Graphs" (NeurIPS 2017).

What SAGE teaches you that GCN does not
---------------------------------------
GCN folds the node's own features into the average via a self-loop and one
shared weight matrix. SAGE **separates the two roles**:

.. math::

    h_i' = W_{\\text{self}}\\, h_i \\;+\\; W_{\\text{neigh}}\\,
           \\text{AGG}\\big(\\{h_j : j \\in N(i)\\}\\big)

Two independent projections, and an explicit choice of aggregator. That small
change is the source of both of SAGE's big ideas:

* **Inductive use.** Because aggregation is over whatever neighbours happen to
  be present, a node unseen at training time can still be embedded by running
  the same forward pass on its own neighbourhood. GCN's fixed
  ``D^-1/2(A+I)D^-1/2`` is tied to one specific graph.
* **Sampling.** Neighbour *sets* can be subsampled, so cost per node is bounded
  and no longer scales with degree -- which is what makes billion-edge graphs
  trainable.

The aggregator is a hyper-parameter here, which makes SAGE the best place to
feel the message-passing abstraction: same layer, swap ``mean`` for ``max``,
watch what happens.

Parameter naming matches ``torch_geometric.nn.SAGEConv`` (``lin_l`` for the
neighbour branch, ``lin_r`` for the root/self branch) so weights copy across
directly for the equivalence test.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import scatter_add, scatter_max, scatter_mean

__all__ = ["SAGEConvScratch", "GraphSAGEScratch", "AGGREGATORS"]

AGGREGATORS = ("mean", "max", "sum")


def _aggregate(msg: torch.Tensor, dst: torch.LongTensor, num_nodes: int, how: str) -> torch.Tensor:
    """The single line that differs between SAGE variants."""
    if how == "mean":
        return scatter_mean(msg, dst, num_nodes)
    if how == "sum":
        return scatter_add(msg, dst, num_nodes)
    if how == "max":
        return scatter_max(msg, dst, num_nodes)
    raise ValueError(f"unknown aggregator {how!r}; choose from {AGGREGATORS}")


class SAGEConvScratch(nn.Module):
    """One GraphSAGE layer.

    Parameters
    ----------
    aggr : {"mean", "max", "sum"}
        Neighbourhood combiner. ``mean`` is the paper's default and the
        strongest baseline; ``max`` sharpens on distinctive neighbours;
        ``sum`` is degree-sensitive (a hub neighbour dominates).
    root_weight : bool
        Include the ``W_self h_i`` term. Setting it False gives a *pure*
        neighbourhood model -- useful in an ablation to show how much of the
        accuracy is coming from a node's own features vs its context.
    normalize : bool
        L2-normalise the output. PyG's default is False; the original SAGE paper
        normalises between layers. Left as a switch so you can test both.

    Notes
    -----
    No self-loops are added here on purpose. ``lin_r`` already handles the node
    itself, and adding a self-loop *and* a root weight would double-count it --
    a subtle bug that silently changes what "mean" means.
    """

    def __init__(self, in_channels: int, out_channels: int, aggr: str = "mean",
                 root_weight: bool = True, normalize: bool = False, bias: bool = True):
        super().__init__()
        if aggr not in AGGREGATORS:
            raise ValueError(f"aggr must be one of {AGGREGATORS}, got {aggr!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.root_weight = root_weight
        self.normalize = normalize

        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)   # neighbour branch
        if root_weight:
            self.lin_r = nn.Linear(in_channels, out_channels, bias=False)  # self branch
        else:
            self.lin_r = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin_l.reset_parameters()
        if self.lin_r is not None:
            self.lin_r.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                num_nodes: int | None = None) -> torch.Tensor:
        num_nodes = num_nodes if num_nodes is not None else x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # MESSAGE: SAGE's message is just the neighbour's own features.
        # All the modelling happens in the aggregator and the two projections.
        msg = x.index_select(0, src)

        # AGGREGATE over the receiver's neighbourhood.
        agg = _aggregate(msg, dst, num_nodes, self.aggr)

        # UPDATE: two independent linear branches, summed.
        out = self.lin_l(agg)
        if self.root_weight:
            out = out + self.lin_r(x)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out

    def extra_repr(self) -> str:
        return (f"in={self.in_channels}, out={self.out_channels}, aggr={self.aggr}, "
                f"root_weight={self.root_weight}, normalize={self.normalize}")


class GraphSAGEScratch(nn.Module):
    """Stacked GraphSAGE classifier with ReLU + dropout between layers."""

    def __init__(self, num_features: int, hidden_channels: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5, aggr: str = "mean",
                 normalize: bool = False, root_weight: bool = True):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        dims = [num_features] + [hidden_channels] * (num_layers - 1) + [num_classes]
        for i in range(num_layers):
            self.convs.append(SAGEConvScratch(
                dims[i], dims[i + 1], aggr=aggr,
                root_weight=root_weight, normalize=normalize))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def embeddings(self, x, edge_index, upto: int | None = None) -> torch.Tensor:
        n = len(self.convs) if upto is None else upto
        for i, conv in enumerate(self.convs[:n]):
            x = conv(x, edge_index)
            if i < n - 1:
                x = F.relu(x)
        return x
