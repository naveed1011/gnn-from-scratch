"""
GIN layer -- Xu, Hu, Leskovec & Jegelka, "How Powerful are Graph Neural
Networks?" (ICLR 2019).

Why GIN is in this repo even though it is the "fourth" model
------------------------------------------------------------
GCN and GraphSAGE-mean both use an aggregation that is a *weighted average*.
The GIN paper proves something uncomfortable about that: mean and max
aggregation cannot distinguish certain non-isomorphic graphs. Concretely, they
cannot tell apart a node with one neighbour of feature 1 from a node with two
neighbours of feature 0.5 -- same mean, different structure.

GIN's answer is to use the **sum** and an MLP instead of a single linear layer:

.. math::

    h_i' = \\text{MLP}\\Big((1 + \\epsilon)\\, h_i + \\sum_{j \\in N(i)} h_j\\Big)

Sum-aggregation is provably as expressive as the Weisfeiler-Lehman graph test,
which is the standard upper bound for message-passing GNNs. The ``(1+eps)``
term is a learnable interpolation between "count my neighbours" (eps=0) and
"ignore my neighbours" (eps=-1).

This matters practically, not just theoretically: for **graph-level** tasks
(where you compare whole molecules to each other) GIN is usually clearly
stronger than GCN, while for node classification on Cora the difference is
small. That split is exactly what notebooks 02 and 05 measure.

Parameter naming matches ``torch_geometric.nn.GINConv`` (``nn``, ``eps``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import scatter_add
from .pooling import global_pool

__all__ = ["GINConvScratch", "GINScratch"]


class GINConvScratch(nn.Module):
    """One GIN layer: sum-aggregate the neighbourhood, then apply an MLP.

    Parameters
    ----------
    nn : nn.Module
        The MLP applied after aggregation. Must map ``in_channels -> out_channels``.
        GIN is unusually sensitive to this: a *single linear layer* here is
        strictly weaker than an MLP with one hidden layer and ReLU.
    eps : float
        Initial value of the ``(1 + eps)`` self-term coefficient.
    train_eps : bool
        Learn ``eps`` (paper's recommendation) or freeze it at the initial value.

    Notes
    -----
    No self-loops are added. The ``(1+eps) h_i`` term *is* the self-loop, written
    explicitly so it gets its own coefficient instead of being averaged in with
    the neighbours.
    """

    def __init__(self, nn_module: nn.Module, eps: float = 0.0, train_eps: bool = True):
        super().__init__()
        # named `nn` to match PyG's GINConv state_dict key exactly
        self.nn = nn_module
        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer("eps", torch.tensor([eps]))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                num_nodes: int | None = None) -> torch.Tensor:
        num_nodes = num_nodes if num_nodes is not None else x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # MESSAGE + AGGREGATE: sum of neighbour features (no scaling at all).
        msg = x.index_select(0, src)
        agg = scatter_add(msg, dst, num_nodes)

        # UPDATE: (1 + eps) * self + sum(neighbours), then the MLP.
        out = (1.0 + self.eps) * x + agg
        return self.nn(out)

    def extra_repr(self) -> str:
        return f"eps={float(self.eps):.4f}, train_eps={isinstance(self.eps, nn.Parameter)}"


def _mlp(in_c: int, hid_c: int, out_c: int, norm: bool = True) -> nn.Module:
    """The standard 2-layer GIN MLP, optionally with BatchNorm.

    BatchNorm inside the MLP is not cosmetic -- on graph-level tasks it does most
    of the work of stabilising the sum-aggregation, whose scale grows with degree.
    """
    layers: list[nn.Module] = [nn.Linear(in_c, hid_c)]
    if norm:
        layers.append(nn.BatchNorm1d(hid_c))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(hid_c, out_c))
    return nn.Sequential(*layers)


class GINScratch(nn.Module):
    """Stacked GIN for **graph-level** classification.

    Unlike the node-classification models, this one takes a ``batch`` vector so
    it can process many small graphs in a single forward pass. See
    :mod:`gnn_lab.layers.pooling` for how that works.
    """

    def __init__(self, num_features: int, hidden_channels: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.5, train_eps: bool = True,
                 use_batchnorm: bool = True, pool: str = "sum"):
        super().__init__()
        self.dropout = dropout
        self.pool = pool

        self.convs = nn.ModuleList()
        self.convs.append(GINConvScratch(
            _mlp(num_features, hidden_channels, hidden_channels, norm=use_batchnorm),
            train_eps=train_eps))
        for _ in range(num_layers - 1):
            self.convs.append(GINConvScratch(
                _mlp(hidden_channels, hidden_channels, hidden_channels, norm=use_batchnorm),
                train_eps=train_eps))

        # JK-style readout: concatenate the embeddings of *every* layer, then an MLP.
        # This is the "jumping knowledge" trick from the GIN paper and it matters
        # a lot on small datasets -- it lets the classifier pick whichever
        # neighbourhood radius is most informative.
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels * num_layers, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                batch: torch.LongTensor) -> torch.Tensor:
        hs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hs.append(global_pool(x, batch, method=self.pool))

        return self.readout(torch.cat(hs, dim=-1))
