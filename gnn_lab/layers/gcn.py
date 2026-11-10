"""
GCN layer -- Kipf & Welling, "Semi-Supervised Classification with Graph
Convolutional Networks" (ICLR 2017).

The layer in one line of math
-----------------------------
.. math::

    X' = \\hat{D}^{-1/2}(A + I)\\hat{D}^{-1/2}\\, X\\, W

Read it right-to-left and it is exactly three steps:

1. ``X W``            -- a linear map, identical to a dense layer.
2. ``(A + I)``        -- add self-loops so a node keeps its own signal.
3. ``D^-1/2 ... D^-1/2`` -- weighted *mean* over the neighbourhood.

Step 3 is the only part that is graph-shaped, and it is a
:func:`~gnn_lab.ops.scatter_add`. That is the whole trick.

Index convention (this is where hand-written GNNs break)
--------------------------------------------------------
``edge_index[0]`` = **src**, the neighbour *sending* the message.
``edge_index[1]`` = **dst**, the node *receiving* it.

So a message pass is: gather from ``src``, scatter into ``dst``.

    msg[e] = x[ src[e] ] * norm[e]        # gather from the sender
    out[ dst[e] ] += msg[e]               # accumulate on the receiver

Swap those two and the code still runs, still trains, and still gives you a
plausible-looking accuracy -- just a wrong one. Keep ``src``/``dst`` named
explicitly rather than ``row``/``col``.

Parameter naming
----------------
The names ``lin`` and ``bias`` deliberately match
``torch_geometric.nn.GCNConv`` so that ``load_state_dict`` can copy weights
straight across. That is what makes the numerical equivalence test in
``gnn_lab/equivalence.py`` a one-liner instead of a fragile mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import add_self_loops, gcn_norm, scatter_add

__all__ = ["GCNConvScratch", "GCNScratch"]


class GCNConvScratch(nn.Module):
    """One graph-convolution layer, written with nothing but ``index_add_``.

    Parameters
    ----------
    in_channels, out_channels : int
        Feature dimensions.
    bias : bool
        Add a learned per-output-channel bias after propagation (default True).
    add_self_loops : bool
        Append ``(i, i)`` edges (default True). Turn this off in an ablation to
        see how much of GCN's performance is just "keep my own features".
    normalize : bool
        Apply the symmetric ``D^-1/2`` scaling (default True). Turning it off
        gives raw *sum* aggregation, which usually diverges -- a good demo of
        why the normalisation exists.
    cached_norm : bool
        Cache ``(edge_index, norm)`` for a fixed graph. Cora is trained
        full-batch on one graph, so recomputing degrees every epoch is pure
        waste. Must be left False if the graph changes between calls.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
                 add_self_loops: bool = True, normalize: bool = True,
                 cached_norm: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops_flag = add_self_loops
        self.normalize = normalize
        self.cached_norm = cached_norm
        self._cache = None

        # bias=False here; the bias is added after propagation (matches PyG)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()
        self._cache = None
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def clear_cache(self) -> None:
        self._cache = None

    def _prepare(self, edge_index: torch.LongTensor, num_nodes: int, x: torch.Tensor):
        """Build (edge_index_with_loops, per_edge_coefficient), optionally cached."""
        if self.cached_norm and self._cache is not None:
            return self._cache

        if self.normalize:
            ei, norm = gcn_norm(edge_index, num_nodes, add_loops=self.add_self_loops_flag,
                                dtype=x.dtype)
        else:
            ei = add_self_loops(edge_index, num_nodes) if self.add_self_loops_flag else edge_index
            norm = torch.ones(ei.size(1), device=x.device, dtype=x.dtype)

        norm = norm.to(x.dtype)
        if self.cached_norm:
            self._cache = (ei, norm)
        return ei, norm

    # -- the three-step message pass -------------------------------------- #
    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                num_nodes: int | None = None) -> torch.Tensor:
        num_nodes = num_nodes if num_nodes is not None else x.size(0)

        ei, norm = self._prepare(edge_index, num_nodes, x)
        src, dst = ei[0], ei[1]

        # ---- step 1: linear map (identical to a dense layer) -------------
        x = self.lin(x)

        # ---- step 2: MESSAGE -- gather from the sender, scale -------------
        msg = x.index_select(0, src) * norm.unsqueeze(-1)

        # ---- step 3: AGGREGATE -- unsorted group-by onto the receiver ----
        out = scatter_add(msg, dst, num_nodes)

        # ---- UPDATE: add the bias ---------------------------------------
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"normalize={self.normalize}, self_loops={self.add_self_loops_flag}")


class GCNScratch(nn.Module):
    """A complete GCN classifier: stack of :class:`GCNConvScratch`.

    This is the model Kipf & Welling used to get 81.5% on Cora, and two layers
    is still the right answer -- see the over-smoothing notebook for why more
    layers actively hurt.
    """

    def __init__(self, num_features: int, hidden_channels: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5, **conv_kwargs):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        dims = [num_features] + [hidden_channels] * (num_layers - 1) + [num_classes]
        for i in range(num_layers):
            self.convs.append(GCNConvScratch(dims[i], dims[i + 1], **conv_kwargs))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:      # no activation on the output layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def embeddings(self, x, edge_index, upto: int | None = None) -> torch.Tensor:
        """Return intermediate node embeddings -- used for t-SNE and over-smoothing plots.

        ``upto=k`` stops after layer ``k`` (1-indexed), so you can compare the
        geometry of layer-1 vs layer-8 representations of the same trained net.
        """
        n = len(self.convs) if upto is None else upto
        for i, conv in enumerate(self.convs[:n]):
            x = conv(x, edge_index)
            if i < n - 1:
                x = F.relu(x)
        return x
