"""
GAT layer -- Velickovic et al., "Graph Attention Networks" (ICLR 2018).

The idea
--------
GCN gives every neighbour a weight fixed by the *graph structure* (degrees).
GAT lets the model *learn* the weight from the *features*:

.. math::

    e_{ij} = \\text{LeakyReLU}\\big(a^\\top [W h_i \\,\\|\\, W h_j]\\big)

    \\alpha_{ij} = \\frac{\\exp(e_{ij})}{\\sum_{k \\in N(i)} \\exp(e_{ik})}

    h_i' = \\sigma\\Big(\\sum_{j \\in N(i)} \\alpha_{ij} W h_j\\Big)

Four things to notice, in order of how often they trip people up:

1. **One shared projection ``W``, not two.** The paper transforms every node
   with the same ``W`` and *then* scores pairs. ``torch_geometric`` does the
   same when you pass a plain ``int`` for ``in_channels`` (it only splits into
   ``lin_src``/``lin_dst`` for the heterogeneous tuple case). Splitting them by
   accident doubles the parameters and silently changes the model.
2. **The softmax is per-node, not global.** Each node normalises over *its own*
   incoming edges. That is a group-by, so it is :func:`~gnn_lab.ops.scatter_softmax`
   -- not ``torch.softmax``.
3. **The score is a sum of two one-sided dot products.** ``a_s . Wh_j`` depends
   only on the sender and ``a_d . Wh_i`` only on the receiver, so the
   concatenated form ``a^T[Wh_i || Wh_j]`` factorises into a sum. You never
   form the quadratic term -- that would be O(E*d^2).
4. **Multi-head means parallel independent attentions**, concatenated (or, for
   the final layer, averaged). Each head can specialise, which is exactly what
   the attention visualisation in notebook 04 shows.

Why it is worth the extra complexity
------------------------------------
Learned weights make the layer robust to noisy or irrelevant edges. On Cora the
gain over GCN is modest (~1-2 points); on heterophilous graphs, where "neighbour
implies similar" is false, attention is what keeps the model usable.

Parameter names (``lin``, ``att_src``, ``att_dst``, ``bias``) match
``torch_geometric.nn.GATConv`` exactly, so weights copy across with a plain
``load_state_dict`` for the equivalence test.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import add_self_loops, scatter_add, scatter_softmax

__all__ = ["GATConvScratch", "GATScratch", "GATConvScratch_forward_with_attention"]


class GATConvScratch(nn.Module):
    """Multi-head graph attention layer.

    Parameters
    ----------
    heads : int
        Number of independent attention heads (paper uses 8).
    concat : bool
        ``True`` concatenates head outputs -> ``heads * out_channels``.
        ``False`` averages them -> ``out_channels``. Use ``False`` for the final
        layer, as in the paper. The bias size follows this choice, which is a
        detail that is easy to get backwards.
    negative_slope : float
        LeakyReLU slope for the raw scores (paper: 0.2).
    dropout : float
        Dropout applied **to the attention coefficients**, not to features.
        This regularises *which* neighbours the model listens to -- a different
        kind of noise injection from ordinary dropout.
    add_self_loops : bool
        A node must be able to attend to itself, otherwise it loses its own
        features entirely. Keep True.
    separate_lin : bool
        Use two projections instead of one shared ``W``. Left as a switch purely
        so you can measure the difference; default False matches both the paper
        and PyG.
    return_attention : bool
        If set, ``forward`` also returns the per-edge ``[E, heads]`` attention
        matrix. Being able to inspect these weights is the main practical
        advantage of writing the layer yourself.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True,
                 separate_lin: bool = False, return_attention: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops_flag = add_self_loops
        self.return_attention = return_attention

        hid = heads * out_channels

        if separate_lin:
            self.lin_src = nn.Linear(in_channels, hid, bias=False)
            self.lin_dst = nn.Linear(in_channels, hid, bias=False)
        else:
            # One shared W. `lin` is the only *registered* submodule, so the
            # state_dict exposes a single `lin.weight` -- exactly PyG's key.
            #
            # `lin_src` / `lin_dst` are plain attribute aliases to that same
            # object so `forward` reads identically in both modes. They are set
            # through `object.__setattr__` on purpose: `self.lin_src = self.lin`
            # would make nn.Module register the *same* Linear under three names,
            # and state_dict() lists every registered name, so we would get
            # `lin.weight`, `lin_src.weight` and `lin_dst.weight` -- three keys
            # for one tensor, which breaks the weight copy into PyG.
            self.lin = nn.Linear(in_channels, hid, bias=False)
            object.__setattr__(self, "lin_src", self.lin)
            object.__setattr__(self, "lin_dst", self.lin)

        # Per-head scoring vectors, shape [1, heads, out_channels].
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))

        # NOTE the direction of this choice: concat -> heads*out, else -> out.
        total_out = hid if concat else out_channels
        if bias:
            self.bias = nn.Parameter(torch.empty(total_out))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        seen = set()
        for lin in (self.lin_src, self.lin_dst):
            if id(lin) in seen:
                continue
            seen.add(id(lin))
            nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                num_nodes: int | None = None):
        num_nodes = num_nodes if num_nodes is not None else x.size(0)

        if self.add_self_loops_flag:
            edge_index = add_self_loops(edge_index, num_nodes)
        src, dst = edge_index[0], edge_index[1]

        H, C = self.heads, self.out_channels

        # ---- project, then view as [N, heads, out] ------------------------
        h_src = self.lin_src(x).view(num_nodes, H, C)
        h_dst = self.lin_dst(x).view(num_nodes, H, C) if self.lin_dst is not self.lin_src else h_src

        # ---- raw scores: two one-sided dot products, summed ---------------
        # h_src is gathered with `src` (the sender); h_dst with `dst` (receiver).
        score_src = (h_src.index_select(0, src) * self.att_src).sum(dim=-1)   # [E, H]
        score_dst = (h_dst.index_select(0, dst) * self.att_dst).sum(dim=-1)   # [E, H]
        alpha = F.leaky_relu(score_src + score_dst, self.negative_slope)

        # ---- softmax *within each receiving node's edge set* --------------
        alpha = scatter_softmax(alpha, dst, num_nodes, dim=0)                 # [E, H]
        attention = alpha

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # ---- MESSAGE: neighbour features scaled by attention --------------
        msg = h_src.index_select(0, src) * alpha.unsqueeze(-1)                # [E, H, C]

        # ---- AGGREGATE: plain sum; the weighting already happened ---------
        out = scatter_add(msg, dst, num_nodes)                                # [N, H, C]

        # ---- head combination --------------------------------------------
        out = out.view(num_nodes, H * C) if self.concat else out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if self.return_attention:
            return out, attention
        return out

    def extra_repr(self) -> str:
        return (f"in={self.in_channels}, out={self.out_channels}, heads={self.heads}, "
                f"concat={self.concat}, slope={self.negative_slope}")


def GATConvScratch_forward_with_attention(conv: GATConvScratch, x, edge_index):
    """Run a conv with ``return_attention`` temporarily enabled."""
    old, conv.return_attention = conv.return_attention, True
    try:
        return conv(x, edge_index)
    finally:
        conv.return_attention = old


class GATScratch(nn.Module):
    """Two-layer GAT exactly as in the paper: 8 heads hidden, 1 head output.

    The output layer averages its heads (``concat=False``) rather than
    concatenating them, because the hidden dimension has already been grown 8x
    and we need to land on ``num_classes``.
    """

    def __init__(self, num_features: int, hidden_channels: int, num_classes: int,
                 heads: int = 8, dropout: float = 0.6, num_layers: int = 2,
                 attention_dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        self.convs.append(GATConvScratch(
            num_features, hidden_channels, heads=heads, concat=True,
            dropout=attention_dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConvScratch(
                hidden_channels * heads, hidden_channels, heads=heads, concat=True,
                dropout=attention_dropout))
        self.convs.append(GATConvScratch(
            hidden_channels * heads if num_layers > 1 else num_features,
            num_classes, heads=1, concat=False, dropout=attention_dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        return x

    def attention_weights(self, x: torch.Tensor, edge_index: torch.LongTensor,
                          layer: int = 0) -> tuple[torch.LongTensor, torch.Tensor]:
        """Return ``(edge_index_with_self_loops, attention[E, heads])`` for one layer.

        Feeds all preceding layers first, so the attention you inspect is the
        one the trained model actually used rather than the attention of random
        weights.
        """
        if not 0 <= layer < len(self.convs):
            raise IndexError(f"layer {layer} out of range (model has {len(self.convs)})")
        with torch.no_grad():
            for i, conv in enumerate(self.convs):
                if i == layer:
                    _, att = GATConvScratch_forward_with_attention(conv, x, edge_index)
                    n = x.size(0)
                    ei = add_self_loops(edge_index, n) if conv.add_self_loops_flag else edge_index
                    return ei, att
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
        raise AssertionError("unreachable")
