"""
Scatter primitives -- the actual mechanics underneath every GNN.
=================================================================

A graph convolution is *not* a matrix multiply over a dense adjacency matrix.
It is an **unsorted group-by**: for every target node, gather the messages
coming from its neighbours and combine them.

PyTorch spells that as ``index_add_`` / ``index_reduce_``. PyTorch Geometric
spells it as ``scatter()``. They are the same operation. This module writes it
once, from scratch, and everything else in ``gnn_lab`` builds on it.

Why this matters for learning
-----------------------------
Once you see that ``GCNConv``, ``SAGEConv``, ``GATConv`` and ``GINConv`` are
all the same three lines --

    1. build a message per edge              ``msg[e] = f(x_src[e], x_dst[e])``
    2. group messages by target node         ``out  = scatter(msg, dst, op)``
    3. update the node                       ``out  = update(out, x)``

-- then "GNN architectures" stop being a zoo of papers and become a choice of
``f``, a choice of ``op``, and a choice of ``update``.

Notation used throughout
------------------------
``edge_index`` is a ``LongTensor`` of shape ``[2, num_edges]``:

* ``edge_index[0]`` = **source** (the neighbour sending the message, often ``j``)
* ``edge_index[1]`` = **target** (the node receiving it, often ``i``)

This is the COO / PyG convention. A directed edge ``(s -> t)`` is one column.
An undirected graph is stored as both directions.
"""

from __future__ import annotations

import torch

__all__ = [
    "scatter_add",
    "scatter_mean",
    "scatter_max",
    "scatter_sum",
    "scatter_softmax",
    "add_self_loops",
    "remove_self_loops",
    "to_undirected",
    "degree",
    "gcn_norm",
    "edge_order",
    "dense_to_edge_index",
]


# --------------------------------------------------------------------------- #
# 1. The group-by primitives
# --------------------------------------------------------------------------- #
def scatter_add(src: torch.Tensor, index: torch.LongTensor, dim_size: int, dim: int = 0) -> torch.Tensor:
    """``out[k] = sum of src[i] for all i where index[i] == k``.

    This single function is the entire engine of message passing. Every GNN
    aggregation -- sum, mean, max, attention-weighted -- is a variation on it.

    Parameters
    ----------
    src : Tensor
        One value per *edge*, shape ``[num_edges, ...]``.
    index : LongTensor
        Target group (node) id per edge, shape ``[num_edges]``.
    dim_size : int
        Number of groups = number of nodes. Needed because isolated nodes would
        otherwise silently shrink the output.

    Notes
    -----
    ``index_add_`` is atomic on CUDA, so this is safe under multi-threading and
    runs as a single fused kernel -- it is *not* a Python loop.
    """
    index = index.long()
    shape = list(src.shape)
    shape[dim] = dim_size
    out = src.new_zeros(shape)
    # No special case for an empty index: index_add_ with zero rows is a valid
    # no-op, and unlike returning a bare ``new_zeros`` it keeps ``out`` attached
    # to ``src`` in the autograd graph, so ``.backward()`` still works on a graph
    # that happens to have no edges.
    return out.index_add_(dim, index, src)


def scatter_sum(src: torch.Tensor, index: torch.LongTensor, dim_size: int, dim: int = 0) -> torch.Tensor:
    """Alias of :func:`scatter_add` (sum is the canonical name in the GIN paper)."""
    return scatter_add(src, index, dim_size, dim)


def scatter_mean(src: torch.Tensor, index: torch.LongTensor, dim_size: int, dim: int = 0) -> torch.Tensor:
    """Mean over each group. Isolated nodes get ``0`` (not ``nan``).

    Implemented as ``scatter_add(x) / count``. The count is clamped at 1 so a
    node with no incoming edges contributes a zero vector rather than NaN --
    NaNs here are the single most common source of "loss went to nan" bugs in
    hand-written GNNs.
    """
    summed = scatter_add(src, index, dim_size, dim)

    # `ones` has the same trailing shape as `src` but with every feature dim
    # collapsed to 1, so `count` broadcasts against `summed` for free.
    ones = src.new_ones((index.numel(),) + (1,) * max(src.dim() - 1, 0))
    count = scatter_add(ones, index.long(), dim_size, dim).clamp_(min=1.0)

    return summed / count


def scatter_max(src: torch.Tensor, index: torch.LongTensor, dim_size: int, dim: int = 0) -> torch.Tensor:
    """Max over each group. Isolated nodes get ``0``.

    Uses ``index_reduce_(..., 'amax')`` which is the fused CUDA group-max.
    """
    index = index.long()
    shape = list(src.shape)
    shape[dim] = dim_size

    if index.numel() == 0:
        return src.new_zeros(shape)

    out = src.new_full(shape, float("-inf"))
    out = out.index_reduce_(dim, index, src, "amax", include_self=True)
    # -inf means "no edge pointed here"; replace with 0 so downstream math is finite
    return out.masked_fill(torch.isinf(out), 0.0)


# --------------------------------------------------------------------------- #
# 2. Softmax over a group -- what makes attention an attention
# --------------------------------------------------------------------------- #
def scatter_softmax(src: torch.Tensor, index: torch.LongTensor, dim_size: int, dim: int = 0) -> torch.Tensor:
    """Softmax normalised *within each group* of ``index``.

    This is the step that turns GAT's raw scores into an attention distribution:
    every node's incoming edge weights sum to 1.

    Numerical stability
    -------------------
    We subtract the per-group max before exponentiating, exactly like the
    textbook softmax trick. The max is ``.detach()``-ed on purpose. That is
    **mathematically exact**, not an approximation: softmax is invariant to a
    constant shift of its inputs, so the shift contributes zero to the
    gradient. Detaching also sidesteps any question of whether the fused
    group-max kernel exposes a gradient.
    """
    index = index.long()

    # Every tensor here keeps `src`'s trailing shape, so a plain index_select on
    # the grouping dim is all that is needed -- no reshaping. That is what lets
    # this work unchanged for a [E] score vector *and* a [E, heads] matrix.
    group_max = scatter_max(src.detach(), index, dim_size, dim)          # [dim_size, ...]
    out = (src - group_max.index_select(dim, index)).exp()               # [E, ...]

    denom = scatter_add(out, index, dim_size, dim) + 1e-16               # [dim_size, ...]
    return out / denom.index_select(dim, index)                          # [E, ...]


# --------------------------------------------------------------------------- #
# 3. Edge-index surgery
# --------------------------------------------------------------------------- #
def add_self_loops(edge_index: torch.LongTensor, num_nodes: int) -> torch.LongTensor:
    """Append ``(i, i)`` for every node.

    GCN, GAT and GIN all do this. It is what lets a node keep its *own*
    representation in the update -- without it, a node never sees itself and
    information can only reach it through neighbours.
    """
    loop = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    loop = loop.unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, loop], dim=1)


def remove_self_loops(edge_index: torch.LongTensor) -> torch.LongTensor:
    """Drop ``(i, i)`` edges."""
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def to_undirected(edge_index: torch.LongTensor, num_nodes: int) -> torch.LongTensor:
    """Symmetrise an edge list and de-duplicate it.

    Cora's citation file is directed ("paper A cites paper B") but citation
    *similarity* is symmetric, so every GNN paper on Cora symmetrises first.
    Forgetting this step is a classic silent bug: you get a plausible-looking
    accuracy that is a few points below the published number.
    """
    row, col = edge_index[0], edge_index[1]
    both = torch.cat([torch.stack([row, col]), torch.stack([col, row])], dim=1)
    # encode (u, v) as a single int64 so `unique` de-duplicates pairs
    flat = both[0] * num_nodes + both[1]
    uniq = torch.unique(flat)
    return torch.stack([uniq // num_nodes, uniq % num_nodes], dim=0)


def degree(index: torch.LongTensor, num_nodes: int, dtype=None) -> torch.Tensor:
    """Node degree = count of edges in ``index``. Returns a dense ``[num_nodes]`` vector."""
    dtype = dtype or torch.float
    return scatter_add(torch.ones(index.numel(), dtype=dtype, device=index.device), index, num_nodes)


# --------------------------------------------------------------------------- #
# 4. The GCN normalisation coefficients
# --------------------------------------------------------------------------- #
def gcn_norm(edge_index: torch.LongTensor, num_nodes: int, add_loops: bool = True,
             edge_weight: torch.Tensor | None = None,
             dtype: torch.dtype | None = None) -> tuple[torch.LongTensor, torch.Tensor]:
    """Return ``(edge_index_with_self_loops, per-edge coefficient)``.

    Computes the coefficients of

        \\hat{A} = \\tilde{D}^{-1/2} (A + I) \\tilde{D}^{-1/2}

    where ``\\tilde{D}`` is the degree matrix of ``A + I``. Concretely, edge
    ``(s -> t)`` gets weight ``1 / sqrt(deg_t * deg_s)`` *after* self-loops are
    added, so a node's own degree includes the ``+1``.

    ``dtype`` is the dtype the coefficients are computed in. It matters: if you
    compute them in float32 and cast to float64 afterwards, every coefficient
    already carries ~1e-8 of rounding error, and a float64 equivalence check
    against PyG can never get below that. The layers pass ``x.dtype`` here.

    Intuition
    ---------
    Plain averaging (``1/deg``) makes a high-degree neighbour and a low-degree
    neighbour contribute equally, which lets hub nodes dominate after several
    layers. The symmetric ``D^-1/2 A D^-1/2`` splits the discount between the
    two endpoints, which is the same normalisation a random walk with restart
    uses and is why GCN trains stably.
    """
    if dtype is None:
        dtype = edge_weight.dtype if edge_weight is not None else torch.float
    device = edge_index.device

    w = (torch.ones(edge_index.size(1), dtype=dtype, device=device)
         if edge_weight is None else edge_weight.to(dtype))

    if add_loops:
        edge_index = add_self_loops(edge_index, num_nodes)
        # the new (i, i) edges each get weight 1, so ``w`` must grow with the edge list
        w = torch.cat([w, torch.ones(num_nodes, dtype=dtype, device=device)])

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(w, col, num_nodes)                 # in-degree including self-loop
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)

    w = deg_inv_sqrt.index_select(0, row) * w * deg_inv_sqrt.index_select(0, col)
    return edge_index, w


def edge_order(edge_index: torch.LongTensor, num_nodes: int) -> torch.LongTensor:
    """Permutation that sorts edges by ``(dst, src)`` -- i.e. by column, then row.

    This is what ``torch.lexsort((src, dst))`` would do if PyTorch had one (NumPy
    does; PyTorch does not). Encoding the pair as a single int64 ``dst * N + src``
    and doing an ordinary sort is the same thing. It is also the ordering PyG
    produces from ``coalesce(..., sort_by_row=False)``, which is why the Cora
    loader and the equivalence checks need it.
    """
    key = edge_index[1] * num_nodes + edge_index[0]
    return torch.argsort(key, stable=True)


# --------------------------------------------------------------------------- #
# 5. Convenience conversions
# --------------------------------------------------------------------------- #
def dense_to_edge_index(adj: torch.Tensor) -> torch.LongTensor:
    """``[N, N]`` dense adjacency -> ``[2, E]`` COO edge index (nonzero positions)."""
    return adj.nonzero(as_tuple=False).t().contiguous()
