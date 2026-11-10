"""
Batching and pooling -- the part of GNNs that has no CNN analogue.

The block-diagonal trick
------------------------
CNNs get batching for free: stack 32 images of shape ``[3, H, W]`` into
``[32, 3, H, W]``. Every image has the same shape, so a 4-D tensor works.

Graphs do not. Molecule 1 has 17 nodes, molecule 2 has 28. You cannot stack
ragged shapes. So instead of a batch dimension you build **one big disconnected
graph**: concatenate the node features, concatenate the edges, and offset each
graph's edge indices so nothing accidentally connects across graphs.

    graph A: 3 nodes, edges (0,1) (1,2)
    graph B: 2 nodes, edges (0,1)
    ------------------------------------
    batched: 5 nodes, edges (0,1) (1,2) (3,4)      <- B's edge offset by +3
             batch  = [0, 0, 0, 1, 1]               <- which graph each node belongs to

Because message passing only ever looks at ``edge_index``, no information can
leak between graphs -- they are physically disconnected. One forward pass
processes all of them, with the same cost profile as a single big graph.

The ``batch`` vector is then all you need for pooling: "sum every node whose
batch id is k". That is :func:`~gnn_lab.ops.scatter_add` again -- the same
primitive as message passing, one level up.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

from ..ops import scatter_add, scatter_max, scatter_mean

__all__ = ["global_pool", "GlobalPooling", "collate_graphs", "graph_batches", "BatchedGraph"]


# --------------------------------------------------------------------------- #
# Readout / pooling
# --------------------------------------------------------------------------- #
def global_pool(x: torch.Tensor, batch: torch.LongTensor, method: str = "mean",
                num_graphs: int | None = None) -> torch.Tensor:
    """Collapse ``[total_nodes, F]`` node embeddings to ``[num_graphs, F]``.

    * ``sum``  -- degree-sensitive; keeps "how much stuff is in this graph".
      The GIN paper argues sum is the right choice when graph *size* is signal.
    * ``mean`` -- size-invariant; safer when graphs vary a lot in size.
    * ``max``  -- keeps only the most strongly activated node per channel.
      Behaves like a soft "does this graph contain substructure X?" detector,
      which is often what molecular classification actually needs.
    """
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0

    if method == "sum":
        return scatter_add(x, batch, num_graphs)
    if method == "mean":
        return scatter_mean(x, batch, num_graphs)
    if method == "max":
        return scatter_max(x, batch, num_graphs)
    raise ValueError(f"unknown pooling method {method!r}")


class GlobalPooling(nn.Module):
    """``global_pool`` as a module, so it can sit inside ``nn.Sequential``."""

    def __init__(self, method: str = "mean"):
        super().__init__()
        self.method = method

    def forward(self, x: torch.Tensor, batch: torch.LongTensor, num_graphs: int | None = None):
        return global_pool(x, batch, self.method, num_graphs)


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
class BatchedGraph:
    """A minimal stand-in for ``torch_geometric.data.Batch``.

    Deliberately small on purpose: ``x``, ``edge_index``, ``batch``, ``y`` and
    ``num_graphs`` are the only five things a graph-level GNN needs. Seeing that
    list makes it obvious there is no magic in PyG's ``DataLoader``.
    """

    __slots__ = ("x", "edge_index", "batch", "y", "num_graphs", "num_nodes")

    def __init__(self, x, edge_index, batch, y, num_graphs):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
        self.y = y
        self.num_graphs = num_graphs
        self.num_nodes = x.size(0)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self

    def __repr__(self) -> str:
        return (f"BatchedGraph(graphs={self.num_graphs}, nodes={self.num_nodes}, "
                f"edges={self.edge_index.size(1)}, x={tuple(self.x.shape)})")


def collate_graphs(graphs: Sequence[dict], device=None) -> BatchedGraph:
    """Build one block-diagonal :class:`BatchedGraph` from a list of graphs.

    Each element of ``graphs`` is a dict with keys ``x`` (``[n_i, F]``) and
    ``edge_index`` (``[2, e_i]``, **0-indexed within its own graph**), plus an
    optional ``y``.

    The only real work is the running node offset: graph ``k``'s edges must be
    shifted by the total number of nodes in graphs ``0..k-1``.
    """
    xs, eis, batches, ys = [], [], [], []
    offset = 0

    for k, g in enumerate(graphs):
        n = g["x"].size(0)
        xs.append(g["x"])
        eis.append(g["edge_index"] + offset)             # <- the offset
        batches.append(torch.full((n,), k, dtype=torch.long))
        if g.get("y") is not None:
            y_k = g["y"] if torch.is_tensor(g["y"]) else torch.tensor(g["y"])
            ys.append(y_k.reshape(-1))      # a graph label is 0-dim; cat needs >= 1-dim
        offset += n

    x = torch.cat(xs, dim=0) if xs else torch.zeros(0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros(2, 0, dtype=torch.long)
    batch = torch.cat(batches, dim=0) if batches else torch.zeros(0, dtype=torch.long)
    y = torch.cat(ys, dim=0) if ys else None

    out = BatchedGraph(x, edge_index, batch, y, num_graphs=len(graphs))
    return out.to(device) if device is not None else out


def graph_batches(graphs: Iterable[dict], batch_size: int, shuffle: bool = True,
                  generator: torch.Generator | None = None) -> Iterable[BatchedGraph]:
    """A ~10-line DataLoader for graphs, so the batching logic stays visible."""
    graphs = list(graphs)
    idx = torch.randperm(len(graphs), generator=generator) if shuffle else torch.arange(len(graphs))
    for start in range(0, len(graphs), batch_size):
        sel = idx[start:start + batch_size]
        yield collate_graphs([graphs[int(i)] for i in sel])
