"""
Small deterministic graphs for teaching and testing.

Two reasons these exist:

1. **Tests must not need the network.** Every unit test in ``tests/`` runs on
   graphs built here, so ``pytest`` works on a fresh clone with no downloads.
2. **Some concepts are only visible on a graph you can see.** Over-smoothing on
   Cora is a curve on a plot; on a 12-node path graph it is something you can
   literally watch happen node by node.

The interesting one is :func:`triangle_vs_square`, which constructs a graph
classification task where the label *is* a substructure count. That is the
concrete version of the GIN paper's expressiveness argument: mean aggregation
cannot count, sum aggregation can. Run notebook 05 and the difference is not
subtle.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .tu import GraphDataset

__all__ = ["karate_club", "path_graph", "two_communities", "triangle_vs_square",
           "toy_graph", "triangle_square_dataset", "KARATE_EDGES", "KARATE_CLUB_LABELS"]


# Zachary's karate club (1977): 34 members, 78 undirected ties. The standard
# "can you see the social structure?" dataset, and it fits on one screen.
KARATE_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10),
    (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
    (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
    (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
    (3, 7), (3, 12), (3, 13),
    (4, 6), (4, 10),
    (5, 6), (5, 10), (5, 16),
    (6, 16),
    (8, 30), (8, 32), (8, 33),
    (9, 33),
    (13, 33),
    (14, 32), (14, 33),
    (15, 32), (15, 33),
    (18, 32), (18, 33),
    (19, 33),
    (20, 32), (20, 33),
    (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
    (24, 25), (24, 27), (24, 31),
    (25, 31),
    (26, 29), (26, 33),
    (27, 33),
    (28, 31), (28, 33),
    (29, 32), (29, 33),
    (30, 32), (30, 33),
    (31, 32), (31, 33),
    (32, 33),
]

# The historical split into "Mr. Hi" (0) and "Officer" (1) factions: 17 + 17.
KARATE_CLUB_LABELS: List[int] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1,   # nodes  0- 9
    0, 0, 0, 0, 1, 1, 0, 0, 1, 0,   # nodes 10-19
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1,   # nodes 20-29
    1, 1, 1, 1,                     # nodes 30-33
]

assert len(KARATE_CLUB_LABELS) == 34, "karate labels must cover all 34 nodes"
assert len({frozenset(e) for e in KARATE_EDGES}) == 78, "karate club has 78 undirected ties"
assert KARATE_CLUB_LABELS.count(0) == 17 and KARATE_CLUB_LABELS.count(1) == 17


def _undirected(pairs: List[Tuple[int, int]]) -> torch.LongTensor:
    """Pairs -> ``[2, 2E]`` edge_index containing both directions, de-duplicated."""
    s = {tuple(sorted(p)) for p in pairs}
    both = [e for p in s for e in (p, (p[1], p[0]))]
    return torch.tensor(both, dtype=torch.long).t().contiguous()


def karate_club(feature_dim: int = 8, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Zachary's karate club with random features and the true faction labels.

    Only 2 of the 34 labels are normally "known" in the semi-supervised setup.
    Here we expose the full label vector so you can run it as a plain
    node-classification toy problem and *see* the two factions separate.
    """
    g = torch.Generator().manual_seed(seed)
    n = 34
    # features carry a weak signal about the faction so the task is learnable
    y = torch.tensor(KARATE_CLUB_LABELS, dtype=torch.long)
    centre = torch.randn(2, feature_dim, generator=g)
    x = centre[y] + 0.35 * torch.randn(n, feature_dim, generator=g)
    return {"x": x.float(), "edge_index": _undirected(KARATE_EDGES), "y": y,
            "num_nodes": n, "num_classes": 2}


def path_graph(n: int = 12, feature_dim: int = 4, seed: int = 0) -> Dict[str, torch.Tensor]:
    """A chain ``0 - 1 - 2 - ... - n-1``.

    The cleanest possible demonstration of receptive field growth: after ``k``
    layers, node ``i`` has only seen nodes ``i-k .. i+k``. Set ``k = n`` and
    every node sees the entire graph, so every node's embedding converges to the
    same vector -- over-smoothing, with no ambiguity about the cause.
    """
    g = torch.Generator().manual_seed(seed)
    pairs = [(i, i + 1) for i in range(n - 1)]
    return {"x": torch.randn(n, feature_dim, generator=g).float(),
            "edge_index": _undirected(pairs),
            "y": torch.zeros(n, dtype=torch.long),
            "num_nodes": n, "num_classes": 1}


def toy_graph() -> Dict[str, torch.Tensor]:
    """A 5-node graph with hand-checkable arithmetic.

        0 -- 1 -- 2
        |         |
        3 ------- 4

    Used by ``tests/test_scatter.py`` because every expected value can be
    written down on paper, which is the only way to be sure a test is testing
    the right thing.
    """
    x = torch.tensor([[1.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 1.0],
                      [2.0, 0.0],
                      [0.0, 2.0]])
    ei = _undirected([(0, 1), (1, 2), (0, 3), (2, 4), (3, 4)])
    return {"x": x, "edge_index": ei,
            "y": torch.tensor([0, 1, 1, 0, 0], dtype=torch.long),
            "num_nodes": 5, "num_classes": 2}


def two_communities(n_each: int = 60, p_in: float = 0.10, p_out: float = 0.01,
                    feature_dim: int = 8, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Two planted cliques with sparse inter-community edges.

    Homophilous by construction, so any GNN solves it -- which makes it the
    right substrate for measuring *how fast* representations collapse as depth
    grows, without confounding that with the task being hard.
    """
    g = torch.Generator().manual_seed(seed)
    n = 2 * n_each
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            same = (i < n_each) == (j < n_each)
            p = p_in if same else p_out
            if torch.rand(1, generator=g).item() < p:
                pairs.append((i, j))

    y = torch.tensor([0] * n_each + [1] * n_each, dtype=torch.long)
    centre = torch.randn(2, feature_dim, generator=g) * 2.0
    x = centre[y] + torch.randn(n, feature_dim, generator=g) * 0.5
    return {"x": x.float(), "edge_index": _undirected(pairs), "y": y,
            "num_nodes": n, "num_classes": 2}


def triangle_vs_square(num_graphs: int = 400, min_nodes: int = 12, max_nodes: int = 22,
                       feature_dim: int = 4, seed: int = 0) -> List[Dict[str, torch.Tensor]]:
    """A graph-classification task whose label *is* a substructure count.

    Label 1 if the graph contains at least one triangle, label 0 otherwise.

    How the two classes are matched
    -------------------------------
    A first version of this generator drew Erdos-Renyi graphs and rejected the
    ones with the wrong label. That looked balanced but was not: triangle-free
    graphs only survive rejection when they are sparse, so the "no triangle" class
    had ~21 edges on average versus ~34, and **edge density alone predicted the
    label with 88% accuracy** -- no GNN needed. A "mean readout fails" result on
    that data would have been a density story, not an expressiveness story.

    So each graph is now built in pairs:

    1. grow a random **triangle-free** graph with exactly ``m`` edges
       (add random edges, skip any that would close a triangle);
    2. label 0 = that graph; label 1 = the same graph with **one edge moved** so
       that it closes a triangle.

    Node count and edge count are identical within a pair, so size and density carry
    no signal (``tests/test_data.py`` checks this with a logistic regression).
    Degrees differ only by the one moved edge.

    Why this dataset is in the repo
    -------------------------------
    Counting triangles requires distinguishing a node whose neighbours are
    connected to each other from one whose neighbours are not. Mean- and
    max-*aggregation* cannot do this (the GIN paper's Theorem 3); sum can. Note
    that this is a statement about the aggregator *inside* each layer. The GIN
    model here always sums neighbours in its convolutions and only the final
    *readout* varies, so readout choice is a weaker lever than the theorem and
    you should not expect mean/max readouts to collapse to chance. Measure it,
    do not assume it.

    Returns a list of graph dicts in the shape :func:`gnn_lab.data.tu.GraphDataset`
    expects, so it drops straight into the graph-level trainer.
    """
    import random

    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)

    def grow_triangle_free(n: int, m: int):
        """Random triangle-free edge set with exactly ``m`` edges, or None if stuck."""
        adj = [set() for _ in range(n)]
        edges = []
        for _ in range(60 * m):
            if len(edges) == m:
                return adj, edges
            u, v = rng.sample(range(n), 2)
            if v in adj[u] or (adj[u] & adj[v]):      # duplicate, or would close a triangle
                continue
            adj[u].add(v); adj[v].add(u)
            edges.append((u, v))
        return (adj, edges) if len(edges) == m else None

    def close_a_triangle(n: int, adj, edges):
        """Move one edge so the graph gains a triangle. Returns the new edge list or None."""
        cands = [(u, v) for u in range(n) for v in range(u + 1, n)
                 if v not in adj[u] and (adj[u] & adj[v])]      # non-edges with a common neighbour
        if not cands:
            return None
        u, v = cands[rng.randrange(len(cands))]
        new_edge = (u, v)
        # remove a random edge that is not needed for the triangle (u-w, w-v for some common w)
        w = rng.choice(sorted(adj[u] & adj[v]))
        protected = {frozenset((u, w)), frozenset((w, v))}
        removable = [e for e in edges if frozenset(e) not in protected]
        drop = removable[rng.randrange(len(removable))]
        return [e for e in edges if e != drop] + [new_edge]

    def has_triangle(n: int, edges) -> bool:
        adj = [set() for _ in range(n)]
        for a_, b_ in edges:
            adj[a_].add(b_); adj[b_].add(a_)
        return any(adj[a_] & adj[b_] for a_, b_ in edges)

    def to_graph(n: int, edges, label: int) -> Dict[str, torch.Tensor]:
        pairs = [(a_, b_) for a_, b_ in edges] + [(b_, a_) for a_, b_ in edges]
        return {"x": torch.randn(n, feature_dim, generator=g).float(),
                "edge_index": torch.tensor(pairs, dtype=torch.long).t().contiguous(),
                "y": torch.tensor(label, dtype=torch.long)}

    graphs: List[Dict[str, torch.Tensor]] = []
    pair_id = 0
    while len(graphs) < num_graphs:
        n = rng.randint(min_nodes, max_nodes)
        m = round(n * rng.uniform(1.3, 1.9))
        built = grow_triangle_free(n, m)
        if built is None:
            continue
        adj, edges = built
        twin = close_a_triangle(n, adj, edges)
        if twin is None or not has_triangle(n, twin) or has_triangle(n, edges):
            continue
        # alternate which twin is emitted first so that a truncated list stays balanced
        pair = [(edges, 0), (twin, 1)]
        if pair_id % 2:
            pair.reverse()
        for e, lab in pair:
            if len(graphs) < num_graphs:
                graphs.append(to_graph(n, e, lab))
        pair_id += 1
    return graphs


def triangle_square_dataset(num_graphs: int = 400, seed: int = 0):
    """``triangle_vs_square`` wrapped as a :class:`~gnn_lab.data.tu.GraphDataset`."""

    graphs = triangle_vs_square(num_graphs=num_graphs, seed=seed)
    return GraphDataset(
        graphs=graphs, name="TriangleVsSquare",
        num_features=graphs[0]["x"].size(1), num_classes=2,
        class_names=["no_triangle", "has_triangle"],
        notes=[f"synthetic, seed {seed}; label = presence of a 3-cycle",
               "classes matched on node count and edge count (density carries no signal)"],
    )
