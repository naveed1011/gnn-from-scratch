"""
Dataset tests: structural invariants that must hold for any graph loader.

The Cora/MUTAG tests that need a download are marked ``network`` so a normal
``pytest`` run stays fast and offline. Run everything with::

    pytest -m "network or not network"
"""

from __future__ import annotations

import torch
import pytest

from gnn_lab.data.synthetic import (KARATE_CLUB_LABELS, KARATE_EDGES, karate_club,
                                    path_graph, toy_graph, triangle_square_dataset,
                                    triangle_vs_square, two_communities)
from gnn_lab.layers.pooling import collate_graphs
from gnn_lab.ops import to_undirected

network = pytest.mark.network


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def assert_valid_graph(edge_index: torch.LongTensor, num_nodes: int, label: str = ""):
    """The invariants every loader in this repo must satisfy."""
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, f"{label}: edge_index must be [2, E]"
    assert edge_index.dtype == torch.long, f"{label}: edge_index must be int64, got {edge_index.dtype}"
    if edge_index.numel():
        assert int(edge_index.min()) >= 0, f"{label}: negative node index"
        assert int(edge_index.max()) < num_nodes, f"{label}: node index out of range"

    pairs = {tuple(c) for c in edge_index.t().tolist()}
    assert len(pairs) == edge_index.size(1), f"{label}: duplicate directed edges present"
    assert not any(a == b for a, b in pairs), f"{label}: self-loops present"
    assert all((b, a) in pairs for a, b in pairs), f"{label}: graph is not symmetric"


# --------------------------------------------------------------------------- #
# synthetic
# --------------------------------------------------------------------------- #
def test_toy_graph_is_two_regular():
    g = toy_graph()
    assert g["x"].shape == (5, 2)
    assert g["edge_index"].size(1) == 10          # 5 undirected edges
    deg = torch.zeros(5).index_add_(0, g["edge_index"][1], torch.ones(10))
    assert deg.tolist() == [2.0] * 5, "the hand-computed layer tests rely on this"
    assert_valid_graph(g["edge_index"], 5, "toy_graph")


def test_karate_club_structure():
    g = karate_club()
    assert g["x"].shape[0] == 34 and g["y"].shape == (34,)
    assert len({frozenset(e) for e in KARATE_EDGES}) == 78
    assert g["edge_index"].size(1) == 156
    assert_valid_graph(g["edge_index"], 34, "karate")
    assert KARATE_CLUB_LABELS.count(0) == 17 and KARATE_CLUB_LABELS.count(1) == 17

    # the graph must be connected, otherwise community detection is meaningless
    adj = {i: set() for i in range(34)}
    for a, b in KARATE_EDGES:
        adj[a].add(b); adj[b].add(a)
    seen, stack = {0}, [0]
    while stack:
        n = stack.pop()
        for m in adj[n]:
            if m not in seen:
                seen.add(m); stack.append(m)
    assert len(seen) == 34, "karate club graph is disconnected"


def test_karate_club_is_homophilous():
    """67 within-faction ties vs 11 across; a GNN should find this easy."""
    g = karate_club()
    lab = g["y"]
    same = int((lab[g["edge_index"][0]] == lab[g["edge_index"][1]]).sum())
    assert same / g["edge_index"].size(1) > 0.8


def test_path_graph_receptive_field_structure():
    g = path_graph(n=12)
    assert g["x"].shape[0] == 12
    assert g["edge_index"].size(1) == 22              # 11 undirected edges
    deg = torch.zeros(12).index_add_(0, g["edge_index"][1], torch.ones(22))
    assert deg[0] == 1 and deg[11] == 1 and deg[5] == 2, "endpoints must have degree 1"
    assert_valid_graph(g["edge_index"], 12, "path")


def test_two_communities_shapes_and_labels():
    g = two_communities(n_each=25, seed=3)
    assert g["x"].shape == (50, 8)
    assert g["y"].tolist() == [0] * 25 + [1] * 25
    assert_valid_graph(g["edge_index"], 50, "two_communities")


def test_triangle_vs_square_labels_are_actually_correct():
    """Independently re-detect triangles and check the generator's labels."""
    graphs = triangle_vs_square(num_graphs=40, seed=7)
    assert len(graphs) == 40
    for k, g in enumerate(graphs):
        n = g["x"].size(0)
        adj = [set() for _ in range(n)]
        for a, b in g["edge_index"].t().tolist():
            adj[a].add(b)
        found = any(c in adj[a] for a in range(n) for b in adj[a] for c in adj[b] if c != a)
        assert int(g["y"]) == int(found), f"graph {k}: label {int(g['y'])} but triangle={found}"


def test_triangle_vs_square_is_balanced_and_valid():
    ds = triangle_square_dataset(num_graphs=60, seed=1)
    assert len(ds) == 60
    y = ds.labels()
    assert int((y == 0).sum()) == int((y == 1).sum()) == 30
    assert abs(ds.majority_accuracy() - 0.5) < 1e-9
    for g in ds.graphs:
        assert_valid_graph(g["edge_index"], g["x"].size(0), "triangle_vs_square")


def test_collate_round_trip_preserves_each_graph():
    graphs = triangle_vs_square(num_graphs=8, seed=2)
    batched = collate_graphs(graphs)
    assert batched.num_nodes == sum(g["x"].size(0) for g in graphs)
    assert batched.num_graphs == 8
    assert batched.batch.numel() == batched.num_nodes
    assert batched.batch.min() == 0 and batched.batch.max() == 7
    assert_valid_graph(batched.edge_index, batched.num_nodes, "collated")


# --------------------------------------------------------------------------- #
# Cora (needs a download the first time)
# --------------------------------------------------------------------------- #
@network
def test_cora_planetoid_structure(tmp_path):
    from gnn_lab.data.cora import load_cora_planetoid

    ds = load_cora_planetoid(root=str(tmp_path / "planetoid"), verbose=False)
    assert ds.num_nodes == 2708
    assert ds.num_features == 1433
    assert ds.num_classes == 7
    assert_valid_graph(ds.edge_index, ds.num_nodes, "cora/planetoid")
    assert ds.num_edges == 10556, f"expected the de-duplicated edge count, got {ds.num_edges}"


@network
def test_cora_planetoid_split_sizes(tmp_path):
    from gnn_lab.data.cora import load_cora_planetoid

    ds = load_cora_planetoid(root=str(tmp_path / "planetoid"), verbose=False)
    assert int(ds.train_mask.sum()) == 140
    assert int(ds.val_mask.sum()) == 500
    assert int(ds.test_mask.sum()) == 1000
    # the three splits must be disjoint. They do NOT cover every node: the Planetoid
    # protocol labels only 140 + 500 + 1000 = 1640 of the 2708 papers; the other 1068
    # are simply unused (they still take part in message passing).
    overlap = (ds.train_mask & ds.val_mask) | (ds.val_mask & ds.test_mask) | (ds.train_mask & ds.test_mask)
    assert not bool(overlap.any()), "splits overlap"
    assert int((ds.train_mask | ds.val_mask | ds.test_mask).sum()) == 1640


@network
def test_cora_planetoid_labels_are_valid(tmp_path):
    from gnn_lab.data.cora import load_cora_planetoid

    ds = load_cora_planetoid(root=str(tmp_path / "planetoid"), verbose=False)
    assert ds.y.min() >= 0 and ds.y.max() < 7
    counts = torch.bincount(ds.y, minlength=7)
    assert int(counts.sum()) == 2708
    assert int(counts.min()) > 0, "some class has no members"


@network
def test_cora_dedupe_switch_changes_edge_count(tmp_path):
    """The 302 duplicate adjacency entries are real and the switch controls them."""
    from gnn_lab.data.cora import load_cora_planetoid

    a = load_cora_planetoid(root=str(tmp_path / "planetoid"), dedupe=True, verbose=False)
    b = load_cora_planetoid(root=str(tmp_path / "planetoid"), dedupe=False, verbose=False)
    assert b.num_edges == 10858
    assert a.num_edges == 10556
    assert b.num_edges - a.num_edges == 302


@network
def test_cora_linqs_matches_planetoid_graph_statistics(tmp_path):
    """Different node ordering, same graph: node count, edge count, class sizes."""
    from gnn_lab.data.cora import load_cora_linqs, load_cora_planetoid

    lin = load_cora_linqs(root=str(tmp_path / "linqs"), verbose=False)
    pl = load_cora_planetoid(root=str(tmp_path / "planetoid"), row_normalize=False, verbose=False)

    assert lin.num_nodes == pl.num_nodes == 2708
    assert lin.num_features == pl.num_features == 1433
    assert lin.num_edges == pl.num_edges == 10556
    assert torch.equal(torch.bincount(lin.y, minlength=7).sort().values,
                       torch.bincount(pl.y, minlength=7).sort().values)
    assert lin.class_names == sorted(lin.class_names)
    assert_valid_graph(lin.edge_index, lin.num_nodes, "cora/linqs")


@network
def test_cora_linqs_public_split_protocol(tmp_path):
    from gnn_lab.data.cora import load_cora_linqs

    ds = load_cora_linqs(root=str(tmp_path / "linqs"), split="public", verbose=False)
    assert int(ds.train_mask.sum()) == 140
    assert int(ds.val_mask.sum()) == 500
    assert int(ds.test_mask.sum()) == 1000
    # 20 labelled nodes per class, exactly like the Planetoid protocol
    per_class = torch.bincount(ds.y[ds.train_mask], minlength=7)
    assert per_class.tolist() == [20] * 7


@network
def test_cora_linqs_split_is_reproducible(tmp_path):
    from gnn_lab.data.cora import load_cora_linqs

    a = load_cora_linqs(root=str(tmp_path / "linqs"), seed=123, verbose=False)
    b = load_cora_linqs(root=str(tmp_path / "linqs"), seed=123, verbose=False)
    assert torch.equal(a.train_mask, b.train_mask)
    c = load_cora_linqs(root=str(tmp_path / "linqs"), seed=999, verbose=False)
    assert not torch.equal(a.train_mask, c.train_mask)


# --------------------------------------------------------------------------- #
# MUTAG (needs a download the first time)
# --------------------------------------------------------------------------- #
@network
def test_mutag_structure(tmp_path):
    from gnn_lab.data.tu import load_mutag

    ds = load_mutag(root=str(tmp_path), verbose=False)
    assert len(ds) == 188
    assert ds.num_nodes == 3371
    assert ds.num_edges == 7442
    assert ds.num_features == 7
    assert ds.num_classes == 2
    assert abs(ds.majority_accuracy() - 125 / 188) < 1e-6

    for g in ds.graphs:
        n = g["x"].size(0)
        assert 10 <= n <= 28
        # node features are one-hot atom types
        assert torch.allclose(g["x"].sum(dim=1), torch.ones(n))
        assert_valid_graph(g["edge_index"], n, "mutag")
        # every edge must stay inside its own graph
        assert int(g["edge_index"].max()) < n


@network
def test_mutag_collates_without_cross_graph_edges(tmp_path):
    from gnn_lab.data.tu import load_mutag

    ds = load_mutag(root=str(tmp_path), verbose=False)
    batched = collate_graphs(ds.graphs[:16])
    assert batched.num_graphs == 16
    assert_valid_graph(batched.edge_index, batched.num_nodes, "mutag batch")

    # no edge may join two different graphs
    g_of_node = batched.batch
    assert bool((g_of_node[batched.edge_index[0]] == g_of_node[batched.edge_index[1]]).all())


def test_triangle_classes_are_matched_on_size_and_density():
    """Regression: the first generator let edge density predict the label at 88%.

    Rejection-sampling triangle-free graphs only keeps the sparse ones, so the classes
    had ~21 vs ~34 edges. The generator now builds matched pairs, so the per-class means
    of node count and edge count must be identical.
    """
    graphs = triangle_vs_square(num_graphs=200, seed=3)
    by = {0: [], 1: []}
    for g in graphs:
        by[int(g["y"])].append((g["x"].size(0), g["edge_index"].size(1) // 2))
    assert len(by[0]) == len(by[1]) == 100
    for k in (0, 1):
        assert sorted(by[0]) == sorted(by[1]), "classes differ in (nodes, edges) multiset"
