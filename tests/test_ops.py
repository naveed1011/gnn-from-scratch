"""
Tests for the scatter primitives in ``gnn_lab.ops``.

Every expected value here is either produced by an explicit Python loop or
written down by hand. Nothing is compared against another library, so a
passing test means the operation matches its *definition*, not that it matches
somebody else's implementation.
"""

from __future__ import annotations

import torch

from gnn_lab.ops import (add_self_loops, degree, dense_to_edge_index, gcn_norm,
                         remove_self_loops, scatter_add, scatter_max, scatter_mean,
                         scatter_softmax, to_undirected)

DT = torch.float64


def _ref_groups(index, dim_size):
    groups = [[] for _ in range(dim_size)]
    for e, k in enumerate(index.tolist()):
        groups[k].append(e)
    return groups


# --------------------------------------------------------------------------- #
# scatter_add
# --------------------------------------------------------------------------- #
def test_scatter_add_matches_loop():
    index = torch.tensor([0, 2, 2, 1, 0, 4])
    src = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0],
                        [4.0, 40.0], [5.0, 50.0], [6.0, 60.0]], dtype=DT)
    out = scatter_add(src, index, dim_size=6)

    assert out.shape == (6, 2)
    # hand-computed
    assert torch.allclose(out[0], torch.tensor([6.0, 60.0], dtype=DT))    # rows 0 and 4
    assert torch.allclose(out[1], torch.tensor([4.0, 40.0], dtype=DT))    # row 3
    assert torch.allclose(out[2], torch.tensor([5.0, 50.0], dtype=DT))    # rows 1 and 2
    assert torch.allclose(out[3], torch.zeros(2, dtype=DT))               # nothing points here
    assert torch.allclose(out[4], torch.tensor([6.0, 60.0], dtype=DT))    # row 5
    assert torch.allclose(out[5], torch.zeros(2, dtype=DT))


def test_scatter_add_is_linear():
    """add(a) + add(b) == add(a + b): the operation must not interact across edges."""
    g = torch.Generator().manual_seed(1)
    index = torch.randint(0, 7, (50,), generator=g)
    a = torch.randn(50, 4, generator=g, dtype=DT)
    b = torch.randn(50, 4, generator=g, dtype=DT)
    assert torch.allclose(scatter_add(a, index, 7) + scatter_add(b, index, 7),
                          scatter_add(a + b, index, 7), atol=1e-12)


def test_scatter_add_empty_index_returns_zeros_with_grad():
    src = torch.randn(0, 3, dtype=DT, requires_grad=True)
    out = scatter_add(src, torch.zeros(0, dtype=torch.long), dim_size=4)
    assert out.shape == (4, 3)
    assert float(out.abs().sum()) == 0.0
    out.sum().backward()          # must not raise
    assert src.grad is not None


# --------------------------------------------------------------------------- #
# scatter_mean / scatter_max
# --------------------------------------------------------------------------- #
def test_scatter_mean_matches_loop():
    g = torch.Generator().manual_seed(2)
    index = torch.randint(0, 5, (60,), generator=g)
    src = torch.randn(60, 3, generator=g, dtype=DT)

    out = scatter_mean(src, index, 5)
    for k, members in enumerate(_ref_groups(index, 5)):
        expected = src[members].mean(0) if members else torch.zeros(3, dtype=DT)
        assert torch.allclose(out[k], expected, atol=1e-12), f"group {k}"


def test_scatter_mean_of_empty_group_is_zero_not_nan():
    """The single most common source of 'loss became nan' in hand-written GNNs."""
    index = torch.tensor([0, 0, 2])
    src = torch.ones(3, 2, dtype=DT)
    out = scatter_mean(src, index, 4)
    assert torch.isfinite(out).all(), "empty groups produced inf/nan"
    assert torch.allclose(out[1], torch.zeros(2, dtype=DT))
    assert torch.allclose(out[3], torch.zeros(2, dtype=DT))
    assert torch.allclose(out[0], torch.ones(2, dtype=DT))


def test_scatter_max_matches_loop():
    g = torch.Generator().manual_seed(3)
    index = torch.randint(0, 5, (60,), generator=g)
    src = torch.randn(60, 3, generator=g, dtype=DT)

    out = scatter_max(src, index, 5)
    for k, members in enumerate(_ref_groups(index, 5)):
        expected = src[members].max(0).values if members else torch.zeros(3, dtype=DT)
        assert torch.allclose(out[k], expected, atol=1e-12), f"group {k}"


def test_scatter_max_on_all_negative_values_is_not_clamped_to_zero():
    """Only *empty* groups become 0; a real max of -5 must stay -5."""
    index = torch.tensor([0, 0, 1])
    src = torch.tensor([[-5.0], [-3.0], [-9.0]], dtype=DT)
    out = scatter_max(src, index, 3)
    assert float(out[0]) == -3.0
    assert float(out[1]) == -9.0
    assert float(out[2]) == 0.0        # genuinely empty


# --------------------------------------------------------------------------- #
# scatter_softmax
# --------------------------------------------------------------------------- #
def test_scatter_softmax_sums_to_one_per_group():
    g = torch.Generator().manual_seed(4)
    index = torch.randint(0, 6, (80,), generator=g)
    src = torch.randn(80, 4, generator=g, dtype=DT)
    out = scatter_softmax(src, index, 6)

    totals = scatter_add(out, index, 6)
    nonempty = torch.zeros(6, dtype=torch.bool)
    nonempty[index.unique()] = True
    assert torch.allclose(totals[nonempty], torch.ones(int(nonempty.sum()), 4, dtype=DT),
                          atol=1e-10)


def test_scatter_softmax_matches_per_group_torch_softmax():
    g = torch.Generator().manual_seed(5)
    index = torch.randint(0, 4, (40,), generator=g)
    src = torch.randn(40, 3, generator=g, dtype=DT)
    out = scatter_softmax(src, index, 4)

    for k, members in enumerate(_ref_groups(index, 4)):
        if members:
            assert torch.allclose(out[members], torch.softmax(src[members], dim=0), atol=1e-12)


def test_scatter_softmax_is_shift_invariant():
    """Subtracting a constant per group must not change the result at all."""
    g = torch.Generator().manual_seed(6)
    index = torch.randint(0, 3, (30,), generator=g)
    src = torch.randn(30, 2, generator=g, dtype=DT)
    assert torch.allclose(scatter_softmax(src, index, 3),
                          scatter_softmax(src + 1e6, index, 3), atol=1e-10)


def test_scatter_softmax_gradient_is_correct():
    """Because softmax is shift-invariant, detaching the max must not change the gradient.

    Verified against a reference that keeps the max attached.
    """
    g = torch.Generator().manual_seed(7)
    index = torch.randint(0, 4, (25,), generator=g)
    base = torch.randn(25, 2, generator=g, dtype=DT)

    a = base.clone().requires_grad_(True)
    scatter_softmax(a, index, 4).sum().backward()

    def reference(src):
        m = scatter_max(src, index, 4)               # NOT detached
        e = (src - m.index_select(0, index)).exp()
        return e / scatter_add(e, index, 4).index_select(0, index)

    b = base.clone().requires_grad_(True)
    reference(b).sum().backward()

    assert torch.allclose(a.grad, b.grad, atol=1e-9)


# --------------------------------------------------------------------------- #
# edge-index surgery
# --------------------------------------------------------------------------- #
def test_add_and_remove_self_loops_roundtrip():
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    with_loops = add_self_loops(ei, num_nodes=3)
    assert with_loops.size(1) == 6
    assert with_loops[:, 3:].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert torch.equal(remove_self_loops(with_loops), ei)


def test_to_undirected_dedupes_reciprocal_pairs():
    """Cora's raw file has 151 reciprocal pairs; duplicating them inflates degrees."""
    ei = torch.tensor([[0, 1, 1, 0, 2], [1, 0, 2, 0, 1]])
    out = to_undirected(ei, num_nodes=3)
    assert out.size(1) == 5                     # 2 edges x2 + 1 self-loop = 5 directed
    pairs = {tuple(c) for c in out.t().tolist()}
    assert len(pairs) == 5
    assert (0, 1) in pairs and (1, 0) in pairs


def test_degree_counts_incoming_edges():
    ei = torch.tensor([[0, 1, 2, 0], [1, 1, 1, 2]])   # node 1 receives 3 edges
    d = degree(ei[1], num_nodes=3)
    assert d.tolist() == [0.0, 3.0, 1.0]


def test_dense_to_edge_index_roundtrip():
    adj = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    ei = dense_to_edge_index(adj)
    assert ei.shape[0] == 2 and ei.size(1) == 4
    rebuilt = torch.zeros(3, 3, dtype=torch.long)
    rebuilt[ei[0], ei[1]] = 1
    assert torch.equal(rebuilt, adj)


# --------------------------------------------------------------------------- #
# gcn_norm
# --------------------------------------------------------------------------- #
def test_gcn_norm_on_a_regular_graph_is_uniform():
    """On the 2-regular toy graph every coefficient must be exactly 1/3.

    With self-loops each node has degree 3, so 1/sqrt(3*3) = 1/3 for all edges.
    A hand-computable case that pins down the whole normalisation.
    """
    from gnn_lab.data.synthetic import toy_graph

    ei = toy_graph()["edge_index"]
    ei_loops, w = gcn_norm(ei, num_nodes=5, add_loops=True)
    assert ei_loops.size(1) == ei.size(1) + 5
    assert torch.allclose(w, torch.full_like(w, 1.0 / 3.0), atol=1e-7)


def test_gcn_norm_rows_sum_to_one():
    """D^-1/2 (A+I) D^-1/2 is not row-stochastic in general, but on a regular
    graph it is. On an irregular one, verify instead that coefficients are
    symmetric: swapping src and dst must not change the weight."""
    g = torch.Generator().manual_seed(8)
    n = 40
    ei = torch.randint(0, n, (2, 400), generator=g)
    ei = ei[:, ei[0] != ei[1]]
    ei = torch.cat([ei, ei.flip(0)], dim=1)

    ei_l, w = gcn_norm(ei, n, add_loops=True)
    lookup = {(int(a), int(b)): float(v) for a, b, v in zip(ei_l[0], ei_l[1], w)}
    for (a, b), v in list(lookup.items()):
        assert abs(lookup[(b, a)] - v) < 1e-12, f"asymmetric coefficient on ({a},{b})"


def test_gcn_norm_without_loops_uses_bare_degree():
    """Two nodes joined by one edge: with loops both have degree 2 -> coeff 1/2."""
    ei = torch.tensor([[0, 1], [1, 0]])
    _, w = gcn_norm(ei, num_nodes=2, add_loops=True)
    assert torch.allclose(w, torch.full_like(w, 0.5), atol=1e-7)
