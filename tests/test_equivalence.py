"""
Pytest wrapper around :mod:`gnn_lab.equivalence`.

Run with ``-s`` to see the full table::

    pytest tests/test_equivalence.py -s

The scatter checks always run. The PyG checks skip cleanly if
``torch_geometric`` is not installed -- that is a supported configuration, since
the from-scratch code has no PyG dependency at all.
"""

from __future__ import annotations

import pytest
import torch

from gnn_lab import equivalence as eq

network = pytest.mark.network
needs_pyg = pytest.mark.skipif(not eq.PYG_AVAILABLE,
                               reason=f"torch_geometric unavailable ({eq.PYG_IMPORT_ERROR})")


# --------------------------------------------------------------------------- #
# Scatter primitives -- always run, no third-party library needed
# --------------------------------------------------------------------------- #
def test_scatter_primitives_match_brute_force():
    results = eq.check_scatter_vs_loop(n=64, dim_size=13, seed=0)
    for r in results:
        assert r.passed, r.line()


def test_scatter_ops_are_differentiable():
    """Gradients must be finite and non-trivial through every reduction."""
    index = torch.tensor([0, 0, 1, 2, 2, 2])
    for fn in (eq.scatter_add, eq.scatter_mean, eq.scatter_max, eq.scatter_softmax):
        src = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
        fn(src, index, 3).sum().backward()
        assert src.grad is not None, f"{fn.__name__} produced no gradient"
        assert torch.isfinite(src.grad).all(), f"{fn.__name__} produced non-finite gradient"
        assert float(src.grad.abs().sum()) > 0, f"{fn.__name__} produced a zero gradient"


# --------------------------------------------------------------------------- #
# Layer-by-layer equivalence with PyTorch Geometric
# --------------------------------------------------------------------------- #
@needs_pyg
def test_gcn_norm_matches_pyg():
    for r in eq.check_gcn_norm():
        assert r.passed is not False, r.line()


@needs_pyg
def test_gcn_matches_pyg():
    for r in eq.check_gcn():
        assert r.passed, r.line()


@needs_pyg
@pytest.mark.parametrize("aggr", ["mean", "max", "sum"])
def test_sage_matches_pyg(aggr):
    for r in eq.check_sage(aggr):
        assert r.passed, r.line()


@needs_pyg
@pytest.mark.parametrize("heads,concat", [(1, True), (4, True), (8, True), (8, False)])
def test_gat_matches_pyg(heads, concat):
    for r in eq.check_gat(heads=heads, concat=concat):
        assert r.passed, r.line()


@needs_pyg
def test_gin_matches_pyg():
    for r in eq.check_gin():
        assert r.passed, r.line()


@needs_pyg
def test_pooling_matches_pyg():
    for r in eq.check_pooling():
        assert r.passed, r.line()


# --------------------------------------------------------------------------- #
# End-to-end: the same training run, two implementations
# --------------------------------------------------------------------------- #
@needs_pyg
@network
def test_gcn_trains_to_the_same_accuracy_as_pyg(tmp_path):
    """Layer-level equality is necessary but not sufficient; this closes the loop.

    Same data, same architecture, same seed, same hyper-parameters, 60 epochs.
    The two test accuracies should land within a couple of points of each other
    -- they will not be bitwise identical because dropout and the optimiser
    introduce chaotic divergence, but a systematic gap would mean one of the two
    implementations is computing something different.
    """
    from gnn_lab.data.cora import load_cora_planetoid
    from gnn_lab.layers.gcn import GCNScratch
    from gnn_lab.train import train_node
    from gnn_lab.utils import set_seed

    data = load_cora_planetoid(root=str(tmp_path / "cora"), row_normalize=False,
                              dedupe=True, verbose=False)

    set_seed(0)
    mine = GCNScratch(data.num_features, 16, data.num_classes, num_layers=2, dropout=0.5)
    r_mine = train_node(mine, data, model_name="scratch", epochs=60, lr=1e-2, weight_decay=5e-4)

    from torch_geometric.nn import GCNConv
    import torch.nn.functional as F

    class PyGGCN(torch.nn.Module):
        def __init__(self, f, h, c):
            super().__init__()
            self.c1 = GCNConv(f, h)
            self.c2 = GCNConv(h, c)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.c1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            return self.c2(x, edge_index)

    set_seed(0)
    ref = PyGGCN(data.num_features, 16, data.num_classes)
    r_ref = train_node(ref, data, model_name="pyg", epochs=60, lr=1e-2, weight_decay=5e-4)

    print(f"\n  scratch test acc {r_mine.test_acc_at_best_val:.4f}  "
          f"pyg test acc {r_ref.test_acc_at_best_val:.4f}")
    assert abs(r_mine.test_acc_at_best_val - r_ref.test_acc_at_best_val) < 0.05


# --------------------------------------------------------------------------- #
# Dataset equivalence
# --------------------------------------------------------------------------- #
@needs_pyg
@network
def test_cora_loader_matches_pyg(tmp_path):
    results = eq.check_cora_loader(root=str(tmp_path))
    for r in results:
        assert r.passed is not False, r.line()


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #
@needs_pyg
def test_run_all_reports_no_failures(tmp_path, capsys):
    results = eq.run_all(root=str(tmp_path), include_data=False, verbose=True)
    captured = capsys.readouterr()
    print(captured.out)
    assert eq.all_passed(results), "at least one equivalence check FAILED (see table above)"
    assert sum(1 for r in results if r.passed is True) >= 8, \
        "suspiciously few checks ran -- the suite may be silently skipping"


# --------------------------------------------------------------------------- #
# The checks themselves must be able to fail
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not eq.PYG_AVAILABLE, reason="needs torch_geometric")
@pytest.mark.parametrize("name", ["gcn", "sage", "gat"])
def test_equivalence_graph_is_directed_so_a_src_dst_swap_is_detected(name):
    """Mutation test: feed the scratch layer a flipped edge_index and demand a big error.

    If ``_random_graph`` ever goes back to symmetric edges this fails, because a
    src<->dst swap is invisible on a symmetric graph.
    """
    import torch
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv

    from gnn_lab.layers.gat import GATConvScratch
    from gnn_lab.layers.gcn import GCNConvScratch
    from gnn_lab.layers.sage import SAGEConvScratch

    x, ei = eq._random_graph(60, 300, 16, seed=0)
    mine, ref = {
        "gcn": (GCNConvScratch(16, 8), GCNConv(16, 8)),
        "sage": (SAGEConvScratch(16, 8), SAGEConv(16, 8)),
        "gat": (GATConvScratch(16, 8, heads=4), GATConv(16, 8, heads=4)),
    }[name]
    mine, ref = mine.double().eval(), ref.double().eval()
    ref.load_state_dict(mine.state_dict())
    with torch.no_grad():
        assert (mine(x, ei) - ref(x, ei)).abs().max() < 1e-10             # correct code passes
        assert (mine(x, ei.flip(0)) - ref(x, ei)).abs().max() > 1e-3      # swapped code is caught
