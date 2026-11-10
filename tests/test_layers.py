"""
Layer tests with **hand-computed** expected outputs.

The fixture is ``toy_graph()``::

    0 -- 1 -- 2
    |         |
    3 ------- 4

with features ``x = [[1,0],[0,1],[1,1],[2,0],[0,2]]``.

Two properties make it ideal for testing:

* Every node has degree exactly 2. With self-loops that becomes 3, so the GCN
  coefficient ``1/sqrt(deg_i * deg_j)`` is ``1/3`` for **every** edge. No
  per-node arithmetic, no floating-point wiggle.
* The features are small integers, so every expected value below was written
  down on paper before the test was run. If the test passes, the layer matches
  its mathematical definition -- not merely another implementation of it.

Expected values (weights set to identity, biases to zero):

    GCN  (1/3) * sum over {self} U neighbours
        node 0: (x0+x1+x3)/3 = [3,1]/3
        node 1: (x1+x0+x2)/3 = [2,2]/3
        node 2: (x2+x1+x4)/3 = [1,4]/3
        node 3: (x3+x0+x4)/3 = [3,2]/3
        node 4: (x4+x2+x3)/3 = [3,3]/3

    SAGE mean (no self-loop; separate self branch zeroed)
        node 0: mean(x1,x3) = [1.0, 0.5]
        node 1: mean(x0,x2) = [1.0, 0.5]
        node 2: mean(x1,x4) = [0.0, 1.5]
        node 3: mean(x0,x4) = [0.5, 1.0]
        node 4: mean(x2,x3) = [1.5, 0.5]

    GIN, eps = 0, MLP = Identity
        node 0: x0+x1+x3 = [3,1]
        node 1: x1+x0+x2 = [2,2]
        node 2: x2+x1+x4 = [1,4]
        node 3: x3+x0+x4 = [3,2]
        node 4: x4+x2+x3 = [3,3]

    GAT with att_src = att_dst = 0 -> uniform 1/3 attention -> identical to GCN
        (on a degree-regular graph, uniform attention and symmetric GCN
         normalisation coincide; that equivalence is itself worth asserting)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gnn_lab.data.synthetic import toy_graph
from gnn_lab.layers.gat import GATConvScratch
from gnn_lab.layers.gcn import GCNConvScratch, GCNScratch
from gnn_lab.layers.gin import GINConvScratch
from gnn_lab.layers.pooling import collate_graphs, global_pool
from gnn_lab.layers.sage import SAGEConvScratch

DT = torch.float64
ATOL = 1e-10


# --------------------------------------------------------------------------- #
# fixture
# --------------------------------------------------------------------------- #
def _toy(dtype=DT):
    g = toy_graph()
    return g["x"].to(dtype), g["edge_index"]


def _identity(layer: nn.Module) -> nn.Module:
    """Set every linear map to the identity and every bias to zero."""
    layer = layer.double().eval()
    for m in layer.modules():
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.copy_(torch.eye(m.out_features, m.in_features, dtype=DT))
                if m.bias is not None:
                    m.bias.zero_()
    for name, p in layer.named_parameters():
        if name.endswith("bias") and p.dim() == 1:
            with torch.no_grad():
                p.zero_()
    return layer


GCN_EXPECTED = torch.tensor([[3, 1], [2, 2], [1, 4], [3, 2], [3, 3]], dtype=DT) / 3.0
SAGE_MEAN_EXPECTED = torch.tensor([[1.0, 0.5], [1.0, 0.5], [0.0, 1.5],
                                   [0.5, 1.0], [1.5, 0.5]], dtype=DT)
GIN_EXPECTED = torch.tensor([[3, 1], [2, 2], [1, 4], [3, 2], [3, 3]], dtype=DT)


# --------------------------------------------------------------------------- #
# GCN
# --------------------------------------------------------------------------- #
def test_gcn_matches_hand_computed_values():
    x, ei = _toy()
    conv = _identity(GCNConvScratch(2, 2))
    with torch.no_grad():
        out = conv(x, ei, num_nodes=5)
    assert torch.allclose(out, GCN_EXPECTED, atol=1e-9), f"\n{out}\nvs\n{GCN_EXPECTED}"


def test_gcn_without_normalization_is_plain_sum_with_self_loop():
    x, ei = _toy()
    conv = _identity(GCNConvScratch(2, 2, normalize=False))
    with torch.no_grad():
        out = conv(x, ei, num_nodes=5)
    assert torch.allclose(out, GIN_EXPECTED, atol=ATOL)


def test_gcn_without_self_loops_excludes_the_node_itself():
    x, ei = _toy()
    conv = _identity(GCNConvScratch(2, 2, normalize=False, add_self_loops=False))
    with torch.no_grad():
        out = conv(x, ei, num_nodes=5)
    # sum over strict neighbours only = GIN_EXPECTED - x
    assert torch.allclose(out, GIN_EXPECTED - x, atol=ATOL)


def test_gcn_bias_is_added_after_propagation():
    x, ei = _toy()
    conv = _identity(GCNConvScratch(2, 2))
    with torch.no_grad():
        conv.bias.copy_(torch.tensor([10.0, -5.0], dtype=DT))
        out = conv(x, ei, num_nodes=5)
    assert torch.allclose(out, GCN_EXPECTED + torch.tensor([10.0, -5.0], dtype=DT), atol=1e-9)


def test_gcn_cached_norm_matches_uncached():
    x, ei = _toy()
    a = _identity(GCNConvScratch(2, 2, cached_norm=False))
    b = _identity(GCNConvScratch(2, 2, cached_norm=True))
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        o1 = a(x, ei, num_nodes=5)
        o2 = b(x, ei, num_nodes=5)
        o3 = b(x, ei, num_nodes=5)      # second call must hit the cache
    assert torch.allclose(o1, o2, atol=ATOL) and torch.allclose(o1, o3, atol=ATOL)


def test_gcn_gradient_flows_to_every_parameter():
    x, ei = _toy()
    conv = GCNConvScratch(2, 3).double()
    conv(x, ei, num_nodes=5).sum().backward()
    for name, p in conv.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


# --------------------------------------------------------------------------- #
# GraphSAGE
# --------------------------------------------------------------------------- #
def test_sage_mean_matches_hand_computed_values():
    x, ei = _toy()
    conv = _identity(SAGEConvScratch(2, 2, aggr="mean"))
    with torch.no_grad():
        conv.lin_r.weight.zero_()          # isolate the neighbour branch
        out = conv(x, ei, num_nodes=5)
    assert torch.allclose(out, SAGE_MEAN_EXPECTED, atol=ATOL), f"\n{out}"


def test_sage_sum_is_degree_times_mean():
    x, ei = _toy()
    mean = _identity(SAGEConvScratch(2, 2, aggr="mean"))
    summ = _identity(SAGEConvScratch(2, 2, aggr="sum"))
    with torch.no_grad():
        mean.lin_r.weight.zero_(); summ.lin_r.weight.zero_()
        o_mean = mean(x, ei, num_nodes=5)
        o_sum = summ(x, ei, num_nodes=5)
    # every node has exactly 2 neighbours on this graph
    assert torch.allclose(o_sum, 2.0 * o_mean, atol=ATOL)


def test_sage_max_takes_elementwise_maximum_of_neighbours():
    x, ei = _toy()
    conv = _identity(SAGEConvScratch(2, 2, aggr="max"))
    with torch.no_grad():
        conv.lin_r.weight.zero_()
        out = conv(x, ei, num_nodes=5)
    expected = torch.tensor([[2.0, 1.0],   # node 0: max(x1,x3) = max([0,1],[2,0])
                             [1.0, 1.0],   # node 1: max(x0,x2) = max([1,0],[1,1])
                             [0.0, 2.0],   # node 2: max(x1,x4) = max([0,1],[0,2])
                             [1.0, 2.0],   # node 3: max(x0,x4) = max([1,0],[0,2])
                             [2.0, 1.0]],  # node 4: max(x2,x3) = max([1,1],[2,0])
                            dtype=DT)
    assert torch.allclose(out, expected, atol=ATOL), f"\n{out}"


def test_sage_root_weight_adds_the_node_itself():
    x, ei = _toy()
    conv = _identity(SAGEConvScratch(2, 2, aggr="mean", root_weight=True))
    conv_no_root = _identity(SAGEConvScratch(2, 2, aggr="mean", root_weight=False))
    with torch.no_grad():
        out_with = conv(x, ei, num_nodes=5)
        out_without = conv_no_root(x, ei, num_nodes=5)
    assert torch.allclose(out_with - out_without, x, atol=ATOL)


def test_sage_rejects_unknown_aggregator():
    try:
        SAGEConvScratch(2, 2, aggr="median")
    except ValueError:
        return
    raise AssertionError("SAGEConvScratch accepted an invalid aggregator")


# --------------------------------------------------------------------------- #
# GIN
# --------------------------------------------------------------------------- #
def test_gin_with_zero_eps_and_identity_mlp_is_pure_sum():
    x, ei = _toy()
    conv = GINConvScratch(nn.Identity(), eps=0.0, train_eps=False).double().eval()
    with torch.no_grad():
        out = conv(x, ei, num_nodes=5)
    assert torch.allclose(out, GIN_EXPECTED, atol=ATOL), f"\n{out}"


def test_gin_eps_scales_the_self_term():
    x, ei = _toy()
    conv = GINConvScratch(nn.Identity(), eps=2.0, train_eps=False).double().eval()
    with torch.no_grad():
        out = conv(x, ei, num_nodes=5)
    # (1 + eps) * x_i + sum_{j in N(i)} x_j  with eps = 2 -> 3*x + (sum - x)
    assert torch.allclose(out, 3.0 * x + (GIN_EXPECTED - x), atol=ATOL)


def test_gin_eps_is_trainable_when_requested():
    conv = GINConvScratch(nn.Linear(2, 2), eps=0.0, train_eps=True)
    assert isinstance(conv.eps, nn.Parameter)
    frozen = GINConvScratch(nn.Linear(2, 2), eps=0.0, train_eps=False)
    assert not isinstance(frozen.eps, nn.Parameter)


# --------------------------------------------------------------------------- #
# GAT
# --------------------------------------------------------------------------- #
def test_gat_with_zero_attention_vectors_equals_gcn_on_regular_graph():
    """Uniform attention over {self} U N(i) is exactly the GCN average here."""
    x, ei = _toy()
    conv = _identity(GATConvScratch(2, 2, heads=1, concat=True))
    with torch.no_grad():
        conv.att_src.zero_()
        conv.att_dst.zero_()
        out = conv(x, ei, num_nodes=5)
    assert torch.allclose(out, GCN_EXPECTED, atol=1e-9), f"\n{out}"


def test_gat_attention_rows_sum_to_one():
    x, ei = _toy()
    conv = GATConvScratch(2, 2, heads=4, concat=True).double().eval()
    conv.return_attention = True
    with torch.no_grad():
        _, att = conv(x, ei, num_nodes=5)

    # self-loops are appended inside forward, so the attention covers E + N edges
    assert att.shape == (ei.size(1) + 5, 4)

    dst_with_loops = torch.cat([ei[1], torch.arange(5, dtype=ei.dtype)])
    totals = torch.zeros(5, 4, dtype=DT).index_add_(0, dst_with_loops, att)
    assert torch.allclose(totals, torch.ones(5, 4, dtype=DT), atol=1e-10)


def test_gat_output_dimension_follows_concat_flag():
    x, ei = _toy()
    for concat, expected in ((True, 3 * 4), (False, 4)):
        conv = GATConvScratch(2, 4, heads=3, concat=concat).double().eval()
        with torch.no_grad():
            out = conv(x, ei, num_nodes=5)
        assert out.shape == (5, expected), f"concat={concat} gave {out.shape}"


def test_gat_bias_size_matches_concat_choice():
    """Easy to get backwards: concat -> heads*out, else -> out."""
    assert GATConvScratch(2, 4, heads=3, concat=True).bias.numel() == 12
    assert GATConvScratch(2, 4, heads=3, concat=False).bias.numel() == 4


def test_gat_uses_a_single_shared_projection_by_default():
    conv = GATConvScratch(2, 4, heads=3)
    assert conv.lin_src is conv.lin_dst
    keys = set(conv.state_dict())
    assert "lin.weight" in keys
    assert "lin_src.weight" not in keys and "lin_dst.weight" not in keys


def test_gat_separate_lin_creates_two_projections():
    conv = GATConvScratch(2, 4, heads=3, separate_lin=True)
    assert conv.lin_src is not conv.lin_dst
    assert {"lin_src.weight", "lin_dst.weight"} <= set(conv.state_dict())


# --------------------------------------------------------------------------- #
# Pooling and batching
# --------------------------------------------------------------------------- #
def test_global_pool_hand_computed():
    x = torch.tensor([[1.0, 1.0],
                      [3.0, 3.0],
                      [5.0, 5.0],
                      [10.0, 10.0]], dtype=DT)
    batch = torch.tensor([0, 0, 1, 1])
    assert torch.allclose(global_pool(x, batch, "sum"), torch.tensor([[4.0, 4.0], [15.0, 15.0]], dtype=DT))
    assert torch.allclose(global_pool(x, batch, "mean"), torch.tensor([[2.0, 2.0], [7.5, 7.5]], dtype=DT))
    assert torch.allclose(global_pool(x, batch, "max"), torch.tensor([[3.0, 3.0], [10.0, 10.0]], dtype=DT))


def test_collate_graphs_offsets_are_correct():
    """The block-diagonal trick: graph k's edges shift by the running node count."""
    gA = {"x": torch.zeros(3, 2, dtype=DT),
          "edge_index": torch.tensor([[0, 1], [1, 2]]),
          "y": torch.tensor(0)}
    gB = {"x": torch.ones(2, 2, dtype=DT),
          "edge_index": torch.tensor([[0], [1]]),
          "y": torch.tensor(1)}
    b = collate_graphs([gA, gB])

    assert b.num_nodes == 5 and b.num_graphs == 2
    assert b.batch.tolist() == [0, 0, 0, 1, 1]
    assert b.x.shape == (5, 2)
    # gB's single edge (0->1) must become (3->4)
    assert b.edge_index.t().tolist() == [[0, 1], [1, 2], [3, 4]]
    assert b.y.tolist() == [0, 1]


def test_collated_graphs_cannot_exchange_information():
    """Message passing on a batch must equal per-graph message passing exactly."""
    g = torch.Generator().manual_seed(11)
    graphs = []
    for _ in range(4):
        n = int(torch.randint(4, 9, (1,), generator=g).item())
        ei = torch.randint(0, n, (2, 3 * n), generator=g)
        ei = ei[:, ei[0] != ei[1]]
        ei = torch.cat([ei, ei.flip(0)], dim=1)
        graphs.append({"x": torch.randn(n, 5, generator=g, dtype=DT),
                       "edge_index": ei, "y": torch.tensor(0)})

    conv = GCNConvScratch(5, 5).double().eval()
    batched = collate_graphs(graphs)
    with torch.no_grad():
        joint = conv(batched.x, batched.edge_index, batched.num_nodes)
        for k, gr in enumerate(graphs):
            alone = conv(gr["x"], gr["edge_index"], gr["x"].size(0))
            sel = (batched.batch == k)
            assert torch.allclose(joint[sel], alone, atol=1e-12), f"graph {k} leaked information"


# --------------------------------------------------------------------------- #
# Full models
# --------------------------------------------------------------------------- #
def test_gcn_model_output_shape_and_no_nan():
    x, ei = _toy()
    model = GCNScratch(2, 8, 3, num_layers=2, dropout=0.0).double().eval()
    with torch.no_grad():
        out = model(x, ei)
    assert out.shape == (5, 3)
    assert torch.isfinite(out).all()


def test_deep_gcn_still_produces_finite_output():
    x, ei = _toy()
    model = GCNScratch(2, 8, 3, num_layers=16, dropout=0.0).double().eval()
    with torch.no_grad():
        out = model(x, ei)
    assert torch.isfinite(out).all(), "16-layer GCN produced inf/nan"
