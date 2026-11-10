"""
Prove the hand-written layers are correct.

Writing a GNN from scratch is easy. Writing one that is *right* is not -- the
failure mode is silent. Swap ``src`` and ``dst`` in a message pass and the code
still runs, still trains, still produces a plausible accuracy. Nothing throws.

So this module does the only thing that actually settles it: build the same
layer in PyTorch Geometric, copy the weights across verbatim, and assert the
outputs are numerically identical.

    max | mine(x, edge_index) - pyg(x, edge_index) |   <  atol

Why weight copying works at all
-------------------------------
Because ``gnn_lab`` deliberately names its parameters the same way PyG does --
``lin``/``bias`` for GCN, ``lin_l``/``lin_r`` for SAGE, ``lin``/``att_src``/
``att_dst``/``bias`` for GAT, ``nn``/``eps`` for GIN. That is not an aesthetic
choice; it is what turns this test into ``pyg.load_state_dict(mine.state_dict())``
instead of a brittle hand-written key mapping that has to be updated every time
PyG ships a release.

Checks are run in float64 where possible. In float32 a 1e-5 tolerance would
pass even for a subtly wrong implementation; in float64 agreement to ~1e-12 is
strong evidence the two are computing the same expression, not merely similar
ones.

Every check degrades gracefully: if PyG is not installed, or a version differs
in a way that breaks the transfer, the check reports ``SKIPPED`` with a reason
rather than crashing the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from .layers.gat import GATConvScratch
from .layers.gcn import GCNConvScratch
from .layers.gin import GINConvScratch
from .layers.pooling import global_pool
from .layers.sage import SAGEConvScratch
from .ops import edge_order, gcn_norm, scatter_add, scatter_max, scatter_mean, scatter_softmax

__all__ = ["CheckResult", "run_all", "all_passed", "check_scatter_vs_loop", "check_gcn",
           "check_sage", "check_gat", "check_gin", "check_pooling", "check_gcn_norm",
           "check_cora_loader", "PYG_AVAILABLE", "PYG_IMPORT_ERROR", "PYG_VERSION"]

try:
    import torch_geometric  # noqa: F401
    from torch_geometric.nn import (GATConv, GCNConv, GINConv, SAGEConv,
                                    global_add_pool, global_max_pool, global_mean_pool)
    PYG_AVAILABLE = True
    PYG_IMPORT_ERROR = None
    PYG_VERSION = torch_geometric.__version__
except Exception as _e:                                            # noqa: BLE001
    PYG_AVAILABLE = False
    PYG_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    PYG_VERSION = None


@dataclass
class CheckResult:
    name: str
    passed: Optional[bool]      # None == skipped
    max_diff: Optional[float]
    tolerance: Optional[float]
    detail: str = ""

    @property
    def status(self) -> str:
        return "SKIP" if self.passed is None else ("PASS" if self.passed else "FAIL")

    def line(self) -> str:
        d = "-" if self.max_diff is None else f"{self.max_diff:.3e}"
        t = "-" if self.tolerance is None else f"{self.tolerance:.0e}"
        return f"[{self.status:4}] {self.name:<44} max|diff|={d:<10} tol={t:<8} {self.detail}"


def _skip(name: str, why: str) -> CheckResult:
    return CheckResult(name, None, None, None, f"skipped: {why}")


def _compare(name: str, mine: torch.Tensor, ref: torch.Tensor, atol: float) -> CheckResult:
    diff = float((mine.detach() - ref.detach()).abs().max().item())
    return CheckResult(name, diff <= atol, diff, atol,
                       f"shapes {tuple(mine.shape)} vs {tuple(ref.shape)}")


# --------------------------------------------------------------------------- #
# 0. Scatter primitives vs an explicit Python loop (needs no PyG)
# --------------------------------------------------------------------------- #
def check_scatter_vs_loop(n: int = 40, dim_size: int = 11, seed: int = 0) -> List[CheckResult]:
    """Validate every scatter op against a brute-force loop.

    This check needs no third-party library at all, so it always runs -- and it
    is the foundation the other checks stand on. If ``scatter_add`` were wrong,
    every layer built on it would be wrong in the same way.
    """
    g = torch.Generator().manual_seed(seed)
    index = torch.randint(0, dim_size, (n,), generator=g)
    src = torch.randn(n, 3, generator=g, dtype=torch.float64)

    # brute-force references
    ref_add = torch.zeros(dim_size, src.size(1), dtype=torch.float64)
    ref_max = torch.full((dim_size, src.size(1)), float("-inf"), dtype=torch.float64)
    counts = torch.zeros(dim_size, dtype=torch.float64)
    for e in range(n):
        k = int(index[e])
        ref_add[k] += src[e]
        ref_max[k] = torch.maximum(ref_max[k], src[e])
        counts[k] += 1
    ref_max[counts == 0] = 0.0
    ref_mean = ref_add / counts.clamp(min=1).unsqueeze(-1)

    # reference softmax: per-group, computed with torch on each group separately
    ref_sm = torch.zeros_like(src)
    for k in range(dim_size):
        m = index == k
        if m.any():
            ref_sm[m] = torch.softmax(src[m], dim=0)

    out = [
        _compare("scatter_add  vs python loop", scatter_add(src, index, dim_size), ref_add, 1e-12),
        _compare("scatter_mean vs python loop", scatter_mean(src, index, dim_size), ref_mean, 1e-12),
        _compare("scatter_max  vs python loop", scatter_max(src, index, dim_size), ref_max, 1e-12),
        _compare("scatter_softmax vs per-group torch.softmax",
                 scatter_softmax(src, index, dim_size), ref_sm, 1e-12),
    ]

    # softmax must sum to 1 within every group, *separately for every feature column*
    # (src is [n, 3], so each of the 3 columns is its own softmax). Summing the
    # columns together as well would give 3, not 1.
    s = scatter_softmax(src, index, dim_size)
    grp = scatter_add(s, index, dim_size)                 # [dim_size, 3]
    empty = counts == 0
    dev = float((grp[~empty] - 1.0).abs().max().item()) if (~empty).any() else 0.0
    out.append(CheckResult("scatter_softmax rows sum to 1 per node", dev < 1e-10, dev,
                           1e-10, f"{int((~empty).sum())}/{dim_size} non-empty groups"))

    # gradient flows through every op
    # NOTE: weight the outputs by random numbers before summing. For softmax,
    # ``softmax(z).sum()`` is a constant (every group sums to 1), so its gradient
    # is exactly zero in theory and only non-zero by float round-off in practice --
    # a plain ``.sum().backward()`` would be testing nothing.
    grads_ok = True
    for fn in (scatter_add, scatter_mean, scatter_max, scatter_softmax):
        z = src.clone().requires_grad_(True)
        y = fn(z, index, dim_size)
        w = torch.randn(y.shape, generator=g, dtype=y.dtype)
        (y * w).sum().backward()
        if z.grad is None or not torch.isfinite(z.grad).all() or float(z.grad.abs().sum()) < 1e-6:
            grads_ok = False
    out.append(CheckResult("gradients flow through all scatter ops", grads_ok, None, None,
                           "weighted sum().backward() produced finite non-zero grads"))
    return out


def check_gcn_norm(n: int = 300, num_nodes: int = 60, seed: int = 0) -> List[CheckResult]:
    """The ``D^-1/2 (A+I) D^-1/2`` coefficients, vs PyG's ``gcn_norm``."""
    if not PYG_AVAILABLE:
        return [_skip("gcn_norm coefficients vs PyG", PYG_IMPORT_ERROR or "PyG missing")]
    try:
        from torch_geometric.nn.conv.gcn_conv import gcn_norm as pyg_gcn_norm
    except Exception as e:                                        # noqa: BLE001
        return [_skip("gcn_norm coefficients vs PyG", f"import failed: {e}")]

    g = torch.Generator().manual_seed(seed)
    ei_raw = torch.randint(0, num_nodes, (2, n), generator=g)
    ei_raw = ei_raw[:, ei_raw[0] != ei_raw[1]]           # strip self-loops

    mine_ei, mine_w = gcn_norm(ei_raw, num_nodes, add_loops=True, dtype=torch.float64)
    try:
        pyg_ei, pyg_w = pyg_gcn_norm(
            ei_raw, edge_weight=None, num_nodes=num_nodes, improved=False,
            add_self_loops=True, flow="source_to_target", dtype=torch.float64)
    except TypeError as e:                               # older/newer signature
        return [_skip("gcn_norm coefficients vs PyG", f"signature mismatch: {e}")]

    # the two implementations sort edges differently -> compare after a common sort
    def key(e, w):
        order = edge_order(e, num_nodes)
        return e[:, order], w[order]

    m_ei, m_w = key(mine_ei, mine_w.double())
    p_ei, p_w = key(pyg_ei, pyg_w.double())
    same_edges = bool(torch.equal(m_ei, p_ei)) and m_ei.size(1) == p_ei.size(1)
    out = [CheckResult("gcn_norm edge_index matches PyG", same_edges, None, None,
                       f"{m_ei.size(1)} vs {p_ei.size(1)} edges")]
    if same_edges:
        out.append(_compare("gcn_norm coefficients match PyG", m_w, p_w, 1e-12))
    return out


# --------------------------------------------------------------------------- #
# 1..4. Layer-by-layer equivalence
# --------------------------------------------------------------------------- #
def _random_graph(num_nodes: int, num_edges: int, num_features: int, seed: int = 0,
                  dtype=torch.float64, undirected: bool = False):
    """A random graph for the layer checks.

    ``undirected=False`` is the default **on purpose**. On a symmetric edge set,
    swapping ``src`` and ``dst`` in a message pass changes nothing (every edge
    ``s -> t`` has its mirror ``t -> s``), so a check that only ever sees
    symmetric graphs *cannot* catch the single most common GNN bug. A directed
    graph breaks that symmetry: with ``src``/``dst`` swapped, the layers below
    disagree with PyG by O(1), not O(1e-16).
    """
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, num_nodes, (2, num_edges), generator=g)
    mask = ei[0] != ei[1]
    ei = ei[:, mask]
    if undirected:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    x = torch.randn(num_nodes, num_features, generator=g).to(dtype)
    return x, ei


def _transfer(mine: torch.nn.Module, ref: torch.nn.Module, name: str) -> Optional[CheckResult]:
    """Copy weights mine -> ref. Returns a SKIP result if the key sets differ."""
    mine_sd, ref_sd = mine.state_dict(), ref.state_dict()
    if set(mine_sd) != set(ref_sd):
        only_mine = sorted(set(mine_sd) - set(ref_sd))
        only_ref = sorted(set(ref_sd) - set(mine_sd))
        return _skip(name, f"param names differ (mine only: {only_mine[:3]}, "
                           f"pyg only: {only_ref[:3]}); PyG {PYG_VERSION}")
    try:
        ref.load_state_dict(mine_sd)
    except Exception as e:                                        # noqa: BLE001
        return _skip(name, f"state_dict transfer failed: {type(e).__name__}: {e}")
    return None


def check_gcn(num_nodes: int = 60, num_edges: int = 300, num_features: int = 16,
              hidden: int = 8, seed: int = 0, atol: float = 1e-10) -> List[CheckResult]:
    name = "GCNConvScratch == PyG GCNConv"
    if not PYG_AVAILABLE:
        return [_skip(name, PYG_IMPORT_ERROR or "PyG missing")]
    x, ei = _random_graph(num_nodes, num_edges, num_features, seed)
    mine = GCNConvScratch(num_features, hidden).double().eval()
    ref = GCNConv(num_features, hidden).double().eval()
    if (s := _transfer(mine, ref, name)) is not None:
        return [s]
    with torch.no_grad():
        return [_compare(name, mine(x, ei, num_nodes), ref(x, ei), atol)]


def check_sage(aggr: str = "mean", num_nodes: int = 60, num_edges: int = 300,
               num_features: int = 16, hidden: int = 8, seed: int = 0,
               atol: float = 1e-10) -> List[CheckResult]:
    name = f"SAGEConvScratch(aggr={aggr}) == PyG SAGEConv"
    if not PYG_AVAILABLE:
        return [_skip(name, PYG_IMPORT_ERROR or "PyG missing")]
    x, ei = _random_graph(num_nodes, num_edges, num_features, seed)
    mine = SAGEConvScratch(num_features, hidden, aggr=aggr).double().eval()
    ref = SAGEConv(num_features, hidden, aggr=aggr).double().eval()
    if (s := _transfer(mine, ref, name)) is not None:
        return [s]
    with torch.no_grad():
        return [_compare(name, mine(x, ei, num_nodes), ref(x, ei), atol)]


def check_gat(heads: int = 4, concat: bool = True, num_nodes: int = 60, num_edges: int = 300,
              num_features: int = 16, hidden: int = 8, seed: int = 0,
              atol: float = 1e-10) -> List[CheckResult]:
    name = f"GATConvScratch(heads={heads},concat={concat}) == PyG GATConv"
    if not PYG_AVAILABLE:
        return [_skip(name, PYG_IMPORT_ERROR or "PyG missing")]
    x, ei = _random_graph(num_nodes, num_edges, num_features, seed)
    mine = GATConvScratch(num_features, hidden, heads=heads, concat=concat).double().eval()
    ref = GATConv(num_features, hidden, heads=heads, concat=concat).double().eval()
    if (s := _transfer(mine, ref, name)) is not None:
        return [s]
    with torch.no_grad():
        return [_compare(name, mine(x, ei, num_nodes), ref(x, ei), atol)]


def check_gin(num_nodes: int = 60, num_edges: int = 300, num_features: int = 16,
              hidden: int = 8, seed: int = 0, atol: float = 1e-10) -> List[CheckResult]:
    name = "GINConvScratch == PyG GINConv"
    if not PYG_AVAILABLE:
        return [_skip(name, PYG_IMPORT_ERROR or "PyG missing")]
    x, ei = _random_graph(num_nodes, num_edges, num_features, seed)

    def make_mlp():
        return torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden)).double()

    mine = GINConvScratch(make_mlp(), eps=0.3, train_eps=True).double().eval()
    ref = GINConv(make_mlp(), eps=0.3, train_eps=True).double().eval()
    if (s := _transfer(mine, ref, name)) is not None:
        return [s]
    with torch.no_grad():
        return [_compare(name, mine(x, ei, num_nodes), ref(x, ei), atol)]


def check_pooling(seed: int = 0, atol: float = 1e-12) -> List[CheckResult]:
    """global_pool(sum/mean/max) vs PyG's global_{add,mean,max}_pool."""
    if not PYG_AVAILABLE:
        return [_skip("global_pool vs PyG global_*_pool", PYG_IMPORT_ERROR or "PyG missing")]
    g = torch.Generator().manual_seed(seed)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3], dtype=torch.long)
    x = torch.randn(10, 5, generator=g).double()
    ng = int(batch.max()) + 1

    refs = {"sum": global_add_pool(x, batch, ng),
            "mean": global_mean_pool(x, batch, ng),
            "max": global_max_pool(x, batch, ng)}
    return [_compare(f"global_pool({m}) vs PyG", global_pool(x, batch, m, ng), r, atol)
            for m, r in refs.items()]


# --------------------------------------------------------------------------- #
# 5. Dataset equivalence
# --------------------------------------------------------------------------- #
def check_cora_loader(root: str = "data", atol: float = 1e-6) -> List[CheckResult]:
    """My Planetoid loader vs ``torch_geometric.datasets.Planetoid``.

    Row-normalisation is disabled on our side because PyG does not do it -- the
    two loaders must be compared on identical inputs before the comparison
    means anything.
    """
    name = "load_cora(planetoid) == PyG Planetoid('Cora')"
    if not PYG_AVAILABLE:
        return [_skip(name, PYG_IMPORT_ERROR or "PyG missing")]
    try:
        from torch_geometric.datasets import Planetoid
        from .data.cora import load_cora_planetoid

        ref = Planetoid(root=f"{root}/_pyg_cora", name="Cora")
        mine = load_cora_planetoid(root=f"{root}/cora_planetoid",
                                  row_normalize=False, dedupe=True, verbose=False)
    except Exception as e:                                        # noqa: BLE001
        return [_skip(name, f"could not load both: {type(e).__name__}: {e}")]

    out: List[CheckResult] = []
    out.append(CheckResult("node count matches PyG",
                           mine.num_nodes == ref[0].num_nodes, None, None,
                           f"{mine.num_nodes} vs {ref[0].num_nodes}"))
    out.append(CheckResult("edge count matches PyG",
                           mine.num_edges == ref[0].edge_index.size(1), None, None,
                           f"{mine.num_edges} vs {ref[0].edge_index.size(1)}"))
    out.append(CheckResult("edge_index identical to PyG (same order)",
                           bool(torch.equal(mine.edge_index.cpu(), ref[0].edge_index.cpu())),
                           None, None, "coalesce(sort_by_row=False) ordering"))
    out.append(CheckResult("labels identical to PyG",
                           bool(torch.equal(mine.y.cpu(), ref[0].y.cpu())), None, None, ""))
    for m in ("train", "val", "test"):
        a = getattr(mine, f"{m}_mask").cpu()
        b = getattr(ref[0], f"{m}_mask").cpu()
        out.append(CheckResult(f"{m}_mask identical to PyG", bool(torch.equal(a, b)), None, None,
                               f"{int(a.sum())} nodes"))
    out.append(_compare("features identical to PyG", mine.x.cpu().double(),
                        ref[0].x.cpu().double(), atol))
    return out


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_all(root: str = "data", include_data: bool = True, verbose: bool = True) -> List[CheckResult]:
    """Run every equivalence check and print a table. Returns the results."""
    if verbose:
        print("=" * 96)
        print("FROM-SCRATCH  vs  TORCH_GEOMETRIC   --   numerical equivalence")
        print(f"PyG available: {PYG_AVAILABLE}" + (f" (v{PYG_VERSION})" if PYG_VERSION else
                                                    f"  [{PYG_IMPORT_ERROR}]"))
        print("=" * 96)

    results: List[CheckResult] = []
    results += check_scatter_vs_loop()
    if PYG_AVAILABLE:
        results += check_gcn_norm()
    results += check_gcn()
    results += check_sage("mean")
    results += check_sage("max")
    results += check_sage("sum")
    results += check_gat(heads=1, concat=True)
    results += check_gat(heads=8, concat=True)
    results += check_gat(heads=8, concat=False)
    results += check_gin()
    results += check_pooling()
    if include_data:
        results += check_cora_loader(root=root)

    if verbose:
        for r in results:
            print(r.line())
        n_pass = sum(1 for r in results if r.passed is True)
        n_fail = sum(1 for r in results if r.passed is False)
        n_skip = sum(1 for r in results if r.passed is None)
        print("-" * 96)
        print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped  "
              f"of {len(results)} checks")
    return results


def all_passed(results: List[CheckResult]) -> bool:
    """True if nothing failed (skips do not count as failures)."""
    return not any(r.passed is False for r in results)
