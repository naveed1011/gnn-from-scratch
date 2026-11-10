#!/usr/bin/env python3
"""
Build ``kaggle/gnn_from_scratch_t4.ipynb`` from the ``gnn_lab`` package source.

Why a generator instead of a hand-written notebook
---------------------------------------------------
The repo has two audiences: chaptered notebooks under ``notebooks/`` that
``import gnn_lab`` for local study, and one self-contained notebook under
``kaggle/`` that must run on a fresh Kaggle T4 kernel with nothing but pip.

Hand-maintaining both means the Kaggle copy drifts the moment you fix a bug in
the package -- and a drifted copy that still *runs* is worse than one that
fails loudly. So the Kaggle notebook is generated. Edit the package, re-run
this script, and the notebook is correct by construction.

    python scripts/build_kaggle_notebook.py

Inlining rules
--------------
Each module becomes one code cell. Inside it we strip:

* the module docstring   -> re-emitted as the markdown cell above it
* ``from __future__``    -> re-added as the first line of every cell (IPython
  compiles each cell independently, so a future import must lead each one)
* relative imports       -> ``from ..ops import scatter_add`` is meaningless once
  everything shares one flat notebook namespace
* top-level stdlib/torch imports -> hoisted into a single header cell

Indented (deferred) imports are kept: ``import networkx`` inside a plotting
function must stay where it is.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "gnn_lab")
OUT = os.path.join(ROOT, "kaggle", "gnn_from_scratch_t4.ipynb")

# Dependency order: a module may only reference names defined above it.
MODULES = [
    "ops.py",
    "utils.py",
    "metrics.py",
    "layers/gcn.py",
    "layers/sage.py",
    "layers/gat.py",
    "layers/pooling.py",   # before gin.py: GINScratch.forward calls global_pool
    "layers/gin.py",
    "data/tu.py",
    "data/synthetic.py",
    "data/cora.py",
    "train.py",
    "viz.py",
    "equivalence.py",
]

MODULE_TITLES = {
    "ops.py": "Scatter primitives -- the engine of message passing",
    "utils.py": "Seeding and device helpers",
    "metrics.py": "Metrics, including the over-smoothing diagnostics",
    "layers/gcn.py": "GCN layer",
    "layers/sage.py": "GraphSAGE layer",
    "layers/gat.py": "GAT layer",
    "layers/gin.py": "GIN layer",
    "layers/pooling.py": "Batching and pooling",
    "data/tu.py": "TU-format loader (MUTAG / PROTEINS)",
    "data/synthetic.py": "Small synthetic graphs for teaching and testing",
    "data/cora.py": "Cora loaders (Planetoid + LINQS)",
    "train.py": "Training loops",
    "viz.py": "Plotting helpers",
    "equivalence.py": "Equivalence checks against torch_geometric",
}


# --------------------------------------------------------------------------- #
# notebook plumbing
# --------------------------------------------------------------------------- #
def _cell_id(seed: str) -> str:
    return hashlib.sha1(seed.encode()).hexdigest()[:12]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id("md" + text[:80]),
            "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": _cell_id("code" + text[:80]),
            "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


# --------------------------------------------------------------------------- #
# source inlining
# --------------------------------------------------------------------------- #
REL_IMPORT = re.compile(r"^\s*from\s+\.+\S*\s+import\b")
FUTURE = re.compile(r"^\s*from\s+__future__\s+import\b")
TOP_IMPORT = re.compile(r"^(import|from)\s+[\w\.]+")


def split_module(path: str) -> Tuple[str, List[str], List[str]]:
    """Return (docstring, hoisted_imports, body_lines)."""
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)

    doc = ""
    doc_end_line = 0
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        doc = tree.body[0].value.value.strip()
        doc_end_line = tree.body[0].end_lineno          # 1-indexed, inclusive

    hoisted: List[str] = []
    body: List[str] = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        if lineno <= doc_end_line:
            continue
        if FUTURE.match(line):
            continue
        if REL_IMPORT.match(line):
            continue
        if lineno > doc_end_line and TOP_IMPORT.match(line) and not line.startswith((" ", "\t")):
            hoisted.append(line.strip())
            continue
        body.append(line)

    text = "\n".join(body).strip("\n")
    return doc, hoisted, text.splitlines()


def build_library_cells() -> Tuple[List[dict], List[str]]:
    cells: List[dict] = []
    all_imports: List[str] = []

    for rel in MODULES:
        path = os.path.join(PKG, rel)
        if not os.path.exists(path):
            raise SystemExit(f"missing module {path}")
        doc, imports, body = split_module(path)
        all_imports.extend(imports)

        cells.append(md(f"#### `{rel}` &mdash; {MODULE_TITLES.get(rel, rel)}\n\n"
                        f"<details><summary>module docstring</summary>\n\n```\n{doc}\n```\n\n</details>"))
        src = "from __future__ import annotations\n" + "\n".join(body)
        cells.append(code(src))

    return cells, sorted(set(all_imports))


# --------------------------------------------------------------------------- #
# narrative + execution cells
# --------------------------------------------------------------------------- #
def narrative_and_runs() -> List[dict]:
    C: List[dict] = []
    A = C.append

    A(md("""## 0. Environment

Kaggle's T4 image already ships PyTorch, so the only extra install is
`torch_geometric`, and even that is optional -- it is used *solely* to verify
that the hand-written layers are numerically identical to a reference
implementation. Every model in this notebook trains without it."""))
    A(code("""import os, sys, platform, subprocess, time

print("python :", sys.version.split()[0])
print("platform:", platform.platform())

import torch
print("torch  :", torch.__version__)
print("cuda   :", torch.cuda.is_available(), torch.version.cuda or "")
if torch.cuda.is_available():
    print("gpu    :", torch.cuda.get_device_name(0))
    print("memory : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

assert sys.version_info >= (3, 10), (
    "This notebook uses PEP 604 unions (int | None) and needs Python >= 3.10.")"""))

    A(code("""# Optional: install PyTorch Geometric for the equivalence checks in section 5.
# Safe to skip -- everything else runs without it.
try:
    import torch_geometric
    print("torch_geometric already present:", torch_geometric.__version__)
except ImportError:
    print("installing torch_geometric ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"], check=False)
    try:
        import torch_geometric
        print("installed:", torch_geometric.__version__)
    except ImportError:
        print("not available -- equivalence checks will report SKIPPED")

for extra in ("networkx", "scikit-learn", "matplotlib"):
    try:
        __import__(extra)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", extra], check=False)"""))

    A(md("""---
# 1. A graph is three tensors

Before any neural network: what *is* the input? A graph is

| object | shape | meaning |
|---|---|---|
| `x` | `[N, F]` | one feature vector per node |
| `edge_index` | `[2, E]` | row 0 = **source** (sender), row 1 = **destination** (receiver) |
| `y` | `[N]` or `[num_graphs]` | labels |

`edge_index` is COO format: one column per *directed* edge. An undirected graph
stores both directions, so `E` is twice the number of edges you would draw.

That single `src`/`dst` distinction is where hand-written GNNs break. Messages
are gathered from `src` and scattered onto `dst`. Swap them and the shapes still
work, the model still trains, and the accuracy still looks plausible -- it is
just quietly wrong."""))
    A(code("""g = toy_graph()
x, edge_index = g["x"], g["edge_index"]

print("features x       :", tuple(x.shape))
print("edge_index       :", tuple(edge_index.shape))
print("  sources (row 0):", edge_index[0].tolist())
print("  targets (row 1):", edge_index[1].tolist())
print()
print("The graph is:")
print("    0 -- 1 -- 2")
print("    |         |")
print("    3 ------- 4")
print()
print("features per node:")
for i, row in enumerate(x.tolist()):
    print(f"  node {i}: {row}")

deg = degree(edge_index[1], num_nodes=5)
print("\\ndegree of every node:", deg.tolist(), "-> 2-regular, which is what makes")
print("the hand-computed expected values in section 2 exact rather than approximate.")

draw_graph(edge_index, labels=g["y"], title="toy graph (5 nodes, 5 undirected edges)",
           path=f"{RESULTS}/01_toy_graph.png")"""))

    A(md("""---
# 2. Message passing, done by hand

A graph convolution is an **unsorted group-by**. For every node, gather the
messages arriving from its neighbours and combine them. PyTorch spells that as
`index_add_`. Here is a full GCN layer with no classes, no libraries and no
abstraction -- five lines:""" ))
    A(code("""num_nodes = x.size(0)

# --- step 0: the GCN normalisation coefficients -------------------------
# add self-loops, then scale each edge by 1/sqrt(deg_dst * deg_src)
ei, norm = gcn_norm(edge_index, num_nodes)
print("edges after adding self-loops:", ei.size(1), "(10 original + 5 loops)")
print("norm coefficients (unique)   :", sorted({round(float(v), 6) for v in norm}))
print("  -> all 1/3, because every node has degree 3 once the loop is added")

# --- step 1: a linear map, identical to a dense layer -------------------
W = torch.eye(2, dtype=torch.float64)          # identity, so we can read the result
h = x.to(torch.float64) @ W

# --- step 2: MESSAGE. gather from the SOURCE, scale ---------------------
src, dst = ei[0], ei[1]
msg = h.index_select(0, src) * norm.unsqueeze(-1)
print("\\nmessage tensor shape:", tuple(msg.shape), "-- one row per directed edge")

# --- step 3: AGGREGATE. scatter onto the DESTINATION --------------------
out = scatter_add(msg, dst, num_nodes)

print("\\nhand-written GCN output (W = identity, no bias):")
for i, row in enumerate(out.tolist()):
    print(f"  node {i}: [{row[0]:.4f}, {row[1]:.4f}]")

expected = torch.tensor([[3, 1], [2, 2], [1, 4], [3, 2], [3, 3]], dtype=torch.float64) / 3.0
print("\\nmatches the value computed on paper:",
      bool(torch.allclose(out, expected, atol=1e-9)))
print("expected node 0 = (x0 + x1 + x3)/3 = ([1,0]+[0,1]+[2,0])/3 = [1.0000, 0.3333]")"""))

    A(md("""That is the entire mechanism. Everything else in this notebook is a variation
on those three lines:

| model | message `f` | aggregation | update |
|---|---|---|---|
| **GCN** | `W h_j`, scaled by `1/sqrt(d_i d_j)` | sum | add bias |
| **GraphSAGE** | `h_j` (unchanged) | mean / max / sum | `W_self h_i + W_neigh agg` |
| **GAT** | `W h_j`, scaled by learned `alpha_ij` | sum | add bias |
| **GIN** | `h_j` | sum | `MLP((1+eps) h_i + agg)` |

Now let the library version do it and check it agrees:"""))
    A(code("""conv = GCNConvScratch(2, 2).double().eval()
with torch.no_grad():
    conv.lin.weight.copy_(torch.eye(2, dtype=torch.float64))
    conv.bias.zero_()
    out_layer = conv(x.to(torch.float64), edge_index, num_nodes)

print("GCNConvScratch output matches the five lines above:",
      bool(torch.allclose(out_layer, out, atol=1e-12)))

# and the other three aggregators, on the same graph
for aggr in ("mean", "sum", "max"):
    s = SAGEConvScratch(2, 2, aggr=aggr).double().eval()
    with torch.no_grad():
        s.lin_l.weight.copy_(torch.eye(2, dtype=torch.float64)); s.lin_l.bias.zero_()
        s.lin_r.weight.zero_()
        o = s(x.to(torch.float64), edge_index, num_nodes)
    print(f"\\nSAGE aggr={aggr:4} -> node 0 aggregates its neighbours (x1=[0,1], x3=[2,0]) as "
          f"{[round(v, 4) for v in o[0].tolist()]}")"""))

    A(code("""# The scatter primitives themselves, checked against a brute-force Python loop.
for r in check_scatter_vs_loop():
    print(r.line())"""))

    A(md("""---
# 3. Cora

2708 machine-learning papers, 1433 bag-of-words features, 7 classes, 5278
undirected citation edges. The task is **semi-supervised node classification**:
only 140 labelled nodes (20 per class) are used for training, 500 for
validation, 1000 for testing. The other nodes are never labelled, but they are
still *in the graph*, so their structure informs every prediction. That is the
idea that makes GNNs interesting rather than merely different.

Two things about the raw files are worth knowing, both discovered by inspecting
the bytes rather than trusting a README:

1. **`test.index` is not sorted.** Row `k` of the test feature block belongs to
   node `sorted(test_index)[k]`, so concatenating the blocks puts the test rows
   in the wrong order and you must un-permute them.
2. **The adjacency dict contains 302 duplicate entries** -- 10858 entries for a
   graph with 10556 unique directed edges. PyTorch Geometric removes them; the
   original tkipf/gcn code does not. PyG even carries a source comment about it.
   We default to PyG's behaviour so the comparison in section 5 is fair, and
   section 6 measures what the difference is actually worth."""))
    A(code("""cora = load_cora(source="planetoid", row_normalize=False)   # PyG does not row-normalise
cora = cora.to(DEVICE)
print()
print("on device:", DEVICE)"""))

    A(code("""# The same graph, read from the human-readable LINQS text release instead.
# No scipy involved: one line per paper, one line per citation.
cora_text = load_cora(source="linqs", row_normalize=False)
print()
print("Same graph, different node ordering and a reconstructed split:")
print("  planetoid edges :", cora.num_edges)
print("  linqs edges     :", cora_text.num_edges)
print("  class sizes match:",
      torch.equal(torch.bincount(cora.y.cpu(), minlength=7).sort().values,
                  torch.bincount(cora_text.y.cpu(), minlength=7).sort().values))
# `cora` was moved to the GPU above; `cora_text` was just loaded and is still on the CPU.
# torch.equal refuses to compare tensors on different devices, hence the .cpu() calls."""))

    A(code("""draw_graph(cora.edge_index[:, :4000].cpu(), labels=None, node_size=6,
           title="Cora (subset of edges shown)", path=f"{RESULTS}/03_cora.png")"""))

    A(md("""---
# 4. Train the three canonical architectures

Same protocol for all three, so the comparison means something: 2 hidden layers,
Adam, lr 0.01, weight decay 5e-4, dropout, 200 epochs, full-batch.

**Reporting note.** We report the test accuracy *at the epoch where validation
accuracy peaked*, not the best test accuracy ever seen. Taking the max over
epochs is test-set selection and inflates the number by a point or two -- it is
common in GNN papers and it is wrong. Both numbers are printed so you can see
the gap."""))
    A(code("""from collections import OrderedDict

def build(name):
    f, h, c = cora.num_features, 16, cora.num_classes
    if name == "gcn":
        return lambda: GCNScratch(f, h, c, num_layers=2, dropout=0.5)
    if name == "sage":
        return lambda: GraphSAGEScratch(f, h, c, num_layers=2, dropout=0.5, aggr="mean")
    if name == "gat":
        return lambda: GATScratch(f, h, c, heads=8, dropout=0.6)
    raise KeyError(name)

SEEDS = [0, 1, 2]
summary = OrderedDict()
histories = {}

for name in ("gcn", "sage", "gat"):
    print(f"\\n=== {name.upper()} ===")
    res = train_node_multi_seed(build(name), cora, model_name=name.upper(),
                                seeds=SEEDS, epochs=200, lr=1e-2,
                                weight_decay=5e-4, device=DEVICE)
    summary[name] = res
    histories[name.upper()] = res["results"][-1].history
    print(f"  parameters : {count_parameters(build(name)()):,}")
    print(f"  test acc   : {res['test_acc_mean'] * 100:.2f} +/- {res['test_acc_std'] * 100:.2f}%")
    print(f"  macro F1   : {res['test_f1_mean'] * 100:.2f} +/- {res['test_f1_std'] * 100:.2f}%")
    print(f"  majority   : {res['baseline'] * 100:.2f}%  <- the number to beat")

plot_training_curves(histories, title="Cora, 200 epochs, full-batch",
                     path=f"{RESULTS}/04_training_curves.png")"""))

    A(code("""rows = [{"model": v["model"], "test_acc_mean": v["test_acc_mean"],
         "test_acc_std": v["test_acc_std"]} for v in summary.values()]
plot_results_bars(rows, title="Cora test accuracy (mean over 3 seeds)",
                  baseline=next(iter(summary.values()))["baseline"],
                  path=f"{RESULTS}/04_results.png")

print(f"{'model':8} {'test acc':>18} {'macro F1':>18}")
for v in summary.values():
    print(f"{v['model']:8} {v['test_acc_mean']*100:6.2f} +/- {v['test_acc_std']*100:4.2f}%"
          f"     {v['test_f1_mean']*100:6.2f} +/- {v['test_f1_std']*100:4.2f}%")"""))

    A(md("""---
# 5. Is the from-scratch code actually correct?

Writing a GNN from scratch is easy. Writing one that is *right* is not, because
the failure mode is silent. So: build the same layer in PyTorch Geometric, copy
the weights across verbatim, and assert the outputs agree.

This works at all because `gnn_lab` names its parameters the way PyG does --
`lin`/`bias` for GCN, `lin_l`/`lin_r` for SAGE, `lin`/`att_src`/`att_dst` for
GAT, `nn`/`eps` for GIN. That is not an aesthetic choice; it is what turns the
check into `pyg.load_state_dict(mine.state_dict())` instead of a brittle key
mapping that breaks on every PyG release.

Checks run in **float64**. In float32 a 1e-5 tolerance would pass even for a
subtly wrong implementation; agreement to ~1e-12 is evidence the two compute the
same expression, not merely similar ones."""))
    A(code("""eq_results = run_all(root="data", include_data=True, verbose=True)
print()
print("VERDICT:", "all checks passed" if all_passed(eq_results) else "SOMETHING FAILED")"""))

    A(md("""---
# 6. Ablations -- where the understanding actually comes from

Four questions, each with a measurable answer:

1. **Does depth hurt?** Stacking layers grows the receptive field, but message
   passing is diffusion, and diffusion drives every node toward the same value.
   That is over-smoothing, and it is the main reason GNNs are shallow.
2. **Does the aggregator matter?** Same SAGE architecture, `mean` vs `max` vs `sum`.
3. **Do self-loops matter?** Turn them off and a node can no longer see itself.
4. **Do those 302 duplicate edges matter?** A data-hygiene question with a real
   answer, because most tutorials silently pick one behaviour or the other."""))
    A(code("""print("=== depth sweep: over-smoothing ===")
depth_rows = []
for depth in (1, 2, 3, 4, 6, 8, 12):
    set_seed(0)
    model = GCNScratch(cora.num_features, 16, cora.num_classes,
                       num_layers=depth, dropout=0.5)
    r = train_node(model, cora, model_name=f"GCN-{depth}L", epochs=200, device=DEVICE)
    with torch.no_grad():
        model.eval()
        h = model.embeddings(cora.x, cora.edge_index)
    depth_rows.append({"depth": depth, "variant": "plain",
                       "test_acc": r.test_acc_at_best_val,
                       "cosine": mean_pairwise_cosine(h),
                       "energy": dirichlet_energy(h, cora.edge_index, cora.num_nodes)})
    print(f"  {depth:2d} layers  test {r.test_acc_at_best_val*100:5.2f}%  "
          f"cosine {depth_rows[-1]['cosine']:.4f}  energy {depth_rows[-1]['energy']:.4e}")

plot_oversmoothing([d["depth"] for d in depth_rows], depth_rows,
                   path=f"{RESULTS}/06_oversmoothing.png")"""))

    A(code("""print("=== aggregator sweep (GraphSAGE, 2 layers) ===")
agg_rows = []
for aggr in ("mean", "max", "sum"):
    set_seed(0)
    model = GraphSAGEScratch(cora.num_features, 16, cora.num_classes,
                             num_layers=2, dropout=0.5, aggr=aggr)
    r = train_node(model, cora, model_name=f"SAGE-{aggr}", epochs=200, device=DEVICE)
    agg_rows.append({"model": f"SAGE-{aggr}", "test_acc_mean": r.test_acc_at_best_val,
                     "test_acc_std": 0.0})
    print(f"  aggr={aggr:5}  test {r.test_acc_at_best_val*100:5.2f}%")

print("\\n=== self-loop ablation (GCN, 2 layers) ===")
for loops in (True, False):
    set_seed(0)
    model = GCNScratch(cora.num_features, 16, cora.num_classes, num_layers=2,
                       dropout=0.5, add_self_loops=loops)
    r = train_node(model, cora, model_name="GCN", epochs=200, device=DEVICE)
    agg_rows.append({"model": f"GCN self_loops={loops}", "test_acc_mean": r.test_acc_at_best_val,
                     "test_acc_std": 0.0})
    print(f"  add_self_loops={loops!s:5}  test {r.test_acc_at_best_val*100:5.2f}%")

plot_results_bars(agg_rows, title="Ablations on Cora",
                  path=f"{RESULTS}/06_ablations.png")"""))

    A(code("""print("=== duplicate-edge ablation ===")
cora_dup = load_cora(source="planetoid", row_normalize=False, dedupe=False).to(DEVICE)
print("  edges deduped :", cora.num_edges)
print("  edges kept    :", cora_dup.num_edges, f"(+{cora_dup.num_edges - cora.num_edges})")

for label, data in (("deduped (PyG behaviour)", cora), ("with duplicates (tkipf behaviour)", cora_dup)):
    set_seed(0)
    r = train_node(GCNScratch(data.num_features, 16, data.num_classes, num_layers=2,
                              dropout=0.5), data, model_name="GCN", epochs=200, device=DEVICE)
    print(f"  {label:38} test {r.test_acc_at_best_val*100:5.2f}%")"""))

    A(md("""### 6b. What does over-smoothing look like?

The accuracy curve above says *that* it happens. The t-SNE plots say *what*
happens: class clusters that are cleanly separated after one layer blur into a
single blob as layers stack."""))
    A(code("""set_seed(0)
deep = GCNScratch(cora.num_features, 16, cora.num_classes, num_layers=8, dropout=0.0).to(DEVICE)
train_node(deep, cora, model_name="GCN-8L", epochs=200, device=DEVICE, log_every=0)

deep.eval()
hs = [cora.x]
with torch.no_grad():
    h = cora.x
    for i, conv in enumerate(deep.convs):
        h = conv(h, cora.edge_index)
        if i < len(deep.convs) - 1:
            h = torch.nn.functional.relu(h)
        hs.append(h)

plot_layerwise_similarity(hs, cora.edge_index,
                          path=f"{RESULTS}/06b_layerwise.png")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, k in zip(axes, (0, 1, 2, 8)):
    if k < len(hs):
        plot_tsne(hs[k], cora.y, title=f"after {k} layer" + ("" if k == 1 else "s"),
                  ax=ax, seed=0)
fig.tight_layout()
save_or_show(fig, f"{RESULTS}/06b_tsne.png")"""))

    A(md("""---
# 7. What is GAT's attention actually doing?

The claim is that GAT learns to weight informative neighbours more heavily. That
is falsifiable: split the learned attention coefficients into edges that connect
two nodes of the *same* class and edges that connect *different* classes, and
compare the distributions. If attention is doing its job, the same-class
distribution sits to the right.

On homophilous Cora the separation is visible but far from clean. That is the
honest result, and it is a useful corrective to figures that make attention
look like a perfect edge classifier."""))
    A(code("""set_seed(0)
gat = GATScratch(cora.num_features, 8, cora.num_classes, heads=8, dropout=0.0).to(DEVICE)
train_node(gat, cora, model_name="GAT", epochs=200, device=DEVICE)

ei_att, att = gat.attention_weights(cora.x, cora.edge_index, layer=0)
att = att.detach().cpu()
src_i, dst_i = ei_att[0].cpu(), ei_att[1].cpu()
y_cpu = cora.y.cpu()
same = y_cpu[src_i] == y_cpu[dst_i]

per_edge = att.mean(dim=1)          # average across the 8 heads
print("edges with attention:", per_edge.numel())
print(f"  same-class  edges: {int(same.sum()):5d}  mean attention {per_edge[same].mean():.4f}")
print(f"  cross-class edges: {int((~same).sum()):5d}  mean attention {per_edge[~same].mean():.4f}")
print(f"  ratio same/cross : {per_edge[same].mean() / per_edge[~same].mean():.3f}")

plot_attention_distribution(per_edge[same], per_edge[~same],
                            path=f"{RESULTS}/07_attention.png")"""))

    A(code("""# Per-head behaviour: do the 8 heads specialise, or do they all agree?
print("attention mean per head, same-class vs cross-class edges:")
print(f"{'head':>5} {'same':>9} {'cross':>9} {'ratio':>8}")
for hnd in range(att.size(1)):
    s = float(att[same, hnd].mean()); d = float(att[~same, hnd].mean())
    print(f"{hnd:>5} {s:>9.4f} {d:>9.4f} {s / max(d, 1e-12):>8.3f}")

# draw a subgraph with edge width proportional to attention
sub = torch.arange(60)
mask = torch.isin(src_i, sub) & torch.isin(dst_i, sub)
draw_graph(ei_att[:, mask], labels=y_cpu[sub],
           highlight_edges=per_edge[mask].float(),
           title="GAT attention on a Cora subgraph (edge width = attention)",
           path=f"{RESULTS}/07_attention_graph.png", node_size=90)"""))

    A(md("""---
# 8. Graph-level classification: GIN on MUTAG

Everything so far predicted a label per *node* in one fixed graph. Now the task
flips: 188 small molecules, each with a single label (mutagenic or not), and the
model must learn a function from whole-graph structure to that class.

Two new ingredients appear:

* **Batching.** Molecules have 10-28 atoms, so they cannot be stacked into a 4-D
  tensor. Instead they are concatenated into one block-diagonal graph plus a
  `batch` vector saying which molecule each node belongs to. No information can
  leak between molecules because they are physically disconnected.
* **Readout.** Node embeddings must collapse into one vector per molecule. Sum,
  mean or max -- and the choice changes what the model can see.

MUTAG is **imbalanced**: 63 vs 125. The majority-class baseline is **66.5%, not
50%**. Any number below that is worse than a constant."""))
    A(code("""mutag = load_mutag()
print()
print("first molecule:", mutag.graphs[0]["x"].shape[0], "atoms,",
      mutag.graphs[0]["edge_index"].shape[1], "directed bonds, label",
      int(mutag.graphs[0]["y"]))"""))

    A(code("""# Show the batching trick explicitly: two graphs -> one block-diagonal graph.
b = collate_graphs(mutag.graphs[:3])
print(b)
print("batch vector:", b.batch.tolist())
print("edge_index (note the offsets that keep graphs disconnected):")
print(b.edge_index.t().tolist()[:12], "...")

# Proof that batching cannot leak information between graphs.
conv = GINConvScratch(torch.nn.Sequential(torch.nn.Linear(mutag.num_features, 8),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(8, 8))).double().eval()
with torch.no_grad():
    joint = conv(b.x.double(), b.edge_index, b.num_nodes)
    ok = True
    node_ptr = 0
    for k, gr in enumerate(mutag.graphs[:3]):
        alone = conv(gr["x"].double(), gr["edge_index"], gr["x"].size(0))
        n = gr["x"].size(0)
        ok &= bool(torch.allclose(joint[node_ptr:node_ptr + n], alone, atol=1e-12))
        node_ptr += n
print("\\nbatched forward == per-graph forward for every graph:", ok)"""))

    A(code("""print("=== GIN on MUTAG, 5-fold cross-validation ===")
def build_gin(pool="sum"):
    return lambda: GINScratch(mutag.num_features, 32, mutag.num_classes,
                              num_layers=3, dropout=0.5, pool=pool)

mutag_results = []
for pool in ("sum", "mean", "max"):
    r = train_graph_classification(build_gin(pool), mutag, model_name=f"GIN-{pool}",
                                   epochs=80, batch_size=32, folds=5, device=DEVICE)
    mutag_results.append({"model": r["model"], "test_acc_mean": r["acc_mean"],
                          "test_acc_std": r["acc_std"]})

plot_results_bars(mutag_results, title="MUTAG, 5-fold CV (GIN, varying readout)",
                  baseline=mutag.majority_accuracy(),
                  path=f"{RESULTS}/08_mutag.png")"""))

    A(md("""### 8b. The expressiveness argument, made concrete

The GIN paper proves that mean and max aggregation cannot distinguish certain
non-isomorphic graphs -- they cannot *count*. Concretely: a node with one
neighbour of feature 1 and a node with two neighbours of feature 0.5 have the
same mean and the same max, but different structure.

A theorem can feel abstract, so here is a dataset built around it, and an honest
caveat. `triangle_vs_square` makes graphs whose label is "does this graph contain
a triangle?". Classes are built in matched pairs (same node and edge counts; the
positive graph is the negative one with one edge moved), so size and density are
not shortcuts. An earlier version without that matching let edge density predict
the label at 88%, and `sum` appeared to win -- a leak, not expressiveness.

Expected outcome: **every readout near 50%.** The theorem is about the aggregator
inside each layer, this model always sums there, and triangle detection is beyond
1-WL message passing anyway. A clearly positive gap here would mean a shortcut
crept back in."""))
    A(code("""ts = triangle_square_dataset(num_graphs=400, seed=0)
print()
print("majority baseline:", round(ts.majority_accuracy(), 3), "(balanced by construction)")

ts_results = []
for pool in ("mean", "sum"):
    def build(pool=pool):
        return lambda: GINScratch(ts.num_features, 32, ts.num_classes,
                                  num_layers=3, dropout=0.5, pool=pool)
    r = train_graph_classification(build(), ts, model_name=f"GIN-{pool}-pool",
                                   epochs=60, batch_size=32, folds=3, device=DEVICE)
    ts_results.append({"model": r["model"], "test_acc_mean": r["acc_mean"],
                       "test_acc_std": r["acc_std"]})

plot_results_bars(ts_results, title="Triangle detection (matched classes)",
                  baseline=0.5, path=f"{RESULTS}/08b_triangle.png")"""))

    A(md("""---
# 9. What this run showed

**Mechanics.** A graph convolution is `gather from src` + `scatter onto dst` +
`update`. Section 2 wrote one in five lines and matched a value computed on
paper. GCN, GraphSAGE, GAT and GIN differ only in the message function, the
aggregation operator, and the update rule.

**Correctness.** Section 5 compared every hand-written layer against PyTorch
Geometric in float64 with weights copied across verbatim. That is the only check
that catches the silent bugs -- swapping `src` and `dst` produces a model that
trains happily and answers the wrong question.

**Why GNNs are shallow.** Section 6: accuracy falls off as depth grows while
mean pairwise cosine similarity climbs toward 1 and Dirichlet energy falls
toward 0. Message passing is diffusion; enough of it and every node looks the
same. The t-SNE panels show the clusters merging.

**What attention buys.** Section 7: same-class edges do receive more attention
than cross-class ones, but the distributions overlap heavily. Attention is a
soft, learned reweighting -- not an edge classifier.

**Structure vs features.** Section 8: on a task whose label *is* a substructure
(a triangle), with density shortcuts removed, every readout sits near chance --
the limit of 1-WL message passing, and a reminder that the first version of this
experiment "worked" only because of a leak.

**Data hygiene.** Section 6 also measured the effect of the 302 duplicate
adjacency entries in Cora. Small, but real -- and most tutorials pick a side
without mentioning it."""))
    A(code("""print("=" * 72)
print("RESULTS SUMMARY")
print("=" * 72)
print("\\n-- Cora node classification (mean over %d seeds) --" % len(SEEDS))
print(f"{'model':10} {'test acc':>18} {'macro F1':>18}")
for v in summary.values():
    print(f"{v['model']:10} {v['test_acc_mean']*100:6.2f} +/- {v['test_acc_std']*100:4.2f}%"
          f"     {v['test_f1_mean']*100:6.2f} +/- {v['test_f1_std']*100:4.2f}%")
print(f"{'baseline':10} {next(iter(summary.values()))['baseline']*100:6.2f}% (majority class)")

print("\\n-- Cora depth sweep (GCN) --")
for d in depth_rows:
    print(f"  {d['depth']:2d} layers  test {d['test_acc']*100:5.2f}%  "
          f"cosine {d['cosine']:.4f}  energy {d['energy']:.3e}")

print("\\n-- MUTAG graph classification (5-fold CV) --")
for r in mutag_results:
    print(f"  {r['model']:12} {r['test_acc_mean']*100:6.2f} +/- {r['test_acc_std']*100:4.2f}%")
print(f"  {'baseline':12} {mutag.majority_accuracy()*100:6.2f}% (majority class)")

print("\\n-- Triangle detection (3-fold CV) --")
for r in ts_results:
    print(f"  {r['model']:16} {r['test_acc_mean']*100:6.2f} +/- {r['test_acc_std']*100:4.2f}%")

print("\\nequivalence checks:",
      sum(1 for r in eq_results if r.passed is True), "passed /",
      sum(1 for r in eq_results if r.passed is False), "failed /",
      sum(1 for r in eq_results if r.passed is None), "skipped")
print("\\nfigures written to:", os.path.abspath(RESULTS))
print(sorted(os.listdir(RESULTS)))"""))

    return C


# --------------------------------------------------------------------------- #
def main() -> None:
    header_imports = [
        "import os", "import sys", "import time", "import copy", "import math",
        "import random", "import pickle", "import tarfile", "import zipfile",
        "import urllib.request",
        "from dataclasses import dataclass, field",
        "from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
    ]

    cells: List[dict] = []
    cells.append(md(f"""# Graph Neural Networks from Scratch

**Self-contained Kaggle run.** Everything below executes top to bottom on a free
T4 kernel (it also runs fine on CPU -- Cora is small). No repository checkout,
no local package: the whole `gnn_lab` library is inlined into the cells in
section *"The library"*, so this file is the single source of truth for a
reproducible run.

> This notebook is **generated** by `scripts/build_kaggle_notebook.py` from the
> `gnn_lab` package. Do not edit it by hand -- edit the package and re-run the
> builder, otherwise the two copies drift apart.

| | |
|---|---|
| **Task 1** | semi-supervised node classification on Cora (2708 papers, 7 classes) |
| **Task 2** | graph-level classification on MUTAG (188 molecules) |
| **Models** | GCN, GraphSAGE, GAT, GIN -- all hand-written in plain PyTorch |
| **Verification** | every layer checked numerically against `torch_geometric` in float64 |
| **Experiments** | depth/over-smoothing, aggregator, self-loops, duplicate edges, attention |
| **Runtime** | ~10-15 min on T4, ~25 min on CPU |

## Contents
1. A graph is three tensors
2. Message passing, done by hand
3. Cora
4. Train GCN / GraphSAGE / GAT
5. Equivalence with PyTorch Geometric
6. Ablations: depth, aggregator, self-loops, duplicate edges
7. What GAT attention actually does
8. Graph-level GIN on MUTAG + the expressiveness experiment
9. Summary"""))

    lib_cells, hoisted = build_library_cells()

    imports = sorted(set(header_imports + hoisted))
    imports = [i for i in imports if not i.startswith("from __future__")]

    cells.append(md("""---
# The library (inlined from `gnn_lab`)

The cells in this section are the package source, concatenated in dependency
order. They are generated, not written -- see the note at the top. Read them
here or in the repository; they are identical.

Each cell starts with `from __future__ import annotations` because IPython
compiles every cell independently, and the code uses PEP 604 unions
(`int | None`) in annotations."""))
    cells.append(code("# imports shared by every inlined module\n" + "\n".join(imports)))
    cells.extend(lib_cells)

    cells.extend(narrative_and_runs())

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"wrote {OUT}")
    print(f"  {len(cells)} cells ({n_code} code, {n_md} markdown)")
    print(f"  {os.path.getsize(OUT) / 1024:.0f} KB")


if __name__ == "__main__":
    sys.exit(main())
