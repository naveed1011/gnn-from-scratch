"""
TU-format graph datasets (MUTAG, PROTEINS) -- graph-level classification.

Node classification on Cora keeps the graph fixed and predicts a label per
node. Graph-level tasks do the opposite: you get *many small graphs*, each with
one label, and you must learn a function from whole-graph structure to a class.

That difference forces you to learn two things Cora never asks for:

* **Batching.** Graphs have different sizes, so you cannot stack them. You
  build one block-diagonal graph plus a ``batch`` vector (see
  :mod:`gnn_lab.layers.pooling`).
* **Readout/pooling.** Node embeddings have to be collapsed into a single
  graph embedding, and *how* you collapse them (sum / mean / max) changes what
  the model is able to see.

Storage format
--------------
Every TU dataset is the same six plain-text files, all **1-indexed**:

============================  ============================================
``DS_A.txt``                  one ``src,dst`` pair per line, both directions
``DS_graph_indicator.txt``    graph id of node *i*, one line per node
``DS_graph_labels.txt``       one label per graph
``DS_node_labels.txt``        discrete node type per node -> one-hot features
============================  ============================================

Verified against the actual downloads:

* **MUTAG** -- 188 graphs, 3371 nodes, 7442 directed edges (3721 undirected),
  7 atom types, 10-28 nodes/graph, labels ``{-1, +1}`` with an **imbalanced**
  63/125 split. That imbalance matters: a classifier that always says "+1"
  scores 66.5%. Any GIN number below that is worse than doing nothing.
* **PROTEINS** -- 1113 graphs, 39.06 nodes/graph on average, 3 node types.

Source
------
Downloaded from a GitHub mirror first (``nd7141/graph_datasets``), with the
original chrsmrrs.com host as a fallback. Both are free.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

__all__ = ["GraphDataset", "load_tu", "load_mutag", "load_proteins", "TU_SPECS"]

# Tried in order. ``raw.githubusercontent.com`` goes first: the ``github.com/.../raw/...``
# form of the same file is a redirect that is much slower (and returned a 504 when this
# was tested), while the raw host answered in ~0.15 s.
GH_RAW_MIRROR = "https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets/{name}.zip"
GH_MIRROR = "https://github.com/nd7141/graph_datasets/raw/master/datasets/{name}.zip"
CHRSMRRS = "https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"

# name -> (expected graphs, expected nodes, description)
TU_SPECS = {
    "MUTAG": {
        "description": "188 mutagenic aromatic/nitro compounds; predict mutagenicity",
        "expected_graphs": 188,
        "expected_nodes": 3371,
    },
    "PROTEINS": {
        "description": "1113 protein structures; predict enzyme vs non-enzyme",
        "expected_graphs": 1113,
        "expected_nodes": 43_471,
    },
}


@dataclass
class GraphDataset:
    """A list of small graphs plus the metadata needed to train on them."""

    graphs: List[Dict[str, torch.Tensor]]   # each: {"x": [n_i,F], "edge_index": [2,e_i], "y": tensor}
    name: str
    num_features: int
    num_classes: int
    class_names: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.graphs)

    @property
    def num_nodes(self) -> int:
        return sum(g["x"].size(0) for g in self.graphs)

    @property
    def num_edges(self) -> int:
        return sum(g["edge_index"].size(1) for g in self.graphs)

    def labels(self) -> torch.Tensor:
        return torch.stack([g["y"].reshape(()) for g in self.graphs])

    def majority_accuracy(self) -> float:
        """The baseline you must beat. On MUTAG this is 0.665, not 0.5."""
        y = self.labels()
        counts = torch.bincount(y, minlength=self.num_classes)
        return float(counts.max()) / float(y.numel())

    def summary(self) -> str:
        sizes = torch.tensor([g["x"].size(0) for g in self.graphs], dtype=torch.float)
        y = self.labels()
        counts = torch.bincount(y, minlength=self.num_classes)
        lines = [
            f"{self.name}",
            f"  graphs            : {len(self)}",
            f"  total nodes       : {self.num_nodes}",
            f"  total dir. edges  : {self.num_edges}",
            f"  node features     : {self.num_features}",
            f"  nodes/graph       : min {int(sizes.min())} / mean {float(sizes.mean()):.2f} / "
            f"max {int(sizes.max())}",
            f"  classes           : {self.num_classes} {list(counts.tolist())}",
            f"  majority baseline : {self.majority_accuracy():.3f}",
        ]
        lines += [f"  note              : {n}" for n in self.notes]
        return "\n".join(lines)

    def to(self, device):
        for g in self.graphs:
            g["x"] = g["x"].to(device)
            g["edge_index"] = g["edge_index"].to(device)
            g["y"] = g["y"].to(device)
        return self


# --------------------------------------------------------------------------- #
def _read_lines(path: str) -> List[str]:
    with open(path) as fh:
        return [l.strip() for l in fh if l.strip()]


# Last resort for MUTAG only: the GIN paper's repo ships the real 188-graph MUTAG
# (same counts as TU: 3371 nodes, 7442 directed edges, 63/125) in its own text format.
# It is a 33 MB zip, so it is tried only after every TU-format mirror has failed.
GIN_REPO_ZIP = "https://raw.githubusercontent.com/weihua916/powerful-gnns/master/dataset.zip"
_UA = {"User-Agent": "Mozilla/5.0 (gnn-lab)"}


def _count_lines(path: str) -> int:
    with open(path) as fh:
        return sum(1 for l in fh if l.strip())


def _check_counts(name: str, d: str) -> Optional[str]:
    """None if the files in ``d`` match ``TU_SPECS``, otherwise a human-readable reason.

    A mirror can serve a *different version* of a dataset under the same name. The
    nd7141 mirror's MUTAG has 135 graphs, not 188, and every downstream number
    (including the 66.5% majority baseline) silently changes. So we check.
    """
    try:
        g = _count_lines(os.path.join(d, f"{name}_graph_labels.txt"))
        n = _count_lines(os.path.join(d, f"{name}_graph_indicator.txt"))
    except OSError as e:
        return f"missing file ({e})"
    spec = TU_SPECS.get(name)
    if spec and (g != spec["expected_graphs"] or n != spec["expected_nodes"]):
        return (f"got {g} graphs / {n} nodes, expected {spec['expected_graphs']} / "
                f"{spec['expected_nodes']}")
    return None


def _cache_ok(name: str, d: str) -> bool:
    required = [f"{name}_A.txt", f"{name}_graph_indicator.txt", f"{name}_graph_labels.txt"]
    if not all(os.path.exists(os.path.join(d, f)) for f in required):
        return False
    why = _check_counts(name, d)
    if why:
        print(f"  cached {name} is the wrong version ({why}); re-downloading")
    return why is None


def _download_to(url: str, dest: str, timeout: int = 90) -> None:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest + ".part", "wb") as fh:
        fh.write(r.read())
    os.replace(dest + ".part", dest)


def _extract_flat(zip_path: str, d: str) -> None:
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            base = os.path.basename(member)
            if not base or member.endswith("/"):
                continue
            with z.open(member) as src, open(os.path.join(d, base), "wb") as dst:
                dst.write(src.read())


def _mutag_from_gin_repo(root: str, d: str, verbose: bool) -> None:
    """Download the GIN repo zip, convert its MUTAG.txt into TU-format files."""
    zip_path = os.path.join(root, "gin_dataset.zip")
    if not os.path.exists(zip_path):
        if verbose:
            print(f"  downloading {GIN_REPO_ZIP} (33 MB, MUTAG only)")
        _download_to(GIN_REPO_ZIP, zip_path, timeout=300)
    with zipfile.ZipFile(zip_path) as z:
        lines = z.read("dataset/MUTAG/MUTAG.txt").decode().strip().split("\n")
    n_graphs, i = int(lines[0]), 1
    labels, node_types, indicator, edges, base = [], [], [], [], 0
    for g in range(n_graphs):
        n_nodes, lab = map(int, lines[i].split()); i += 1
        labels.append(lab)
        for v in range(n_nodes):
            parts = lines[i].split(); i += 1
            node_types.append(int(parts[0])); indicator.append(g + 1)
            edges += [(base + v + 1, base + int(u) + 1) for u in parts[2:2 + int(parts[1])]]
        base += n_nodes
    # TU convention: label -1 = 63 non-mutagenic, +1 = 125 mutagenic. The GIN file uses
    # its own ids, so identify the classes by size (the minority one is -1).
    minority = min(set(labels), key=labels.count)
    tu_labels = [-1 if l == minority else 1 for l in labels]
    def w(suffix, rows):
        with open(os.path.join(d, f"MUTAG_{suffix}.txt"), "w") as fh:
            fh.write("\n".join(rows) + "\n")
    w("A", [f"{a}, {b}" for a, b in edges]); w("graph_indicator", map(str, indicator))
    w("graph_labels", map(str, tu_labels)); w("node_labels", map(str, node_types))


def _fetch_tu(name: str, root: str, d: str, verbose: bool) -> None:
    """Try each mirror in turn; accept the first whose counts match ``TU_SPECS``."""
    problems = []
    for tmpl in (CHRSMRRS, GH_RAW_MIRROR, GH_MIRROR):
        url = tmpl.format(name=name)
        zip_path = os.path.join(root, f"{name}.zip")
        try:
            if verbose:
                print(f"  downloading {url}")
            _download_to(url, zip_path)
            _extract_flat(zip_path, d)
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"{url} -> {type(e).__name__}: {e}")
            continue
        why = _check_counts(name, d)
        if why is None:
            return
        problems.append(f"{url} -> wrong version: {why}")
        if verbose:
            print(f"  rejected: {why}; trying next mirror")
        os.remove(zip_path)
    if name == "MUTAG":
        try:
            _mutag_from_gin_repo(root, d, verbose)
            why = _check_counts(name, d)
            if why is None:
                return
            problems.append(f"GIN repo -> wrong version: {why}")
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"GIN repo -> {type(e).__name__}: {e}")
    raise RuntimeError(f"could not get a correct copy of {name}:\n  " + "\n  ".join(problems))



def load_tu(name: str, root: str = "data", verbose: bool = True,
            max_graphs: Optional[int] = None) -> GraphDataset:
    """Download (if needed) and parse a TU-format dataset into a list of graphs.

    Parameters
    ----------
    max_graphs : int | None
        Keep only the first ``k`` graphs. Handy for a fast smoke test -- PROTEINS
        trains in a few seconds anyway, but 50 graphs trains in under one.
    """
    name = name.upper()
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)

    if not _cache_ok(name, d):
        _fetch_tu(name, root, d, verbose)

    # ---- parse --------------------------------------------------------------
    graph_indicator = [int(v) - 1 for v in _read_lines(os.path.join(d, f"{name}_graph_indicator.txt"))]
    node_labels_path = os.path.join(d, f"{name}_node_labels.txt")
    node_labels = ([int(v) for v in _read_lines(node_labels_path)]
                   if os.path.exists(node_labels_path) else [])
    raw_graph_labels = [int(v) for v in _read_lines(os.path.join(d, f"{name}_graph_labels.txt"))]

    edges_raw = [tuple(int(v) for v in line.split(",")) for line in _read_lines(os.path.join(d, f"{name}_A.txt"))]

    num_nodes = len(graph_indicator)
    num_graphs = len(raw_graph_labels)

    # labels can be {-1,+1} (MUTAG) or {1,2} (PROTEINS); remap to 0..C-1
    uniq = sorted(set(raw_graph_labels))
    label_map = {v: i for i, v in enumerate(uniq)}
    graph_labels = [label_map[v] for v in raw_graph_labels]

    # discrete node types -> one-hot. If the file is missing, fall back to a
    # constant feature so the loader still works for attribute-only datasets.
    n_types = max(node_labels) + 1 if node_labels else 1
    x = torch.zeros(num_nodes, n_types)
    if node_labels:
        x[torch.arange(num_nodes), torch.tensor(node_labels, dtype=torch.long)] = 1.0
    else:
        x[:, 0] = 1.0                      # no node types -> constant feature

    # first node index belonging to each graph (nodes are stored contiguously)
    offsets = [0] * (num_graphs + 1)
    for gi in graph_indicator:
        offsets[gi + 1] += 1
    for g in range(num_graphs):
        offsets[g + 1] += offsets[g]

    # per-graph edge lists, re-indexed locally (A.txt is 1-indexed globally)
    per_graph: List[List[List[int]]] = [[] for _ in range(num_graphs)]
    for s, t in edges_raw:
        g = graph_indicator[s - 1]
        per_graph[g].append([s - 1 - offsets[g], t - 1 - offsets[g]])

    graphs = []
    for g in range(num_graphs):
        n = offsets[g + 1] - offsets[g]
        e = per_graph[g]
        ei = (torch.tensor(e, dtype=torch.long).t().contiguous()
              if e else torch.zeros(2, 0, dtype=torch.long))
        graphs.append({
            "x": x[offsets[g]:offsets[g + 1]].clone(),
            "edge_index": ei,
            "y": torch.tensor(graph_labels[g], dtype=torch.long),
        })

    if max_graphs is not None:
        graphs = graphs[:max_graphs]

    notes = [TU_SPECS.get(name, {}).get("description", "")]
    ds = GraphDataset(
        graphs=graphs, name=name, num_features=n_types,
        num_classes=len(uniq), class_names=[str(v) for v in uniq], notes=[n for n in notes if n],
    )
    if verbose:
        print(ds.summary())
    return ds


def load_mutag(root: str = "data", **kw) -> GraphDataset:
    return load_tu("MUTAG", root=root, **kw)


def load_proteins(root: str = "data", **kw) -> GraphDataset:
    return load_tu("PROTEINS", root=root, **kw)
