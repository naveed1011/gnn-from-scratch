"""
Cora -- the "MNIST of graph learning".

2708 machine-learning papers, 1433 bag-of-words features, 7 classes, 5278
undirected citation edges. Tiny enough to train in seconds, real enough that
every published GNN number you will ever compare against was produced on it.

Two loaders, on purpose
-----------------------
``source="planetoid"``
    The canonical Planetoid pickle files, downloaded straight from GitHub
    (``kimiyoung/planetoid``). This is the exact byte stream that produced the
    81.5% GCN number in the original paper, and the exact thing
    ``torch_geometric.datasets.Planetoid`` reads -- which is what makes the
    numerical equivalence test in ``gnn_lab.equivalence`` meaningful.

``source="linqs"``
    The official plain-text release from the LINQS group at UC Santa Cruz.
    Human-readable: one line per paper with 1433 ``0``/``1`` columns, and a
    separate edge list. Needs no scipy at all. Use this one when you want to
    actually *look* at the data, and it is the fallback if the pickle path
    cannot resolve.

Both describe the same graph. They do **not** share a node ordering, so a model
trained on one cannot be compared node-for-node with the other.

Verified facts about the raw files
----------------------------------
The numbers below were measured against the actual downloads, not copied from a
README:

============================  ==============================================
``ind.cora.x``                ``(140, 1433)``   train features
``ind.cora.allx``             ``(1708, 1433)``  train+val features
``ind.cora.tx``               ``(1000, 1433)``  test features
``ind.cora.y / .ally / .ty``  ``(140/1708/1000, 7)`` one-hot, dtype ``i4``
``ind.cora.graph``            ``defaultdict`` with 2708 keys, symmetric
``ind.cora.test.index``       1000 ints in ``[1708, 2707]``, **not sorted**
``cora.content`` (LINQS)      2708 lines x 1435 tab-separated fields
``cora.cites`` (LINQS)        5429 lines -> 5278 unique undirected edges
============================  ==============================================

Two gotchas that are worth reading before you write your own loader
-------------------------------------------------------------------
1. **``test.index`` is unsorted.** Row ``k`` of ``tx`` is the features of node
   ``sorted(test_index)[k]``, not of node ``test_index[k]``. Concatenating
   ``allx`` and ``tx`` therefore puts the test rows in the *wrong* order, and
   you have to un-permute them: ``x[test_index] = x[sorted_test_index]``.
   Skip this and you still train fine -- you just train on a silently
   relabelled test set. Citeseer needs an extra fix on top of this (it has
   isolated nodes); Cora does not.

2. **The ``graph`` dict contains duplicate adjacency entries.** 10858 entries
   for 5278 undirected edges = 10556 unique directed + 302 duplicates.
   PyTorch Geometric removes them and strips self-loops; the original tkipf/gcn
   code does not. PyG even carries a source comment about it: *"There are some
   duplicated edges and self loops in the datasets. Other implementations do
   not remove them!"* We default to PyG's behaviour (``dedupe=True``) so the
   equivalence test is apples-to-apples, and keep the switch so you can measure
   what the 302 stray edges are actually worth.
"""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass, field
from typing import List

import torch

from ..ops import edge_order, remove_self_loops, to_undirected

__all__ = [
    "NodeDataset",
    "load_cora",
    "load_cora_planetoid",
    "load_cora_linqs",
    "CORA_CLASS_NAMES",
]

CORA_CLASS_NAMES = [
    "Case_Based", "Genetic_Algorithms", "Neural_Networks",
    "Probabilistic_Methods", "Reinforcement_Learning", "Rule_Learning", "Theory",
]

# Primary: GitHub raw, exactly as the user asked (free + on GitHub).
# ``raw.githubusercontent.com`` first: the ``github.com/.../raw/...`` form of the same
# file goes through a redirect that took ~11 s per file (x8 files) when tested, versus
# ~0.07 s for the raw host. Both serve identical bytes; the second is just the fallback.
PLANETOID_URLS = [
    "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/",
    "https://github.com/kimiyoung/planetoid/raw/master/data/",
]
PLANETOID_FILES = ["ind.cora.x", "ind.cora.tx", "ind.cora.allx", "ind.cora.y",
                   "ind.cora.ty", "ind.cora.ally", "ind.cora.graph",
                   "ind.cora.test.index"]

LINQS_URLS = [
    "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
]


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
@dataclass
class NodeDataset:
    """Everything a semi-supervised node-classification run needs.

    Attribute names deliberately mirror ``torch_geometric.data.Data`` so the
    equivalence test can hand the same object to either model.
    """

    x: torch.Tensor                      # [N, F] float32
    edge_index: torch.LongTensor         # [2, E] int64, src on row 0, dst on row 1
    y: torch.LongTensor                  # [N] int64 class ids
    train_mask: torch.BoolTensor         # [N]
    val_mask: torch.BoolTensor
    test_mask: torch.BoolTensor
    num_classes: int
    class_names: List[str] = field(default_factory=list)
    source: str = "?"
    notes: List[str] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return int(self.x.size(0))

    @property
    def num_features(self) -> int:
        return int(self.x.size(1))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    def to(self, device) -> "NodeDataset":
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        return self

    def summary(self) -> str:
        # Build the degree vector on the SAME device as edge_index: after `.to("cuda")` the
        # index lives on the GPU, and index_add_ into a CPU tensor would raise a device error.
        dev = self.edge_index.device
        deg = torch.zeros(self.num_nodes, device=dev).index_add_(
            0, self.edge_index[1], torch.ones(self.num_edges, device=dev))
        lines = [
            f"Cora ({self.source})",
            f"  nodes          : {self.num_nodes}",
            f"  features       : {self.num_features} "
            f"(density {float((self.x != 0).float().mean()):.4f})",
            f"  directed edges : {self.num_edges} "
            f"({self.num_edges // 2} undirected)",
            f"  classes        : {self.num_classes}",
            f"  degree         : min {int(deg.min())} / mean {float(deg.mean()):.2f} / "
            f"max {int(deg.max())}",
            f"  train/val/test : {int(self.train_mask.sum())} / "
            f"{int(self.val_mask.sum())} / {int(self.test_mask.sum())}",
        ]
        if self.class_names:
            lines.append(f"  class names    : {', '.join(self.class_names)}")
        lines += [f"  note           : {n}" for n in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def _download(url: str, dest: str, timeout: int = 60) -> None:
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (gnn-lab)"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as fh:
        fh.write(r.read())
    os.replace(tmp, dest)


def _download_with_mirrors(candidates: List[str], dest: str) -> str:
    errs = []
    for u in candidates:
        try:
            _download(u, dest)
            return u
        except Exception as e:                              # noqa: BLE001
            errs.append(f"{u} -> {type(e).__name__}: {e}")
    raise RuntimeError("all download mirrors failed:\n  " + "\n  ".join(errs))


# --------------------------------------------------------------------------- #
# Loader 1 -- Planetoid (canonical split, literature-comparable)
# --------------------------------------------------------------------------- #
def _pickle_planetoid(path: str):
    """``pickle.load`` with ``encoding='latin1'`` (these are Python-2 pickles)."""
    with open(path, "rb") as fh:
        return pickle.load(fh, encoding="latin1")


def load_cora_planetoid(root: str = "data/cora_planetoid",
                        row_normalize: bool = True,
                        dedupe: bool = True,
                        verbose: bool = True) -> NodeDataset:
    """Load Cora from the Planetoid pickle files.

    Parameters
    ----------
    row_normalize : bool
        Divide each feature row by its sum. The original GCN code does this and
        the published 81.5% assumes it. PyG's ``Planetoid`` does *not*, so set
        this to ``False`` when cross-checking against PyG.
    dedupe : bool
        Remove duplicate/self-loop adjacency entries to match PyG. See the
        module docstring for why this is worth knowing about.
    """
    import numpy as np

    try:
        import scipy.sparse as sp
    except ImportError as e:                                 # pragma: no cover
        raise ImportError(
            "The Planetoid files are scipy-sparse pickles, so this loader needs "
            "scipy. Either `pip install scipy` or call load_cora(source='linqs') "
            "for the dependency-free plain-text loader."
        ) from e

    os.makedirs(root, exist_ok=True)
    notes: List[str] = []

    for fname in PLANETOID_FILES:
        path = os.path.join(root, fname)
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            url = f"{PLANETOID_URLS[0]}{fname}"
            if verbose:
                print(f"  downloading {url}")
            _download_with_mirrors([f"{base}{fname}" for base in PLANETOID_URLS], path)

    P = lambda n: os.path.join(root, f"ind.cora.{n}")          # noqa: E731

    x = _pickle_planetoid(P("x"))            # (140, 1433) csr
    tx = _pickle_planetoid(P("tx"))          # (1000, 1433) csr
    allx = _pickle_planetoid(P("allx"))      # (1708, 1433) csr
    y = _pickle_planetoid(P("y"))            # (140, 7) one-hot
    ty = _pickle_planetoid(P("ty"))          # (1000, 7)
    ally = _pickle_planetoid(P("ally"))      # (1708, 7)
    graph = _pickle_planetoid(P("graph"))    # defaultdict(list), 2708 keys

    with open(P("test.index")) as fh:
        test_index = torch.tensor([int(l) for l in fh.read().split()], dtype=torch.long)
    sorted_test_index = test_index.sort()[0]

    # --- features: allx covers nodes 0..1707, tx covers the 1000 test nodes ---
    features = np.asarray(sp.vstack([allx, tx]).todense(), dtype=np.float32)
    # un-permute the test rows (see module docstring, gotcha #1)
    features[test_index.numpy()] = features[sorted_test_index.numpy()]

    labels_onehot = np.vstack([ally, ty]).astype(np.int64)
    labels_onehot[test_index.numpy()] = labels_onehot[sorted_test_index.numpy()]
    y_all = torch.from_numpy(labels_onehot.argmax(axis=1))

    num_classes = int(labels_onehot.shape[1])
    num_nodes, num_features = features.shape

    if row_normalize:
        rowsum = features.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        features = features / rowsum

    # --- adjacency dict -> edge_index ---------------------------------------
    rows, cols = [], []
    for k, nbrs in graph.items():
        rows.extend([k] * len(nbrs))
        cols.extend(nbrs)
    edge_index = torch.stack([torch.tensor(rows, dtype=torch.long),
                              torch.tensor(cols, dtype=torch.long)], dim=0)

    raw_edges = edge_index.size(1)
    if dedupe:
        edge_index = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes)
        # match PyG's ordering: coalesce(..., sort_by_row=False) => sort by (dst, src)
        edge_index = edge_index[:, edge_order(edge_index, num_nodes)]
        notes.append(f"deduped {raw_edges} -> {edge_index.size(1)} directed edges")
    else:
        notes.append(f"kept all {raw_edges} adjacency entries (tkipf/gcn behaviour)")

    # --- canonical public split: 140 train / 500 val / 1000 test -------------
    def mask(idx, n):
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        return m

    n_train = int(y.shape[0])
    train_mask = mask(torch.arange(n_train), num_nodes)
    val_mask = mask(torch.arange(n_train, n_train + 500), num_nodes)
    test_mask = mask(test_index, num_nodes)

    ds = NodeDataset(
        x=torch.from_numpy(features),
        edge_index=edge_index,
        y=y_all,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        num_classes=num_classes,
        class_names=[],   # the pickle carries no label->name mapping; see module docs
        source="planetoid",
        notes=notes,
    )
    if verbose:
        print(ds.summary())
    return ds


# --------------------------------------------------------------------------- #
# Loader 2 -- LINQS plain text (no scipy, human-readable)
# --------------------------------------------------------------------------- #
def load_cora_linqs(root: str = "data/cora_linqs",
                    row_normalize: bool = True,
                    split: str = "public",
                    seed: int = 42,
                    verbose: bool = True) -> NodeDataset:
    """Load Cora from the official LINQS plain-text release.

    ``cora.content``  one line per paper:
    ``<paper_id>\\t<f1>\\t...\\t<f1433>\\t<label>``
    ``cora.cites``    one line per citation: ``<src_paper_id>\\t<dst_paper_id>``

    Note that ``cora.cites`` contains 5429 lines but only 5278 unique undirected
    edges -- 151 pairs appear in both directions. De-duplicating is not optional
    here: leaving them in inflates the degree of ~300 nodes and changes every
    GCN normalisation coefficient.

    Parameters
    ----------
    split : {"public", "random"}
        ``public`` reproduces the Planetoid *protocol* (20 labelled nodes per
        class, 500 validation, 1000 test) with a fixed seed. ``random`` is a
        60/20/20 split, i.e. fully supervised -- useful for seeing how much
        those 140 labels are really carrying.
    """
    import numpy as np

    os.makedirs(root, exist_ok=True)
    notes: List[str] = []

    content_path = os.path.join(root, "cora.content")
    cites_path = os.path.join(root, "cora.cites")
    if not (os.path.exists(content_path) and os.path.exists(cites_path)):
        tgz = os.path.join(root, "cora.tgz")
        if not os.path.exists(tgz):
            _download_with_mirrors(LINQS_URLS, tgz)
        with tarfile.open(tgz, "r:gz") as t:
            for member in t.getmembers():
                if member.name.endswith(("cora.content", "cora.cites")):
                    data = t.extractfile(member).read()
                    with open(os.path.join(root, os.path.basename(member.name)), "wb") as fh:
                        fh.write(data)

    # ---- nodes, features, labels ------------------------------------------
    ids: List[str] = []
    feats: List[List[float]] = []
    labels: List[str] = []
    with open(content_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            ids.append(parts[0])
            feats.append([float(v) for v in parts[1:-1]])
            labels.append(parts[-1])

    index_of = {pid: i for i, pid in enumerate(ids)}
    names = sorted(set(labels))
    label_of = {n: i for i, n in enumerate(names)}
    x_np = np.asarray(feats, dtype=np.float32)
    y = torch.tensor([label_of[l] for l in labels], dtype=torch.long)

    # ---- edges: de-duplicate and symmetrise --------------------------------
    raw = 0
    pairs = []
    seen = set()
    with open(cites_path) as fh:
        for line in fh:
            a, b = line.split()
            if a not in index_of or b not in index_of:
                continue
            raw += 1
            u, v = index_of[a], index_of[b]
            key = (min(u, v), max(u, v))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((u, v))
            pairs.append((v, u))
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    notes.append(f"cora.cites had {raw} lines -> {len(pairs) // 2} unique undirected edges")

    num_nodes = len(ids)
    if row_normalize:
        s = x_np.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        x_np = x_np / s

    # ---- split -------------------------------------------------------------
    g = torch.Generator().manual_seed(seed)
    if split == "public":
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for c in range(len(names)):
            idx = (y == c).nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(idx.numel(), generator=g)]
            train_mask[perm[:20]] = True
        rest = (~train_mask).nonzero(as_tuple=True)[0]
        rest = rest[torch.randperm(rest.numel(), generator=g)]
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[rest[:500]] = True
        test_mask[rest[500:1500]] = True
        notes.append(f"public split (seed {seed}): same protocol as Planetoid")
    elif split == "random":
        perm = torch.randperm(num_nodes, generator=g)
        n_tr, n_va = int(0.6 * num_nodes), int(0.2 * num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool); train_mask[perm[:n_tr]] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool); val_mask[perm[n_tr:n_tr + n_va]] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool); test_mask[perm[n_tr + n_va:]] = True
        notes.append(f"random 60/20/20 split (seed {seed})")
    else:
        raise ValueError(f"split must be 'public' or 'random', got {split!r}")

    ds = NodeDataset(
        x=torch.from_numpy(x_np), edge_index=edge_index, y=y,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        num_classes=len(names), class_names=names, source="linqs", notes=notes,
    )
    if verbose:
        print(ds.summary())
    return ds


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_cora(root: str = "data", source: str = "planetoid", **kwargs) -> NodeDataset:
    """Load Cora from either source, downloading on first use.

    ``source="planetoid"`` falls back to ``"linqs"`` automatically if scipy is
    unavailable, so a notebook never dies on a missing optional dependency.
    """
    if source == "planetoid":
        try:
            return load_cora_planetoid(root=os.path.join(root, "cora_planetoid"), **kwargs)
        except ImportError as e:
            print(f"[load_cora] planetoid source unavailable ({e}); falling back to linqs")
            kwargs.pop("dedupe", None)
            return load_cora_linqs(root=os.path.join(root, "cora_linqs"), **kwargs)
    if source == "linqs":
        kwargs.pop("dedupe", None)
        return load_cora_linqs(root=os.path.join(root, "cora_linqs"), **kwargs)
    raise ValueError(f"source must be 'planetoid' or 'linqs', got {source!r}")
