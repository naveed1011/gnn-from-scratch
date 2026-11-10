"""
Training loops.

Two very different regimes, and the difference is instructive:

**Node classification (Cora)** runs *full-batch*. All 2708 nodes, all 10556
edges, one forward pass per epoch. This is possible because the graph is small,
and it is why Cora trains in a couple of seconds. The loss is masked: only the
140 training nodes contribute a gradient, but information still flows through
the validation and test nodes because they are part of the graph. That is the
whole point of semi-supervised learning on graphs -- the unlabelled nodes are
not thrown away, they are used as structure.

**Graph classification (MUTAG)** runs *mini-batch* over graphs, because each
graph is an independent training example. Here you need the block-diagonal
batching trick and a real DataLoader.

Protocol note
-------------
For Cora we report the test accuracy **at the epoch where validation accuracy
peaked**, not the best test accuracy ever seen. Reporting the max test accuracy
over epochs is test-set selection and inflates results by a point or two; it is
also, unfortunately, common in GNN papers. ``train_node`` records both so you
can see the gap.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.pooling import collate_graphs
from .metrics import accuracy, macro_f1, majority_baseline

__all__ = ["NodeRunResult", "train_node", "evaluate_node", "train_graph_classification"]


# --------------------------------------------------------------------------- #
# Node classification
# --------------------------------------------------------------------------- #
@dataclass
class NodeRunResult:
    model_name: str
    seed: int
    epochs: int
    best_val_acc: float
    test_acc_at_best_val: float          # <- the honest number
    best_test_acc: float                 # <- the inflated number, kept for comparison
    val_acc_at_best_val: float
    test_f1_at_best_val: float
    train_time_s: float
    history: Dict[str, List[float]] = field(default_factory=dict)
    extra: Dict = field(default_factory=dict)

    def summary_line(self) -> str:
        return (f"{self.model_name:<12} seed={self.seed}  "
                f"val={self.best_val_acc * 100:.2f}%  "
                f"test={self.test_acc_at_best_val * 100:.2f}%  "
                f"(max_test={self.best_test_acc * 100:.2f}%)  "
                f"f1={self.test_f1_at_best_val * 100:.2f}%  "
                f"{self.train_time_s:.1f}s")


def evaluate_node(model: nn.Module, data, num_classes: int) -> Dict[str, float]:
    """Full-batch inference; returns acc/F1 for each split."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=-1)

    out = {}
    for split, mask in (("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)):
        p, t = pred[mask], data.y[mask]
        out[f"{split}_acc"] = accuracy(p, t)
        out[f"{split}_f1"] = macro_f1(p, t, num_classes)
    return out


def train_node(model: nn.Module, data, model_name: str = "model", seed: int = 0,
               epochs: int = 200, lr: float = 1e-2, weight_decay: float = 5e-4,
               dropout_ignored: bool = False, device=None, log_every: int = 0,
               on_epoch: Optional[Callable[[int, Dict[str, float]], None]] = None) -> NodeRunResult:
    """Train a node classifier with the standard Cora protocol.

    ``weight_decay`` is applied **only to the weights of the first layer** in
    the original GCN implementation, but applying it to every parameter is the
    common simplification and what PyG's examples do. We follow the latter and
    note it, because the difference is under a point of accuracy.
    """
    device = device or next(model.parameters()).device
    data = data.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    hist = {"epoch": [], "loss": [], "train_acc": [], "val_acc": [], "test_acc": [],
            "val_f1": [], "test_f1": []}

    best = {"val_acc": -1.0, "state": None, "epoch": -1, "test_acc": 0.0, "test_f1": 0.0}
    best_test = 0.0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        m = evaluate_node(model, data, data.num_classes)
        hist["epoch"].append(epoch)
        hist["loss"].append(float(loss.item()))
        for k in ("train_acc", "val_acc", "test_acc", "val_f1", "test_f1"):
            hist[k].append(m[k])

        best_test = max(best_test, m["test_acc"])
        if m["val_acc"] > best["val_acc"]:
            best = {"val_acc": m["val_acc"], "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch, "test_acc": m["test_acc"], "test_f1": m["test_f1"]}

        if on_epoch is not None:
            on_epoch(epoch, m)
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}  loss {float(loss.item()):.4f}  "
                  f"train {m['train_acc'] * 100:5.1f}%  val {m['val_acc'] * 100:5.1f}%  "
                  f"test {m['test_acc'] * 100:5.1f}%")

    elapsed = time.perf_counter() - t0
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    return NodeRunResult(
        model_name=model_name, seed=seed, epochs=epochs,
        best_val_acc=best["val_acc"],
        test_acc_at_best_val=best["test_acc"],
        best_test_acc=best_test,
        val_acc_at_best_val=best["val_acc"],
        test_f1_at_best_val=best["test_f1"],
        train_time_s=elapsed,
        history=hist,
        extra={"best_epoch": best["epoch"],
               "majority_baseline": majority_baseline(data.y[data.test_mask], data.num_classes)},
    )


def train_node_multi_seed(build_model: Callable[[], nn.Module], data, model_name: str,
                          seeds: List[int] = (0, 1, 2), **kwargs) -> Dict[str, float]:
    """Run the same config over several seeds and report mean +/- std.

    A single Cora run has enough seed-to-seed variance (~1%) that comparing two
    architectures on one seed each is close to meaningless. Three seeds is the
    minimum for a claim; the ablation script uses five.
    """
    results = []
    for s in seeds:
        from .utils import set_seed
        set_seed(s)
        results.append(train_node(build_model(), data, model_name=model_name, seed=s, **kwargs))
        print("   ", results[-1].summary_line())

    accs = torch.tensor([r.test_acc_at_best_val for r in results])
    f1s = torch.tensor([r.test_f1_at_best_val for r in results])
    return {
        "model": model_name, "seeds": list(seeds), "results": results,
        "test_acc_mean": float(accs.mean()), "test_acc_std": float(accs.std(unbiased=False)),
        "test_f1_mean": float(f1s.mean()), "test_f1_std": float(f1s.std(unbiased=False)),
        "baseline": results[0].extra["majority_baseline"],
    }


# --------------------------------------------------------------------------- #
# Graph classification
# --------------------------------------------------------------------------- #
def _stratified_folds(labels: torch.Tensor, k: int, seed: int) -> List[torch.Tensor]:
    """Assign each graph to one of ``k`` folds, keeping class proportions."""
    g = torch.Generator().manual_seed(seed)
    folds = torch.zeros(labels.numel(), dtype=torch.long)
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        for i, node in enumerate(perm):
            folds[node] = i % k
    return folds


def train_graph_classification(build_model: Callable[[], nn.Module], dataset,
                               model_name: str = "model", epochs: int = 100,
                               batch_size: int = 32, lr: float = 1e-2,
                               weight_decay: float = 1e-5, folds: int = 5,
                               seed: int = 42, device=None, verbose: bool = True) -> Dict[str, float]:
    """K-fold cross-validated graph classification.

    K-fold rather than a single split because MUTAG has only 188 graphs: one
    80/20 split leaves ~38 test graphs, so a single lucky fold swings accuracy
    by 5+ points. Five folds use every graph for testing exactly once.
    """
    from .utils import set_seed, get_device

    device = device or get_device()
    set_seed(seed)
    labels = dataset.labels()
    fold_of = _stratified_folds(labels, folds, seed)

    per_fold_acc, per_fold_f1 = [], []
    for f in range(folds):
        test_idx = (fold_of == f).nonzero(as_tuple=True)[0]
        train_idx = (fold_of != f).nonzero(as_tuple=True)[0]
        # carve a validation set out of the training folds for early stopping
        g = torch.Generator().manual_seed(seed + f)
        perm = train_idx[torch.randperm(train_idx.numel(), generator=g)]
        n_val = max(1, int(0.1 * len(train_idx)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        train_graphs = [dataset.graphs[int(i)] for i in tr_idx]
        val_graphs = [dataset.graphs[int(i)] for i in val_idx]
        test_graphs = [dataset.graphs[int(i)] for i in test_idx]

        model = build_model().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        def run_eval(graphs):
            model.eval()
            with torch.no_grad():
                preds, tgts = [], []
                for start in range(0, len(graphs), batch_size):
                    b = collate_graphs(graphs[start:start + batch_size]).to(device)
                    preds.append(model(b.x, b.edge_index, b.batch).argmax(-1))
                    tgts.append(b.y)
                p = torch.cat(preds); t = torch.cat(tgts)
            return accuracy(p, t), macro_f1(p, t, dataset.num_classes)

        best_val, best_state = -1.0, None
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_graphs), generator=g)
            total = 0.0
            for start in range(0, len(train_graphs), batch_size):
                sel = order[start:start + batch_size]
                b = collate_graphs([train_graphs[int(i)] for i in sel]).to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(b.x, b.edge_index, b.batch), b.y)
                loss.backward()
                opt.step()
                total += float(loss.item()) * b.num_graphs
            va, _ = run_eval(val_graphs)
            if va > best_val:
                best_val, best_state = va, copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)
        te, tf1 = run_eval(test_graphs)
        per_fold_acc.append(te)
        per_fold_f1.append(tf1)
        if verbose:
            print(f"  fold {f + 1}/{folds}  val {best_val * 100:5.1f}%  "
                  f"test {te * 100:5.1f}%  f1 {tf1 * 100:5.1f}%")

    accs = torch.tensor(per_fold_acc)
    f1s = torch.tensor(per_fold_f1)
    out = {
        "model": model_name,
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std(unbiased=False)),
        "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std(unbiased=False)),
        "per_fold_acc": per_fold_acc,
        "majority_baseline": dataset.majority_accuracy(),
    }
    if verbose:
        print(f"  => {model_name}: acc {out['acc_mean'] * 100:.2f} +/- "
              f"{out['acc_std'] * 100:.2f}%  (majority baseline "
              f"{out['majority_baseline'] * 100:.1f}%)")
    return out
