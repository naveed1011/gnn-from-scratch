"""
Metrics -- and why accuracy alone is the wrong headline number here.

Neither dataset in this repo is balanced:

* **Cora** class sizes range from 180 (Rule_Learning) to 818 (Neural_Networks).
  A model that always predicts Neural_Networks scores 30% -- low, but not
  zero, and a GNN that collapses to the majority class looks "80% accurate"
  only because it also happens to be right about the big classes.
* **MUTAG** is 63 vs 125. **Majority-class baseline is 66.5%, not 50%.** A GIN
  reporting 65% accuracy on MUTAG is worse than a constant.

So every table in this repo reports macro-F1 and the majority baseline next to
accuracy. Macro-F1 averages the per-class F1 *unweighted*, which is what you
want when you care about the rare classes too.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

__all__ = ["accuracy", "macro_f1", "per_class_report", "confusion_matrix",
           "dirichlet_energy", "mean_pairwise_cosine", "majority_baseline"]


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred == target).float().mean().item()) if target.numel() else 0.0


def majority_baseline(target: torch.Tensor, num_classes: int) -> float:
    """Accuracy of a constant predictor. The number your model has to beat."""
    counts = torch.bincount(target, minlength=num_classes)
    return float(counts.max().item()) / float(max(target.numel(), 1))


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``[num_classes, num_classes]`` where entry ``[t, p]`` counts true ``t`` predicted ``p``."""
    # Allocate it FLAT: (true, pred) is encoded as the single index ``true * C + pred``,
    # so a 1-D accumulator is what ``index_add_`` along dim 0 expects. Reshape at the end.
    cm = torch.zeros(num_classes * num_classes, dtype=torch.long, device=pred.device)
    cm.index_add_(0, (target * num_classes + pred).long(), torch.ones_like(target, dtype=torch.long))
    return cm.view(num_classes, num_classes)


def macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Unweighted mean of per-class F1. Classes absent from ``target`` are skipped."""
    cm = confusion_matrix(pred, target, num_classes).float()
    tp = cm.diagonal()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    prec = tp / (tp + fp).clamp(min=1e-12)
    rec = tp / (tp + fn).clamp(min=1e-12)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-12)

    support = cm.sum(dim=1)
    present = support > 0
    return float(f1[present].mean().item()) if present.any() else 0.0


def per_class_report(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                     class_names: Optional[List[str]] = None) -> Dict[str, List]:
    """Per-class precision / recall / F1 / support, as plain python lists."""
    cm = confusion_matrix(pred, target, num_classes).float()
    tp = cm.diagonal()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    prec = (tp / (tp + fp).clamp(min=1e-12)).tolist()
    rec = (tp / (tp + fn).clamp(min=1e-12)).tolist()
    f1 = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(prec, rec)]
    support = cm.sum(dim=1).long().tolist()
    names = class_names or [f"class_{i}" for i in range(num_classes)]
    return {"name": names, "precision": prec, "recall": rec, "f1": f1, "support": support}


# --------------------------------------------------------------------------- #
# Over-smoothing diagnostics
# --------------------------------------------------------------------------- #
def mean_pairwise_cosine(h: torch.Tensor) -> float:
    """Average cosine similarity between all pairs of node embeddings.

    Rises toward 1.0 as layers stack: that *is* over-smoothing, stated as a
    single number. On Cora a 2-layer GCN sits around 0.5-0.7 and an 8-layer one
    approaches 1.0, at which point every node has the same representation and
    no classifier can separate them.
    """
    with torch.no_grad():
        hn = torch.nn.functional.normalize(h, dim=-1)
        sim = hn @ hn.t()
        n = sim.size(0)
        if n < 2:
            return 1.0
        off_diag = (sim.sum() - sim.diagonal().sum()) / (n * n - n)
        return float(off_diag.item())


def dirichlet_energy(h: torch.Tensor, edge_index: torch.LongTensor,
                     num_nodes: Optional[int] = None) -> float:
    """Graph Dirichlet energy: ``sum_{(i,j) in E} ||h_i - h_j||^2``.

    The standard quantitative measure of over-smoothing. Message passing is
    literally a diffusion process, and diffusion monotonically *decreases* this
    energy -- neighbouring nodes are pulled together every layer. When it
    approaches 0 the features carry no class information left.

    Normalised by the number of edges so it is comparable across depths.
    """
    with torch.no_grad():
        num_nodes = num_nodes if num_nodes is not None else h.size(0)
        if edge_index.numel() == 0:
            return 0.0
        src, dst = edge_index[0], edge_index[1]
        diff = h.index_select(0, src) - h.index_select(0, dst)
        return float(diff.pow(2).sum().item() / edge_index.size(1))
