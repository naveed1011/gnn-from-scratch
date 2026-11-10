"""
Plotting helpers.

Every function returns the matplotlib ``Figure`` and optionally saves it, so a
notebook can display it inline and a script can write it to ``results/``.

Optional dependencies are imported lazily and degrade gracefully:

* ``networkx`` -- only for :func:`draw_graph`. Without it you get a message
  instead of an exception.
* ``scikit-learn`` -- only for the t-SNE in :func:`plot_tsne`.

matplotlib is required by everything here; the rest of ``gnn_lab`` never
imports it, so training still works in a headless environment.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import torch

__all__ = [
    "save_or_show", "draw_graph", "plot_training_curves", "plot_oversmoothing",
    "plot_tsne", "plot_attention_distribution", "plot_confusion",
    "plot_layerwise_similarity", "plot_results_bars",
]

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")          # safe headless default; notebooks override inline
    import matplotlib.pyplot as plt
    return plt


def save_or_show(fig, path: Optional[str] = None, dpi: int = 150, show: bool = True):
    """Save to ``path`` if given, otherwise display inline. Returns the figure."""
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"  saved {path}")
    elif show:
        _plt().show()
    return fig


# --------------------------------------------------------------------------- #
def draw_graph(edge_index: torch.LongTensor, labels: Optional[torch.Tensor] = None,
               pos: Optional[dict] = None, title: str = "", path: Optional[str] = None,
               node_size: int = 120, figsize=(8, 6), highlight_edges: Optional[torch.Tensor] = None,
               ax=None):
    """Draw a graph with networkx, coloured by node label.

    ``highlight_edges`` takes a per-edge weight in ``[0, 1]``; edges are then
    drawn with width and opacity proportional to it, which is how the GAT
    attention visualisation works.
    """
    try:
        import networkx as nx
    except ImportError:
        print("draw_graph needs networkx: pip install networkx")
        return None

    plt = _plt()
    ei = edge_index.detach().cpu()
    n = int(ei.max().item()) + 1 if ei.numel() else (len(labels) if labels is not None else 1)
    if labels is not None and len(labels) < n:
        raise ValueError(f"draw_graph: got {len(labels)} labels for {n} nodes. `labels` must "
                         "be one entry per NODE, not per edge.")
    n = max(n, len(labels)) if labels is not None else n
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(int(a), int(b)) for a, b in ei.t().tolist() if a != b])

    if pos is None:
        pos = nx.spring_layout(G, seed=0, k=0.45)

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if labels is not None:
        lab = labels.detach().cpu().numpy()
        colours = [_PALETTE[int(v) % len(_PALETTE)] for v in lab]
    else:
        colours = "#9bb7d4"

    if highlight_edges is None:
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.6, edge_color="#888888")
    else:
        w = highlight_edges.detach().cpu().numpy()
        seen = set()
        for k, (a, b) in enumerate(ei.t().tolist()):
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            v = float(w[k])
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color="#C44E52" if v > 0.5 else "#4C72B0",
                    alpha=0.15 + 0.85 * v, linewidth=0.4 + 3.0 * v, zorder=1)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color=colours,
                           linewidths=0.4, edgecolors="#333333")
    if n <= 40:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    if title:
        ax.set_title(title)
    ax.axis("off")
    return save_or_show(fig, path, show=own_ax)


# --------------------------------------------------------------------------- #
def plot_training_curves(histories: Dict[str, Dict[str, List[float]]],
                         title: str = "Training curves", path: Optional[str] = None,
                         figsize=(12, 4), show: bool = True):
    """Loss / train / val / test accuracy, one panel each, one line per model."""
    plt = _plt()
    keys = ["loss", "train_acc", "val_acc", "test_acc"]
    titles = ["Loss", "Train acc", "Val acc", "Test acc"]
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)

    for c, (name, hist) in enumerate(histories.items()):
        ep = hist.get("epoch", list(range(len(hist.get("loss", [])))))
        for ax, key, t in zip(axes, keys, titles):
            if key in hist and len(hist[key]) == len(ep):
                ax.plot(ep, hist[key], label=name, color=_PALETTE[c % len(_PALETTE)], lw=1.6)
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if "acc" in t:
            ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return save_or_show(fig, path, show=show)


def plot_oversmoothing(depths: Sequence[int], rows: Sequence[dict],
                       path: Optional[str] = None, figsize=(13, 4), show: bool = True):
    """The over-smoothing picture: accuracy collapses as similarity saturates.

    ``rows`` is a list of dicts with keys ``test_acc``, ``cosine``, ``energy``,
    ``residual`` (bool) -- one per depth.
    """
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    groups: Dict[str, List[dict]] = {}
    for d, r in zip(depths, rows):
        groups.setdefault(str(r.get("variant", "plain")), []).append({"depth": d, **r})

    for c, (variant, rs) in enumerate(groups.items()):
        rs = sorted(rs, key=lambda z: z["depth"])
        d = [z["depth"] for z in rs]
        col = _PALETTE[c % len(_PALETTE)]
        axes[0].plot(d, [z["test_acc"] for z in rs], "-o", color=col, label=variant, ms=4)
        axes[1].plot(d, [z["cosine"] for z in rs], "-o", color=col, ms=4)
        axes[2].plot(d, [z["energy"] for z in rs], "-o", color=col, ms=4)

    axes[0].set_title("Test accuracy vs depth")
    axes[0].set_ylabel("accuracy")
    axes[1].set_title("Mean pairwise cosine similarity")
    axes[1].set_ylabel("cosine")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_title("Dirichlet energy (per edge)")
    axes[2].set_ylabel("energy")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("number of GCN layers")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return save_or_show(fig, path, show=show)


def plot_tsne(h: torch.Tensor, y: torch.Tensor, title: str = "", path: Optional[str] = None,
              perplexity: float = 30.0, ax=None, seed: int = 0):
    """t-SNE of node embeddings coloured by class. Needs scikit-learn."""
    plt = _plt()
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("plot_tsne needs scikit-learn: pip install scikit-learn")
        return None

    X = h.detach().cpu().numpy()
    perp = max(5.0, min(perplexity, (X.shape[0] - 1) / 3.0))
    Z = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed,
             learning_rate="auto").fit_transform(X)
    lab = y.detach().cpu().numpy()

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    for c in sorted(set(lab.tolist())):
        m = lab == c
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.65, color=_PALETTE[c % len(_PALETTE)],
                   label=f"class {c}")
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=7, frameon=False)
    ax.set_xticks([]); ax.set_yticks([])
    return save_or_show(fig, path, show=own)


def plot_attention_distribution(att_same: torch.Tensor, att_diff: torch.Tensor,
                                title: str = "GAT attention: same-class vs cross-class edges",
                                path: Optional[str] = None, bins: int = 60, show: bool = True):
    """Do learned attention weights separate same-class from cross-class edges?

    If GAT is doing what it claims, the same-class histogram should sit to the
    right of the cross-class one. On homophilous Cora the separation is visible
    but far from clean -- which is an honest result worth reporting rather than
    a figure worth doctoring.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    a = att_same.detach().cpu().numpy().ravel()
    b = att_diff.detach().cpu().numpy().ravel()
    ax.hist(a, bins=bins, alpha=0.6, color=_PALETTE[0], density=True,
            label=f"same class (mean {a.mean():.4f})")
    ax.hist(b, bins=bins, alpha=0.6, color=_PALETTE[3], density=True,
            label=f"different class (mean {b.mean():.4f})")
    ax.set_xlabel("attention coefficient")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_or_show(fig, path, show=show)


def plot_confusion(cm: torch.Tensor, class_names: Optional[List[str]] = None,
                   title: str = "Confusion matrix", path: Optional[str] = None,
                   normalize: bool = True, show: bool = True):
    plt = _plt()
    C = cm.detach().cpu().float()
    if normalize:
        C = C / C.sum(dim=1, keepdim=True).clamp(min=1)
    names = class_names or [str(i) for i in range(C.size(0))]

    fig, ax = plt.subplots(figsize=(1.4 * C.size(0) + 1.5, 1.3 * C.size(0) + 1.2))
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=float(C.max()))
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(C.size(0)):
        for j in range(C.size(1)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="white" if C[i, j] > C.max() / 2 else "#222222", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return save_or_show(fig, path, show=show)


def plot_layerwise_similarity(hs: Sequence[torch.Tensor], edge_index: torch.LongTensor,
                              title: str = "How representations collapse layer by layer",
                              path: Optional[str] = None, show: bool = True):
    """Cosine similarity and Dirichlet energy after each layer of one forward pass.

    ``hs`` is the list of hidden states from a single forward pass (layer 0 = the
    input features) and ``edge_index`` is the graph they were computed on. Both
    diagnostics should move monotonically in opposite directions: similarity up,
    energy down. That is diffusion, and it is why stacking layers has a limit.
    """
    from .metrics import dirichlet_energy, mean_pairwise_cosine

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    layers = list(range(len(hs)))

    cos = [mean_pairwise_cosine(h) for h in hs]
    energy = [dirichlet_energy(h, edge_index, num_nodes=h.size(0)) for h in hs]

    axes[0].plot(layers, cos, "-o", color=_PALETTE[0])
    axes[0].set_ylabel("mean pairwise cosine")
    axes[0].set_title("Similarity between node embeddings")
    axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(layers, energy, "-o", color=_PALETTE[3])
    axes[1].set_ylabel("Dirichlet energy")
    axes[1].set_title("Energy of the feature signal")
    axes[1].set_yscale("log")

    for ax in axes:
        ax.set_xlabel("layer"); ax.grid(alpha=0.3)
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    return save_or_show(fig, path, show=show)


def plot_results_bars(results: Sequence[dict], metric: str = "test_acc_mean",
                      std: Optional[str] = "test_acc_std", label_key: str = "model",
                      title: str = "", path: Optional[str] = None, baseline: Optional[float] = None, show: bool = True):
    """Horizontal bar chart of model results with error bars and a baseline line."""
    plt = _plt()
    labels = [r[label_key] for r in results]
    vals = [float(r[metric]) for r in results]
    errs = [float(r[std]) for r in results] if std and std in results[0] else None

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 1.8))
    y = range(len(labels))
    ax.barh(list(y), vals, xerr=errs, color=[_PALETTE[i % len(_PALETTE)] for i in y],
            alpha=0.9, error_kw=dict(lw=1, capsize=3))
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1.0)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v * 100:.2f}%", va="center", fontsize=9)
    if baseline is not None:
        ax.axvline(baseline, color="#C44E52", ls="--", lw=1.2,
                   label=f"majority baseline {baseline * 100:.1f}%")
        ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return save_or_show(fig, path, show=show)
