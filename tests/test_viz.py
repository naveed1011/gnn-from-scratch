"""
Smoke tests for ``gnn_lab.viz``.

The training scripts call every ``plot_*`` helper with ``show=False`` so they can run
headless on Kaggle. Before these tests existed, most helpers did not accept ``show``,
so a script would train for minutes and then crash while saving its figure.
"""

from __future__ import annotations

import inspect

import matplotlib
import pytest

matplotlib.use("Agg")

from gnn_lab import viz  # noqa: E402

PLOTS = ["plot_training_curves", "plot_oversmoothing", "plot_attention_distribution",
         "plot_confusion", "plot_layerwise_similarity", "plot_results_bars"]


@pytest.mark.parametrize("name", PLOTS)
def test_every_plot_helper_accepts_show(name):
    assert "show" in inspect.signature(getattr(viz, name)).parameters


def test_results_bars_and_curves_write_files(tmp_path):
    rows = [{"model": "GCN", "test_acc_mean": 0.8, "test_acc_std": 0.01},
            {"model": "GAT", "test_acc_mean": 0.82, "test_acc_std": 0.01}]
    p1 = tmp_path / "bars.png"
    viz.plot_results_bars(rows, title="t", path=str(p1), show=False)
    assert p1.exists() and p1.stat().st_size > 0

    hist = {"GCN": {"train_loss": [1.0, 0.5], "val_acc": [0.5, 0.7]}}
    p2 = tmp_path / "curves.png"
    viz.plot_training_curves(hist, title="t", path=str(p2), show=False)
    assert p2.exists() and p2.stat().st_size > 0


def test_draw_graph_rejects_per_edge_labels_with_a_clear_message(tmp_path):
    """Regression: a notebook passed one label per edge; matplotlib's error was cryptic."""
    import torch
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    with pytest.raises(ValueError, match="one entry per NODE"):
        viz.draw_graph(ei, labels=torch.zeros(2, dtype=torch.long), path=str(tmp_path / "g.png"))
    # per-node labels work, even when the highest-numbered node is isolated
    viz.draw_graph(ei, labels=torch.zeros(6, dtype=torch.long), path=str(tmp_path / "g.png"))
    assert (tmp_path / "g.png").exists()
