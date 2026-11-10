#!/usr/bin/env python3
"""
Train GCN / GraphSAGE / GAT on Cora and write results + figures.

    python scripts/train_node_classification.py
    python scripts/train_node_classification.py --models gcn gat --seeds 0 1 2 3 4
    python scripts/train_node_classification.py --source linqs --device cpu

Writes ``results/node_classification.json`` and the plots into ``results/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build(name: str, num_features: int, hidden: int, num_classes: int):
    from gnn_lab.layers import GATScratch, GCNScratch, GraphSAGEScratch
    if name == "gcn":
        return lambda: GCNScratch(num_features, hidden, num_classes, num_layers=2, dropout=0.5)
    if name == "sage":
        return lambda: GraphSAGEScratch(num_features, hidden, num_classes,
                                        num_layers=2, dropout=0.5, aggr="mean")
    if name == "gat":
        return lambda: GATScratch(num_features, hidden, num_classes, heads=8, dropout=0.6)
    raise KeyError(f"unknown model {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["gcn", "sage", "gat"],
                    choices=["gcn", "sage", "gat"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--source", default="planetoid", choices=["planetoid", "linqs"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    import torch

    from gnn_lab.data import load_cora
    from gnn_lab.metrics import per_class_report
    from gnn_lab.train import evaluate_node, train_node_multi_seed
    from gnn_lab.utils import count_parameters, get_device, set_seed

    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    print(f"device: {device}")
    print(f"loading Cora from {args.source!r} ...\n")
    data = load_cora(root=args.root, source=args.source).to(device)
    print()

    payload = {"dataset": "cora", "source": args.source, "device": str(device),
               "hidden": args.hidden, "epochs": args.epochs, "lr": args.lr,
               "weight_decay": args.weight_decay, "seeds": args.seeds, "models": {}}

    histories = {}
    bar_rows = []
    for name in args.models:
        print(f"=== {name.upper()} ({len(args.seeds)} seeds x {args.epochs} epochs) ===")
        t0 = time.perf_counter()
        res = train_node_multi_seed(
            build(name, data.num_features, args.hidden, data.num_classes),
            data, model_name=name.upper(), seeds=args.seeds, epochs=args.epochs,
            lr=args.lr, weight_decay=args.weight_decay, device=device)
        n_params = count_parameters(build(name, data.num_features, args.hidden,
                                          data.num_classes)())
        res["num_parameters"] = n_params
        # capture the last seed's curves for plotting *before* dropping the
        # non-serialisable dataclass objects out of the JSON payload
        histories[name.upper()] = res["results"][-1].history
        res.pop("results", None)
        payload["models"][name] = res
        bar_rows.append({"model": name.upper(), "test_acc_mean": res["test_acc_mean"],
                         "test_acc_std": res["test_acc_std"]})
        print(f"  parameters: {n_params:,}   wall: {time.perf_counter() - t0:.1f}s")
        print(f"  test acc  : {res['test_acc_mean'] * 100:.2f} +/- "
              f"{res['test_acc_std'] * 100:.2f}%")
        print(f"  macro F1  : {res['test_f1_mean'] * 100:.2f} +/- "
              f"{res['test_f1_std'] * 100:.2f}%\n")

    baseline = next(iter(payload["models"].values()))["baseline"]
    print("=" * 62)
    print(f"{'model':8}{'test acc':>20}{'macro F1':>20}")
    for m in payload["models"].values():
        print(f"{m['model']:8}{m['test_acc_mean']*100:>10.2f} +/- {m['test_acc_std']*100:>5.2f}%"
              f"{m['test_f1_mean']*100:>11.2f} +/- {m['test_f1_std']*100:>5.2f}%")
    print(f"{'baseline':8}{baseline*100:>10.2f}%  (majority class on the test split)")
    print("=" * 62)

    # per-class breakdown for the first model, which is where Cora's imbalance shows
    set_seed(args.seeds[0])
    model = build(args.models[0], data.num_features, args.hidden, data.num_classes)()
    from gnn_lab.train import train_node
    train_node(model, data, model_name=args.models[0].upper(), epochs=args.epochs,
               device=device, seed=args.seeds[0])
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(-1)
    report = per_class_report(pred[data.test_mask], data.y[data.test_mask],
                              data.num_classes, data.class_names or None)
    payload["per_class"] = report
    print(f"\nper-class breakdown ({args.models[0].upper()}, seed {args.seeds[0]}):")
    print(f"  {'class':<24}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    for i, nm in enumerate(report["name"]):
        print(f"  {nm:<24}{report['precision'][i]:>7.3f}{report['recall'][i]:>8.3f}"
              f"{report['f1'][i]:>7.3f}{report['support'][i]:>9}")

    out_path = os.path.join(args.out, "node_classification.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_path}")

    if not args.no_plots:
        try:
            from gnn_lab.viz import plot_results_bars, plot_training_curves
            if histories:
                plot_training_curves(histories, title="Cora training curves",
                                     path=os.path.join(args.out, "training_curves.png"),
                                     show=False)
            plot_results_bars(bar_rows, title="Cora test accuracy",
                              baseline=baseline,
                              path=os.path.join(args.out, "node_results.png"), show=False)
        except ImportError as e:
            print(f"skipping plots ({e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
