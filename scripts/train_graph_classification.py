#!/usr/bin/env python3
"""
Graph-level classification: GIN on MUTAG, plus the expressiveness experiment.

    python scripts/train_graph_classification.py
    python scripts/train_graph_classification.py --dataset PROTEINS --folds 3
    python scripts/train_graph_classification.py --only triangle --quick

Two runs, because they answer different questions:

``mutag``     A real dataset. 188 molecules, 5-fold CV, three readout operators.
              The headline number to remember is that MUTAG is **imbalanced**
              (63 vs 125), so the majority-class baseline is 66.5% and any
              result below that is worse than predicting a constant.

``triangle``  A synthetic dataset whose label *is* "does this graph contain a
              3-cycle?", with classes matched on size and density. Every readout is
              expected near chance (3-cycle detection is beyond 1-WL message
              passing). It is a shortcut detector: a clear gap means a leak crept
              into the data, as one did in an earlier version.

Results go to ``results/graph_classification.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="MUTAG", choices=["MUTAG", "PROTEINS"])
    ap.add_argument("--only", nargs="+", choices=["real", "triangle"], default=None)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--pools", nargs="+", default=["sum", "mean", "max"],
                    choices=["sum", "mean", "max"])
    ap.add_argument("--triangle-graphs", type=int, default=400)
    ap.add_argument("--triangle-folds", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="fewer epochs/folds -- smoke test, not a result")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 25)
        args.folds = min(args.folds, 3)
        args.triangle_graphs = min(args.triangle_graphs, 120)
        args.triangle_folds = min(args.triangle_folds, 3)

    from gnn_lab.data import load_tu, triangle_square_dataset
    from gnn_lab.layers import GINScratch
    from gnn_lab.train import train_graph_classification
    from gnn_lab.utils import count_parameters, get_device

    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    runs = args.only or ["real", "triangle"]
    payload: dict = {"device": str(device), "hidden": args.hidden, "layers": args.layers,
                     "epochs": args.epochs, "folds": args.folds}

    def build_gin(num_features, num_classes, pool):
        return lambda: GINScratch(num_features, args.hidden, num_classes,
                                  num_layers=args.layers, dropout=args.dropout,
                                  pool=pool).to(device)

    bar_rows = []

    # ---------------------------------------------------------------- real data
    if "real" in runs:
        print(f"=== {args.dataset}: GIN, {args.folds}-fold CV ===")
        ds = load_tu(args.dataset, root=args.root)
        print()
        payload["dataset"] = {"name": ds.name, "graphs": len(ds), "nodes": ds.num_nodes,
                              "features": ds.num_features, "classes": ds.num_classes,
                              "majority_baseline": ds.majority_accuracy()}
        results = []
        for pool in args.pools:
            t0 = time.perf_counter()
            r = train_graph_classification(
                build_gin(ds.num_features, ds.num_classes, pool), ds,
                model_name=f"GIN-{pool}", epochs=args.epochs, batch_size=args.batch_size,
                folds=args.folds, lr=args.lr, device=device)
            r["seconds"] = round(time.perf_counter() - t0, 1)
            r["num_parameters"] = count_parameters(build_gin(ds.num_features, ds.num_classes,
                                                             pool)())
            results.append(r)
            bar_rows.append({"model": f"{ds.name}/GIN-{pool}", "test_acc_mean": r["acc_mean"],
                             "test_acc_std": r["acc_std"]})
        payload["real"] = results
        print(f"\n  majority-class baseline: {ds.majority_accuracy() * 100:.2f}%  "
              f"<- the number every row above must beat")

    # ------------------------------------------------------------ expressiveness
    if "triangle" in runs:
        print(f"\n=== triangle detection: {args.triangle_graphs} synthetic graphs, "
              f"{args.triangle_folds}-fold CV ===")
        ts = triangle_square_dataset(num_graphs=args.triangle_graphs, seed=0)
        print(ts.summary(), "\n")
        payload["triangle"] = {"graphs": len(ts), "majority_baseline": ts.majority_accuracy()}
        results = []
        # all three readouts are expected near 0.5 (see the module docstring).
        for pool in ("mean", "max", "sum"):
            r = train_graph_classification(
                build_gin(ts.num_features, ts.num_classes, pool), ts,
                model_name=f"GIN-{pool}-pool", epochs=args.epochs,
                batch_size=args.batch_size, folds=args.triangle_folds, lr=args.lr,
                device=device)
            results.append(r)
            bar_rows.append({"model": f"triangle/GIN-{pool}", "test_acc_mean": r["acc_mean"],
                             "test_acc_std": r["acc_std"]})
        payload["triangle"]["results"] = results

        gap = next(r["acc_mean"] for r in results if r["model"] == "GIN-sum-pool") - \
              next(r["acc_mean"] for r in results if r["model"] == "GIN-mean-pool")
        print(f"\n  sum - mean readout gap: {gap * 100:+.2f} points")
        print("  Expected: near zero, with every readout near 50%. Classes are matched on size")
        print("  and density, and 3-cycle detection is beyond 1-WL message passing. A clearly")
        print("  positive gap would suggest a shortcut in the data, so check for one first.")
        payload["triangle"]["sum_minus_mean_gap"] = gap

    out = os.path.join(args.out, "graph_classification.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out}")

    if not args.no_plots and bar_rows:
        try:
            from gnn_lab.viz import plot_results_bars
            plot_results_bars(bar_rows, title="Graph classification (cross-validated)",
                              show=False, path=os.path.join(args.out, "graph_results.png"))
        except ImportError as e:
            print(f"skipping plots ({e})")
    return 0


if __name__ == "__main__":
    t = time.perf_counter()
    code = main()
    print(f"total wall time {time.perf_counter() - t:.1f}s")
    raise SystemExit(code)
