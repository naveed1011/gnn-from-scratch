#!/usr/bin/env python3
"""
The experiments that actually teach you something.

    python scripts/run_ablations.py                  # everything
    python scripts/run_ablations.py --only depth     # one sweep
    python scripts/run_ablations.py --depths 1 2 4 8 16 --epochs 100

Four sweeps, each answering one question:

``depth``      Does stacking layers help? Measures test accuracy alongside mean
               pairwise cosine similarity and Dirichlet energy, which is how you
               tell "the model is underfitting" apart from "the representations
               have collapsed". Compares plain vs residual vs jumping-knowledge.

``aggregator`` Same GraphSAGE, three aggregation operators. The message-passing
               abstraction reduced to its one degree of freedom.

``self_loops`` GCN and GAT both rely on a node being able to attend to itself.
               Remove it and see how much of the accuracy was the node's own
               features all along.

``dedupe``     Cora's raw adjacency dict has 302 duplicate entries. PyG removes
               them, tkipf/gcn does not. This measures what the choice is worth,
               which is the kind of thing nobody tells you.

Results go to ``results/ablations.json`` plus figures in ``results/``.
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
    ap.add_argument("--only", nargs="+",
                    choices=["depth", "aggregator", "self_loops", "dedupe"],
                    default=None, help="run a subset of the sweeps")
    ap.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8, 12, 16])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from gnn_lab.data import load_cora
    from gnn_lab.layers import GCNScratch, GraphSAGEScratch
    from gnn_lab.metrics import dirichlet_energy, mean_pairwise_cosine
    from gnn_lab.train import train_node
    from gnn_lab.utils import get_device, set_seed

    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    sweeps = args.only or ["depth", "aggregator", "self_loops", "dedupe"]
    payload: dict = {"device": str(device), "epochs": args.epochs, "seeds": args.seeds}

    data = load_cora(root=args.root, source="planetoid", row_normalize=False).to(device)
    print(data.summary(), "\n")

    class ResidualGCN(nn.Module):
        """GCN with a skip connection per layer: ``h <- h + dropout(relu(conv(h)))``.

        The cheapest known defence against over-smoothing: even if a layer has
        diffused everything into mush, the residual path still carries the
        shallower representation forward. Not a cure, but enough to make the
        mechanism obvious.

        ``depth`` is the number of **conv layers**, the same meaning it has for the
        plain and jumping-knowledge variants. (An earlier version built ``depth - 1``
        convs, so "depth 1" was a graph-free MLP at ~54%.) It also had a LayerNorm
        after each add, which let the raw-feature path drown out the graph signal and
        held the model at 50-58% at every depth; without it the model keeps ~75% out
        to 16 layers. Dropout on the input features matters too: with 140 labels the
        input projection overfits badly without it.
        """

        def __init__(self, f, h, c, depth, dropout=0.5):
            super().__init__()
            from gnn_lab.layers import GCNConvScratch
            self.dropout = dropout
            self.input_proj = nn.Linear(f, h)
            self.convs = nn.ModuleList([GCNConvScratch(h, h) for _ in range(depth)])
            self.out = nn.Linear(h, c)

        def forward(self, x, edge_index):
            h = self.input_proj(F.dropout(x, p=self.dropout, training=self.training))
            for conv in self.convs:
                h = h + F.dropout(F.relu(conv(h, edge_index)),
                                  p=self.dropout, training=self.training)
            return self.out(h)

        def embeddings(self, x, edge_index):
            h = self.input_proj(x)
            for conv in self.convs:
                h = h + F.relu(conv(h, edge_index))
            return h

    class JumpingKnowledgeGCN(nn.Module):
        """Concatenate every layer's output and classify from the concatenation.

        Lets the classifier *choose* the receptive field instead of committing to
        the last layer's. This is the JK-Net idea and it is what makes deep GNNs
        usable at all.
        """

        def __init__(self, f, h, c, depth, dropout=0.5):
            super().__init__()
            from gnn_lab.layers import GCNConvScratch
            self.dropout = dropout
            self.convs = nn.ModuleList()
            dims = [f] + [h] * depth
            for i in range(depth):
                self.convs.append(GCNConvScratch(dims[i], dims[i + 1]))
            self.out = nn.Linear(h * depth, c)

        def forward(self, x, edge_index):
            hs = []
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
                hs.append(x)
            return self.out(torch.cat(hs, dim=-1))

        def embeddings(self, x, edge_index):
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            return x

    # ------------------------------------------------------------------ depth
    if "depth" in sweeps:
        print("=== depth sweep ===")
        rows = []
        variants = {
            "plain": lambda d: GCNScratch(data.num_features, args.hidden, data.num_classes,
                                          num_layers=d, dropout=0.5),
            "residual": lambda d: ResidualGCN(data.num_features, args.hidden,
                                              data.num_classes, d),
            "jumping_knowledge": lambda d: JumpingKnowledgeGCN(data.num_features, args.hidden,
                                                               data.num_classes, d),
        }
        for depth in args.depths:
            for vname, factory in variants.items():
                accs, coss, ens = [], [], []
                for seed in args.seeds:
                    set_seed(seed)
                    model = factory(depth)
                    r = train_node(model, data, model_name=f"GCN-{depth}L",
                                   epochs=args.epochs, device=device, seed=seed)
                    accs.append(r.test_acc_at_best_val)
                    model.eval()
                    with torch.no_grad():
                        # every variant exposes .embeddings(), so the cosine and
                        # energy numbers are measured on comparable representations
                        h = model.embeddings(data.x, data.edge_index)
                    coss.append(mean_pairwise_cosine(h))
                    ens.append(dirichlet_energy(h, data.edge_index, data.num_nodes))
                row = {"depth": depth, "variant": vname,
                       "test_acc": sum(accs) / len(accs),
                       "cosine": sum(coss) / len(coss),
                       "energy": sum(ens) / len(ens)}
                rows.append(row)
                print(f"  {depth:2d} layers  {vname:17} test {row['test_acc']*100:5.2f}%  "
                      f"cosine {row['cosine']:.4f}  energy {row['energy']:.3e}")
        payload["depth"] = rows

        if not args.no_plots:
            try:
                from gnn_lab.viz import plot_oversmoothing
                plot_oversmoothing([r["depth"] for r in rows], rows, show=False,
                                   path=os.path.join(args.out, "ablation_depth.png"))
            except ImportError as e:
                print(f"  (no plots: {e})")

    # ------------------------------------------------------------- aggregator
    if "aggregator" in sweeps:
        print("\n=== aggregator sweep (GraphSAGE, 2 layers) ===")
        rows = []
        for aggr in ("mean", "max", "sum"):
            accs, f1s = [], []
            for seed in args.seeds:
                set_seed(seed)
                model = GraphSAGEScratch(data.num_features, args.hidden, data.num_classes,
                                         num_layers=2, dropout=0.5, aggr=aggr)
                r = train_node(model, data, model_name=f"SAGE-{aggr}",
                               epochs=args.epochs, device=device, seed=seed)
                accs.append(r.test_acc_at_best_val)
                f1s.append(r.test_f1_at_best_val)
            rows.append({"aggr": aggr,
                         "test_acc": sum(accs) / len(accs),
                         "test_f1": sum(f1s) / len(f1s)})
            print(f"  aggr={aggr:5} test {rows[-1]['test_acc']*100:5.2f}%  "
                  f"f1 {rows[-1]['test_f1']*100:5.2f}%")
        payload["aggregator"] = rows

    # ------------------------------------------------------------ self loops
    if "self_loops" in sweeps:
        print("\n=== self-loop ablation ===")
        rows = []
        for loops in (True, False):
            accs = []
            for seed in args.seeds:
                set_seed(seed)
                model = GCNScratch(data.num_features, args.hidden, data.num_classes,
                                   num_layers=2, dropout=0.5, add_self_loops=loops)
                r = train_node(model, data, model_name="GCN", epochs=args.epochs,
                               device=device, seed=seed)
                accs.append(r.test_acc_at_best_val)
            rows.append({"add_self_loops": loops, "test_acc": sum(accs) / len(accs)})
            print(f"  add_self_loops={loops!s:5} test {rows[-1]['test_acc']*100:5.2f}%")
        payload["self_loops"] = rows

    # --------------------------------------------------------------- dedupe
    if "dedupe" in sweeps:
        print("\n=== duplicate-edge ablation ===")
        rows = []
        for dedupe in (True, False):
            d = load_cora(root=args.root, source="planetoid", row_normalize=False,
                          dedupe=dedupe, verbose=False).to(device)
            accs = []
            for seed in args.seeds:
                set_seed(seed)
                model = GCNScratch(d.num_features, args.hidden, d.num_classes,
                                   num_layers=2, dropout=0.5)
                r = train_node(model, d, model_name="GCN", epochs=args.epochs,
                               device=device, seed=seed)
                accs.append(r.test_acc_at_best_val)
            rows.append({"dedupe": dedupe, "num_edges": d.num_edges,
                         "test_acc": sum(accs) / len(accs)})
            print(f"  dedupe={dedupe!s:5} edges {d.num_edges:5d}  "
                  f"test {rows[-1]['test_acc']*100:5.2f}%")
        payload["dedupe"] = rows

    out = os.path.join(args.out, "ablations.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out}")

    if not args.no_plots and "aggregator" in payload:
        try:
            from gnn_lab.viz import plot_results_bars
            rows = [{"model": f"SAGE-{r['aggr']}", "test_acc_mean": r["test_acc"],
                     "test_acc_std": 0.0} for r in payload["aggregator"]]
            plot_results_bars(rows, title="GraphSAGE aggregator ablation (Cora)", show=False,
                              path=os.path.join(args.out, "ablation_aggregator.png"))
        except ImportError:
            pass
    return 0


if __name__ == "__main__":
    t = time.perf_counter()
    code = main()
    print(f"total wall time {time.perf_counter() - t:.1f}s")
    raise SystemExit(code)
