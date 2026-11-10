# `results/` — figures from the Kaggle T4 run

The `*.png` files in this directory are **committed**. They are the reference
output of the executed Kaggle notebook (`kaggle/gnn_from_scratch_t4_executed.ipynb`)
run on a free Tesla T4, and they are linked directly from the main
`README.md`.

| figure | section | what it shows |
|---|---|---|
| `01_toy_graph.png` | 1. A graph is three tensors | the 5-node 2-regular toy graph used for hand-computed outputs |
| `03_cora.png` | 3. Cora | first 4000 edges of the Cora citation graph |
| `04_training_curves.png` | 4. Train | loss/val/test curves for GCN, SAGE, GAT (3 seeds) |
| `04_results.png` | 4. Train | Cora test accuracy + macro-F1 bar chart with error bars |
| `06_oversmoothing.png` | 6. Ablations | depth sweep 1→12 layers: accuracy, mean cosine similarity, Dirichlet energy |
| `06_ablations.png` | 6. Ablations | aggregator (mean/max/sum) and self-loop sweeps |
| `06b_layerwise.png` | 6b. Over-smoothing | per-layer similarity + energy for an 8-layer GCN |
| `06b_tsne.png` | 6b. Over-smoothing | t-SNE of node embeddings after 0, 1, 2, 8 layers — clusters merge |
| `07_attention.png` | 7. GAT attention | same-class vs cross-class attention coefficient distributions |
| `07_attention_graph.png` | 7. GAT attention | subgraph with edge width proportional to learned attention |
| `08_mutag.png` | 8. GIN on MUTAG | 5-fold CV accuracy per readout (sum / mean / max) |
| `08b_triangle.png` | 8b. Expressiveness | triangle-detection negative result — every readout near chance |

you can run this:

```bash
python scripts/train_node_classification.py --models gcn sage gat --seeds 0 1 2
python scripts/train_graph_classification.py --dataset MUTAG --folds 5
python scripts/run_ablations.py --depths 1 2 3 4 6 8 12
```

## Environment requirements of the committed run

- Python 3.13.16, Linux x86_64
- PyTorch 2.11.0+cu130, CUDA 13.0, Tesla T4 (15.6 GB)
- torch_geometric 2.8.1 (only for the equivalence section)

Scatter reductions on CUDA are not bitwise deterministic in float order, so
expect small run-to-run variation even with a fixed seed.
