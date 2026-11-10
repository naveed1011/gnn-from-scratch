#!/usr/bin/env python3
"""
Run the from-scratch vs torch_geometric equivalence checks.

    python scripts/run_selftest.py              # offline checks only
    python scripts/run_selftest.py --with-data  # also compare the Cora loader to PyG

Exits non-zero if anything FAILS (skips are not failures), so this doubles as a
CI gate.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-data", action="store_true",
                    help="also compare load_cora() against PyG's Planetoid (downloads ~1 MB)")
    ap.add_argument("--root", default="data", help="where to cache datasets")
    args = ap.parse_args()

    import torch
    from gnn_lab import equivalence as eq

    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"cuda {torch.cuda.is_available()}")
    results = eq.run_all(root=args.root, include_data=args.with_data, verbose=True)

    n_pass = sum(1 for r in results if r.passed is True)
    n_fail = sum(1 for r in results if r.passed is False)
    n_skip = sum(1 for r in results if r.passed is None)
    print()
    if n_fail:
        print(f"FAILED: {n_fail} check(s) did not match torch_geometric.")
        for r in results:
            if r.passed is False:
                print("   ", r.line())
        return 1
    if n_pass < 5:
        print(f"WARNING: only {n_pass} checks ran. Is torch_geometric installed?")
        return 2
    print(f"OK: {n_pass} checks passed, {n_skip} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
