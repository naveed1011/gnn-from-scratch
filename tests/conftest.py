"""
pytest configuration.

Adds the repository root to ``sys.path`` so ``import gnn_lab`` works when you run
``pytest`` from anywhere, and registers the ``network`` marker used to separate
the offline tests from the ones that download a dataset.

Default behaviour
-----------------
    pytest                       # fast, offline, no downloads
    pytest -m network            # only the dataset tests (downloads ~1.5 MB once)
    pytest -m "not network"      # same as the default, stated explicitly
    pytest tests/test_equivalence.py -s    # see the from-scratch vs PyG table
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: test needs to download a dataset (deselect with `-m 'not network'`)",
    )
