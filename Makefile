# Graph Neural Networks from Scratch -- developer entry points.
#
#   make help        list targets
#   make install     create the environment and install the package
#   make selftest    prove the from-scratch layers match torch_geometric
#   make test        run the offline test suite
#   make results     regenerate every number and figure in the README
#   make notebook    regenerate kaggle/gnn_from_scratch_t4.ipynb from the package
#   make all         selftest + test + results + notebook

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
OUT    ?= results

.DEFAULT_GOAL := help
.PHONY: help install install-dev selftest test test-all results node graph ablations \
        notebook clean clean-data lint check

help:  ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## install runtime deps and the package in editable mode
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install-dev:  ## install everything including torch_geometric and pytest
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e ".[all]"

selftest:  ## numerical equivalence checks vs torch_geometric (float64)
	$(PYTHON) scripts/run_selftest.py --with-data

test:  ## offline test suite -- no downloads, no torch_geometric needed
	$(PYTEST)

test-all:  ## offline + network tests (downloads ~1.5 MB of datasets once)
	$(PYTEST) -m "network or not network"

lint:  ## style + static checks
	$(PYTHON) -m ruff check gnn_lab scripts tests
	$(PYTHON) -m ruff format --check gnn_lab scripts tests || true

check:  ## the AST convention checks on their own (no torch required)
	$(PYTEST) tests/test_index_convention.py -v

node:  ## train GCN / GraphSAGE / GAT on Cora
	$(PYTHON) scripts/train_node_classification.py --out $(OUT)

graph:  ## GIN on MUTAG + the triangle-detection expressiveness experiment
	$(PYTHON) scripts/train_graph_classification.py --out $(OUT)

ablations:  ## depth / aggregator / self-loops / duplicate-edge sweeps
	$(PYTHON) scripts/run_ablations.py --out $(OUT)

results: node graph ablations  ## regenerate every number and figure in the README
	@echo
	@echo "results written to $(OUT)/"
	@ls -1 $(OUT) 2>/dev/null || true

notebook:  ## regenerate the self-contained Kaggle notebook from gnn_lab
	$(PYTHON) scripts/build_kaggle_notebook.py

clean:  ## remove caches and generated results (keeps downloaded datasets)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build *.egg-info
	rm -f $(OUT)/*.json $(OUT)/*.png

clean-data: clean  ## also remove downloaded datasets
	rm -rf data
