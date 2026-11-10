"""
Static check on the src/dst index convention in every hand-written layer.

This test exists because of a real bug caught while writing this repo. The first
draft of ``GCNConvScratch`` did::

    row, col = ei[0], ei[1]
    msg = x.index_select(0, col) * norm      # <-- gathers the RECEIVER
    out = scatter_add(msg, row, num_nodes)   # <-- scatters onto the SENDER

That is backwards: messages must be gathered from the **source** and scattered
onto the **destination**. And it is the worst kind of bug, because *nothing
fails*. The shapes stay valid, the model still trains, the accuracy still looks
plausible. It is just wrong.

The invariants enforced here
----------------------------
A. **Every ``scatter_*`` call must aggregate onto a destination index.** This is
   the invariant the original draft violated, and it is the one that matters.
B. **Every layer's ``forward`` must gather messages from a source index.** A
   layer that never reads ``edge_index[0]`` is not doing message passing.
C. ``index_select(0, dst)`` is *allowed*, not flagged. GAT legitimately gathers
   the receiver's own features to compute the ``a_dst . Wh_i`` score term, so a
   blanket "gather must be src" rule produces false positives.

The test parses the AST instead of importing torch, so it runs in a bare
environment and cannot be broken by a dependency change.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "gnn_lab"
LAYER_DIR = PKG / "layers"

# file -> (class.forward to inspect, module-level aggregation helpers to inspect)
TARGETS = {
    "gcn.py":  (["GCNConvScratch.forward"], []),
    "sage.py": (["SAGEConvScratch.forward"], ["_aggregate"]),
    "gat.py":  (["GATConvScratch.forward"], []),
    "gin.py":  (["GINConvScratch.forward"], []),
}

DST_NAMES = {"dst", "col", "target", "dst_idx", "i", "index"}
SRC_NAMES = {"src", "row", "source", "src_idx", "j"}


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #
def _role_bindings(fn: ast.AST) -> dict[str, str]:
    """Find ``src, dst = ei[0], ei[1]`` style bindings -> {var_name: 'src'|'dst'}."""
    roles: dict[str, str] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Tuple) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)):
            continue
        names, values = node.targets[0].elts, node.value.elts
        if len(names) != len(values):
            continue
        for nm, val in zip(names, values):
            if isinstance(nm, ast.Name) and isinstance(val, ast.Subscript) \
                    and isinstance(val.slice, ast.Constant):
                if val.slice.value == 0:
                    roles[nm.id] = "src"
                elif val.slice.value == 1:
                    roles[nm.id] = "dst"
    return roles


def _resolve_roles(fn: ast.AST) -> dict[str, str]:
    """Bindings from tuple-unpacking, plus any parameter/variable literally named src/dst."""
    roles = _role_bindings(fn)
    for node in ast.walk(fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for a in node.args.args:
                if a.arg in SRC_NAMES:
                    roles[a.arg] = "src"
                elif a.arg in DST_NAMES:
                    roles[a.arg] = "dst"
    return roles


def _calls(fn: ast.AST):
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name) else None)
            if name:
                yield name, node


def _find_class_fn(tree: ast.Module, qualified: str) -> ast.FunctionDef:
    cls_name, fn_name = qualified.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == fn_name:
                    return sub
    raise AssertionError(f"{qualified} not found in module")


def _find_module_fn(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"module-level function {name} not found")


def _arg_role(call: ast.Call, pos: int, roles: dict[str, str]) -> str | None:
    if len(call.args) <= pos:
        return None
    arg = call.args[pos]
    if isinstance(arg, ast.Name):
        return roles.get(arg.id)
    return None


# --------------------------------------------------------------------------- #
# Invariant A -- scatter always targets a destination
# --------------------------------------------------------------------------- #
def test_scatter_always_targets_destination():
    violations = []
    for filename, (fns, helpers) in TARGETS.items():
        tree = ast.parse((LAYER_DIR / filename).read_text())
        scopes = [_find_class_fn(tree, f) for f in fns] + \
                 [_find_module_fn(tree, h) for h in helpers]
        for scope in scopes:
            roles = _resolve_roles(scope)
            for name, call in _calls(scope):
                if not name.startswith("scatter_"):
                    continue
                role = _arg_role(call, 1, roles)
                if role == "src":
                    violations.append(f"{filename}:{scope.name} -> {name}(..., <src>, ...)")
    assert not violations, (
        "scatter_* must aggregate onto the DESTINATION index, but these aggregate onto a "
        "source index (messages would be delivered to the wrong node):\n  "
        + "\n  ".join(violations))


# --------------------------------------------------------------------------- #
# Invariant B -- every layer gathers from a source
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("filename,forward", [(f, q) for f, (qs, _) in TARGETS.items() for q in qs])
def test_forward_gathers_from_source(filename, forward):
    tree = ast.parse((LAYER_DIR / filename).read_text())
    fn = _find_class_fn(tree, forward)
    roles = _resolve_roles(fn)

    gathers_src = any(
        name == "index_select" and _arg_role(call, 1, roles) == "src"
        for name, call in _calls(fn)
        if len(call.args) >= 1 and isinstance(call.args[0], ast.Constant) and call.args[0].value == 0
    )
    # SAGE delegates its gather to a helper; accept a helper call that receives src.
    passes_src_to_helper = any(
        name not in ("index_select",) and any(_arg_role(call, p, roles) == "src"
                                              for p in range(len(call.args)))
        for name, call in _calls(fn)
        if name.startswith("_")
    )
    assert gathers_src or passes_src_to_helper, (
        f"{filename}:{forward} never reads edge_index[0] (the source). "
        f"A message-passing layer must gather neighbour features from the source index.")


# --------------------------------------------------------------------------- #
# Invariant C is documented, not asserted: gather-from-dst is legal (GAT scores)
# --------------------------------------------------------------------------- #
def test_gat_uses_dst_gather_only_for_receiver_scores():
    """GAT's ``index_select(0, dst)`` must feed the score, not the message."""
    text = (LAYER_DIR / "gat.py").read_text()
    tree = ast.parse(text)
    fn = _find_class_fn(tree, "GATConvScratch.forward")
    roles = _resolve_roles(fn)

    dst_gathers = sum(1 for name, call in _calls(fn)
                      if name == "index_select" and _arg_role(call, 1, roles) == "dst")
    assert dst_gathers >= 1, "GAT should gather receiver features for the a_dst score term"

    # The message itself must come from the source gather. Compare on whitespace-
    # normalised source so reformatting never breaks this.
    flat = "".join(text.split())
    assert "h_src.index_select(0,src)*alpha.unsqueeze(-1)" in flat, \
        "GAT message must be `h_src.index_select(0, src) * alpha`, i.e. gathered from the source"


def test_helper_aggregate_forwards_dst():
    """sage._aggregate must pass its dst argument straight into a scatter call."""
    tree = ast.parse((LAYER_DIR / "sage.py").read_text())
    fn = _find_module_fn(tree, "_aggregate")
    roles = _resolve_roles(fn)
    scatters = [(n, c) for n, c in _calls(fn) if n.startswith("scatter_")]
    assert scatters, "_aggregate contains no scatter call"
    assert all(_arg_role(c, 1, roles) == "dst" for _, c in scatters), \
        "_aggregate does not forward its dst argument to scatter_*"


# --------------------------------------------------------------------------- #
# Coverage guards
# --------------------------------------------------------------------------- #
def test_all_layer_files_are_covered():
    covered = set(TARGETS)
    on_disk = {p.name for p in LAYER_DIR.glob("*.py")
               if p.name not in {"__init__.py", "pooling.py"}}
    assert covered == on_disk, (
        f"layer files not covered by the convention test: {sorted(on_disk - covered)}. "
        f"Add them to TARGETS rather than deleting this test.")


def test_ops_exports_the_primitives():
    tree = ast.parse((PKG / "ops.py").read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    required = {
        "scatter_add": ["src", "index", "dim_size"],
        "scatter_sum": ["src", "index", "dim_size"],
        "scatter_mean": ["src", "index", "dim_size"],
        "scatter_max": ["src", "index", "dim_size"],
        "scatter_softmax": ["src", "index", "dim_size"],
        "gcn_norm": ["edge_index", "num_nodes"],
        "add_self_loops": ["edge_index", "num_nodes"],
        "remove_self_loops": ["edge_index"],
        "to_undirected": ["edge_index", "num_nodes"],
        "degree": ["index", "num_nodes"],
    }
    for name, args in required.items():
        assert name in funcs, f"ops.{name} is missing"
        actual = [a.arg for a in funcs[name].args.args]
        assert actual[:len(args)] == args, f"ops.{name} signature changed: {actual}"


def test_scatter_softmax_detaches_the_max():
    """The shift must be treated as a constant; see ops.scatter_softmax for why that is exact."""
    src = (PKG / "ops.py").read_text()
    fn = src.split("def scatter_softmax")[1].split("\ndef ")[0]
    assert ".detach()" in fn, "scatter_softmax no longer detaches the per-group max"
    assert "index_select" in fn, "scatter_softmax must broadcast via index_select, not reshape"


def test_no_dense_adjacency_in_layers():
    """Layers must stay sparse. A dense [N, N] matmul would silently OOM on big graphs."""
    for path in LAYER_DIR.glob("*.py"):
        src = path.read_text()
        for banned in ("torch.sparse.mm", "to_dense()", "dense()"):
            assert banned not in src, f"{path.name} uses {banned}; keep message passing sparse"
