"""The vendored copies must not quietly drift from the modules they came from.

The analyser vendors what it needs so it can be lifted out whole. The price of
that is drift, and this repository has already paid it once: an earlier
vendoring went stale, a ``success`` green that stopped matching the rig's for
months, so "saved OK" was a different colour in the two apps. That is why the
copies are only defensible while something fails when they diverge.

What is compared is the **definitions**, not the files. Rewriting an import to
point at a vendored sibling IS the vendoring, so imports are stripped at every
depth before comparing; what must match is the body of every function and
class. A difference that is deliberate goes in ``EXPECTED_DIFFERENCES`` with
its reason, so the next reader learns it was a decision rather than a mistake.

Away from the repository the rig is simply absent and these skip, which is the
right behaviour for a copy that has been deliberately lifted out.
"""
from __future__ import annotations

import ast
import copy
import pathlib

import pytest

_VENDOR = pathlib.Path(__file__).resolve().parents[1] / "vendor"
_REPO = pathlib.Path(__file__).resolve().parents[3]

#: vendored file -> the rig module it was copied from.
ORIGINS = {
    "theme.py": "source/gui/theme.py",
    "style_builders.py": "source/gui/style_builders.py",
    "zone_editor.py": "source/gui/widgets/zone_editor.py",
    "numeric_line_edit.py": "source/gui/widgets/numeric_line_edit.py",
    "zone_overlay.py": "source/gui/widgets/zone_overlay.py",
    "zone_coords.py": "source/video/zones/coords.py",
    "zone_geometry.py": "source/video/zones/geometry.py",
    "zone_io.py": "source/video/zones/io.py",
}

#: vendored file -> {definition name: why this one is allowed to differ}.
#: Empty braces mean every definition must match and the file's known
#: divergence is outside any of them.
EXPECTED_DIFFERENCES: dict[str, dict[str, str]] = {
    # The rig reaches for the Cython accelerator (source.cython.zone_math) at
    # module level; carrying that would tie the copy back to the thing it
    # replaces, and the pure-Python path below it is what the rig itself falls
    # back to uncompiled. No definition differs, so nothing is listed here.
    "zone_geometry.py": {},
}


class _DropImports(ast.NodeTransformer):
    """Remove every import, at any depth.

    Rewriting an import to point at the vendored sibling IS the vendoring, and
    the rig writes some of them inside methods, a lazy import of a heavy
    module, so a top-level-only sweep would leave those behind and report the
    rewrite as drift.
    """

    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


def _defs(path: pathlib.Path) -> dict[str, str]:
    """Top-level function and class names -> their source, imports excluded.

    Comparing bodies rather than whole files lets the import rewrites, the
    deliberate part, pass while a changed behaviour fails.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stripped = _DropImports().visit(copy.deepcopy(node))
            ast.fix_missing_locations(stripped)
            out[node.name] = ast.dump(stripped)
    return out


def _origin(name: str) -> pathlib.Path:
    return _REPO / ORIGINS[name]


@pytest.mark.parametrize("name", sorted(ORIGINS))
def test_the_vendored_copy_still_matches_the_rig(name):
    """Same functions, same bodies, or a stated reason why not."""
    rig = _origin(name)
    if not rig.exists():
        pytest.skip(f"lifted out of the repo: {ORIGINS[name]} is not here")

    copy_defs = _defs(_VENDOR / name)
    rig_defs = _defs(rig)
    allowed = EXPECTED_DIFFERENCES.get(name, {})

    missing = sorted(set(rig_defs) - set(copy_defs) - set(allowed))
    assert not missing, (
        f"vendor/{name} is missing {missing}, the rig's {ORIGINS[name]} grew "
        "and the copy did not follow")

    differing = sorted(
        d for d in set(copy_defs) & set(rig_defs)
        if copy_defs[d] != rig_defs[d] and d not in allowed)
    assert not differing, (
        f"vendor/{name}: {differing} has drifted from {ORIGINS[name]}. "
        "Re-copy it, or record the difference in EXPECTED_DIFFERENCES with "
        "the reason.")


def test_every_vendored_file_names_where_it_came_from():
    """A copy whose origin is not written down is a copy nobody will update."""
    text = (_VENDOR / "__init__.py").read_text(encoding="utf-8")
    for name, origin in sorted(ORIGINS.items()):
        assert name in text, f"vendor/__init__.py does not list {name}"
        assert origin in text, (
            f"vendor/__init__.py does not say where {name} came from")


def test_the_vendor_package_imports_nothing_from_the_rig():
    """The whole point. Enforced here as well as by the perimeter test, because
    this is the file a future vendoring will be read alongside."""
    for path in _VENDOR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
            elif isinstance(node, ast.Import):
                root = node.names[0].name.split(".")[0]
            else:
                continue
            assert root != "source", (
                f"{path.name}:{node.lineno} imports source, the copy still "
                "depends on the thing it was copied to replace")


def test_no_vendored_file_is_unused():
    """A copy nobody imports is drift waiting to happen with no upside: it
    still has to be kept in step, and nothing exercises it."""
    users = [p for p in (_VENDOR.parent).rglob("*.py")
             if _VENDOR not in p.parents]
    text = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                     for p in users)
    text += "\n".join(p.read_text(encoding="utf-8", errors="replace")
                      for p in _VENDOR.glob("*.py"))
    unused = [n for n in sorted(ORIGINS)
              if f"vendor.{n[:-3]}" not in text and f"from .{n[:-3]}" not in text]
    assert not unused, (
        f"vendored but imported nowhere: {unused}. Delete the copy, or import "
        "it where the rig module would have been used.")
