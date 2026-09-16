"""Every host module must load on the interpreter the app actually runs.

pytest here runs on a newer Python than the app launcher does, so anything
the test interpreter tolerates and the app's rejects ships green and breaks
on launch. Two such gaps have bitten:

* **Syntax.** A backslash inside an f-string expression is legal from 3.12
  and a SyntaxError before it.
* **Annotations.** From 3.14 (PEP 649) annotations are evaluated lazily, so
  an annotation naming something the module never imports is inert unless
  something reads ``__annotations__``, which nothing does. On 3.9 the same
  annotation is evaluated when the ``def`` executes, so importing the module
  raises ``NameError`` and the subsystem is dead at startup. A return
  annotation of ``Optional[Tuple[int, int]]`` in ``video/tracking/blob.py``,
  with only ``Optional`` imported, passed the entire suite this way.

The syntax check needs the older interpreter and skips without it. The
annotation check is static, so it always runs.
"""
import ast
import builtins
import subprocess
import sys
from pathlib import Path

import pytest

# Oldest interpreter the app is expected to launch under.
RUNTIME_PY = "3.9"

SOURCE = Path(__file__).resolve().parents[2]

# MicroPython, not host code, uploaded to the board, never imported here.
EXCLUDED_DIRS = {"pyControl", "board_tests", "framework_tests"}


def _runtime_interpreter():
    """Path to the runtime Python via the py launcher, or None."""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(["py", f"-{RUNTIME_PY}", "-c",
                              "import sys; print(sys.executable)"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _host_modules():
    for path in SOURCE.rglob("*.py"):
        if EXCLUDED_DIRS.isdisjoint(p.name for p in path.parents):
            yield path


def test_host_modules_parse_on_the_runtime_interpreter():
    interp = _runtime_interpreter()
    if interp is None:
        pytest.skip(f"Python {RUNTIME_PY} not installed")

    files = sorted(str(p) for p in _host_modules())
    assert files, "found no host modules to check"

    # One subprocess for the whole tree: per-file spawn costs seconds.
    script = (
        "import ast, sys\n"
        "bad = []\n"
        "for f in sys.argv[1:]:\n"
        "    try:\n"
        "        ast.parse(open(f, encoding='utf-8').read())\n"
        "    except SyntaxError as e:\n"
        "        bad.append('%s:%s: %s' % (f, e.lineno, e.msg))\n"
        "print('\\n'.join(bad))\n"
    )
    out = subprocess.run([interp, "-c", script, *files],
                         capture_output=True, text=True, timeout=600)
    failures = out.stdout.strip()
    assert not failures, (
        f"these modules do not parse on Python {RUNTIME_PY}, which the app "
        f"launches with:\n{failures}")


# --- annotations must resolve, because 3.9 evaluates them eagerly -----------

def _bound_names(tree) -> set:
    """Every name the module binds, by any means, anywhere in the file.

    Deliberately over-permissive, no scope analysis, so a name bound inside
    some unrelated function counts as bound. That direction is the safe one:
    it cannot fail a working module, and a name bound *nowhere* is still
    unambiguously undefined, which is the bug this catches.
    """
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs,
                        a.vararg, a.kwarg):
                if arg is not None:
                    names.add(arg.arg)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def _eager_annotations(tree):
    """(line, annotation-node) for every annotation 3.9 evaluates on import.

    Argument and return annotations run when the ``def`` executes; a bare
    ``x: T`` at class or module level runs where it sits. Both are reached on
    import, which is what makes an unresolvable name fatal rather than
    latent.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs,
                        a.vararg, a.kwarg):
                if arg is not None and arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            yield node.annotation


def _names_used(annotation):
    """Names an annotation reads. Quoted parts are skipped, a string
    annotation is never evaluated, so it cannot raise on import."""
    for n in ast.walk(annotation):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            yield n.id


def test_annotations_resolve_on_the_runtime_interpreter():
    """No annotation may name something the module never binds.

    Static, so it runs everywhere and needs no second interpreter. Modules
    carrying ``from __future__ import annotations`` are exempt: their
    annotations are strings on every version, so nothing is evaluated.
    """
    known = set(dir(builtins))
    failures = []

    for path in sorted(_host_modules()):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue                      # the parse test above owns this
        if any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names)
               for n in tree.body):
            continue
        bound = _bound_names(tree) | known
        for annotation in _eager_annotations(tree):
            for name in _names_used(annotation):
                if name not in bound:
                    failures.append(
                        f"{path.relative_to(SOURCE.parent)}:"
                        f"{annotation.lineno}: {name}")

    assert not failures, (
        "these annotations name something the module never imports or "
        "defines. Python "
        f"{RUNTIME_PY} evaluates annotations when the module is imported, so "
        "each one is a NameError on launch, invisible here because this "
        "interpreter evaluates them lazily (PEP 649):\n  "
        + "\n  ".join(failures))
