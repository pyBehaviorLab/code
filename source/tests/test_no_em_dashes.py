"""No em dash anywhere: code, comments, docstrings, GUI text, docs.

Asked for twice, and cleared twice by hand before this test existed, which is
why it exists. A house style that is only enforced by remembering is a style
that comes back one commit later.

An em dash is doing one of four jobs, and each has a plainer mark that says the
same thing: a comma between clauses, a semicolon between two that could stand
alone, a colon before a definition, a hyphen in a range. The replacement is
never "some dash character that is not U+2014", so the horizontal bar and the
figure dash are refused here too.

The en dash is NOT refused. It has one legitimate use in this tree, a numeric
range, and rewriting those would be a different decision from the one asked
for.
"""
from __future__ import annotations

import pathlib

import pytest

#: Every dash that reads as an em dash. The horizontal bar and the figure dash
#: are visually identical in most fonts, so allowing them would let the style
#: back in through a different code point.
#:
#: By code point, not by literal: a guard that spells out what it forbids is
#: caught by itself, and excluding this one file from the scan would leave a
#: hole exactly where someone would think to put an exception.
FORBIDDEN = {
    chr(0x2014): "em dash",
    chr(0x2015): "horizontal bar",
    chr(0x2012): "figure dash",
}

#: Box-drawing characters (U+2500) are deliberately NOT here. They rule off a
#: section header in a comment block; they are not punctuation and they are not
#: read as a dash.

ROOT = pathlib.Path(__file__).resolve().parents[2]

SKIP_DIRS = {".git", "_build", "__pycache__", "node_modules", ".venv", "data",
             "models", "vendor", ".claude", ".pytest_cache", ".ruff_cache",
             "htmlcov", "dist", "build", ".mypy_cache"}
#: The text this project writes. Generated C from Cython carries whatever the
#: generator emitted and is not edited by hand, so ``.c`` stays out; ``.pyx``
#: and ``.pxd`` are hand-written and were missing from this list, which is
#: exactly where the last nine em dashes were still sitting.
EXTS = (".py", ".pyx", ".pxd", ".md", ".html", ".json", ".txt", ".rst",
        ".css", ".cfg", ".toml", ".yml", ".yaml", ".bat")


def _files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(p in SKIP_DIRS for p in path.parts):
            continue
        if path.suffix.lower() in EXTS:
            yield path


def _offences():
    out = []
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if not any(ch in text for ch in FORBIDDEN):
            continue
        for n, line in enumerate(text.split("\n"), start=1):
            for ch, name in FORBIDDEN.items():
                if ch in line:
                    out.append((path.relative_to(ROOT), n, name, line.strip()))
    return out


def test_the_tree_carries_no_em_dashes():
    found = _offences()
    if not found:
        return
    listing = "\n".join(
        f"  {p}:{n}  ({name})  {line[:90]}" for p, n, name, line in found[:40])
    more = f"\n  ... and {len(found) - 40} more" if len(found) > 40 else ""
    pytest.fail(
        f"{len(found)} em dash(es) in the tree:\n{listing}{more}\n\n"
        "Use a comma between clauses, a semicolon between two that could "
        "stand alone, a colon before a definition, or a hyphen in a range.")


def test_the_check_actually_looks_at_something():
    """A guard whose file list is empty passes for the wrong reason."""
    seen = list(_files())
    assert len(seen) > 300, f"only {len(seen)} files scanned; the filter is wrong"
    assert any(p.suffix == ".py" for p in seen)
    assert any(p.suffix == ".md" for p in seen)


def test_it_would_catch_one():
    """The detector, on a line that has one. Built rather than written, for
    the reason given above ``FORBIDDEN``."""
    line = "the model is loaded " + chr(0x2014) + " and the box is ready"
    assert any(ch in line for ch in FORBIDDEN)
    assert not any(ch in line.replace(chr(0x2014), ",") for ch in FORBIDDEN)
