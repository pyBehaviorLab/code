"""Every file destined for the board is at least syntactically valid Python.

The MCU tree is excluded from pytest collection, correctly: it imports ``pyb``
and cannot run on the host. But *parsing* is not *importing*, and skipping the
parse too left a real gap. Two task files sat in the tree with a tab where
their neighbours used eight spaces, so neither could be uploaded, and nothing
said so until the moment somebody tried, which is at the rig with an animal
already in the chamber.

This is deliberately the weakest possible check. It cannot know whether a task
does the right thing, or whether an event name matches a hardware definition
(``AI_AUTHORING_GUIDE.md`` covers the contracts that fail silently). It only
proves the file is Python, which is the failure that costs a session for the
dullest possible reason.

Parsed under the interpreter the BOARD runs, as far as the host can tell: no
host imports, no execution, just ``ast.parse``.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]

#: Everything uploaded to, or run on, the microcontroller.
MCU_DIRS = ("tasks", "devices", "hardware_definitions", "api_classes",
            "source/pyControl")


def _mcu_files():
    for rel in MCU_DIRS:
        base = ROOT / rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_there_is_an_mcu_tree_to_check():
    """A sweep that finds nothing passes for the wrong reason."""
    found = list(_mcu_files())
    # The device drivers alone clear this. Tasks and hardware
    # definitions are written per lab and ship empty, so the guard
    # asks whether a tree was found at all, not how big it is.
    assert len(found) > 20, f"only {len(found)} MCU files found, is ROOT wrong?"


@pytest.mark.parametrize("path", list(_mcu_files()),
                         ids=lambda p: str(p.relative_to(ROOT)).replace("\\", "/"))
def test_the_file_is_valid_python(path):
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        ast.parse(src)
    except SyntaxError as exc:
        rel = path.relative_to(ROOT)
        pytest.fail(
            f"{rel}:{exc.lineno} cannot be parsed: {exc.msg}\n"
            "This file cannot be uploaded to a board. A mix of tabs and "
            "spaces is the usual cause; match the indentation the rest of "
            "the file uses.")
