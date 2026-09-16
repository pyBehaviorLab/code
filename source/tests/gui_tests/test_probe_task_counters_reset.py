"""A probe task must start each run with fresh counters.

These files run on the MCU and cannot be imported here, so they are read as
source. That is enough: what must hold is structural.

Module-level assignment runs when the task is UPLOADED, not when Record is
pressed. A board that still holds the task carries the previous run's totals
into the next one, and the consequence is not cosmetic. The stop-when-broken
guard is ``n_lost_run >= max_lost_run and n_done == 0``, so one recorded trial
from any earlier run disables it for ever.

Measured on the rig: on 2026-09-10 box 1 began a session already holding
``n_done=3``, never armed, missed all 150 flashes, and ran the full ten
minutes instead of stopping at 45 s. Its end-of-run report summed two runs.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

TASKS = sorted((pathlib.Path(__file__).resolve().parents[3] / "tasks"
                / "Latency").glob("*Probe.py"))


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _reset_fn(tree):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_reset_counters":
            return node
    return None


def _assigned_vars(node):
    """``v.x = ...`` targets anywhere under ``node``."""
    out = set()
    for n in ast.walk(node):
        if not isinstance(n, ast.Assign):
            continue
        for t in n.targets:
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "v"):
                out.add(t.attr)
    return out


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.name)
def test_counters_live_in_one_reset_function(path):
    tree = _tree(path)
    fn = _reset_fn(tree)
    assert fn is not None, (
        f"{path.name} has no _reset_counters(); its counters are assigned at "
        f"module level and so survive from one run into the next")
    assert _assigned_vars(fn), "_reset_counters() sets nothing"


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.name)
def test_run_start_resets_them(path):
    tree = _tree(path)
    run_start = next((n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "run_start"),
                     None)
    assert run_start is not None, f"{path.name} has no run_start()"
    calls = [n.func.id for n in ast.walk(run_start)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "_reset_counters" in calls, (
        f"{path.name}: run_start() does not reset the counters, so a second "
        f"Record on a board that still holds the task continues the first "
        f"run's totals")


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.name)
def test_no_counter_is_also_set_at_module_level(path):
    """Two places to set one counter is two places for them to disagree."""
    tree = _tree(path)
    fn = _reset_fn(tree)
    in_reset = _assigned_vars(fn)
    module_level = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            module_level |= _assigned_vars(node)
    clash = in_reset & module_level
    assert not clash, (
        f"{path.name}: {sorted(clash)} are set both at module level and in "
        f"_reset_counters(); the module-level value is the one that persists "
        f"across runs, so the two will drift")


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.name)
def test_the_stop_guard_reads_a_counter_that_is_reset(path):
    """The guard that stops a broken run must not read a stale counter."""
    tree = _tree(path)
    fn = _reset_fn(tree)
    reset = _assigned_vars(fn)
    for needed in ("n_done", "n_lost_run", "n_timeout"):
        assert needed in reset, (
            f"{path.name}: {needed} is not reset, and the stop-when-broken "
            f"guard depends on it")
