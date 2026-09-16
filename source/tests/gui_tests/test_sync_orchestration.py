"""Phase 3, hardware-sync start ordering.

A secondary must be armed before the primary free-runs and strobes, or the
opening trigger edges are lost. ``plan_sync_start_order`` enforces that:
secondaries first, standalone cameras next, primaries last.
"""
from __future__ import annotations

from source.video.cameras.capture import plan_sync_start_order


def _order(ids, roles):
    return plan_sync_start_order(ids, lambda c: roles.get(c, "none"))


def test_secondary_before_primary():
    out = _order(["p", "s"], {"p": "primary", "s": "secondary"})
    assert out.index("s") < out.index("p")


def test_full_bucket_order():
    roles = {"p": "primary", "s1": "secondary", "s2": "secondary", "n": "none"}
    out = _order(["p", "n", "s1", "s2"], roles)
    # secondaries, then standalone, then primary
    assert out == ["s1", "s2", "n", "p"]


def test_stable_within_group():
    roles = {"a": "secondary", "b": "secondary", "c": "secondary"}
    assert _order(["b", "a", "c"], roles) == ["b", "a", "c"]


def test_all_standalone_unchanged():
    ids = ["x", "y", "z"]
    assert _order(ids, {}) == ids          # no roles → identity order


def test_none_role_from_callable():
    # role_of returning None is treated as standalone, not an error.
    assert plan_sync_start_order(["a", "b"], lambda c: None) == ["a", "b"]


def test_empty():
    assert plan_sync_start_order([], lambda c: "none") == []


def test_multiple_primaries_last():
    roles = {"p1": "primary", "p2": "primary", "s": "secondary"}
    out = _order(["p1", "s", "p2"], roles)
    assert out.index("s") < out.index("p1")
    assert out.index("s") < out.index("p2")
