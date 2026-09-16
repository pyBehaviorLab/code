"""The settle window must end when the batch cannot get any fuller.

The window exists to absorb the other boxes' submits when a producer iterates
them back-to-back. Its comment promised "≤ BATCH_SETTLE_S" of added latency,
but the loop paid it in full every cycle: once the queued wake tokens were
drained, the next ``get()`` blocked until the deadline whether or not anything
was still coming. On a shared camera, where every box's frame is segmented in
the same call, that is a flat tax on every pose for a wait that can never
collect anything.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque

from source.video.framebus.pose_sink import PoseSink


def _sink(settle_s: float, enabled: dict, pending: set):
    """A PoseSink carrying only the state ``_run``'s settle window reads."""
    s = PoseSink.__new__(PoseSink)
    s._nworkers = 1
    s._wakes = [queue.Queue()]
    s._lock = threading.RLock()
    s._running = True
    s.BATCH_SETTLE_S = settle_s
    s._enabled = dict(enabled)
    s._pending = set(pending)
    # One frame waiting per pending box, so the drain has something to take.
    s._per_box = {bid: deque([object()]) for bid in pending}
    s.dispatched = []

    def _dispatch(batch):
        s.dispatched.append(batch)
        s._running = False          # one pass, then let _run fall out

    s._dispatch_batch = _dispatch
    return s


def _time_one_pass(s) -> float:
    """Run ``_run`` for a single wake and return how long it took, seconds."""
    s._wake.put(next(iter(s._pending)) if s._pending else 1)
    t0 = time.monotonic()
    t = threading.Thread(target=s._run, daemon=True)
    t.start()
    t.join(timeout=10)
    return time.monotonic() - t0


def test_a_complete_batch_does_not_wait_out_the_window():
    """Both enabled boxes already have a frame: there is nothing to absorb."""
    s = _sink(settle_s=0.5, enabled={1: True, 2: True}, pending={1, 2})
    elapsed = _time_one_pass(s)
    assert elapsed < 0.25, (
        f"waited {elapsed:.3f}s of a 0.5s window with the batch already full")
    assert len(s.dispatched) == 1
    assert {bid for bid, _ in s.dispatched[0]} == {1, 2}


def test_an_incomplete_batch_still_waits_for_the_stragglers():
    """Box 2 is enabled but has not submitted, so the window must be paid."""
    s = _sink(settle_s=0.25, enabled={1: True, 2: True}, pending={1})
    elapsed = _time_one_pass(s)
    assert elapsed >= 0.2, (
        f"gave up after {elapsed:.3f}s; box 2 had not arrived yet")


def test_a_disabled_box_is_not_waited_for():
    """Only enabled boxes can contribute to a batch, so only they are counted."""
    s = _sink(settle_s=0.5, enabled={1: True, 2: False}, pending={1})
    elapsed = _time_one_pass(s)
    assert elapsed < 0.25, (
        f"waited {elapsed:.3f}s for box 2, which pose is switched off for")


def test_no_enabled_boxes_keeps_the_old_behaviour():
    """Nothing is enabled, so the early exit has no opinion and the window
    runs as before rather than short-circuiting on an empty comparison."""
    s = _sink(settle_s=0.2, enabled={}, pending={1})
    elapsed = _time_one_pass(s)
    assert elapsed >= 0.15
