"""The in-flight gate must outlive the forward pass, not end with it.

Post-processing was moved off the inference thread so the next batch can start
while the previous batch's boxes are still being finished. That only stays
correct if the per-box gate now closes on POST-PROCESSING completion: a box
whose callback is still running owns its smoother, its crop state and its
previous-centroid slot, and handing it a second frame would run two writers
over all three.

These pin that contract, and the two ways it could be broken silently: a box
gated forever because its marker never resolved, and a box let back in early.
"""
from __future__ import annotations

import threading

from source.video.tracking.inference import ModelHandle, ThreadInferenceBackend

KEY = ("stub", 224, 176, 1.0, "sleap", "sig")


class _Tracker:
    """Minimal PoseTracker: one pose per frame, or a raising predict."""

    is_initialized = True

    def __init__(self, raises: bool = False):
        self.raises = raises

    def predict_batch(self, frames):
        if self.raises:
            raise RuntimeError("predict exploded")
        return [{"nose": (1.0, 2.0, 0.9)} for _ in frames]

    def get_body_parts(self):
        return ["nose"]

    def close(self):
        pass


def _backend_and_handle(raises: bool = False):
    b = ThreadInferenceBackend()
    return b, ModelHandle(key=KEY, model=_Tracker(raises))


def test_a_box_is_refused_while_its_own_callback_is_still_running():
    """The forward pass being over is not enough to reopen the gate."""
    backend, handle = _backend_and_handle()
    in_callback = threading.Event()
    release = threading.Event()

    def on_done(bid, pose):
        in_callback.set()
        release.wait(timeout=5)

    try:
        assert backend.submit_batch(handle, [(1, object())], on_done) == 1
        assert in_callback.wait(timeout=5), "callback never ran"
        # The pass has finished by now, so a gate keyed on the pass would let
        # this through and run a second callback for box 1 concurrently.
        assert backend.submit_batch(handle, [(1, object())], on_done) == 0
    finally:
        release.set()
        backend.shutdown(wait=True)


def test_the_gate_reopens_once_the_callback_finishes():
    backend, handle = _backend_and_handle()
    done = threading.Event()

    def on_done(bid, pose):
        done.set()

    try:
        assert backend.submit_batch(handle, [(1, object())], on_done) == 1
        assert done.wait(timeout=5)
        # The marker resolves inside the callback's own finally, so poll
        # rather than assuming this thread sees it on the first try.
        for _ in range(500):
            if backend.submit_batch(handle, [(1, object())], on_done) == 1:
                break
            threading.Event().wait(0.01)
        else:
            raise AssertionError("gate never reopened after the callback ended")
    finally:
        backend.shutdown(wait=True)


def test_a_failed_forward_pass_still_reopens_the_gate():
    """A box gated forever looks like one camera going blind, not an error."""
    backend, handle = _backend_and_handle(raises=True)
    seen = threading.Event()

    def on_done(bid, pose):
        # A failed batch still calls back, with an empty pose.
        assert pose == {}
        seen.set()

    try:
        assert backend.submit_batch(handle, [(1, object())], on_done) == 1
        assert seen.wait(timeout=5), "no callback after a failed predict"
        for _ in range(500):
            if backend.submit_batch(handle, [(1, object())], on_done) == 1:
                break
            threading.Event().wait(0.01)
        else:
            raise AssertionError("a failed batch left the box gated out")
    finally:
        backend.shutdown(wait=True)


def test_boxes_are_gated_independently():
    """Box 2 must not wait behind box 1's post-processing."""
    backend, handle = _backend_and_handle()
    entered = threading.Event()
    release = threading.Event()

    def blocking(bid, pose):
        entered.set()
        release.wait(timeout=5)

    try:
        assert backend.submit_batch(handle, [(1, object())], blocking) == 1
        assert entered.wait(timeout=5)
        # Different box, different state: nothing about box 1 being busy makes
        # box 2 unsafe.
        assert backend.submit_batch(handle, [(2, object())], lambda b, p: None) == 1
    finally:
        release.set()
        backend.shutdown(wait=True)


def test_shutdown_does_not_hang_on_a_running_callback():
    """Teardown drains through the post pool; ordering it wrongly would make
    every shutdown wait out the full drain timeout."""
    backend, handle = _backend_and_handle()
    ran = threading.Event()
    backend.submit_batch(handle, [(1, object())], lambda b, p: ran.set())
    assert ran.wait(timeout=5)
    backend.shutdown(wait=True)
    # Idempotent, and does not raise the second time.
    backend.shutdown(wait=True)
