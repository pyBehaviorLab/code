"""Changing a pose setting must tear the old model down safely and completely.

Two failures this guards, both of which surface far from their cause:

  * **Use-after-close.** ``_evict`` closed the model without waiting for work
    already on the executor. Dropping a reference does not stop a running
    thread, so the close landed under the thread still inside the model, a
    use-after-free in a C extension, not a Python error.

  * **Memory that never comes back.** PyTorch keeps freed blocks in its caching
    allocator, and NVIDIA's own tracker documents repeated TensorRT context
    create/destroy as a steady climb. An operator trying three precisions and
    two scales is six teardowns; without a release each one keeps its VRAM and
    the session eventually dies of something unrelated-looking.
"""
import threading
import time

import numpy as np
import pytest

from source.video.tracking import inference as inf


class _SlowModel:
    """A model whose forward pass takes long enough to still be running when
    the teardown starts, which is the whole scenario under test."""

    def __init__(self, delay=0.25):
        self.delay = delay
        self.closed = False
        self.closed_while_running = False
        self.running = False

    def predict_batch(self, frames):
        self.running = True
        try:
            time.sleep(self.delay)
            if self.closed:
                self.closed_while_running = True
            return [{} for _ in frames]
        finally:
            self.running = False

    def get_body_parts(self):
        return ["a"]

    def close(self):
        if self.running:
            self.closed_while_running = True
        self.closed = True

    @property
    def is_initialized(self):
        return True


def _handle(key, model):
    return inf.ModelHandle(key=key, model=model)


KEY_A = ("m", 64, 64, 1.0, "dlc", "")
KEY_B = ("m", 64, 64, 0.5, "dlc", "")


@pytest.fixture
def backend():
    b = inf.ThreadInferenceBackend()
    yield b
    b.shutdown(wait=True)


# ── drain before close ───────────────────────────────────────────────────

def test_eviction_waits_for_work_already_on_the_gpu(backend):
    """The close must not land while a batch is inside the model."""
    model = _SlowModel(delay=0.3)
    backend._models[KEY_A] = _handle(KEY_A, model)
    backend._active_key = KEY_A

    accepted = backend.submit_batch(backend._models[KEY_A],
                                    [(1, np.zeros((8, 8, 3), np.uint8))],
                                    lambda bid, pose: None)
    assert accepted == 1
    time.sleep(0.05)                      # let the worker enter predict_batch

    backend._evict(KEY_A)

    assert model.closed is True
    assert model.closed_while_running is False, \
        "closed the model while the executor was still inside it"
    assert backend._inflight == {}, "in-flight entries must not outlive the model"


def test_drain_gives_up_rather_than_hanging(monkeypatch):
    """A wedged worker must not hold the GUI thread for ever, the teardown
    warns and proceeds, because a stuck model is not a reason to freeze."""
    monkeypatch.setattr(inf, "_DRAIN_TIMEOUT_S", 0.15)
    started = threading.Event()
    stop = threading.Event()

    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(lambda: (started.set(), stop.wait(5))[1])
        started.wait(2)
        t0 = time.monotonic()
        inf._drain_inflight({1: fut})
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, "drain must bound its wait"
    finally:
        stop.set()
        ex.shutdown(wait=True)


def test_drain_of_nothing_is_free():
    inf._drain_inflight({})              # must not raise or block


def test_a_failed_batch_still_counts_as_finished(backend):
    """A batch that raised is done; the teardown must not wait on it twice or
    let the exception escape into the operator's settings change."""
    class _Boom(_SlowModel):
        def predict_batch(self, frames):
            raise RuntimeError("kaboom")

    model = _Boom(delay=0)
    backend._models[KEY_A] = _handle(KEY_A, model)
    backend._active_key = KEY_A
    backend.submit_batch(backend._models[KEY_A],
                         [(1, np.zeros((8, 8, 3), np.uint8))],
                         lambda bid, pose: None)
    time.sleep(0.1)
    backend._evict(KEY_A)                # must not raise
    assert model.closed is True


# ── device memory release ────────────────────────────────────────────────

class TestReleaseDeviceMemory:
    def test_no_framework_loaded_is_a_no_op(self, monkeypatch):
        """Importing torch in order to free memory would allocate far more
        than it releases, so an absent framework means do nothing."""
        monkeypatch.delitem(inf.sys.modules, "torch", raising=False)
        inf.release_device_memory()       # must not raise or import anything

    def test_empty_cache_is_called_when_cuda_is_present(self, monkeypatch):
        calls = []

        class _Cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                calls.append(1)

        class _Torch:
            cuda = _Cuda()

        monkeypatch.setitem(inf.sys.modules, "torch", _Torch())
        inf.release_device_memory()
        assert calls == [1]

    def test_cpu_only_torch_is_left_alone(self, monkeypatch):
        class _Cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def empty_cache():
                raise AssertionError("must not empty a cache that cannot exist")

        class _Torch:
            cuda = _Cuda()

        monkeypatch.setitem(inf.sys.modules, "torch", _Torch())
        inf.release_device_memory()

    def test_a_broken_framework_does_not_break_teardown(self, monkeypatch):
        class _Torch:
            @property
            def cuda(self):
                raise RuntimeError("driver gone")

        monkeypatch.setitem(inf.sys.modules, "torch", _Torch())
        inf.release_device_memory()       # swallowed: teardown must finish


def test_eviction_releases_device_memory(backend, monkeypatch):
    calls = []
    monkeypatch.setattr(inf, "release_device_memory", lambda: calls.append(1))
    backend._models[KEY_A] = _handle(KEY_A, _SlowModel(delay=0))
    backend._active_key = KEY_A
    backend._evict(KEY_A)
    assert calls == [1], "every teardown must hand the memory back"


# ── repeated re-init, the scenario that motivated all of this ────────────

def test_twenty_re_inits_release_every_time(backend, monkeypatch):
    """Alternating two settings 20 times must produce 20 closes and 20
    releases, nothing accumulating, nothing skipped.

    This is the shape of a real session: an operator trying scales and
    precisions until the tracking looks right.
    """
    releases = []
    monkeypatch.setattr(inf, "release_device_memory", lambda: releases.append(1))
    models = []

    for i in range(20):
        key = KEY_A if i % 2 == 0 else KEY_B
        model = _SlowModel(delay=0)
        models.append(model)
        backend._models[key] = _handle(key, model)
        backend._active_key = key
        backend._evict(key)

    assert len(releases) == 20
    assert all(m.closed for m in models), "every model must be closed, not dropped"
    assert backend._models == {}, "no handle may survive its eviction"
    assert backend._active_key is None
