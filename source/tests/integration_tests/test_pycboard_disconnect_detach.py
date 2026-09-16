"""Tests for two pipeline invariants.

  1. ``Pipeline.update_box_pycboard(box_id, None)`` detaches
     the board from MCUPusher.
  2. ``Pycboard._pending_writes`` queue: worker-thread
     ``queue_trigger_event`` / ``queue_set_coordinates`` calls enqueue
     a request that ``process_data`` (GUI thread) drains and writes, so
     every ``serial.write`` happens on the GUI thread. No locks, no
     thread-touching of serial.
"""

from __future__ import annotations

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import threading
from unittest.mock import MagicMock

import pytest

from source.communication.pycboard import Pycboard, PyboardError


# update_box_pycboard(None) detach


class TestUpdateBoxPycboardDetach:
    @pytest.fixture
    def pipeline(self):
        from source.video.framebus.controller import Pipeline
        return Pipeline(target_fps=30)

    def test_attaching_then_detaching_clears_pushsink(self, pipeline):
        """mcu_disconnect calls update_box_pycboard(box_id, None); this
        must clear the Pycboard reference from MCUPusher._pycboards so the
        next zone event from the camera worker doesn't write to a dead
        serial port."""
        b = MagicMock(spec=Pycboard)
        pipeline.update_box_pycboard(7, b)
        assert pipeline.push._pycboards.get(7) is b
        assert pipeline._pycboards.get(7) is b

        pipeline.update_box_pycboard(7, None)
        assert pipeline.push._pycboards.get(7) is None
        assert pipeline._pycboards.get(7) is None

    def test_detach_unknown_box_does_not_raise(self, pipeline):
        """Detaching a box that was never registered must be a quiet
        no-op (matches the dict-style ``pop(..., default)`` semantics)."""
        pipeline.update_box_pycboard(404, None)  # must not raise


# queue_* enqueues; process_data drains on GUI thread


class TestPendingWritesQueue:
    """``queue_trigger_event`` / ``queue_set_coordinates`` MUST NOT touch
    the serial port from the calling thread, ``self.serial`` belongs to
    the GUI thread. The queue_* methods enqueue a request; ``process_data``
    (GUI-thread QTimer tick) drains and writes."""

    def _make_pycboard_without_open(self):
        """Construct a Pycboard without opening a real port; record
        every serial.write as a single bytes blob."""
        import queue as _q
        pyc = Pycboard.__new__(Pycboard)
        pyc.serial = MagicMock()
        pyc.serial.in_waiting = 0           # no incoming bytes
        pyc.framework_running = True
        pyc._pending_writes = _q.Queue()
        pyc.sm_info = MagicMock()
        pyc.sm_info.events = {"zone_changed": 7, "reward": 8}
        pyc.data_logger = MagicMock()
        pyc.data_consumers = []
        pyc.timestamp = 0
        pyc.last_message_time = 0
        # Record every write as one atomic bytes blob.
        pyc._writes = []
        pyc.serial.write.side_effect = (
            lambda b: pyc._writes.append(bytes(b))
        )
        return pyc

    def test_queue_trigger_event_does_not_touch_serial(self):
        """The whole point: calling queue_trigger_event from a worker
        thread MUST NOT call serial.write directly."""
        pyc = self._make_pycboard_without_open()
        pyc.queue_trigger_event("zone_changed", source="t")
        assert pyc._writes == [], (
            "queue_trigger_event wrote to serial directly, should "
            "have only enqueued"
        )
        # And the request is sitting in the queue.
        assert pyc._pending_writes.qsize() == 1

    def test_queue_set_coordinates_does_not_touch_serial(self):
        pyc = self._make_pycboard_without_open()
        pyc.queue_set_coordinates("loc_x", 1.5)
        assert pyc._writes == []
        assert pyc._pending_writes.qsize() == 1

    def test_process_data_drains_pending_writes(self):
        """``process_data`` runs on the GUI thread; its first action is
        to drain any pending writes the worker threads enqueued."""
        pyc = self._make_pycboard_without_open()
        pyc.queue_trigger_event("zone_changed", source="t")
        pyc.queue_set_coordinates("loc_x", 2.0)
        pyc.process_data()
        # Both writes happened, exactly two atomic blobs.
        assert len(pyc._writes) == 2
        assert pyc._pending_writes.empty()

    def test_pending_writes_dropped_after_framework_stops(self):
        """If the framework was stopped between enqueue and drain, the
        request is silently dropped, matching the underlying
        ``trigger_event`` / ``set_coordinates`` no-op when the framework
        isn't running."""
        pyc = self._make_pycboard_without_open()
        pyc.queue_trigger_event("zone_changed")
        pyc.framework_running = False
        pyc.process_data()
        assert pyc._writes == []
        assert pyc._pending_writes.empty()

    def test_concurrent_enqueue_does_not_lose_requests(self):
        """The queue is the synchronisation primitive, multi-thread
        enqueue + GUI-thread drain must produce N writes for N
        enqueues."""
        pyc = self._make_pycboard_without_open()
        N = 200

        def worker(name: str):
            for _ in range(N):
                pyc.queue_trigger_event(name)

        a = threading.Thread(target=worker, args=("zone_changed",))
        b = threading.Thread(target=worker, args=("reward",))
        a.start(); b.start(); a.join(); b.join()
        # Drain on the "GUI thread" (this main test thread).
        pyc.process_data()
        assert len(pyc._writes) == 2 * N


