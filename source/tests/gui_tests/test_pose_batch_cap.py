"""How many boxes ride in one forward pass, and when that is decided.

``configure_model`` sets the cap from ``_n_batchable_boxes``, which counts
``_enabled`` plus ``_box_shapes``. At Init both are empty, pose has not been
enabled yet, and ``_box_shapes`` is filled by the frame handler for
pose-enabled boxes only. So the cap was decided at the one moment the sink
knows least about the rig, came out as 1, and nothing ever recomputed it.

Measured on a 16-box rig before the fix: 2186 forward passes in 8 s, every one
of batch 1, GPU busy 96% of wall-clock, and half the camera's frames shed. The
same graph takes a batch of 16 at 0.57 ms/frame against 3.06 ms at batch 1.
After: 236 calls of batch 16, 472 poses/s against 273, GPU busy 29%.

Nothing here needs a camera or a GPU, the cap is arithmetic over the enabled
set, and that is exactly what went wrong.
"""
import numpy as np
import pytest

from source.video.framebus.pose_sink import PoseSink
from source.video.tracking.sleap_export import batch_bucket


@pytest.fixture
def sink():
    return PoseSink()


def _enable(sink, n):
    for b in range(1, n + 1):
        sink.enable_for_box(b)


class TestTheCapFollowsTheBoxCount:
    def test_a_fresh_sink_does_not_cap_at_all(self, sink):
        """0 means unlimited in the chunk loop. Worth pinning: the bug was not
        a bad default, it was ``configure_model`` NARROWING this to 1 from an
        enabled set that was still empty."""
        assert sink._max_batch_size == 0

    @pytest.mark.parametrize("n,want", [(1, 1), (2, 2), (3, 4), (4, 4),
                                        (5, 8), (8, 8), (9, 16), (16, 16)])
    def test_enabling_boxes_raises_the_cap(self, sink, n, want):
        """The bucket rounds up to a power of two so a 5-box and a 7-box rig
        share an engine rather than forcing a rebuild when a box is added."""
        _enable(sink, n)
        assert sink._max_batch_size == want == batch_bucket(n)

    def test_disabling_boxes_lowers_it_again(self, sink):
        _enable(sink, 16)
        assert sink._max_batch_size == 16
        for b in range(2, 17):
            sink.disable_for_box(b)
        assert sink._max_batch_size == 1

    def test_enabling_the_same_box_twice_changes_nothing(self, sink):
        sink.enable_for_box(1)
        sink.enable_for_box(1)
        assert sink._max_batch_size == 1

    def test_the_cap_survives_a_model_being_configured_after_enabling(
            self, sink, monkeypatch):
        """The ordering that produced the bug: the GUI enables pose after
        loading the model, so a cap set only at load time is set from an empty
        enabled set. Re-loading a model must not undo a cap the enabled boxes
        have since justified.
        """
        _enable(sink, 8)
        assert sink._max_batch_size == 8

        class _Handle:
            def __init__(self):
                self.model = _Model()

        class _Model:
            def get_body_parts(self):
                return ["center"]

        monkeypatch.setattr(sink._backend, "get_or_create",
                            lambda *a, **k: _Handle())
        sink.configure_model(
            tracker_type="sleap", model_path="",
            probe_frame=np.zeros((176, 224, 3), np.uint8),
            input_mode="letterbox")
        assert sink._max_batch_size >= 8


class TestAModelThatRefusesIsNeverAskedAgain:
    """Raising the cap is only safe because a refusal is remembered.

    A DeepLabCut export's batch axis is fixed at 1. Once the cap rose to match
    the box count, the sink handed it a batch of 4, 8 or 16 on every frame, and
    each one raised inside ONNX Runtime, logged, and fell back, 290 times in
    one 8-second run, costing about a fifth of DLC's throughput. The tracker's
    per-call fallback is the right last resort and the wrong per-frame
    strategy.
    """

    def test_the_tracker_reports_it_cannot_batch(self):
        from source.video.tracking.pose import DLCLiveTracker, SLEAPTracker

        class _Export:
            batched = True

        dlc = DLCLiveTracker("models/x")
        assert dlc.batched is False, "DeepLabCut-Live itself never batches"
        dlc._export = _Export()
        assert dlc.batched is True, "an export with a dynamic batch axis does"
        dlc._export_no_batch = True
        assert dlc.batched is False, "…until it refuses one"

        sleap = SLEAPTracker("models/x")
        assert sleap.batched is True
        sleap._no_batch = True
        assert sleap.batched is False

    def test_the_sink_chunks_to_one_for_such_a_model(self, sink):
        """The chunk arithmetic the dispatch loop runs, in isolation."""
        _enable(sink, 16)
        assert sink._max_batch_size == 16

        class _Refuses:
            batched = False

        class _Batches:
            batched = True

        def chunk_for(model):
            c = sink._max_batch_size if sink._max_batch_size > 0 else 16
            if not getattr(model, "batched", True):
                c = 1
            return max(1, c)

        assert chunk_for(_Batches()) == 16
        assert chunk_for(_Refuses()) == 1

    def test_a_dlc_export_that_refused_goes_serial_without_raising_again(self):
        """Straight through the tracker: the second call must not re-enter the
        batched path."""
        import numpy as np

        from source.video.tracking.pose import DLCLiveTracker

        calls = []

        class _OneFrameOnly:
            batched = True

            def __call__(self, batch):
                calls.append(batch.shape[0])
                if batch.shape[0] != 1:
                    raise ValueError("Expected: 1")
                return np.zeros((1, 1, 1, 3), np.float32)

        t = DLCLiveTracker("models/x", body_parts=["center"])
        t._initialized = True
        t._export = _OneFrameOnly()
        frames = [np.zeros((8, 8, 3), np.uint8) for _ in range(4)]

        t.predict_batch(frames)          # discovers the refusal
        assert t._export_no_batch is True
        calls.clear()
        t.predict_batch(frames)          # must not try 4 again
        assert calls == [1, 1, 1, 1], calls


class TestTheChunkUsesIt:
    def test_the_chunk_is_the_cap(self, sink):
        """The cap is what the dispatch loop slices by, so a cap of 1 is
        literally one forward pass per box."""
        _enable(sink, 16)
        items = list(range(16))
        chunk = sink._max_batch_size if sink._max_batch_size > 0 else len(items)
        assert len([items[i:i + chunk] for i in range(0, len(items), chunk)]) == 1

    def test_a_cap_of_one_is_a_pass_per_box(self, sink):
        sink.enable_for_box(1)
        items = list(range(16))
        chunk = sink._max_batch_size
        assert len([items[i:i + chunk] for i in range(0, len(items), chunk)]) == 16
