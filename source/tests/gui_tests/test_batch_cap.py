"""One number decides how many boxes ride in a forward pass.

An exported SLEAP engine has its maximum batch fixed at export. If the sink
can submit more frames than that, inference fails outright, so the cap the
sink applies and the size the engine is built for cannot be two decisions.

``controller.py`` constructs ``PoseSink()`` bare, so a cap that is only a
documented ``__init__`` default is a cap nothing ever sets: it sits at 0
(unlimited) while the export is built for a fixed size. A rig with more boxes
than that size then submits a batch the engine cannot take.
"""
import datetime

import numpy as np

from source.video.framebus.pose_sink import PoseSink, _sleap_sig
from source.video.framebus.types import BoxFrame
from source.video.tracking.inference import ModelHandle
from source.video.tracking.sleap_export import batch_bucket


class _Tracker:
    is_initialized = True

    def __init__(self):
        self.batches = []
        self.body_parts = ["nose"]

    def get_body_parts(self):
        return ["nose"]

    def predict(self, f):
        return {"nose": [1.0, 1.0, 0.9]}

    def predict_batch(self, frames):
        self.batches.append(len(frames))
        return [self.predict(f) for f in frames]


class _Backend:
    def __init__(self):
        self.keys = []
        self.opts = []
        self.model = _Tracker()

    def get_or_create(self, key, **kw):
        self.keys.append(key)
        self.opts.append(kw.get("sleap_opts"))
        return ModelHandle(key=key, model=self.model)

    def submit_batch(self, handle, items, on_done):
        poses = handle.model.predict_batch([f for _, f in items])
        for bid, pose in zip([b for b, _ in items], poses):
            on_done(bid, pose)
        return len(items)

    def shutdown(self, *a, **k):
        pass


def _frame(sid, w=64, h=64):
    return BoxFrame(image=np.zeros((h, w, 3), np.uint8), setup_id=sid,
                    cam_frame_id=1, camera_id=0, capture_host_ns=1,
                    capture_wall=datetime.datetime.now(),
                    is_shared_camera=True, poll_host_ns=1)


def _sink(n_boxes, sleap=True):
    s = PoseSink(backend=_Backend())
    for sid in range(1, n_boxes + 1):
        s.enable_for_box(sid)
    s.configure_model(
        tracker_type="sleap" if sleap else "dlc", model_path="models/none",
        probe_frame=np.zeros((64, 64, 3), np.uint8), resize_factor=1.0,
        body_parts=["nose"], confidence=0.5,
        sleap_opts={"runtime": "tensorrt"} if sleap else None)
    return s


def test_the_cap_is_set_from_the_box_count():
    """Not left at 0 (unlimited), which is what a cap nothing sets means."""
    assert _sink(3)._max_batch_size == 4


def test_the_engine_is_asked_for_the_same_size_the_sink_will_send():
    """Two decisions here means an engine that cannot take the batch."""
    s = _sink(3)
    assert s._backend.opts[-1]["max_batch_size"] == s._max_batch_size


def test_box_count_rounds_up_so_adding_a_box_is_not_a_rebuild():
    assert _sink(5)._max_batch_size == batch_bucket(5) == 8
    assert _sink(7)._max_batch_size == 8


def test_a_bigger_rig_is_a_different_engine():
    """Not merely a surprise: an engine built for 4 cannot run a batch of 8."""
    assert _sink(3)._backend.keys[-1] != _sink(12)._backend.keys[-1]


def test_the_batch_size_is_in_the_model_signature():
    assert _sink(3)._backend.keys[-1][5] != _sink(12)._backend.keys[-1][5]
    assert "max_batch_size" in _sleap_sig({"max_batch_size": 8})


def test_dlc_is_not_handed_sleap_options():
    """DLC's constructor does not take them, and never should."""
    assert _sink(3, sleap=False)._backend.opts[-1] is None


def test_a_batch_never_exceeds_what_the_engine_was_built_for():
    """Boxes can be enabled after Init, so the cap has to hold at dispatch
    time; this is the case that would otherwise reach the GPU too large."""
    s = _sink(2)                       # engine built for a bucket of 2
    assert s._max_batch_size == 2
    for sid in range(3, 8):            # five more boxes join afterwards
        s.enable_for_box(sid)
    s._dispatch_batch([(sid, _frame(sid)) for sid in range(1, 8)])
    assert s._backend.model.batches, "nothing was submitted"
    assert max(s._backend.model.batches) <= 2, s._backend.model.batches
    assert sum(s._backend.model.batches) == 7, "a box was silently dropped"
