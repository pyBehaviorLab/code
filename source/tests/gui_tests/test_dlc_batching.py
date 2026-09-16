"""When DLC can run a batch, and when it genuinely cannot.

This replaces ``test_dlc_batch_is_serial.py``, whose premise was that DLC's
batch path is serial and must stay so. That was true of *DeepLabCut-Live's
API* and never of the model underneath it: its PyTorch runner holds an ordinary
``nn.Module`` whose forward has always accepted ``(N, C, H, W)``. The library
simply has no entry point that passes one, its own documentation lists
``resize``, ``cropping``, ``dynamic`` and ``model_type`` as the speed levers and
no batching among them.

So the rig ran sixteen forward passes for one camera frame. Measured on this
project's ResNet-50 at 360x202: 18.60 ms/frame one at a time against 3.25 ms
batched at sixteen, 5.7x, for identical keypoints.

Two cases still cannot batch, and both are pinned below:

* the **TensorFlow** runners, which compile the input placeholder as
  ``[1, H, W, 3]`` at ``init_inference``, so batch size one is baked into the
  graph and into any TensorRT engine or tflite interpreter built from it;
* a runner doing its **own per-frame cropping**: a detector or a
  ``DynamicCropper`` gives each frame a different crop, so the frames stop
  sharing a shape and there is no batch to form.
"""
import logging

import numpy as np

from source.video.tracking import dlc_engine
from source.video.tracking.pose import DLCLiveTracker, PoseTracker, SLEAPTracker


def _tracker():
    """A tracker that answers without dlclive, which may not be installed."""
    t = DLCLiveTracker("models/nonexistent")
    seen = []

    def _fake_predict(frame):
        seen.append(frame)
        return {"nose": [float(frame[0, 0, 0]), 0.0, 0.9]}

    t.predict = _fake_predict
    return t, seen


def _frames(n):
    return [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]


class _Runner:
    """Stands in for dlclive's runner: what ``batchable`` inspects."""

    def __init__(self, kind="pytorch", detector=None, dynamic=None):
        self.model = object() if kind else None
        self.detector = detector
        self.dynamic = dynamic
        if kind == "pytorch":
            self.pose_transform = lambda x: x


class _Live:
    def __init__(self, runner):
        self.runner = runner


class TestWhoCanBatch:
    def test_a_plain_pytorch_runner_can(self):
        assert dlc_engine.batchable(_Live(_Runner("pytorch")))

    def test_a_tensorflow_runner_cannot(self):
        """It has no ``pose_transform``; its placeholder is compiled at 1."""
        assert not dlc_engine.batchable(_Live(_Runner("tensorflow")))

    def test_a_runner_with_a_detector_cannot(self):
        """Top-down: each frame yields its own crops, so no shared shape."""
        assert not dlc_engine.batchable(
            _Live(_Runner("pytorch", detector=object())))

    def test_a_runner_doing_dynamic_cropping_cannot(self):
        """The cropper carries per-frame offset state a batch would unpick."""
        assert not dlc_engine.batchable(
            _Live(_Runner("pytorch", dynamic=object())))

    def test_a_model_that_has_not_loaded_yet_cannot(self):
        assert not dlc_engine.batchable(_Live(_Runner(None)))

    def test_no_runner_at_all_is_not_an_error(self):
        assert not dlc_engine.batchable(object())


class TestTheContractHoldsEitherWay:
    """Whether it batches or not, the per-box callback contract is the same:
    one result per frame, in the order given."""

    def test_every_frame_is_inferred_in_order_when_serial(self):
        t, seen = _tracker()
        out = t.predict_batch(_frames(5))
        assert len(out) == 5
        assert len(seen) == 5, "a dropped frame would silently blank that box"
        assert [r["nose"][0] for r in out] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_an_empty_batch_is_not_an_error(self):
        t, _ = _tracker()
        assert t.predict_batch([]) == []

    def test_a_failed_batch_falls_back_and_stays_fallen_back(self, caplog):
        """One refusal is a finding; a refusal per frame is log spam on the
        inference path."""
        t, seen = _tracker()
        t._dlc_live = _Live(_Runner("pytorch"))

        def _boom(frames):
            raise RuntimeError("no")

        t._torch_batch = _boom
        with caplog.at_level(logging.WARNING,
                             logger="source.video.tracking.pose"):
            first = t.predict_batch(_frames(4))
            second = t.predict_batch(_frames(4))
        assert len(first) == len(second) == 4
        assert t._no_torch_batch is True
        assert len([r for r in caplog.records
                    if "batched forward failed" in r.message]) == 1


class TestTheSerialNoticeStillExists:
    def test_the_cost_is_announced_once_not_per_batch(self, caplog):
        t, _ = _tracker()
        with caplog.at_level(logging.INFO, logger="source.video.tracking.pose"):
            t.predict_batch(_frames(4))
            t.predict_batch(_frames(4))
        notices = [r for r in caplog.records
                   if "forward passes" in r.message]
        assert len(notices) == 1
        assert "cannot batch" in notices[0].getMessage(), (
            "the notice must name WHICH runners cannot, now that one can")

    def test_a_single_box_says_nothing(self, caplog):
        """One box is one forward pass either way."""
        t, _ = _tracker()
        with caplog.at_level(logging.INFO, logger="source.video.tracking.pose"):
            t.predict_batch(_frames(1))
        assert not [r for r in caplog.records if "forward passes" in r.message]


def test_both_backends_override_the_serial_default():
    """Each override carries a finding about its SDK; deleting one loses it."""
    assert DLCLiveTracker.predict_batch is not PoseTracker.predict_batch
    assert SLEAPTracker.predict_batch is not PoseTracker.predict_batch
