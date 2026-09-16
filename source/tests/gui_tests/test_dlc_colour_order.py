"""DLC must not convert the colour we already converted.

``DLCLive.convert2rgb`` reads like "ensure RGB". It is not: dlclive's
``img_to_rgb`` REVERSES the channels of any 3-D array (``cv2.cvtColor(...,
COLOR_BGR2RGB)``, no guard) and only *promotes* a 2-D one. The pipeline
converts BGR→RGB once per camera frame and hands pose an RGB frame, so leaving
the default ``True`` fed the network BGR.

That is the failure mode a detection rate cannot see: on near-grayscale operant
footage a channel-swapped model scores *higher* frame coverage while its
keypoint error nearly doubles. Only ground truth catches it, so these tests
pin the flag itself.
"""
import sys
import types

import numpy as np
import pytest

from source.video.tracking.model_config import GRAYSCALE, RGB, ModelInfo
from source.video.tracking.pose import DLCLiveTracker


class _FakeDLCLive:
    """Captures what the tracker asked for."""
    last_kwargs = None

    def __init__(self, model_path, **kwargs):
        _FakeDLCLive.last_kwargs = dict(kwargs)
        self.cfg = {}

    def init_inference(self, frame):
        return None

    def close(self):
        return None


@pytest.fixture
def fake_dlclive(monkeypatch):
    """dlclive is not installed here; the constructor contract is still ours."""
    mod = types.ModuleType("dlclive")
    mod.DLCLive = _FakeDLCLive
    mod.Processor = type("Processor", (), {})
    monkeypatch.setitem(sys.modules, "dlclive", mod)
    _FakeDLCLive.last_kwargs = None
    return mod


def _init(tracker, channels):
    """Initialise with the model claiming a channel expectation."""
    tracker._model_info = ModelInfo(backend="dlc", channels=channels)
    frame = (np.zeros((32, 32), np.uint8) if channels == GRAYSCALE
             else np.zeros((32, 32, 3), np.uint8))
    assert tracker.initialize(frame) is True
    return _FakeDLCLive.last_kwargs


def test_a_colour_model_is_not_asked_to_convert_again(fake_dlclive):
    """We hand it RGB; converting again would hand the network BGR."""
    kwargs = _init(DLCLiveTracker("models/x"), RGB)
    assert kwargs["convert2rgb"] is False


def test_an_unstated_expectation_is_treated_as_colour(fake_dlclive):
    """A DLC config that says nothing still gets a 3-channel frame from us."""
    kwargs = _init(DLCLiveTracker("models/x"), None)
    assert kwargs["convert2rgb"] is False


def test_a_grayscale_model_still_gets_the_promotion(fake_dlclive):
    """For a 2-D frame the same flag PROMOTES rather than reverses, which is
    what a one-channel frame needs."""
    kwargs = _init(DLCLiveTracker("models/x", colour_mode="grayscale"), GRAYSCALE)
    assert kwargs["convert2rgb"] is True


def test_the_flag_is_always_stated(fake_dlclive):
    """Never left to the default: the default is the bug."""
    kwargs = _init(DLCLiveTracker("models/x"), RGB)
    assert "convert2rgb" in kwargs


def test_the_flag_is_not_relied_on_because_dlclive_overrides_it(fake_dlclive):
    """The flag above is necessary and NOT sufficient, and that is the point.

    ``DLCLive.get_pose`` opens with

        if frame.ndim >= 2:
            self.convert2rgb = True

    and ``ndim >= 2`` holds for every frame there has ever been. So whatever is
    asked for at construction is forced back on at each call, ``process_frame``
    runs ``img_to_rgb``, and the channels are reversed, with the flag set to
    False and this file asserting it was set.

    Which is why the tracker no longer goes through ``DLCLive.get_pose`` at
    all. It calls the RUNNER, where the model lives and no colour conversion
    happens. Measured: the two paths disagreed by 484 px before, and by
    0.017 px after.
    """
    t = DLCLiveTracker("models/x")
    t._model_info = ModelInfo(backend="dlc", channels=RGB)
    assert t.initialize(np.zeros((32, 32, 3), np.uint8))

    handed = {}

    class _Runner:
        def get_pose(self, frame):
            handed["frame"] = frame
            return np.zeros((1, 3))

    t._dlc_live.runner = _Runner()
    frame = np.zeros((8, 8, 3), np.uint8)
    frame[:, :, 0] = 200                      # red channel, in RGB
    t._run_dlclive(frame)

    assert handed["frame"] is not None
    assert np.array_equal(handed["frame"], frame), (
        "the runner must receive the frame the pipeline produced, unreversed")


def test_a_missing_runner_still_produces_a_pose(fake_dlclive):
    """An older DeepLabCut-Live exposing no runner must still track."""
    t = DLCLiveTracker("models/x")
    t._model_info = ModelInfo(backend="dlc", channels=RGB)
    assert t.initialize(np.zeros((32, 32, 3), np.uint8))
    t._dlc_live.get_pose = lambda f: np.zeros((1, 3))
    assert t._run_dlclive(np.zeros((8, 8, 3), np.uint8)) is not None


def test_the_pipeline_really_hands_pose_rgb():
    """The other half of the contract. If this ever flips, the flag above is
    wrong and the network is fed BGR again, so assert it rather than trust
    the comment."""
    import cv2

    from source.video.framebus.types import convert_color

    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[:, :, 0] = 255                      # pure BLUE in BGR
    rgb = convert_color(bgr, "rgb")
    assert rgb[0, 0, 2] == 255 and rgb[0, 0, 0] == 0, (
        "convert_color no longer yields RGB; DLC's convert2rgb must flip too")
    assert np.array_equal(rgb, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
