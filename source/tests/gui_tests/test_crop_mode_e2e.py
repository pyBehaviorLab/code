"""Crop-track mode, through the real sink.

The unit tests cover the geometry; this covers the wiring, that the model is
fed the window and not the frame, that what comes back out is in the box's own
pixels, that the window follows the animal from frame to frame, and that a
model the mode is wrong for is refused rather than quietly mis-fed.
"""
import datetime

import numpy as np
import pytest

from source.video.framebus.pose_sink import PoseSink
from source.video.framebus.types import BoxFrame
from source.video.tracking.inference import ModelHandle


class _Model:
    """Reports the brightest pixel in whatever it is shown."""
    is_initialized = True

    def __init__(self):
        self.shapes = []
        self.body_parts = ["a", "b"]

    def get_body_parts(self):
        return ["a", "b"]

    def predict(self, frame):
        self.shapes.append(frame.shape[:2])
        grey = frame[:, :, 0] if frame.ndim == 3 else frame
        if grey.max() == 0:
            return {"a": None, "b": None}
        y, x = np.unravel_index(int(np.argmax(grey)), grey.shape)
        return {"a": [float(x), float(y), 0.9], "b": [float(x), float(y), 0.9]}

    def predict_batch(self, frames):
        return [self.predict(f) for f in frames]


class _Backend:
    def __init__(self):
        self.keys = []
        self.model = _Model()
        self.probe_shapes = []

    def get_or_create(self, key, **kw):
        self.keys.append(key)
        probe = kw.get("probe_frame")
        if probe is not None:
            self.probe_shapes.append(probe.shape[:2])
        return ModelHandle(key=key, model=self.model)

    def submit_batch(self, handle, items, on_done):
        poses = handle.model.predict_batch([f for _, f in items])
        for bid, pose in zip([b for b, _ in items], poses):
            on_done(bid, pose)
        return len(items)

    def shutdown(self, *a, **k):
        pass


def _frame(sid, w, h, mark=None):
    img = np.zeros((h, w, 3), np.uint8)
    if mark is not None:
        img[mark[1] - 1:mark[1] + 2, mark[0] - 1:mark[0] + 2] = 255
    return BoxFrame(image=img, setup_id=sid, cam_frame_id=1, camera_id=0,
                    capture_host_ns=1, capture_wall=datetime.datetime.now(),
                    is_shared_camera=True, poll_host_ns=1)


def _sink(mode="crop_track", wh=(64, 64), boxes=(1,), model_path="models/none"):
    s = PoseSink(backend=_Backend())
    for b in boxes:
        s.enable_for_box(b)
    s.configure_model(tracker_type="sleap", model_path=model_path,
                      probe_frame=np.zeros((240, 320, 3), np.uint8),
                      resize_factor=1.0, body_parts=["a", "b"], confidence=0.5,
                      input_mode=mode, input_wh=wh,
                      crop_opts={"conf_min": 0.2, "good_min": 2})
    return s


def _results(sink):
    got = {}
    sink._on_result.append(
        lambda sid, cfid, arr, *a, **k: got.__setitem__(sid, arr))
    return got


class TestTheModelSeesTheWindow:
    def test_the_probe_is_the_window_not_the_frame(self):
        """A TensorFlow session compiles its input placeholder from the probe,
        so a full-frame probe would build a session the real crops cannot be
        fed to."""
        s = _sink()
        assert s._backend.probe_shapes[-1] == (64, 64)

    def test_inference_gets_window_sized_frames(self):
        s = _sink()
        _results(s)
        s._dispatch_batch([(1, _frame(1, 320, 240, mark=(160, 120)))])
        assert s._backend.model.shapes[-1] == (64, 64)

    def test_the_model_key_is_the_window_so_boxes_share_it(self):
        s = _sink(boxes=(1, 2, 3))
        _results(s)
        s._dispatch_batch([(b, _frame(b, 320, 240, mark=(160, 120)))
                           for b in (1, 2, 3)])
        assert len(s._backend.keys) == 1
        assert s._backend.keys[0][1:3] == (64, 64)


class TestCoordinatesComeBackInBoxPixels:
    def test_a_marker_is_reported_where_it_is_in_the_frame(self):
        s = _sink()
        got = _results(s)
        # First frame has no fix, so the window starts at the frame centre.
        s._dispatch_batch([(1, _frame(1, 320, 240, mark=(160, 120)))])
        x, y, _c = got[1][0]
        assert x == pytest.approx(160, abs=2), got[1]
        assert y == pytest.approx(120, abs=2), got[1]

    def test_each_box_answers_in_its_own_pixels(self):
        s = _sink(boxes=(1, 2))
        got = _results(s)
        s._dispatch_batch([(1, _frame(1, 320, 240, mark=(160, 120))),
                           (2, _frame(2, 320, 240, mark=(150, 130)))])
        assert got[1][0][0] == pytest.approx(160, abs=2)
        assert got[2][0][0] == pytest.approx(150, abs=2)
        assert got[2][0][1] == pytest.approx(130, abs=2)


class TestTheWindowFollows:
    def test_it_re_centres_on_the_animal(self):
        s = _sink()
        _results(s)
        s._dispatch_batch([(1, _frame(1, 320, 240, mark=(160, 120)))])
        state = s._crop_states[1]
        assert state.cx == pytest.approx(160, abs=2)
        assert state.cy == pytest.approx(120, abs=2)

    def test_it_tracks_a_moving_animal_across_frames(self):
        """The window must still contain the animal after it has moved further
        than the window is wide."""
        s = _sink()
        got = _results(s)
        for x in range(160, 260, 20):
            s._dispatch_batch([(1, _frame(1, 320, 240, mark=(x, 120)))])
            assert got[1][0][0] == pytest.approx(x, abs=2), (
                f"lost the animal at x={x}")

    def test_an_empty_frame_does_not_move_the_window(self):
        s = _sink()
        _results(s)
        s._dispatch_batch([(1, _frame(1, 320, 240, mark=(160, 120)))])
        s._dispatch_batch([(1, _frame(1, 320, 240))])      # nothing to see
        assert s._crop_states[1].cx == pytest.approx(160, abs=2)
        assert s._crop_states[1].lost_frames == 1


class TestRefusals:
    def test_a_window_the_model_never_declared_falls_back(self, caplog):
        s = _sink(wh=(0, 0))
        assert s._input_mode == "letterbox"

    def test_letterbox_is_still_the_default(self):
        s = _sink(mode="letterbox", wh=None)
        assert s._input_mode == "letterbox"
        assert s._crop_wh is None

    def test_full_mode_applies_no_transform(self):
        s = _sink(mode="full", wh=None)
        got = _results(s)
        s._dispatch_batch([(1, _frame(1, 320, 240, mark=(160, 120)))])
        assert s._backend.model.shapes[-1] == (240, 320)
        assert got[1][0][0] == pytest.approx(160, abs=2)

    def test_a_top_down_model_is_refused_because_it_crops_itself(self, tmp_path):
        """sleap-nn crops around detected centroids inside the predictor, so a
        window of ours would crop a crop and map through two transforms."""
        model = tmp_path / "topdown"
        model.mkdir()
        (model / "training_config.yaml").write_text(
            "model_config:\n"
            "  head_configs:\n"
            "    centered_instance:\n"
            "      confmaps: {sigma: 2.5}\n",
            encoding="utf-8")
        s = _sink(model_path=str(model))
        assert s._input_mode == "letterbox"

    def test_a_stride_violating_window_is_refused(self, tmp_path):
        """A side not divisible by max_stride breaks the encoder arithmetic."""
        model = tmp_path / "single"
        model.mkdir()
        (model / "training_config.yaml").write_text(
            "model_config:\n"
            "  backbone_config:\n"
            "    unet: {max_stride: 16}\n"
            "  head_configs:\n"
            "    single_instance:\n"
            "      confmaps: {sigma: 2.5}\n",
            encoding="utf-8")
        assert _sink(wh=(100, 100), model_path=str(model))._input_mode == "letterbox"
        assert _sink(wh=(224, 176), model_path=str(model))._input_mode == "crop_track"
