"""The direct runners: what they decode, and when they refuse to be used.

Neither TensorRT nor ONNX Runtime is installed here, so what is testable is the
part that decides *whether* to use them and the part that turns their output
into the pipeline's pose contract, which is also where a silent mistake would
live, because a wrongly-decoded keypoint is a plausible number in the wrong
place.
"""
import sys
import types

import numpy as np
import pytest

from source.video.tracking import trt_runner
from source.video.tracking.ort_runner import batch_to_poses, peaks_to_pose, to_nchw


class TestLayout:
    def test_hwc_becomes_nchw_with_a_batch_dimension(self):
        out = to_nchw(np.zeros((176, 224, 3), np.uint8))
        assert out.shape == (1, 3, 176, 224)

    def test_a_grayscale_frame_gains_its_channel(self):
        assert to_nchw(np.zeros((64, 64), np.uint8)).shape == (1, 1, 64, 64)

    def test_the_result_is_contiguous(self):
        """A reversed view has a negative stride, which torch rejects outright
        and other runtimes copy silently, the explicit copy is the cheaper
        surprise."""
        rgb = np.zeros((32, 32, 3), np.uint8)[:, ::-1]
        assert to_nchw(rgb).flags["C_CONTIGUOUS"]

    def test_dtype_is_preserved_because_the_graph_casts(self):
        """Pre-converting to float moves 4x the bytes for nothing."""
        assert to_nchw(np.zeros((8, 8, 3), np.uint8)).dtype == np.uint8


class TestDecoding:
    def test_keypoints_land_on_the_right_body_parts(self):
        pose = peaks_to_pose(np.array([[1.0, 2.0], [3.0, 4.0]]),
                             np.array([0.9, 0.5]), ["nose", "tail"])
        assert pose["nose"] == [1.0, 2.0, 0.9]
        assert pose["tail"] == [3.0, 4.0, 0.5]

    def test_a_nan_peak_is_reported_as_not_found(self):
        pose = peaks_to_pose(np.array([[np.nan, np.nan]]), np.array([0.0]),
                             ["nose"])
        assert pose["nose"] is None

    def test_confidence_is_carried_not_gated(self):
        """The host-side threshold is the operator's live control; applying it
        here as well is how the two get to disagree."""
        pose = peaks_to_pose(np.array([[1.0, 2.0]]), np.array([0.001]), ["nose"])
        assert pose["nose"][2] == pytest.approx(0.001)

    def test_a_missing_keypoint_is_none_not_a_crash(self):
        pose = peaks_to_pose(np.zeros((1, 2)), np.array([0.9]), ["a", "b"])
        assert pose["b"] is None

    def test_a_batch_decodes_one_pose_per_frame_in_order(self):
        peaks = np.array([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]])
        vals = np.array([[0.9], [0.8], [0.7]])
        poses = batch_to_poses(peaks, vals, ["nose"])
        assert [p["nose"][0] for p in poses] == [1.0, 2.0, 3.0]

    def test_a_single_frame_still_yields_one_pose(self):
        poses = batch_to_poses(np.array([[1.0, 1.0]]), np.array([0.9]), ["nose"])
        assert len(poses) == 1


class TestWhenTheEngineIsUsable:
    def test_no_engine_file_means_not_usable(self, tmp_path):
        assert trt_runner.engine_path(str(tmp_path)) is None
        assert trt_runner.usable(str(tmp_path)) is False

    def test_an_engine_alone_is_not_enough_without_tensorrt(self, tmp_path,
                                                            monkeypatch):
        """Present on disk, unusable on this machine, the fallback to the
        library path has to happen for a reason the operator can read."""
        (tmp_path / "model.trt").write_bytes(b"engine")
        assert trt_runner.engine_path(str(tmp_path)) is not None
        from source.video.tracking import capability
        monkeypatch.setattr(capability, "_cached",
                            capability.Capabilities(cuda=False))
        assert trt_runner.usable(str(tmp_path)) is False

    def test_it_is_usable_when_both_halves_hold(self, tmp_path, monkeypatch):
        (tmp_path / "model.trt").write_bytes(b"engine")
        from source.video.tracking import capability
        monkeypatch.setattr(
            capability, "_cached",
            capability.Capabilities(cuda=True, tensorrt=True,
                                    tensorrt_version="10.16.1"))
        assert trt_runner.usable(str(tmp_path)) is True

    def test_a_stale_engine_is_refused_with_a_reason(self, tmp_path, monkeypatch):
        """Deserialise returning None is how a cache stale after a driver
        upgrade becomes a mysterious drop to a slower runtime."""
        fake_trt = types.ModuleType("tensorrt")

        class _Logger:
            ERROR = 0

            def __init__(self, *a):
                pass

        fake_trt.Logger = _Logger
        fake_trt.Runtime = lambda *a: types.SimpleNamespace(
            deserialize_cuda_engine=lambda blob: None)
        fake_trt.TensorIOMode = types.SimpleNamespace(INPUT=0)
        fake_trt.nptype = lambda t: np.uint8
        monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"stale")
        with pytest.raises(RuntimeError, match="different GPU, driver"):
            trt_runner.TRTRunner(str(engine))
