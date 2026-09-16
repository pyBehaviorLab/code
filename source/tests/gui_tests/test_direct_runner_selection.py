"""The tracker must drive the engine itself when it can.

Going through sleap-nn's Predictor is the same weights at roughly four times
the cost, so when an export decodes peaks in-graph the tracker should hold the
engine directly. Two things have to be true for that to be safe: it is chosen
only when the graph really is the decoded kind, and anything unexpected falls
back to the library rather than failing the run.

A model directory that already contains an export is the case worth naming,
re-exporting it would need sleap-nn and minutes to produce what is already
sitting on disk.
"""
import json

import numpy as np
import pytest

from source.video.tracking.pose import SLEAPTracker

_REAL = "models/sleap"


def _export_dir(tmp_path, *, onnx=True, trt=False, meta=None):
    d = tmp_path / "export"
    d.mkdir()
    if onnx:
        (d / "model.onnx").write_bytes(b"not a real graph")
    if trt:
        (d / "model.trt").write_bytes(b"not a real engine")
    if meta is not None:
        (d / "export_metadata.json").write_text(json.dumps(meta),
                                                encoding="utf-8")
    return str(d)


class TestChoosingTheDirectPath:
    def test_a_decoded_onnx_graph_is_driven_directly(self, tmp_path, monkeypatch):
        made = {}

        class _Runner:
            backend = "onnxruntime:cpu/4t"
            threads = 4
            provider = "CPUExecutionProvider"

            # **kw, not a fixed list: the caller passes the rig's batch cap
            # too now, and a stub that refuses an argument makes the code
            # under test look broken when it is the stub that is stale.
            def __init__(self, path, prefer_gpu=False, **kw):
                made["path"] = path
                made.update(kw)

            def __call__(self, x):
                return {}

        import source.video.tracking.ort_runner as ort
        monkeypatch.setattr(ort, "graph_is_decoded", lambda p: True)
        monkeypatch.setattr(ort, "ORTRunner", _Runner)
        t = SLEAPTracker("m", device="cpu", max_batch_size=8)
        got = t._direct_runner(_export_dir(tmp_path), "onnx")
        assert isinstance(got, _Runner)
        assert made["path"].endswith("model.onnx")
        assert "onnxruntime" in t.resolved_backend
        # The rig's batch cap reaches the runner, so a dynamic batch axis can
        # be padded to one steady shape. Without it ONNX Runtime re-plans on
        # every size change: 73 frames/s against 188 on a varying batch.
        assert made.get("pad_to") == 8

    def test_a_heatmap_graph_is_left_to_the_library(self, tmp_path, monkeypatch):
        """Without in-graph decoding the caller would have to reimplement
        sleap-nn's post-processing, which is the thing being avoided."""
        import source.video.tracking.ort_runner as ort
        monkeypatch.setattr(ort, "graph_is_decoded", lambda p: False)
        t = SLEAPTracker("m", device="cpu")
        assert t._direct_runner(_export_dir(tmp_path), "onnx") is None

    def test_no_graph_at_all_is_left_to_the_library(self, tmp_path):
        t = SLEAPTracker("m", device="cpu")
        assert t._direct_runner(_export_dir(tmp_path, onnx=False), "onnx") is None

    def test_a_runner_that_raises_falls_back_rather_than_failing(self, tmp_path,
                                                                 monkeypatch):
        import source.video.tracking.ort_runner as ort
        monkeypatch.setattr(ort, "graph_is_decoded", lambda p: True)
        monkeypatch.setattr(ort, "ORTRunner",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("bad provider")))
        t = SLEAPTracker("m", device="cpu")
        assert t._direct_runner(_export_dir(tmp_path), "onnx") is None

    def test_the_native_runtime_asks_for_no_engine(self, tmp_path, monkeypatch):
        import source.video.tracking.ort_runner as ort
        monkeypatch.setattr(ort, "graph_is_decoded", lambda p: True)
        t = SLEAPTracker("m", device="cpu")
        assert t._direct_runner(_export_dir(tmp_path), "native") is None

    def test_tensorrt_is_not_attempted_without_the_capability(self, tmp_path,
                                                              monkeypatch):
        from source.video.tracking import capability
        monkeypatch.setattr(capability, "_cached",
                            capability.Capabilities(cuda=False))
        import source.video.tracking.ort_runner as ort
        monkeypatch.setattr(ort, "graph_is_decoded", lambda p: False)
        t = SLEAPTracker("m", device="cpu")
        assert t._direct_runner(_export_dir(tmp_path, trt=True), "tensorrt") is None


class TestTheEngineBatchWarning:
    """Asked of the RUNNER that was built, not of the manifest beside it.

    ``export_metadata.json``'s ``max_batch_size`` describes the TensorRT engine
    build. The ONNX graph exported alongside it routinely has a *dynamic* batch
    axis and takes any size, ``models/sleap`` declares 1 and runs a batch of
    16 at 0.57 ms/frame against 3.06 ms at batch 1. Trusting the manifest told
    operators to re-export a model that was already fine, and warned on every
    multi-box run that was working perfectly.
    """

    class _Runner:
        def __init__(self, max_batch):
            self.max_batch = max_batch

    def test_a_fixed_axis_smaller_than_the_rig_is_called_out(self, tmp_path,
                                                             caplog):
        t = SLEAPTracker("m", max_batch_size=8)
        t._runner = self._Runner(1)
        with caplog.at_level("WARNING", logger="source.video.tracking.pose"):
            t._check_engine_batch(_export_dir(tmp_path))
        assert any("accepts a batch of 1" in r.getMessage()
                   for r in caplog.records)

    def test_a_dynamic_axis_says_nothing_whatever_the_manifest_claims(
            self, tmp_path, caplog):
        """The case that mattered: manifest says 1, graph takes anything."""
        d = _export_dir(tmp_path, meta={"max_batch_size": 1})
        t = SLEAPTracker("m", max_batch_size=16)
        t._runner = self._Runner(None)          # symbolic 'batch' dimension
        with caplog.at_level("WARNING", logger="source.video.tracking.pose"):
            t._check_engine_batch(d)
        assert not caplog.records

    def test_a_big_enough_engine_says_nothing(self, tmp_path, caplog):
        t = SLEAPTracker("m", max_batch_size=2)
        t._runner = self._Runner(8)
        with caplog.at_level("WARNING", logger="source.video.tracking.pose"):
            t._check_engine_batch(_export_dir(tmp_path))
        assert not caplog.records

    def test_no_runner_yet_is_not_an_error(self, tmp_path):
        SLEAPTracker("m", max_batch_size=8)._check_engine_batch(
            _export_dir(tmp_path))

    def test_the_real_graph_reports_its_own_batch_axis(self):
        """Ground truth, from the model on disk rather than a stub."""
        pytest.importorskip("onnxruntime")
        import os
        if not os.path.isfile(f"{_REAL}/model.onnx"):
            pytest.skip("the exported v4 model is not present")
        from source.video.tracking.ort_runner import ORTRunner
        r = ORTRunner(f"{_REAL}/model.onnx", prefer_gpu=False)
        assert r.max_batch is None, (
            "this graph's batch axis is dynamic; a fixed int here would mean "
            "the multi-box batching win is genuinely unavailable")


@pytest.mark.skipif(not __import__("os").path.isfile(f"{_REAL}/model.onnx"),
                    reason="the exported v4 model is not present")
class TestTheRealExport:
    """The model directory here IS an export, engine beside the config."""

    def test_the_graph_is_the_decoded_kind(self):
        pytest.importorskip("onnxruntime")
        from source.video.tracking.ort_runner import graph_is_decoded
        assert graph_is_decoded(f"{_REAL}/model.onnx")

    def test_it_is_driven_directly_with_no_export_step(self):
        pytest.importorskip("onnxruntime")
        t = SLEAPTracker(_REAL, device="cpu", runtime="onnx")
        runner = t._direct_runner(_REAL, "onnx")
        assert runner is not None, "would have re-exported what is already here"
        assert "onnxruntime" in t.resolved_backend

    def test_real_inference_gives_one_entry_per_body_part(self):
        pytest.importorskip("onnxruntime")
        t = SLEAPTracker(_REAL, device="cpu", runtime="onnx")
        assert t.initialize(np.zeros((176, 224, 3), np.uint8)) is True
        assert t._runner is not None, "the library path was taken"
        assert t._predictor is None, "both paths were held at once"
        assert list(t.get_body_parts()) == [
            "Snout", "Head", "Left_Ear", "Right_Ear", "center", "Tail_Base"]
        pose = t.predict(np.zeros((176, 224, 3), np.uint8))
        assert set(pose) == set(t.get_body_parts())
        t.close()

    def test_a_batch_returns_one_pose_per_frame_in_order(self):
        pytest.importorskip("onnxruntime")
        t = SLEAPTracker(_REAL, device="cpu", runtime="onnx")
        t.initialize(np.zeros((176, 224, 3), np.uint8))
        frames = [np.full((176, 224, 3), v, np.uint8) for v in (0, 40, 80)]
        out = t.predict_batch(frames)
        assert len(out) == 3
        assert all(set(p) == set(t.get_body_parts()) for p in out)
        t.close()
