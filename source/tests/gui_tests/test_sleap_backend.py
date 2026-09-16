"""S1, corrected native SLEAP backend (host-testable core).

No sleap-nn / GPU / model needed: model-type detection reads a fake config,
output parsing runs against mock predictors/Outputs, and the TrackingConfig
round-trip is pure. The live inference itself is verified on the rig.
"""
from __future__ import annotations

import json

import numpy as np

from source.video.tracking.pose import (
    SLEAPTracker, detect_sleap_model_type)
from source.video.framebus.types import TrackingConfig


# ── model-type detection ───────────────────────────────────────────────

def _write_cfg(tmp_path, heads: dict, name="training_config.json"):
    (tmp_path / name).write_text(
        json.dumps({"model_config": {"head_configs": heads}}), encoding="utf-8")
    return str(tmp_path)


def test_detect_single_instance(tmp_path):
    p = _write_cfg(tmp_path, {"single_instance": {"confmaps": {}}})
    assert detect_sleap_model_type(p) == "single"


def test_detect_centroid_and_centered(tmp_path):
    assert detect_sleap_model_type(
        _write_cfg(tmp_path, {"centroid": {"confmaps": {}}})) == "centroid"


def test_detect_bottomup(tmp_path):
    p = _write_cfg(tmp_path, {"bottomup": {"confmaps": {}, "pafs": {}}})
    assert detect_sleap_model_type(p) == "bottomup"


def test_detect_multiclass_wins(tmp_path):
    # multi-class heads take precedence over the plain head they extend.
    p = _write_cfg(tmp_path, {"multi_class_bottomup": {"confmaps": {}},
                              "bottomup": {"confmaps": {}}})
    assert detect_sleap_model_type(p) == "multi_class_bottomup"


def test_detect_unknown_when_no_config(tmp_path):
    assert detect_sleap_model_type(str(tmp_path)) == "unknown"


# ── top-down two-model ordering (the bug S1 fixes) ─────────────────────

def test_topdown_orders_centroid_then_centered():
    t = SLEAPTracker("centered_model", centroid_path="centroid_model",
                     model_type="topdown")
    assert t._ordered_model_paths() == ["centroid_model", "centered_model"]


def test_single_instance_one_path():
    t = SLEAPTracker("single_model", model_type="single")
    assert t._ordered_model_paths() == ["single_model"]


# ── output parsing (the guesswork S1 replaces) ─────────────────────────

class _FakeOutputs:
    def __init__(self, kpts, vals):
        self.pred_keypoints = kpts       # (B, I, N, 2)
        self.pred_peak_values = vals     # (B, I, N)


class _StreamPredictor:
    def __init__(self, outputs):
        self._o = outputs

    def predict_streaming(self, batch):
        return iter([self._o])


def _tracker_ready(predictor, parts):
    t = SLEAPTracker("m")
    t._body_parts = parts
    t._initialized = True
    t._predictor = predictor
    return t


def test_parse_raw_outputs_to_instance():
    kp = np.array([[[[10, 20], [30, 40], [50, 60]]]], dtype=float)   # (1,1,3,2)
    pv = np.array([[[0.9, 0.8, 0.7]]], dtype=float)                  # (1,1,3)
    t = _tracker_ready(_StreamPredictor(_FakeOutputs(kp, pv)),
                       ["nose", "neck", "tail"])
    res = t.predict(np.zeros((64, 64, 3), np.uint8))
    assert res["nose"] == (10.0, 20.0, 0.9)
    assert res["tail"] == (50.0, 60.0, 0.7)


def test_predict_batch_reads_every_frame():
    # S2: a batched forward pass must yield one result PER frame, not collapse
    # to frame 0. 3 frames, 1 instance each, distinct coords.
    kp = np.array([
        [[[1, 1], [2, 2]]],     # frame 0
        [[[3, 3], [4, 4]]],     # frame 1
        [[[5, 5], [6, 6]]],     # frame 2
    ], dtype=float)             # (3, 1, 2, 2)
    pv = np.array([[[0.9, 0.9]], [[0.8, 0.8]], [[0.7, 0.7]]], dtype=float)  # (3,1,2)
    t = _tracker_ready(_StreamPredictor(_FakeOutputs(kp, pv)), ["a", "b"])
    frames = [np.zeros((16, 16, 3), np.uint8) for _ in range(3)]
    res = t.predict_batch(frames)
    assert len(res) == 3
    assert res[0]["a"] == (1.0, 1.0, 0.9)
    assert res[1]["a"] == (3.0, 3.0, 0.8)
    assert res[2]["a"] == (5.0, 5.0, 0.7)


def test_parse_multi_instance_returns_primary():
    # two animals in the frame → single-animal contract returns instance 0.
    kp = np.array([[[[1, 1], [2, 2]], [[9, 9], [8, 8]]]], dtype=float)  # (1,2,2,2)
    pv = np.array([[[0.99, 0.98], [0.9, 0.9]]], dtype=float)            # (1,2,2)
    t = _tracker_ready(_StreamPredictor(_FakeOutputs(kp, pv)), ["a", "b"])
    res = t.predict(np.zeros((32, 32, 3), np.uint8))
    assert res["a"] == (1.0, 1.0, 0.99)      # primary instance


class _NumpyInstance:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def numpy(self):
        return self._arr


class _LabelsPredictor:
    """Mimics sio.Labels: predict() returns an object with .instances."""
    def __init__(self, instances):
        self._insts = instances

    def predict(self, batch):
        class _L:
            instances = self._insts
        return _L()


def test_parse_sio_labels_numpy_instance():
    inst = _NumpyInstance([[5, 6, 0.95], [7, 8, 0.91]])
    t = _tracker_ready(_LabelsPredictor([inst]), ["p1", "p2"])
    res = t.predict(np.zeros((16, 16, 3), np.uint8))
    assert res["p1"] == (5.0, 6.0, 0.95)
    assert res["p2"] == (7.0, 8.0, 0.91)


def test_grayscale_frame_is_refused():
    t = _tracker_ready(_StreamPredictor(_FakeOutputs(
        np.zeros((1, 1, 1, 2)), np.zeros((1, 1, 1)))), ["a"])
    res = t.predict(np.zeros((16, 16), np.uint8))   # 2D → empty
    assert res == {"a": None}


# ── TrackingConfig round-trip + derived opts ───────────────────────────

def test_sleap_config_roundtrip_and_derived_opts():
    tc = TrackingConfig(setup_id=1, tracker_type="sleap",
                        sleap_model_type="topdown", sleap_centroid_path="/c",
                        sleap_runtime="tensorrt", sleap_device="cuda",
                        sleap_fp16=True, sleap_peak_threshold=0.3)
    back = TrackingConfig.from_json(tc.to_json(), setup_id=1)
    assert back.sleap_runtime == "tensorrt"
    assert back.sleap_centroid_path == "/c"
    assert back.sleap_fp16 is True
    opts = back.derived_sleap_opts()
    assert opts["runtime"] == "tensorrt"
    assert opts["centroid_path"] == "/c"
    assert opts["peak_threshold"] == 0.3


def test_derived_sleap_opts_empty_for_dlc():
    assert TrackingConfig(setup_id=1, tracker_type="dlc").derived_sleap_opts() == {}


# ── S3: export cache + runtime fallback ────────────────────────────────

def test_export_cache_dir_keyed_by_runtime_and_device(tmp_path, monkeypatch):
    from source.video.tracking import sleap_export as se
    monkeypatch.setattr(se, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(se, "_model_sig", lambda *a, **k: "abc123")
    trt = se.export_cache_dir("m", "tensorrt", "cuda")
    onnx = se.export_cache_dir("m", "onnx", "cuda")
    cpu = se.export_cache_dir("m", "tensorrt", "cpu")
    assert trt.name == "abc123-cuda-tensorrt-fp16-b8-p001"
    assert onnx != trt and cpu != trt        # runtime + device fragment the cache
    # cuda:0 normalises to cuda (index doesn't fragment).
    assert se.export_cache_dir("m", "tensorrt", "cuda:0") == trt


def test_export_cache_dir_is_keyed_by_precision(tmp_path, monkeypatch):
    """Precision is baked into the engine at export, so it must fragment the
    cache. Without it, asking for fp32 was served the cached fp16 engine,
    the setting changed, the run did not, and nothing said so."""
    from source.video.tracking import sleap_export as se
    monkeypatch.setattr(se, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(se, "_model_sig", lambda *a, **k: "abc123")
    fp16 = se.export_cache_dir("m", "tensorrt", "cuda", precision="fp16")
    fp32 = se.export_cache_dir("m", "tensorrt", "cuda", precision="fp32")
    assert fp16 != fp32
    assert "-fp32-" in fp32.name
    # is_exported must ask about the same engine the export would produce.
    assert se.is_exported("m", "tensorrt", "cuda", precision="fp32") is False


def test_export_cache_dir_is_keyed_by_max_batch(tmp_path, monkeypatch):
    """An engine built for 4 boxes cannot run a batch of 8, so serving the
    cached one to a grown rig fails at inference, not merely surprises."""
    from source.video.tracking import sleap_export as se
    monkeypatch.setattr(se, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(se, "_model_sig", lambda *a, **k: "abc123")
    four = se.export_cache_dir("m", "tensorrt", "cuda", max_batch_size=4)
    eight = se.export_cache_dir("m", "tensorrt", "cuda", max_batch_size=8)
    assert four != eight
    assert se.is_exported("m", "tensorrt", "cuda", max_batch_size=4) is False


def test_export_cache_dir_is_keyed_by_peak_threshold(tmp_path, monkeypatch):
    """sleap-nn compiles peak decoding INTO the graph, so the threshold is a
    constant in the artifact, not an argument anything can pass later."""
    from source.video.tracking import sleap_export as se
    monkeypatch.setattr(se, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(se, "_model_sig", lambda *a, **k: "abc123")
    loose = se.export_cache_dir("m", "tensorrt", "cuda", peak_threshold=0.01)
    tight = se.export_cache_dir("m", "tensorrt", "cuda", peak_threshold=0.2)
    assert loose != tight
    assert se.is_exported("m", "tensorrt", "cuda", peak_threshold=0.2) is False


def test_a_fixed_peak_family_exports_permissively_so_the_knob_stays_live():
    """One peak per keypoint, each with its own confidence, so the host can
    gate, and the operator's threshold costs no rebuild."""
    from source.video.tracking.sleap_export import PERMISSIVE_PEAK_THRESHOLD
    t = SLEAPTracker("m", model_type="single", peak_threshold=0.45)
    assert t.export_peak_threshold == PERMISSIVE_PEAK_THRESHOLD


def test_a_bottom_up_model_keeps_the_operator_value_baked():
    """There the threshold decides how many candidates exist at all."""
    t = SLEAPTracker("m", model_type="bottomup", peak_threshold=0.45)
    assert t.export_peak_threshold == 0.45


def test_the_batch_bucket_rounds_up_so_one_more_box_is_not_a_rebuild():
    """A TensorRT build is minutes; keying on the exact count would spend them
    every time a box joins or leaves a session."""
    from source.video.tracking.sleap_export import batch_bucket
    assert batch_bucket(1) == 1
    assert batch_bucket(3) == 4
    assert [batch_bucket(n) for n in (5, 6, 7, 8)] == [8, 8, 8, 8]
    assert batch_bucket(0) == 1          # never zero: a batch holds one box
    assert batch_bucket(9999) == 32      # the engine stops being worth it


def test_sleap_export_precision_follows_the_fp16_toggle():
    """The toggle previously applied only to the native torch path, so an
    exported engine was fp16 whatever the dialog said."""
    assert SLEAPTracker("m", fp16=True).export_precision == "fp16"
    assert SLEAPTracker("m", fp16=False).export_precision == "fp32"


def test_ensure_exported_skips_when_cached(tmp_path, monkeypatch):
    from source.video.tracking import sleap_export as se
    monkeypatch.setattr(se, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(se, "_model_sig", lambda *a, **k: "sig")
    out = se.export_cache_dir("m", "tensorrt", "cuda")
    out.mkdir(parents=True)
    (out / "model.trt").write_bytes(b"engine")

    def _boom(*a, **k):
        raise AssertionError("must not export when a cached engine exists")
    monkeypatch.setattr(se.subprocess, "run", _boom)
    assert se.ensure_exported("m", "tensorrt", "cuda") == str(out)


def test_build_predictor_falls_back_to_native(monkeypatch):
    t = SLEAPTracker("m", runtime="tensorrt")
    monkeypatch.setattr(t, "_build_exported", lambda rt: None)   # engine unavailable
    sentinel = object()
    monkeypatch.setattr(t, "_build_native", lambda: sentinel)
    assert t._build_predictor() is sentinel


def test_build_predictor_uses_exported_when_available(monkeypatch):
    t = SLEAPTracker("m", runtime="auto")
    fake = object()
    monkeypatch.setattr(t, "_build_exported", lambda rt: fake)
    # native must NOT be consulted when an exported engine builds.
    monkeypatch.setattr(t, "_build_native",
                        lambda: (_ for _ in ()).throw(AssertionError("native")))
    assert t._build_predictor() is fake


def test_a_folder_without_its_own_engine_never_consults_the_cache(tmp_path, monkeypatch):
    """The training PC: a folder that ships only ONNX keeps using it, even when
    that PC's export cache holds an engine for the same model."""
    (tmp_path / "model.onnx").write_bytes(b"graph")
    t = SLEAPTracker(str(tmp_path), runtime="auto")
    onnx = object()
    monkeypatch.setattr(t, "_direct_runner",
                        lambda path, rt: onnx if rt == "onnx" else None)
    monkeypatch.setattr(t, "_cached_engine_runner",
                        lambda: (_ for _ in ()).throw(AssertionError("cache consulted")))
    monkeypatch.setattr(t, "_check_engine_batch", lambda d: None)
    assert t._build_exported("auto") is onnx


def test_a_foreign_engine_in_the_folder_yields_to_this_machines_cached_one(tmp_path, monkeypatch):
    """The 2026-09-15 Jetson run: the model folder's model.trt was built on the
    training PC, TensorRT on the Orin refused it, and the run settled for the
    folder's ONNX graph although an Orin engine could be in the cache."""
    (tmp_path / "model.trt").write_bytes(b"engine built on the PC")
    t = SLEAPTracker(str(tmp_path), runtime="auto")
    tried = []

    def folder_runner(path, runtime):
        tried.append(runtime)
        return None                       # foreign .trt; ONNX never asked
    cached = object()
    monkeypatch.setattr(t, "_direct_runner", folder_runner)
    monkeypatch.setattr(t, "_cached_engine_runner", lambda: cached)
    assert t._build_exported("auto") is cached
    assert tried == ["tensorrt"], "the folder's ONNX must not be chosen first"


def test_without_a_cached_engine_the_folder_onnx_is_still_used(monkeypatch):
    t = SLEAPTracker("m", runtime="auto")
    onnx = object()
    monkeypatch.setattr(t, "_direct_runner",
                        lambda path, rt: onnx if rt == "onnx" else None)
    monkeypatch.setattr(t, "_cached_engine_runner", lambda: None)
    monkeypatch.setattr(t, "_check_engine_batch", lambda d: None)
    assert t._build_exported("auto") is onnx


def test_the_cached_engine_is_looked_up_under_the_runs_own_key(tmp_path, monkeypatch):
    from source.video.tracking import sleap_export as se
    monkeypatch.setattr(se, "_cache_root", lambda: tmp_path)
    monkeypatch.setattr(se, "_model_sig", lambda *a, **k: "sig")
    monkeypatch.setattr(SLEAPTracker, "export_peak_threshold",
                        property(lambda self: se.PERMISSIVE_PEAK_THRESHOLD))
    t = SLEAPTracker("m", runtime="auto", fp16=True, max_batch_size=4,
                     device="cuda")
    assert t._cached_engine_runner() is None      # nothing cached yet

    out = se.export_cache_dir("m", "tensorrt", "cuda", precision="fp16",
                              max_batch_size=4)
    out.mkdir(parents=True)
    (out / "model.trt").write_bytes(b"engine")
    opened = []
    monkeypatch.setattr(t, "_direct_runner",
                        lambda path, rt: opened.append((path, rt)) or object())
    monkeypatch.setattr(t, "_check_engine_batch", lambda d: None)
    assert t._cached_engine_runner() is not None
    assert opened == [(str(out), "tensorrt")]
