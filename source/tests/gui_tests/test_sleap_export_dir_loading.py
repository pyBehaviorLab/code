"""An export directory is not a training directory.

sleap-nn has two entry points and they take different things:

  * ``Predictor.from_model_paths(...)`` loads a TRAINED model - a directory
    holding ``best.ckpt`` beside ``training_config.yaml``;
  * ``Predictor.from_export_dir(...)`` loads an EXPORT - a directory holding
    ``model.onnx`` / ``model.trt`` beside ``training_config.yaml``.

The rig called the first one unconditionally. Pointed at an export, sleap-nn
did as asked and went looking for a checkpoint that is not there:

    SLEAP init failed: [Errno 2] No such file or directory:
    'models/sleap/best.ckpt'

Two things funnelled the run into that call rather than into the perfectly
good engine sitting in the folder, and both are pinned here too:

  * an explicit ``tensorrt`` runtime walked past a decoded ``model.onnx`` in
    the same directory instead of using it;
  * ``graph_is_decoded`` answered a bare ``False`` when onnxruntime failed to
    IMPORT, so "I cannot tell" was recorded as "this graph is no good".

These use a stub predictor: the decision is the rig's, and it is the decision
that was wrong. Whether sleap-nn then accepts the folder is sleap-nn's own
contract, stated in its export guide.
"""
from __future__ import annotations

import sys
import types

import pytest

from source.video.tracking.pose import SLEAPTracker


@pytest.fixture
def fake_sleap_nn(monkeypatch):
    """A stand-in ``sleap_nn.inference.Predictor`` that records the call."""
    calls = []

    class Predictor:
        @staticmethod
        def from_export_dir(path, runtime=None, **kw):
            calls.append(("from_export_dir", str(path), runtime))
            return object()

        @staticmethod
        def from_model_paths(paths, **kw):
            calls.append(("from_model_paths", list(paths), None))
            # What the real one does with an export directory.
            raise FileNotFoundError(
                f"[Errno 2] No such file or directory: '{paths[0]}/best.ckpt'")

    inference = types.ModuleType("sleap_nn.inference")
    inference.Predictor = Predictor
    package = types.ModuleType("sleap_nn")
    package.inference = inference
    monkeypatch.setitem(sys.modules, "sleap_nn", package)
    monkeypatch.setitem(sys.modules, "sleap_nn.inference", inference)
    return calls


class _AbsentRuntime:
    """Refuses to import an engine driver this fake machine does not have."""

    def __init__(self, names):
        self._names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self._names:
            raise ImportError(f"No module named {fullname!r}")
        return None


@pytest.fixture
def runtimes(monkeypatch):
    """State which engine drivers this machine has, rather than inheriting it.

    Which artifact the rig picks now depends on which runtime can actually
    execute it, so a test that left that to the developer's environment would
    assert one thing on a CI box and another on a workstation, and did: these
    passed only while onnxruntime and tensorrt both happened to be absent.
    """
    def _install(*present):
        for name in ("tensorrt", "onnxruntime"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        for name in present:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setattr(
            sys, "meta_path",
            [_AbsentRuntime({"tensorrt", "onnxruntime"} - set(present)),
             *sys.meta_path])
    return _install


def _export_dir(tmp_path, *names):
    for name in names:
        (tmp_path / name).write_bytes(b"not really an engine")
    (tmp_path / "training_config.yaml").write_text(
        "data_config:\n  skeletons:\n  - nodes: [{name: a}]\n", encoding="utf-8")
    return str(tmp_path)


class TestItPicksTheRightEntryPoint:
    def test_an_onnx_export_loads_through_from_export_dir(self, tmp_path,
                                                          fake_sleap_nn,
                                                          runtimes):
        runtimes("onnxruntime")
        path = _export_dir(tmp_path, "model.onnx")
        tracker = SLEAPTracker(model_path=path)
        assert tracker._build_native() is not None
        assert fake_sleap_nn[0][0] == "from_export_dir"
        assert fake_sleap_nn[0][2] == "onnx"

    def test_a_tensorrt_export_says_tensorrt(self, tmp_path, fake_sleap_nn,
                                             runtimes):
        runtimes("tensorrt")
        path = _export_dir(tmp_path, "model.trt")
        SLEAPTracker(model_path=path)._build_native()
        assert fake_sleap_nn[0][2] == "tensorrt"

    def test_a_trt_export_falls_back_to_the_onnx_beside_it(
            self, tmp_path, fake_sleap_nn, runtimes):
        """Same weights, same coordinates, and a runtime that exists here.

        Preferring the .trt purely because the file is present failed the
        whole recording on a machine without TensorRT, while the .onnx in the
        very same folder would have run.
        """
        runtimes("onnxruntime")
        SLEAPTracker(model_path=_export_dir(
            tmp_path, "model.onnx", "model.trt"))._build_native()
        assert fake_sleap_nn[0][2] == "onnx"

    def test_an_export_with_no_usable_runtime_says_which_one_is_missing(
            self, tmp_path, fake_sleap_nn, runtimes, caplog):
        """The refusal has to name the package, not a file that never existed.

        Falling through to the checkpoint loader reports "No such file or
        directory: .../best.ckpt", a file the operator was never given, for a
        reason that has nothing to do with checkpoints.
        """
        runtimes()                                   # neither runtime present
        path = _export_dir(tmp_path, "model.trt")
        with caplog.at_level("ERROR"):
            assert SLEAPTracker(model_path=path)._build_native() is None
        assert not any(c[0] == "from_model_paths" for c in fake_sleap_nn)
        assert "tensorrt" in caplog.text
        assert "best.ckpt" not in caplog.text

    def test_the_checkpoint_loader_is_never_asked_for_an_export(
            self, tmp_path, fake_sleap_nn, runtimes):
        """An export directory must never reach the checkpoint loader."""
        runtimes("onnxruntime", "tensorrt")
        SLEAPTracker(model_path=_export_dir(
            tmp_path, "model.onnx", "model.trt"))._build_native()
        assert not any(c[0] == "from_model_paths" for c in fake_sleap_nn)

    def test_a_real_training_directory_still_uses_the_checkpoint_loader(
            self, tmp_path, fake_sleap_nn):
        """The fix must not send a trained model down the export path."""
        (tmp_path / "best.ckpt").write_bytes(b"weights")
        (tmp_path / "training_config.yaml").write_text("data_config: {}\n",
                                                       encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            SLEAPTracker(model_path=str(tmp_path))._build_native()
        assert fake_sleap_nn[0][0] == "from_model_paths"

    def test_a_failed_export_load_returns_none_rather_than_raising(
            self, tmp_path, monkeypatch):
        """A box that cannot load its model must not take the run down."""
        inference = types.ModuleType("sleap_nn.inference")

        class Predictor:
            @staticmethod
            def from_export_dir(path, runtime=None, **kw):
                raise RuntimeError("engine built for another GPU")
        inference.Predictor = Predictor
        package = types.ModuleType("sleap_nn")
        package.inference = inference
        monkeypatch.setitem(sys.modules, "sleap_nn", package)
        monkeypatch.setitem(sys.modules, "sleap_nn.inference", inference)
        path = _export_dir(tmp_path, "model.onnx")
        assert SLEAPTracker(model_path=path)._build_native() is None


class TestItDoesNotWalkPastAUsableEngine:
    def test_an_explicit_runtime_still_finds_the_other_export(self, tmp_path,
                                                              monkeypatch):
        """Asking for TensorRT on a host without it must not skip the ONNX
        graph lying beside it - same weights, same coordinates."""
        path = _export_dir(tmp_path, "model.onnx")
        tracker = SLEAPTracker(model_path=path, runtime="tensorrt")

        seen = []

        def fake_direct(export_dir, runtime):
            seen.append(runtime)
            if runtime == "auto":
                tracker._backend = "sleap_nn:onnx"
                return object()
            return None

        monkeypatch.setattr(tracker, "_direct_runner", fake_direct)
        assert tracker._build_exported("tensorrt") is not None
        assert seen == ["tensorrt", "auto"]

    def test_with_no_export_present_it_does_not_retry(self, tmp_path,
                                                      monkeypatch):
        """Nothing on disk to fall back to; the retry would be noise."""
        (tmp_path / "training_config.yaml").write_text("data_config: {}\n",
                                                       encoding="utf-8")
        tracker = SLEAPTracker(model_path=str(tmp_path), runtime="tensorrt")
        seen = []
        monkeypatch.setattr(tracker, "_direct_runner",
                            lambda d, r: seen.append(r))
        tracker._build_exported("tensorrt")
        assert seen.count("auto") == 0


class TestTheProbeSaysWhyItRefused:
    def test_a_missing_onnxruntime_is_reported_not_swallowed(self, monkeypatch,
                                                             caplog):
        """"Cannot tell" was being recorded as "no", with nothing logged - so
        a usable graph looked like a bad one and the run walked on past it."""
        import source.video.tracking.ort_runner as ort_runner

        real_import = __builtins__["__import__"] if isinstance(
            __builtins__, dict) else __builtins__.__import__

        def no_onnxruntime(name, *a, **kw):
            if name == "onnxruntime":
                raise ImportError("no onnxruntime here")
            return real_import(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", no_onnxruntime)
        with caplog.at_level("WARNING"):
            assert ort_runner.graph_is_decoded("anything.onnx") is False
        assert any("onnxruntime" in r.message or "onnxruntime" in r.getMessage()
                   for r in caplog.records)
