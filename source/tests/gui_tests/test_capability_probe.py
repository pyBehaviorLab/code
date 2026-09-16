"""The probe decides what the dialog may offer, so it has to be honest.

Two failure modes it exists to prevent: offering TensorRT on a machine that
cannot run it, and, the quiet one, believing ONNX Runtime is on the GPU when
its CUDA provider failed to load and it silently ran on the CPU.
"""
import logging
import sys
import types

import pytest

from source.video.tracking import capability as cap


@pytest.fixture(autouse=True)
def _no_cache():
    cap._cached = None
    yield
    cap._cached = None


_RUNTIMES = ("torch", "tensorrt", "onnxruntime")


class _Absent:
    """An import hook that refuses the runtimes this fake machine lacks.

    Dropping a name from ``sys.modules`` only forces a re-import; it does not
    make an installed package go away. So "nothing installed" quietly meant
    "nothing installed that this machine happens to lack", and the day
    onnxruntime was installed here the probe correctly reported ONNX as
    available and the test that asserted otherwise began to fail.
    """

    def __init__(self, names):
        self._names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self._names:
            raise ImportError(f"No module named {fullname!r}")
        return None


def _fake(monkeypatch, *, torch_cuda=None, trt_version=None, ort_providers=None):
    """Install exactly the modules a machine would have, and no others."""
    for name in _RUNTIMES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    present = set()
    if torch_cuda is not None:
        t = types.ModuleType("torch")
        t.cuda = types.SimpleNamespace(
            is_available=lambda: torch_cuda,
            get_device_properties=lambda i: types.SimpleNamespace(
                name="Fake GPU", total_memory=8 * 1024 ** 3))
        monkeypatch.setitem(sys.modules, "torch", t)
        present.add("torch")
    if trt_version is not None:
        m = types.ModuleType("tensorrt")
        m.__version__ = trt_version
        monkeypatch.setitem(sys.modules, "tensorrt", m)
        present.add("tensorrt")
    if ort_providers is not None:
        o = types.ModuleType("onnxruntime")
        o.get_available_providers = lambda: list(ort_providers)
        monkeypatch.setitem(sys.modules, "onnxruntime", o)
        present.add("onnxruntime")
    monkeypatch.setattr(
        sys, "meta_path",
        [_Absent(set(_RUNTIMES) - present), *sys.meta_path])


class TestWhatIsOffered:
    def test_a_cpu_machine_cannot_use_tensorrt_or_fp16(self, monkeypatch):
        _fake(monkeypatch, torch_cuda=False, ort_providers=["CPUExecutionProvider"])
        c = cap.probe()
        assert c.can("tensorrt") is False
        assert c.can("fp16") is False
        assert c.can("onnx") is True

    def test_a_gpu_machine_with_tensorrt_can_use_it(self, monkeypatch):
        _fake(monkeypatch, torch_cuda=True, trt_version="10.16.1",
              ort_providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        c = cap.probe()
        assert c.can("tensorrt") and c.can("fp16") and c.can("cuda")
        assert c.gpu_name == "Fake GPU"
        assert c.gpu_memory_gb == pytest.approx(8.0, abs=0.01)

    def test_tensorrt_11_is_refused_because_sleap_nn_cannot_export_to_it(self, monkeypatch):
        """11 removed NetworkDefinitionCreationFlag.EXPLICIT_BATCH; the export
        would fail with an error that never names the cause."""
        _fake(monkeypatch, torch_cuda=True, trt_version="11.0.0")
        c = cap.probe()
        assert c.can("tensorrt") is False
        assert "11" in c.why_not("tensorrt") and ">=10" in c.why_not("tensorrt")

    def test_a_gpu_without_the_tensorrt_package(self, monkeypatch):
        _fake(monkeypatch, torch_cuda=True)
        c = cap.probe()
        assert c.can("tensorrt") is False
        assert "not installed" in c.why_not("tensorrt")


class TestEveryRefusalExplainsItself:
    """A greyed control saying why teaches; one that vanished does not."""

    @pytest.mark.parametrize("what", ["tensorrt", "cuda", "fp16", "onnx"])
    def test_an_unavailable_capability_has_a_reason(self, monkeypatch, what):
        _fake(monkeypatch, torch_cuda=False)          # nothing installed
        c = cap.probe()
        assert c.can(what) is False
        assert c.why_not(what), f"{what} refused with no reason"

    def test_an_available_capability_has_no_reason_to_give(self, monkeypatch):
        _fake(monkeypatch, torch_cuda=True, trt_version="10.16.1")
        assert cap.probe().why_not("tensorrt") == ""


class TestTheSilentFallback:
    def test_ort_gpu_is_false_when_only_cpu_provider_loaded(self, monkeypatch):
        """Installed is not usable: onnxruntime-gpu built against the wrong
        CUDA loads happily and runs on the CPU."""
        _fake(monkeypatch, torch_cuda=True, ort_providers=["CPUExecutionProvider"])
        assert cap.probe().ort_gpu is False

    def test_assert_provider_warns_when_the_gpu_was_asked_for(self, caplog):
        session = types.SimpleNamespace(
            get_providers=lambda: ["CPUExecutionProvider"])
        with caplog.at_level(logging.WARNING,
                             logger="source.video.tracking.capability"):
            active = cap.assert_provider(session, want_gpu=True)
        assert active == "CPUExecutionProvider"
        assert any("fell back" in r.message for r in caplog.records)

    def test_assert_provider_is_quiet_when_it_got_what_it_asked_for(self, caplog):
        session = types.SimpleNamespace(
            get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        with caplog.at_level(logging.WARNING,
                             logger="source.video.tracking.capability"):
            assert cap.assert_provider(session, want_gpu=True) == "CUDAExecutionProvider"
        assert not caplog.records

    def test_a_session_that_cannot_answer_is_not_an_error(self):
        broken = types.SimpleNamespace(
            get_providers=lambda: (_ for _ in ()).throw(RuntimeError("nope")))
        assert cap.assert_provider(broken, want_gpu=True) == ""


class TestThreadBudget:
    def test_the_rig_keeps_cores_for_everything_else(self, monkeypatch):
        c = cap.Capabilities(cpu_cores=20)
        assert cap.inference_threads(c, reserved=4) == 16

    def test_a_small_host_never_drops_below_two(self, monkeypatch):
        """One thread measured 55 ms/frame, slower than the budget it would
        be protecting."""
        assert cap.inference_threads(cap.Capabilities(cpu_cores=2)) == 2
        assert cap.inference_threads(cap.Capabilities(cpu_cores=1)) == 2

    def test_it_is_never_the_default_of_every_core(self, monkeypatch):
        c = cap.Capabilities(cpu_cores=20)
        assert cap.inference_threads(c) < c.cpu_cores


def test_the_probe_is_cached_because_hardware_does_not_change(monkeypatch):
    calls = []
    monkeypatch.setattr(cap, "probe",
                        lambda: calls.append(1) or cap.Capabilities(cpu_cores=8))
    cap.capabilities()
    cap.capabilities()
    assert len(calls) == 1
    cap.capabilities(refresh=True)
    assert len(calls) == 2


def test_probing_a_bare_machine_never_raises(monkeypatch):
    """It runs on the GUI thread when a dialog opens."""
    for name in ("torch", "tensorrt", "onnxruntime"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(cap, "_probe_torch",
                        lambda r: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        cap.probe()          # documents that _probe_* raising IS a real bug
