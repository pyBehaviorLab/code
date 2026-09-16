"""The dialog offers what the machine can run, and says why when it cannot.

Offering TensorRT on a CPU host is not a cosmetic problem: the operator picks
it, Init falls back, and the session runs on something slower than they think.
Greyed with a reason rather than hidden, because a control that has vanished
reads as a missing feature.
"""
import pytest

from source.tests.qt_dispose import WidgetBin
from source.video.tracking.capability import Capabilities

_BIN = WidgetBin()

_CPU_ONLY = Capabilities(
    cpu_cores=8, onnxruntime=True, ort_providers=("CPUExecutionProvider",),
    reasons={"tensorrt": "TensorRT needs a CUDA GPU",
             "fp16": "PyTorch reports no CUDA device",
             "cuda": "PyTorch reports no CUDA device"})

_GPU = Capabilities(
    cpu_cores=20, cuda=True, gpu_name="RTX 2000 Ada", gpu_memory_gb=16.0,
    tensorrt=True, tensorrt_version="10.16.1", onnxruntime=True,
    ort_providers=("CUDAExecutionProvider", "CPUExecutionProvider"))


@pytest.fixture(autouse=True)
def _drain():
    yield
    _BIN.drain()


@pytest.fixture
def qapp():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(qapp):
    from source.gui.widgets.tracking_panel import TrackingSettingsPanel
    p = _BIN.add(TrackingSettingsPanel(box_ids=[1]))
    p.build_event_trigger_widget().setParent(p)
    return p


def _runtimes(panel):
    model = panel.sleap_runtime_combo.model()
    return {panel.sleap_runtime_combo.itemText(i): model.item(i).isEnabled()
            for i in range(panel.sleap_runtime_combo.count())}


class TestOnACpuHost:
    def test_tensorrt_is_offered_but_not_selectable(self, panel, qapp):
        panel.apply_machine_capabilities(_CPU_ONLY)
        qapp.processEvents()
        assert _runtimes(panel)["tensorrt"] is False

    def test_onnx_stays_available_because_it_is_the_cpu_path(self, panel, qapp):
        panel.apply_machine_capabilities(_CPU_ONLY)
        qapp.processEvents()
        assert _runtimes(panel)["onnx"] is True

    def test_half_precision_is_refused_with_a_reason(self, panel, qapp):
        """fp16 on a CPU is slower, not faster."""
        panel.apply_machine_capabilities(_CPU_ONLY)
        qapp.processEvents()
        assert panel.sleap_fp16_cb.isEnabled() is False
        assert "CUDA" in panel.sleap_fp16_cb.toolTip()

    def test_the_refusal_reaches_the_item_tooltip(self, panel, qapp):
        from PySide6 import QtCore
        panel.apply_machine_capabilities(_CPU_ONLY)
        qapp.processEvents()
        idx = panel.sleap_runtime_combo.findText("tensorrt")
        tip = panel.sleap_runtime_combo.itemData(idx, QtCore.Qt.ToolTipRole)
        assert "CUDA GPU" in (tip or "")


class TestOnAGpuHost:
    def test_everything_is_selectable(self, panel, qapp):
        panel.apply_machine_capabilities(_GPU)
        qapp.processEvents()
        assert all(_runtimes(panel).values())
        assert panel.sleap_fp16_cb.isEnabled()

    def test_the_machine_is_stated_not_guessed(self, panel, qapp):
        panel.apply_machine_capabilities(_GPU)
        qapp.processEvents()
        text = panel.machine_caps_label.text()
        assert "RTX 2000 Ada" in text and "TensorRT 10.16.1" in text
        assert "20 cores" in text


def test_the_machine_line_is_never_saved(panel, qapp):
    """It belongs to this PC. A project opened on the Jetson must not carry
    the workstation's answer."""
    panel.apply_machine_capabilities(_GPU)
    qapp.processEvents()
    settings = panel.get_settings()
    assert not [k for k in settings if "capab" in k.lower() or "gpu" in k.lower()]


def test_a_panel_built_on_any_machine_still_builds(panel):
    """The probe runs at build time; a machine with nothing installed must not
    stop the dialog from opening."""
    assert panel.machine_caps_label.text()
