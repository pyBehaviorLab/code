"""The scale control must say what it actually does on the active backend.

It previously did not. ``SLEAPTracker`` accepted ``resize_factor``, handed it to
the base class, and no SLEAP code path ever read it again, so the dialog's
Resize spinner was a labelled, defaulted, silently discarded control on one of
the two backends. Input size is the single largest speed lever there is
(sleap-nn measures 1.8 ms at 192x192 against 11.4 ms at 1024x1024, in PyTorch
alone), which is exactly why a control that appears to move it must not lie.
"""
import logging

import pytest

from source.tests.qt_dispose import WidgetBin
from source.video.tracking.pose import SLEAPTracker


class TestSleapIgnoresResize:
    """SLEAP takes its scale from the model config, and an exported engine
    fixes the input size at export. A host-side factor has nothing to act on,
    so the run must say so rather than imply a downscale that never happens."""

    def _init_with_fake_predictor(self, monkeypatch, tracker):
        monkeypatch.setattr(tracker, "_build_predictor", lambda: object())
        monkeypatch.setattr(tracker, "_extract_body_parts", lambda: ["a"])
        monkeypatch.setattr(tracker, "_warmup", lambda frame: None)
        return tracker.initialize(None)

    def test_non_default_resize_is_reported(self, monkeypatch, caplog):
        t = SLEAPTracker("m", resize_factor=0.5)
        with caplog.at_level(logging.WARNING):
            assert self._init_with_fake_predictor(monkeypatch, t) is True
        assert any("resize" in r.message.lower() and "ignored" in r.message.lower()
                   for r in caplog.records), \
            "a discarded resize must be announced, not silently dropped"

    def test_default_resize_is_not_noise(self, monkeypatch, caplog):
        """1.0 means the operator asked for nothing; warning would be noise."""
        t = SLEAPTracker("m", resize_factor=1.0)
        with caplog.at_level(logging.WARNING):
            self._init_with_fake_predictor(monkeypatch, t)
        assert not any("resize" in r.message.lower() for r in caplog.records)


def test_model_info_is_on_the_base_class():
    """Both backends have a config file, so both can describe themselves,
    without an inference stack installed."""
    t = SLEAPTracker("does/not/exist")
    assert t.model_info.ok is False
    assert t.model_info is t.model_info, "read once, cached"


# ── the dialog control ───────────────────────────────────────────────────

_BIN = WidgetBin()


@pytest.fixture(autouse=True)
def _drain():
    yield
    _BIN.drain()


@pytest.fixture
def qapp_or_skip():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(qapp_or_skip):
    from source.gui.widgets.tracking_panel import TrackingSettingsPanel
    return _BIN.add(TrackingSettingsPanel())


def test_control_is_an_operator_lever_under_dlc(panel, qapp_or_skip):
    panel.mode_dlc.setChecked(True)
    qapp_or_skip.processEvents()
    assert panel.dlc_resize_spin.isEnabled() is True
    assert panel.dlc_resize_label.text() == "Resize:"


def test_control_is_model_owned_under_sleap(panel, qapp_or_skip):
    """Removed and restated: the operator sees the value the model uses and
    is not offered a box to type one that would be thrown away.

    This began as "disabled and retitled". A greyed control still says the
    setting exists on this backend and is merely unavailable, and the rule for
    this panel is that every option shown is one the selected model honours.
    So the spin goes and the number is stated as text; the explanation has to
    follow it onto whatever is still visible.
    """
    panel.mode_sleap.setChecked(True)
    qapp_or_skip.processEvents()
    assert panel.dlc_resize_spin.isEnabled() is False
    assert panel.dlc_resize_spin.isHidden(), \
        "a scale SLEAP discards must not be offered as a control at all"
    assert "model" in panel.dlc_resize_label.text().lower()
    assert "export" in panel.dlc_resize_label.toolTip().lower(), \
        "the tooltip must say how to actually change it, on what is visible"


def test_switching_backends_restores_the_lever(panel, qapp_or_skip):
    panel.mode_sleap.setChecked(True)
    qapp_or_skip.processEvents()
    panel.mode_dlc.setChecked(True)
    qapp_or_skip.processEvents()
    assert panel.dlc_resize_spin.isEnabled() is True
    assert panel.dlc_resize_label.text() == "Resize:"
