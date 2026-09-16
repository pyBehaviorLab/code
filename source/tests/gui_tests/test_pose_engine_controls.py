"""The engine options must be reachable, persistent, and honest about cost.

Precision, engine, device and colour mode are all in the model cache key, so
each one genuinely rebuilds the model, unlike confidence, which is a
post-inference cutoff and applies live. The dialog has to reflect that
difference, or it teaches operators to re-initialise for nothing.
"""
import pytest

from source.tests.qt_dispose import WidgetBin

_BIN = WidgetBin()


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
    """A fully built panel, the dialog always builds the trigger tab, so a
    bare panel is not a configuration the app ever runs."""
    from source.gui.widgets.tracking_panel import TrackingSettingsPanel
    p = _BIN.add(TrackingSettingsPanel(box_ids=[1]))
    p.build_event_trigger_widget().setParent(p)
    return p


def _shown(w):
    """``isVisible()`` is False while no ancestor is shown, so ask the widget
    about its own hidden flag instead."""
    return not w.isHidden()


class TestVisibility:
    """One backend runs at a time, so each row belongs to its own backend."""

    def test_dlc_row_for_dlc_only(self, panel, qapp):
        panel.mode_dlc.setChecked(True)
        qapp.processEvents()
        assert _shown(panel.dlc_opts_widget)
        assert not _shown(panel.sleap_opts_widget)

    def test_sleap_row_for_sleap_only(self, panel, qapp):
        panel.mode_sleap.setChecked(True)
        qapp.processEvents()
        assert not _shown(panel.dlc_opts_widget)
        assert _shown(panel.sleap_opts_widget)

    def test_colour_applies_to_either_pose_backend(self, panel, qapp):
        for chip in (panel.mode_dlc, panel.mode_sleap):
            chip.setChecked(True)
            qapp.processEvents()
            assert _shown(panel.pose_colour_widget)

    def test_nothing_pose_related_shows_for_blob(self, panel, qapp):
        panel.mode_simple.setChecked(True)
        qapp.processEvents()
        assert not _shown(panel.dlc_opts_widget)
        assert not _shown(panel.pose_colour_widget)


class TestRebuildClassification:
    """Everything in the model cache key must say it needs a re-init."""

    @pytest.mark.parametrize("attr,value", [
        # ``pytorch`` rather than ``tensorrt``: the engine combo is now built
        # from what the selected model actually offers, and a name no model
        # offers cannot be selected at all, setCurrentText on a fixed combo
        # with no such item is a silent no-op, so the old parametrisation
        # asserted nothing.
        ("dlc_model_type_combo", "pytorch"),
        ("dlc_precision_combo", "FP16"),
        ("pose_colour_combo", "grayscale"),
    ])
    def test_key_settings_mark_dirty(self, panel, qapp, attr, value):
        panel.mode_dlc.setChecked(True)
        qapp.processEvents()
        if attr == "dlc_model_type_combo":
            panel.dlc_model_type_combo.addItem(value)
        panel._dlc_initialized = True
        getattr(panel, attr).setCurrentText(value)
        qapp.processEvents()
        assert panel._dlc_initialized is False, f"{attr} is in the cache key"

    def test_confidence_does_not(self, panel, qapp):
        """The counter-case: a live setting must not demand a rebuild."""
        panel.mode_dlc.setChecked(True)
        qapp.processEvents()
        panel._dlc_initialized = True
        panel.dlc_confidence_spin.setValue(0.71)
        qapp.processEvents()
        assert panel._dlc_initialized is True


class TestRoundTrip:
    def test_settings_survive_save_and_restore(self, panel, qapp):
        from source.gui.widgets.tracking_panel import TrackingSettingsPanel, UnifiedTrackingDialog
        panel.mode_dlc.setChecked(True)
        qapp.processEvents()
        panel.dlc_model_type_combo.addItem("pytorch")
        panel.dlc_model_type_combo.setCurrentText("pytorch")
        panel.dlc_precision_combo.setCurrentText("FP16")
        panel.dlc_device_combo.setCurrentText("cuda:1")
        panel.pose_colour_combo.setCurrentText("grayscale")
        qapp.processEvents()
        saved = panel.get_settings()
        assert saved["dlc_model_type"] == "pytorch"
        assert saved["dlc_precision"] == "FP16"
        assert saved["dlc_device"] == "cuda:1"
        assert saved["pose_colour_mode"] == "grayscale"

        dst = _BIN.add(TrackingSettingsPanel(box_ids=[1]))
        dst.build_event_trigger_widget().setParent(dst)
        UnifiedTrackingDialog._apply_sleap_params(UnifiedTrackingDialog, dst, saved)
        qapp.processEvents()
        # The destination panel has no model selected, so its engine list holds
        # only "auto". The saved value must survive that anyway, a setting
        # that saves and does not restore is worse than one never offered.
        assert dst.dlc_model_type_combo.currentText() == "pytorch"
        assert dst.dlc_precision_combo.currentText() == "FP16"
        assert dst.dlc_device_combo.currentText() == "cuda:1"
        assert dst.pose_colour_combo.currentText() == "grayscale"


class TestSetComboText:
    """An editable combo's list is suggestions; a fixed combo's list is the
    permitted values. The restore has to tell them apart."""

    def test_editable_combo_accepts_a_value_outside_its_list(self, qapp):
        from PySide6 import QtWidgets

        from source.gui.widgets.tracking_panel import UnifiedTrackingDialog
        c = _BIN.add(QtWidgets.QComboBox())
        c.setEditable(True)
        c.addItems(["auto", "cuda", "cpu"])
        UnifiedTrackingDialog._set_combo_text(c, "cuda:1")
        assert c.currentText() == "cuda:1", \
            "a multi-GPU box would otherwise reset to auto on every load"

    def test_fixed_combo_still_refuses_a_stale_value(self, qapp):
        from PySide6 import QtWidgets

        from source.gui.widgets.tracking_panel import UnifiedTrackingDialog
        c = _BIN.add(QtWidgets.QComboBox())
        c.addItems(["FP32", "FP16"])
        UnifiedTrackingDialog._set_combo_text(c, "FP4-nonsense")
        assert c.currentText() == "FP32", "unknown option must not be forced in"
