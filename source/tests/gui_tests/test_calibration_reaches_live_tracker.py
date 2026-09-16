"""Calibrating the blob tracker has to change what the rig actually runs.

Three things broke the loop between "the preview tracks perfectly" and "the
GUI tracks nothing":

* The calibration dialog tunes ``use_clahe``, ``clahe_clip_limit`` and
  ``use_adaptive_threshold``. The settings panel has no widget for any of
  them, so Apply wrote back the six params it did have widgets for and
  dropped these on the floor. The live tracker then ran on the old values,
  the two settings that made the preview work never left the dialog.

* ``self.backgrounds`` starts empty on every open, and every read went
  through it. A background captured minutes earlier, and still sitting at
  ``<project>/background_images/box{N}.png``: looked missing, so the panel
  asked for a re-take and calibration opened with no reference frame.

* "Copy zones to all boxes" sat in the dialog's bottom button row with
  Save/Load All Config, a tab away from the zone editor, and was hidden
  outright with a single box. It read as removed.
"""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets

from source.gui.widgets.tracking_panel import TrackingSettingsPanel
from source.tests.qt_dispose import WidgetBin

_BIN = WidgetBin()


@pytest.fixture(autouse=True)
def _drain():
    yield
    _BIN.drain()


def _panel(box_ids=(1, 2)):
    """A panel with its Event & Trigger tab built.

    The mode radios drive the coord-mapping rows, which only exist once that
    tab has been built, the dialog always builds it, so a bare panel is not
    a configuration the app ever runs.
    """
    p = _BIN.add(TrackingSettingsPanel(box_ids=list(box_ids)))
    tab = p.build_event_trigger_widget()
    tab.setParent(p)
    return p


# --- calibration params must survive Apply ----------------------------------

def test_every_param_the_dialog_tunes_has_somewhere_to_land():
    """The guard that stops this regressing.

    Whatever ``TrackerCalibrationDialog._current_params`` returns must either
    map to a panel widget or be listed in ``_CALIBRATION_ONLY_KEYS``. A new
    slider added to the dialog and forgotten here is exactly how CLAHE and
    the adaptive threshold came to be discarded.
    """
    import ast
    import inspect
    import textwrap

    from source.gui.dialogs.tracking import TrackerCalibrationDialog

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(TrackerCalibrationDialog._current_params)))
    keys = {k.value for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    assert keys, "could not read the dialog's parameter keys"
    # The self_norm settings are copied in by a loop, not a dict literal.
    keys |= set(TrackerCalibrationDialog._SELF_NORM_PASSTHROUGH)

    widget_backed = {
        "detect_dark", "threshold", "min_area", "max_area",
        "blur_mode", "blur_kernel_size", "bg_mode",
        "open_kernel_size", "close_kernel_size",
        "self_norm_sigma", "self_norm_smooth_sigma", "self_norm_minsize",
    }
    homeless = keys - widget_backed - set(
        TrackingSettingsPanel._CALIBRATION_ONLY_KEYS)
    assert not homeless, (
        f"{sorted(homeless)} are tuned in the calibration dialog but have "
        f"nowhere to go, Apply will discard them and the live tracker will "
        f"run on the old values")


def test_calibration_only_params_reach_the_settings(mock_qapplication):
    p = _panel()
    p.mode_blob.setChecked(True)
    p._blob_calibration_extra.update(
        {"use_clahe": True, "clahe_clip_limit": 4.5,
         "use_adaptive_threshold": True})
    s = p.get_settings()
    assert s["use_clahe"] is True
    assert s["clahe_clip_limit"] == 4.5
    assert s["use_adaptive_threshold"] is True


def test_they_are_absent_until_calibration_sets_them(mock_qapplication):
    """Never invent a value, an untouched rig must keep the tracker's own
    defaults rather than have False forced on it."""
    p = _panel()
    p.mode_blob.setChecked(True)
    s = p.get_settings()
    for key in TrackingSettingsPanel._CALIBRATION_ONLY_KEYS:
        assert key not in s


def test_they_are_not_emitted_in_pose_mode(mock_qapplication):
    p = _panel()
    p._blob_calibration_extra["use_clahe"] = True
    p.mode_dlc.setChecked(True)
    assert "use_clahe" not in p.get_settings()


def test_a_saved_config_restores_them(mock_qapplication):
    """They have no widget, so the normal restore path would skip them and
    the rig would silently lose them on the next project load."""
    from source.gui.widgets.tracking_panel import UnifiedTrackingDialog

    dlg = _BIN.add(UnifiedTrackingDialog(box_ids=[1]))
    dlg._apply_tracking_settings(
        {"mode": "normal", "use_clahe": True, "clahe_clip_limit": 3.5,
         "use_adaptive_threshold": True})
    s = dlg.settings_panel.get_settings()
    assert s["use_clahe"] is True
    assert s["clahe_clip_limit"] == 3.5
    assert s["use_adaptive_threshold"] is True


# --- a captured background must be found again ------------------------------

def test_a_background_on_disk_is_found_without_recapturing(
        mock_qapplication, tmp_path):
    p = _panel(box_ids=(1,))
    png = tmp_path / "box1.png"
    png.write_bytes(b"not really a png, only the path matters")

    # Stand in for the main window's canonical lookup.
    p.window()._background_path_for_box = (
        lambda bid: png if int(bid) == 1 else None)

    assert p.backgrounds == {}, "precondition: nothing captured this session"
    assert p._background_for_box(1) == str(png), (
        "a background sitting on disk was reported missing, so the operator "
        "is asked to capture it again")


def test_the_disk_background_is_adopted_once_found(mock_qapplication, tmp_path):
    p = _panel(box_ids=(1,))
    png = tmp_path / "box1.png"
    png.write_bytes(b"x")
    p.window()._background_path_for_box = lambda bid: png
    p._background_for_box(1)
    assert p.backgrounds.get(1) == str(png), (
        "not adopted, so the status line still reports no background")


def test_a_session_capture_wins_over_disk(mock_qapplication, tmp_path):
    """Just-captured is newer than whatever is on disk."""
    p = _panel(box_ids=(1,))
    p.backgrounds[1] = "captured/this/session.png"
    p.window()._background_path_for_box = lambda bid: tmp_path / "old.png"
    assert p._background_for_box(1) == "captured/this/session.png"


def test_no_background_anywhere_reports_none(mock_qapplication):
    p = _panel(box_ids=(1,))
    p.window()._background_path_for_box = lambda bid: None
    assert p._background_for_box(1) is None


def test_a_host_without_the_lookup_does_not_raise(mock_qapplication):
    """The panel is also built standalone in tests and tooling."""
    p = _panel(box_ids=(1,))
    assert p._background_for_box(1) is None


# --- copy zones is where the zones are --------------------------------------

def test_copy_zones_sits_with_the_zone_buttons(mock_qapplication):
    from source.gui.widgets.tracking_panel import UnifiedTrackingDialog

    dlg = _BIN.add(UnifiedTrackingDialog(box_ids=[1, 2]))
    btn = dlg.copy_zones_btn
    siblings = [b.text() for b in btn.parent().findChildren(QtWidgets.QPushButton)]
    assert "Import Box Zones" in siblings and "Export Box Zones" in siblings, (
        f"copy-zones is not beside the zone buttons; it is among {siblings}")


def test_copy_zones_is_disabled_not_hidden_with_one_box(mock_qapplication):
    """Hiding it is why it was reported missing."""
    from source.gui.widgets.tracking_panel import UnifiedTrackingDialog

    dlg = _BIN.add(UnifiedTrackingDialog(box_ids=[1]))
    assert not dlg.copy_zones_btn.isHidden()
    assert not dlg.copy_zones_btn.isEnabled()
    assert "one box" in dlg.copy_zones_btn.toolTip().lower()


def test_copy_zones_is_usable_with_several_boxes(mock_qapplication):
    from source.gui.widgets.tracking_panel import UnifiedTrackingDialog

    dlg = _BIN.add(UnifiedTrackingDialog(box_ids=[1, 2, 3]))
    assert dlg.copy_zones_btn.isEnabled()
