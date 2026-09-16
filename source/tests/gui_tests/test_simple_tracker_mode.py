"""The Simple tracking mode, the chip, its knobs, and their survival.

"Simple" is the blob tracker running ``bg_mode="self_norm"``: no captured
background, each frame divided by its own blur and auto-thresholded. It is a
mode chip rather than a separate tracker precisely so it rides the blob
config path that already round-trips to ``experiment_config.json``.

What these tests defend is the full loop, because every link in it has
silently dropped a value before:

    chip/spin box → get_settings → TrackingConfig → BlobConfig → JSON
    JSON → BlobConfig → TrackingConfig → settings dict → chip/spin box

plus the one-fact rule: the Simple chip and the BG dropdown encode the same
thing (``bg_mode``), so they must never be observed disagreeing, and keeping
them in step must not recurse.
"""
from __future__ import annotations

import pytest

from source.config.experiment import BlobConfig, TrackingConfig
from source.gui.widgets.tracking_panel import (
    TrackingSettingsPanel,
    UnifiedTrackingDialog,
)
from source.tests.qt_dispose import WidgetBin
from source.video.framebus.types import TrackingConfig as BusTrackingConfig

_BIN = WidgetBin()

#: A value per knob, none of them a default, so a dropped field shows up as a
#: mismatch rather than coincidentally matching zero.
TUNED = {
    "self_norm_ratio": 0.873,
    "self_norm_sigma": 24.0,
    "self_norm_smooth_sigma": 10.0,
    "self_norm_minsize": 7,
}


@pytest.fixture(autouse=True)
def _drain():
    yield
    _BIN.drain()


def _panel(box_ids=(1,)):
    p = _BIN.add(TrackingSettingsPanel(box_ids=list(box_ids)))
    tab = p.build_event_trigger_widget()
    tab.setParent(p)
    return p


# --- the chip and the dropdown are one fact ---------------------------------

def test_the_simple_chip_exists_beside_the_others(mock_qapplication):
    p = _panel()
    assert p.mode_simple.text() == "Simple"
    assert p._mode_group.buttons().count(p.mode_simple) == 1


def test_picking_simple_selects_self_norm(mock_qapplication):
    p = _panel()
    p.mode_simple.setChecked(True)
    assert p.bg_mode_combo.currentText() == "self_norm"
    assert p.get_settings()["bg_mode"] == "self_norm"


def test_picking_self_norm_by_hand_selects_the_simple_chip(mock_qapplication):
    """The dropdown still offers self_norm; choosing it there is choosing
    Simple, or the two controls would sit there contradicting each other."""
    p = _panel()
    p.bg_mode_combo.setCurrentText("self_norm")
    assert p.mode_simple.isChecked()
    assert not p.mode_blob.isChecked()


def test_leaving_simple_restores_a_background_mode(mock_qapplication):
    p = _panel()
    p.mode_simple.setChecked(True)
    p.mode_blob.setChecked(True)
    assert p.bg_mode_combo.currentText() == "static", (
        "Blob was left claiming self_norm, so the mode chip and the "
        "background dropdown disagree")


def test_the_two_way_sync_terminates(mock_qapplication):
    """Chip drives dropdown drives chip. Without the reentrancy guard this
    is an infinite recursion, not a wrong value."""
    p = _panel()
    for _ in range(20):
        p.mode_simple.setChecked(True)
        p.mode_blob.setChecked(True)
    assert p.bg_mode_combo.currentText() == "static"


def test_simple_retires_the_background_controls(mock_qapplication):
    """Simple needs no reference frame, so offering Retake BG would invite a
    step that does nothing."""
    p = _panel()
    p.mode_simple.setChecked(True)
    assert not p.retake_bg_btn.isEnabled()
    assert not p.bg_mode_combo.isEnabled()
    # Simple thresholds a brightness ratio, so the fixed threshold does
    # nothing; area bounds still do.
    assert not p.threshold_spin.isEnabled()
    assert p.min_area_spin.isEnabled() and p.max_area_spin.isEnabled()
    assert p.simple_opts_widget.isVisibleTo(p)

    p.mode_blob.setChecked(True)
    assert p.retake_bg_btn.isEnabled()
    assert p.bg_mode_combo.isEnabled()
    assert p.threshold_spin.isEnabled()
    assert not p.simple_opts_widget.isVisibleTo(p)


# --- the knobs default to inert ---------------------------------------------

def test_the_knobs_default_to_frame_scaled_values(mock_qapplication):
    """Simple ships tuned the way the method's author tuned it, so it works
    out of the box on a textured floor rather than only after someone finds
    these three fields, but the two widths ship as Auto rather than as
    mousefinder's literal 10, because they are absolute pixel counts and a
    fixed 10 erodes the animal away on a small view."""
    p = _panel()
    s = p.get_settings()
    assert s["self_norm_sigma"] == 0.0          # 0 = derive from frame height
    assert s["self_norm_smooth_sigma"] == -1.0  # -1 = scale to frame height
    assert s["self_norm_minsize"] == -1
    assert "self_norm_ratio" not in s, (
        "an uncalibrated rig must keep the tracker's own auto ratio rather "
        "than have a value invented for it")


def test_auto_widths_resolve_to_mousefinders_value_at_its_own_scale():
    """Auto is not a new tuning; it is mousefinder's number expressed as a
    fraction of the view, so a mousefinder-sized frame still gets 10."""
    from source.video.tracking.blob import BlobTracker

    t = BlobTracker(1)
    assert round(t._auto_scaled(-1, 580)) == 10     # the view 10 came from
    assert round(t._auto_scaled(-1, 1080)) == 19    # bigger view, wider blur
    assert round(t._auto_scaled(-1, 290)) == 5      # the ROI that measured best
    # An operator's own number is never rescaled, and 0 still means off.
    assert t._auto_scaled(7, 1080) == 7
    assert t._auto_scaled(0, 1080) == 0


def test_the_defaults_agree_everywhere():
    """Four layers carry these numbers: the detector, the project config, the
    pipeline config and the widgets. blob.py is the owner, the other three
    hold copies because they cannot import it without dragging cv2 into the
    config layer, so nothing but this test stops them drifting apart."""
    from source.gui.widgets import tracking_panel as tp
    from source.video.tracking import blob as bl

    for owner, cfg_name, tc_name, widget in (
            (bl.SELF_NORM_SIGMA, "self_norm_sigma",
             "blob_self_norm_sigma", tp._SN_SIGMA),
            (bl.SELF_NORM_SMOOTH_SIGMA, "self_norm_smooth_sigma",
             "blob_self_norm_smooth_sigma", tp._SN_SMOOTH),
            (bl.SELF_NORM_MINSIZE, "self_norm_minsize",
             "blob_self_norm_minsize", tp._SN_MINSIZE)):
        assert getattr(BlobConfig(), cfg_name) == owner, cfg_name
        assert getattr(BusTrackingConfig(setup_id=1), tc_name) == owner, tc_name
        assert widget == owner, cfg_name
        assert getattr(bl.BlobTracker(1), f"_{cfg_name}") == owner, cfg_name


def test_a_deliberately_zeroed_knob_stays_zero():
    """The defaults are non-zero now, so ``d.get(k, 10) or 10`` would read a
    stored 0 as "absent" and switch the knob back on behind the operator."""
    cfg = BlobConfig.from_dict({"bg_mode": "self_norm",
                                "self_norm_smooth_sigma": 0.0,
                                "self_norm_minsize": 0})
    assert cfg.self_norm_smooth_sigma == 0.0
    assert cfg.self_norm_minsize == 0

    tc = BusTrackingConfig.from_json({"setup_id": 1,
                                      "blob_self_norm_smooth_sigma": 0.0,
                                      "blob_self_norm_minsize": 0})
    assert tc.blob_self_norm_smooth_sigma == 0.0
    assert tc.blob_self_norm_minsize == 0


# --- the loop: panel → config → disk → panel --------------------------------

def test_the_panel_emits_every_knob(mock_qapplication):
    p = _panel()
    p.mode_simple.setChecked(True)
    p.sn_sigma_spin.setValue(TUNED["self_norm_sigma"])
    p.sn_smooth_spin.setValue(TUNED["self_norm_smooth_sigma"])
    p.sn_minsize_spin.setValue(TUNED["self_norm_minsize"])
    p._blob_calibration_extra["self_norm_ratio"] = TUNED["self_norm_ratio"]

    s = p.get_settings()
    for key, want in TUNED.items():
        assert s[key] == want, f"{key} never left the panel"


def test_a_saved_config_comes_back(mock_qapplication):
    """The whole point. Tune, save, reopen, same numbers, same chip."""
    dlg = _BIN.add(UnifiedTrackingDialog(box_ids=[1]))
    dlg._apply_tracking_settings({"mode": "normal", "bg_mode": "self_norm",
                                  **TUNED})
    sp = dlg.settings_panel

    assert sp.mode_simple.isChecked(), (
        "a saved self_norm project reopened on the Blob chip, so the "
        "operator sees a mode they did not choose")
    assert sp.sn_sigma_spin.value() == TUNED["self_norm_sigma"]
    assert sp.sn_smooth_spin.value() == TUNED["self_norm_smooth_sigma"]
    assert sp.sn_minsize_spin.value() == TUNED["self_norm_minsize"]

    s = sp.get_settings()
    for key, want in TUNED.items():
        assert s[key] == want, f"{key} was lost across save → load"


def test_the_ratio_badge_reports_the_calibrated_value(mock_qapplication):
    dlg = _BIN.add(UnifiedTrackingDialog(box_ids=[1]))
    sp = dlg.settings_panel
    assert "auto" in sp.sn_ratio_label.text().lower()
    dlg._apply_tracking_settings({"mode": "normal", "bg_mode": "self_norm",
                                  **TUNED})
    assert "0.873" in sp.sn_ratio_label.text()


def test_the_knobs_cross_the_framebus_config(mock_qapplication):
    """``TrackingConfig`` is what the Pipeline actually reads; a field added
    to the dataclass but missed in to_json/from_json reaches the sinks once
    and never again."""
    tc = BusTrackingConfig(
        setup_id=1,
        blob_bg_mode="self_norm",
        blob_self_norm_ratio=TUNED["self_norm_ratio"],
        blob_self_norm_sigma=TUNED["self_norm_sigma"],
        blob_self_norm_smooth_sigma=TUNED["self_norm_smooth_sigma"],
        blob_self_norm_minsize=TUNED["self_norm_minsize"],
    )
    back = BusTrackingConfig.from_json(tc.to_json())
    assert back.blob_bg_mode == "self_norm"
    assert back.blob_self_norm_ratio == TUNED["self_norm_ratio"]
    assert back.blob_self_norm_sigma == TUNED["self_norm_sigma"]
    assert back.blob_self_norm_smooth_sigma == TUNED["self_norm_smooth_sigma"]
    assert back.blob_self_norm_minsize == TUNED["self_norm_minsize"]


def test_the_knobs_cross_the_project_config():
    """``experiment_config.json`` is the file on disk, no Qt involved."""
    cfg = TrackingConfig(mode="blob", blob=BlobConfig(
        bg_mode="self_norm",
        self_norm_ratio=TUNED["self_norm_ratio"],
        self_norm_sigma=TUNED["self_norm_sigma"],
        self_norm_smooth_sigma=TUNED["self_norm_smooth_sigma"],
        self_norm_minsize=TUNED["self_norm_minsize"],
    ))
    back = TrackingConfig.from_dict(cfg.to_compact())
    assert back.blob.bg_mode == "self_norm"
    assert back.blob.self_norm_ratio == TUNED["self_norm_ratio"]
    assert back.blob.self_norm_sigma == TUNED["self_norm_sigma"]
    assert back.blob.self_norm_smooth_sigma == TUNED["self_norm_smooth_sigma"]
    assert back.blob.self_norm_minsize == TUNED["self_norm_minsize"]


def test_the_gui_dict_and_the_config_agree_on_names():
    """``_BLOB_KEYS`` drives both directions of the GUI↔Config conversion, so
    a name present in one list and absent from the other is a silent drop."""
    from source.config import experiment as exp

    for key in TUNED:
        assert key in exp._BLOB_KEYS, (
            f"{key} is not in _BLOB_KEYS, so it never reaches BlobConfig")
        assert hasattr(BlobConfig(), key)
