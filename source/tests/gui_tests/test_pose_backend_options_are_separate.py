"""A backend's panel shows that backend's settings, and no others.

``crop-track`` is OUR following window, written for SLEAP single-instance, and
``_DLC_INPUT_MODES`` never offers it. The controls that steer it, Follow conf,
min kp and re-acquire, were nevertheless on the panel under DLC: greyed, but
present, captioning a setting DeepLabCut never reads.

Also covered here: the status a box reports after init. ``pose_box_is_ready``
compares the fingerprint the sink stored against the one the host builds, and
the sink fingerprinted the body parts it was ASKED for while
``_reconcile_parts`` then replaced them with the model's own, because "the
MODEL is authoritative". A SLEAP model reports its six keypoint names, the
dialog adopts them, and every later check compared the model's list with the
dialog's old one and answered "not loaded" for ever, while inference ran
perfectly. That is why it read as "SLEAP configured" where DLC read "ready".
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from source.gui.widgets.tracking_panel import TrackingSettingsPanel
from source.video.framebus.pose_sink import pose_fingerprint


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(qapp):
    p = TrackingSettingsPanel()
    try:
        yield p
    finally:
        p.hide()
        p.deleteLater()
        qapp.processEvents()
        qapp.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _select(panel, backend):
    getattr(panel, f"mode_{backend}").setChecked(True)
    panel._on_mode_changed()


def _modes(panel):
    c = panel.pose_input_mode_combo
    return [c.itemText(i) for i in range(c.count())]


# ── the input modes each backend can honour ──────────────────────────────

def test_dlc_is_not_offered_our_sleap_window(panel):
    _select(panel, "dlc")
    assert "crop-track" not in _modes(panel)


def test_sleap_is_offered_our_window(panel):
    _select(panel, "sleap")
    assert "crop-track" in _modes(panel)


def test_the_combo_and_the_visibility_rule_read_one_list(panel):
    """They used to derive the backend's modes separately, which is how the
    panel offered one set and steered by another."""
    for backend in ("dlc", "sleap"):
        _select(panel, backend)
        assert _modes(panel) == panel._input_modes_for_current_backend()


# ── the controls that steer that window ──────────────────────────────────

def test_the_follow_settings_are_absent_under_dlc(panel):
    """Hidden, not greyed: a disabled control still says the setting exists
    for this backend, and it does not."""
    _select(panel, "dlc")
    assert panel.pose_crop_steer_widget.isHidden()


def test_the_follow_settings_are_present_under_sleap(panel):
    _select(panel, "sleap")
    assert not panel.pose_crop_steer_widget.isHidden()


def test_switching_backends_both_ways_keeps_them_in_step(panel):
    """A panel is switched back and forth while a rig is set up."""
    for expected, backend in ((True, "dlc"), (False, "sleap"),
                              (True, "dlc"), (False, "sleap")):
        _select(panel, backend)
        assert panel.pose_crop_steer_widget.isHidden() is expected, backend


def test_the_window_size_is_editable_only_where_a_window_is_used(panel):
    """I first assumed the size fields described the model's input and kept
    them live under letterbox. They do not: the whole row describes the
    FOLLOWING window, and ``test_pose_input_contract`` already fixed that
    contract. Under SLEAP they are live on ``auto`` (which resolves to
    crop-track) and dead on letterbox; under DLC there is no such window."""
    _select(panel, "sleap")
    panel.pose_input_mode_combo.setCurrentText("auto")
    assert panel.pose_input_w_spin.isEnabled()
    panel.pose_input_mode_combo.setCurrentText("letterbox")
    assert not panel.pose_input_w_spin.isEnabled()
    _select(panel, "dlc")
    assert not panel.pose_input_w_spin.isEnabled()


def test_the_whole_following_row_is_gone_under_dlc(panel):
    """Not just its steering: DeepLabCut has no following window at all."""
    _select(panel, "dlc")
    assert panel.pose_crop_widget.isHidden()
    _select(panel, "sleap")
    assert not panel.pose_crop_widget.isHidden()


def test_the_follow_settings_are_dead_under_dlc_even_on_auto(panel):
    """``auto`` resolves to crop-track only where crop-track exists."""
    _select(panel, "dlc")
    panel.pose_input_mode_combo.setCurrentText("auto")
    assert not panel.pose_crop_conf_spin.isEnabled()
    assert not panel.pose_crop_good_spin.isEnabled()
    assert not panel.pose_crop_reacquire_cb.isEnabled()


# ── the status a box reports ─────────────────────────────────────────────

def _fp(parts):
    return pose_fingerprint(
        tracker_type="sleap", model_path="models/x", resize_factor=1.0,
        confidence=0.6, body_parts=parts, sleap_opts={}, dlc_opts=None,
        colour_mode="auto", input_mode="auto", input_wh=(0, 0), crop_opts={})


def test_the_model_s_own_keypoint_names_change_the_fingerprint():
    """The premise: if the sink fingerprints one list and the host another,
    the two can never agree that the loaded model is the wanted one."""
    asked = ("center",)
    reported = ("Snout", "Head", "Left_Ear", "Right_Ear", "center", "Tail_Base")
    assert _fp(asked) != _fp(reported)


def test_a_model_that_renames_its_parts_still_reads_as_ready(monkeypatch):
    """After init the host holds the model's names. The sink must have
    fingerprinted those same names, or readiness is permanently false and the
    box reports 'configured, model not loaded yet' while it is tracking."""
    from source.video.framebus import pose_sink as ps

    reported = ["Snout", "Head", "Left_Ear", "Right_Ear", "center", "Tail_Base"]
    asked = ["center"]
    reconciled = ps.PoseSink._reconcile_parts(asked, reported)
    assert list(reconciled) == reported, (
        "the model is authoritative for the names it labels output with")
    # The fingerprint the sink keeps must describe the SAME list the host
    # will hold once the dialog has read the model.
    assert _fp(tuple(reconciled)) == _fp(tuple(reported))


# ── the whole panel, not just the controls I happened to think of ────────
#
# The first pass fixed the input-mode list and the follow settings because
# those were the two reported. That is not the same as checking. These walk
# every widget on the panel in each backend, so a control added later is
# covered without anyone remembering to add a test for it.

def _shown(panel):
    """Widgets actually on screen. ``isVisible`` and not ``isHidden``: a child
    of a hidden container keeps its own flag clear and would count as shown."""
    return [w for w in panel.findChildren(QtWidgets.QWidget) if w.isVisible()]


@pytest.fixture
def shown_panel(qapp):
    p = TrackingSettingsPanel()
    p.resize(980, 900)
    p.show()
    qapp.processEvents()
    try:
        yield p
    finally:
        p.hide()
        p.deleteLater()
        qapp.processEvents()
        qapp.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _select_shown(panel, backend, qapp):
    getattr(panel, f"mode_{backend}").setChecked(True)
    panel._on_mode_changed()
    for _ in range(4):
        qapp.processEvents()


@pytest.mark.parametrize("backend,foreign", [("dlc", "sleap"), ("sleap", "dlc")])
def test_no_visible_control_names_the_other_backend(shown_panel, qapp,
                                                    backend, foreign):
    """The operator reads labels, not attribute names. A control may be
    shared and still be named ``dlc_*`` in code; what must never appear is
    the other backend's NAME while this one is selected."""
    _select_shown(shown_panel, backend, qapp)
    offenders = []
    for w in _shown(shown_panel):
        if isinstance(w, QtWidgets.QPushButton) and w.text().upper() in ("DLC", "SLEAP"):
            continue                      # the backend picker names both
        # placeholderText included: the Model field's hint named a "DLC
        # exported-model folder" while SLEAP was selected, and a first pass
        # that read only text/currentText walked straight past it.
        for getter in ("text", "currentText", "placeholderText"):
            f = getattr(w, getter, None)
            if not callable(f):
                continue
            t = f()
            if isinstance(t, str) and foreign in t.lower():
                offenders.append(f"{getter}={t!r}")
    assert not offenders, f"{backend} shows {foreign} wording: {offenders}"


def test_the_model_hint_describes_the_selected_backend(shown_panel, qapp):
    """One field, two model layouts; the hint has to say which one to pick."""
    _select_shown(shown_panel, "dlc", qapp)
    assert "dlc" in shown_panel.model_path.placeholderText().lower()
    _select_shown(shown_panel, "sleap", qapp)
    assert "sleap" in shown_panel.model_path.placeholderText().lower()


def test_no_model_means_no_type_badge(shown_panel, qapp):
    """It defaulted to the literal ", ", so an empty field drew a stray comma
    beside "Type:" and read as broken rather than as not-yet-chosen."""
    _select_shown(shown_panel, "sleap", qapp)
    shown_panel.model_path.setText("")
    shown_panel._refresh_sleap_type_badge()
    qapp.processEvents()
    assert not shown_panel.sleap_type_label.isVisible()
    assert not shown_panel.sleap_type_caption.isVisible()
    assert "," not in shown_panel.sleap_type_label.text()


@pytest.mark.parametrize("backend", ["dlc", "sleap"])
def test_nothing_is_shown_greyed_out(shown_panel, qapp, backend):
    """Every option on the panel must be one the selected model honours.

    A disabled control still says the setting exists for this backend, so a
    value the model cannot accept is removed, not greyed. Resize was the last
    one: SLEAP takes its scale from the model config and discards a host-side
    factor, and the spin sat there greyed as if it were merely unavailable.
    """
    _select_shown(shown_panel, backend, qapp)
    dead = []
    for w in _shown(shown_panel):
        if not isinstance(w, (QtWidgets.QComboBox, QtWidgets.QCheckBox,
                              QtWidgets.QAbstractSpinBox, QtWidgets.QLineEdit)):
            continue                      # labels and layout carry no setting
        if not w.isEnabled():
            dead.append(f"{type(w).__name__}({w.objectName() or w.toolTip()[:40]!r})")
    assert not dead, f"{backend} shows disabled controls: {dead}"


def test_resize_is_a_lever_under_dlc_and_a_statement_under_sleap(shown_panel,
                                                                 qapp):
    """DLCLive honours ``resize``; SLEAP reads its scale from the model. The
    number is still reported under SLEAP, as information rather than as a
    control that discards what you type into it."""
    _select_shown(shown_panel, "dlc", qapp)
    assert shown_panel.dlc_resize_spin.isVisible()
    assert shown_panel.dlc_resize_spin.isEnabled()
    assert shown_panel.dlc_resize_label.text() == "Resize:"

    _select_shown(shown_panel, "sleap", qapp)
    assert not shown_panel.dlc_resize_spin.isVisible(), (
        "a scale SLEAP discards must not be offered as a control")
    assert "scale" in shown_panel.dlc_resize_label.text().lower(), (
        shown_panel.dlc_resize_label.text())


def test_each_backend_shows_only_its_own_options_widget(shown_panel, qapp):
    """The per-backend blocks: runtime/fp16/export are SLEAP's, engine and
    precision are DLC's."""
    _select_shown(shown_panel, "dlc", qapp)
    assert shown_panel.dlc_opts_widget.isVisible()
    assert not shown_panel.sleap_opts_widget.isVisible()

    _select_shown(shown_panel, "sleap", qapp)
    assert shown_panel.sleap_opts_widget.isVisible()
    assert not shown_panel.dlc_opts_widget.isVisible()


def test_the_type_separator_goes_with_the_type_it_separates(shown_panel, qapp):
    """A divider with nothing on its left is not a divider, it is a stray
    mark at the edge of the row."""
    _select_shown(shown_panel, "sleap", qapp)
    shown_panel.model_path.setText("")
    shown_panel._refresh_sleap_type_badge()
    qapp.processEvents()
    assert not shown_panel.sleap_type_sep.isVisible()
