"""Each row sets ITS OWN camera's resolution, rate, format and orientation.

Resolution and rate were read-only echoes of a single shared card that showed
whichever row happened to be selected. On a rig with one camera per box that
left no way to give each camera its own mode: you set the card, picked another
row, and set the same widgets again for a different camera, with nothing on
screen saying what the others were on.

The values themselves always had a home, ``cameras.registry[].preference``, and
``update_camera_config`` has always been keyed per camera. What was missing was
a control per camera, which is what these cover.
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from source.gui.dialogs.camera_connect import CameraConnectDialog as _Dlg


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ── the mode lists the table offers ──────────────────────────────────────
#
# Driven through the REAL populator, the one the shared CCTV card uses, with
# the row's two combos passed in. The per-row columns used to have their own
# copy of this logic, and every way the copy drifted from the card became a
# defect on screen: a ladder built for the previous resolution, a saved size
# that never re-selected so the column looked locked, a different default
# format. Asserting against a private duplicate could not have caught any of
# them, because the duplicate was the bug.

def _row(modes, wh=None):
    """Fill a row's Resolution/FPS pair exactly as ``_fill_row_capture`` does.

    Returns ``(res_combo, fps_combo)``.
    """
    d = _Dlg.__new__(_Dlg)          # the populators touch only the combos
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    d._populateResolutionCombo(modes, preselect_wh=wh, res_combo=res,
                               fps_combo=fps, compact=True)
    return res, fps


def _ladder(modes, wh):
    _res, fps = _row(modes, wh)
    return [fps.itemData(i) for i in range(fps.count())
            if fps.itemData(i) is not None]


def test_a_rate_is_a_choice_up_to_the_measured_ceiling(qapp):
    """The bug: the FPS cell offered exactly one value, so it looked broken.

    A rate is a CHOICE. A camera reaching 21 fps at 1280x720 can also run at
    5, 10, 15 or 20, and an experiment often wants a lower, steadier rate.
    """
    rates = _ladder([(1280, 720, 21.8, "dshow")], (1280, 720))
    assert len(rates) > 1, f"only one rate offered: {rates}"
    assert rates == sorted(rates), "not ascending"
    assert max(rates) <= 25, "offered more than the camera reached"


def test_the_ladder_is_capped_by_that_resolution_s_own_ceiling(qapp):
    """Each resolution has its own ceiling, so each has its own ladder."""
    modes = [(1920, 1080, 9.4, "dshow"), (640, 480, 29.5, "dshow")]
    slow = _ladder(modes, (1920, 1080))
    fast = _ladder(modes, (640, 480))
    assert max(slow) < max(fast)
    assert max(slow) <= 10, slow


def test_the_ceiling_rounds_to_the_nearest_step(qapp):
    """29.2 measured is a 30 fps camera, the same rule the card used."""
    rates = _ladder([(640, 480, 29.2, "")], (640, 480))
    assert max(rates) == 30, rates


def test_the_backend_travels_with_the_chosen_mode(qapp):
    """Which door serves a mode is part of the same choice, and it rides in
    the resolution's item data, which is what the row writes to the config."""
    res, _fps = _row([(800, 600, 30.0, "dshow")], (800, 600))
    assert res.currentData()["backend"] == "dshow"


def test_nothing_offered_when_nothing_was_detected(qapp):
    """No modes at all is the only case with no rate to offer.

    This used to assert that a mode measured at 0.0 also offered nothing.
    That was wrong, and it was the defect: a 0.0 mode is one whose backend
    never probed, and it is listed precisely so it can be FORCED. Offering
    no rate made choosing it store ``selected_fps=None``.
    """
    assert _ladder([], (640, 480)) == []
    assert _ladder([(640, 480, 0.0, "msmf")], (640, 480)), (
        "an unprobed door must offer rates to try it at")


def test_a_camera_slower_than_the_smallest_step_still_offers_its_rate(qapp):
    """An empty list would read as 'not detected' for a camera that works."""
    assert _ladder([(1920, 1080, 3.0, "dshow")], (1920, 1080)) == [3]


def test_modes_of_the_wrong_shape_are_skipped_not_fatal(qapp):
    """Machine caches from older versions are on disk and must not crash the
    dialog when it opens."""
    modes = [(640, 480), None, ("x", "y", "z", "w"), (640, 480, 30.0, "dshow")]
    assert max(_ladder(modes, (640, 480))) == 30


def test_the_default_is_not_the_slowest_rung(qapp):
    """The ladder ascends, so falling back to its first entry would quietly
    select the slowest rate the camera offers."""
    modes = [(1280, 720, 21.8, "dshow")]
    _res, fps = _row(modes, (1280, 720))
    assert fps.currentData() > min(_ladder(modes, (1280, 720))), (
        f"defaulted to the slowest rung: {fps.currentData()}")


def test_the_default_is_offered_in_the_ladder(qapp):
    modes = [(640, 480, 29.5, "dshow")]
    _res, fps = _row(modes, (640, 480))
    assert fps.currentData() in _ladder(modes, (640, 480))


def test_the_fastest_door_is_the_default_but_not_the_only_one(qapp):
    """The compact list opens on the fastest route to a size.

    It used to keep ONLY that route, one entry per size. That is what made
    Media Foundation unreachable from a per-box column, and MF is the door
    that reaches 30 fps at 1080p on this rig where DirectShow reaches 5.
    Both are listed now; the fast one is merely selected first.
    """
    modes = [(640, 480, 5.0, "msmf"), (640, 480, 30.0, "dshow")]
    res, _fps = _row(modes, (640, 480))
    assert res.currentData()["backend"] == "dshow", "the fastest door opens"
    assert res.count() == 2, [res.itemText(i) for i in range(res.count())]
    assert _doors(res) == {"dshow", "msmf"}, "and the slower one is still there"


def test_the_row_names_the_ceiling_beside_each_size(qapp):
    """Which sizes are usable is decided by the rate they reach, so the rate
    has to be visible while choosing the size, not only after choosing it.
    The shared card always showed it; the row showed the size alone."""
    modes = [(1920, 1080, 9.4, "dshow"), (640, 480, 29.9, "dshow")]
    res, _fps = _row(modes, (640, 480))
    labels = [res.itemText(i) for i in range(res.count())]
    # Whole numbers, because that is what the FPS cell beside this one
    # offers: 9.4 gives a ladder of [9] and 29.9 one ending at 30, so a
    # decimal here would show a rate the rate cell does not have. Two numbers
    # for one fact is the "(24.6) beside a list of 20 and 30" defect.
    assert any("9" in t and "1920" in t for t in labels), labels
    assert any("30" in t and "640" in t for t in labels), labels


def test_a_whole_number_ceiling_is_not_written_as_a_decimal(qapp):
    """"30" is a frame rate; "30.0" reads as a measurement, and the cell is
    too narrow to spend three characters saying the same thing."""
    res, _fps = _row([(800, 600, 30.0, "dshow")], (800, 600))
    assert "(30)" in res.itemText(0), res.itemText(0)


def test_a_size_nothing_measured_claims_no_rate(qapp):
    """Printing "(0)" would state a rate nobody observed.

    The label says "(try)" instead, which is an invitation rather than a
    measurement. What must never appear is a number.
    """
    res, _fps = _row([(1280, 720, 0.0, "dshow")], (1280, 720))
    label = res.itemText(0)
    assert label.startswith("1280×720"), label
    assert not any(c.isdigit() for c in label.split("720", 1)[1]), label


def test_a_saved_size_re_selects_itself(qapp):
    """The column read as locked because the saved size never matched, so the
    combo fell back to item 0 and the ladder was built for another mode."""
    modes = [(1920, 1080, 9.0, "dshow"), (640, 480, 30.0, "dshow"),
             (800, 600, 25.0, "dshow")]
    for want in ((640, 480), (800, 600), (1920, 1080)):
        res, _fps = _row(modes, want)
        assert _Dlg._row_size(res) == want, res.currentText()


# ── one camera, many boxes ───────────────────────────────────────────────

class _Table:
    """Just enough QTableWidget for the row-sharing lookup."""

    def __init__(self, ids):
        self._ids = ids

    def rowCount(self):
        return len(self._ids)


class _Host:
    _rows_sharing_camera = _Dlg._rows_sharing_camera

    def __init__(self, ids):
        self.table = _Table(ids)
        self._ids = ids

    def _camera_id_text(self, row):
        return self._ids[row]


def test_boxes_sharing_one_camera_are_found_together():
    """A CCTV rig is four boxes on ONE camera: a mode set on any of those
    rows is the same camera's mode, so every one of them has to show it."""
    host = _Host(["camA", "camA", "camA", "camA"])
    assert host._rows_sharing_camera("camA") == [0, 1, 2, 3]


def test_one_camera_per_box_keeps_the_rows_independent():
    """The case that had no working control at all: each row is a different
    camera and must be settable on its own."""
    host = _Host(["camA", "camB", "camC", "camD"])
    assert host._rows_sharing_camera("camB") == [1]
    assert host._rows_sharing_camera("camD") == [3]


def test_rows_without_a_camera_are_not_swept_in():
    host = _Host(["camA", "", None, "camA"])
    assert host._rows_sharing_camera("camA") == [0, 3]
    assert host._rows_sharing_camera("") == []
    assert host._rows_sharing_camera(None) == []


# ── the columns exist and are laid out in a readable order ───────────────

def test_every_capture_setting_has_its_own_column():
    """The complaint was that these were one merged read-only cell."""
    cols = {_Dlg.COL_RES, _Dlg.COL_FPS, _Dlg.COL_FORMAT, _Dlg.COL_FLIP}
    assert len(cols) == 4, "two settings share a column"
    assert not hasattr(_Dlg, "COL_MODE"), (
        "the merged 'Res / rate' echo column should be gone")


def test_the_columns_do_not_collide_with_the_existing_ones():
    named = [_Dlg.COL_BOX, _Dlg.COL_CAMERA_ID, _Dlg.COL_BACKEND,
             _Dlg.COL_SAVE, _Dlg.COL_RES, _Dlg.COL_FPS, _Dlg.COL_FORMAT,
             _Dlg.COL_FLIP, _Dlg.COL_DETECT, _Dlg.COL_READY]
    assert len(set(named)) == len(named), f"two columns share an index: {named}"
    assert sorted(named) == list(range(len(named))), "column indices have a gap"


def test_detect_still_sits_beside_the_values_it_fills():
    """It was already per row and must stay next to what it measures."""
    assert _Dlg.COL_DETECT > _Dlg.COL_RES
    assert _Dlg.COL_DETECT > _Dlg.COL_FPS


@pytest.fixture
def row_dialog(qapp):
    """The real dialog with ONE populated row, shown so the cells have real
    geometry. Anything measuring what a cell renders needs both."""
    import contextlib
    from types import SimpleNamespace

    from source.gui.dialogs.camera_connect import CameraConnectDialog
    from source.video.framebus.controller import Pipeline

    pipe = Pipeline()
    cam = "cam3cbb52d3"
    pipe.install_camera_configs({cam: {"camera_id": cam}})   # no format set
    box = SimpleNamespace(setup_number=1, save_video_enabled=True,
                          _camera_backend="OpenCV",
                          camera_id_edit=SimpleNamespace(text=lambda: cam))

    class _MW(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.pipeline = pipe
            self.video_target_fps = 30
            self._active_config = None

        def get_all_setup_widgets(self):
            return [box]

        def _project_changed(self, **_k):
            pass

    mw = _MW()
    dlg = CameraConnectDialog(mw)
    dlg.resize(1200, 460)
    dlg.show()
    qapp.processEvents()
    try:
        yield dlg
    finally:
        dlg.close()
        dlg.deleteLater()
        mw.deleteLater()
        qapp.processEvents()
        qapp.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        with contextlib.suppress(Exception):
            pipe.shutdown()


def test_every_column_is_wide_enough_for_what_it_shows(row_dialog):
    """Narrowing the columns is a display change, and a combo does not shrink
    its text, it ELIDES it: Format at 90 px kept the value "MJPEG" and painted
    "M". The cell then read as empty while the config was correct, which is
    unfalsifiable from the config side. Widths may be tuned freely; they may
    not be tuned below what the widest real value needs to render.
    """
    dialog = row_dialog
    # Only meaningful where the real font is available. The offscreen platform
    # the suite runs under substitutes one that measures ~1.7x wider, so this
    # would fail on widths that render perfectly on the rig, and "fix" itself
    # by padding columns nobody needs padded. The real check is in
    # tools/check_cell_widths.py, which runs on the actual display.
    if QtWidgets.QApplication.platformName() == "offscreen":
        pytest.skip("text metrics are unreliable without the real font")
    widest = {dialog.COL_CAMERA_ID: "cam3cbb52d3",     # "cam" + djb2, always 11
              dialog.COL_BACKEND: "OpenCV",
              dialog.COL_RES: "1920×1080",
              dialog.COL_FPS: "30",
              dialog.COL_FORMAT: "MJPEG"}
    too_narrow = []
    for col, sample in widest.items():
        # The REAL cell widget: an unparented probe does not inherit the
        # dialog's font, and measuring the wrong font is how a width check
        # passes while the screen still truncates.
        w = dialog.table.cellWidget(0, col)
        assert w is not None, f"column {col} has no cell widget to measure"
        w.addItem(sample)
        w.setCurrentIndex(w.count() - 1)
        # Ask the STYLE where the text goes, rather than assuming padding.
        opt = QtWidgets.QStyleOptionComboBox()
        w.initStyleOption(opt)
        room = w.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_ComboBox, opt,
            QtWidgets.QStyle.SubControl.SC_ComboBoxEditField, w).width()
        adv = w.fontMetrics().horizontalAdvance(sample)
        if adv > room:
            hdr = dialog.table.horizontalHeaderItem(col)
            too_narrow.append(
                f"{(hdr.text() if hdr else col)!r} at "
                f"{dialog.table.columnWidth(col)}px draws {room}px of the "
                f"{adv}px {sample!r}")
    assert not too_narrow, "; ".join(too_narrow)


# ── the rate must belong to the resolution it is stored with ─────────────

def test_the_rate_offered_follows_the_resolution(qapp):
    """Caught on the rig: picking 1280x720 stored 9 fps, the rate the camera
    reached at 1920x1080, because the rate list still held the previous
    resolution's options when the write happened. A mode is a (resolution,
    rate) pair; storing a rate the camera cannot reach at that resolution
    makes the recorder drop frames to a number nothing measured."""
    modes = [(1920, 1080, 9.4, "dshow"),
             (1280, 720, 21.8, "dshow"),
             (640, 480, 29.5, "dshow")]
    at_720 = _ladder(modes, (1280, 720))
    at_1080 = _ladder(modes, (1920, 1080))
    assert max(at_720) > max(at_1080), "720p must offer more than 1080p here"
    assert max(at_1080) != max(at_720), "the two must not be the same"


def test_switching_resolution_rebuilds_the_ladder_before_it_is_read(qapp):
    """The write path reads the FPS combo, so the ladder has to belong to the
    resolution just chosen, not the one before it."""
    modes = [(1920, 1080, 9.4, "dshow"), (640, 480, 29.5, "dshow")]
    d = _Dlg.__new__(_Dlg)
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    d._populateResolutionCombo(modes, preselect_wh=(1920, 1080),
                               res_combo=res, fps_combo=fps, compact=True)
    assert max(fps.itemData(i) for i in range(fps.count())) <= 10
    for i in range(res.count()):          # move to 640x480 as the GUI would
        if _Dlg._row_size(res) != (640, 480):
            res.setCurrentIndex(i)
    d._repopulate_fps_combo(res_combo=res, fps_combo=fps)
    assert _Dlg._row_size(res) == (640, 480)
    assert max(fps.itemData(i) for i in range(fps.count())) == 30


def test_the_ceiling_comes_from_the_best_backend(qapp):
    """Two backends serve one resolution; the ladder must reach the higher
    ceiling and name the backend that achieved it."""
    modes = [(640, 480, 15.0, "msmf"), (640, 480, 30.0, "dshow")]
    res, _fps = _row(modes, (640, 480))
    assert max(_ladder(modes, (640, 480))) == 30
    assert res.currentData()["backend"] == "dshow"


# ── CCTV: one camera, so the settings appear once, and must appear ───────

@pytest.fixture
def dialog(qapp):
    """The real dialog, with no boxes; enough to check its chrome."""
    from source.gui.dialogs.camera_connect import CameraConnectDialog

    class _MW(QtWidgets.QWidget):
        pipeline = None
        video_target_fps = 30
        _active_config = None

        def get_all_setup_widgets(self):
            return []

        def _project_changed(self, **_k):
            pass

    mw = _MW()
    dlg = CameraConnectDialog(mw)
    try:
        yield dlg
    finally:
        dlg.deleteLater()
        mw.deleteLater()
        qapp.processEvents()
        qapp.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


def _in_shared_row(dlg, widget):
    parent = widget.parent() if widget is not None else None
    while parent is not None:
        if parent is dlg._shared_capture_row:
            return True
        parent = parent.parent()
    return False


def test_cctv_hides_the_per_camera_columns(dialog):
    """One shared camera means these describe the camera, not the box, so
    four copies are four chances to disagree."""
    dialog._set_shared_camera_row_visible(True)
    for name, col in (("Camera", dialog.COL_CAMERA_ID),
                      ("Backend", dialog.COL_BACKEND),
                      ("Resolution", dialog.COL_RES), ("FPS", dialog.COL_FPS),
                      ("Format", dialog.COL_FORMAT), ("Flip", dialog.COL_FLIP),
                      ("Detect", dialog.COL_DETECT)):
        assert dialog.table.isColumnHidden(col), f"{name} still per row in CCTV"


def test_cctv_keeps_the_columns_that_really_are_per_box(dialog):
    dialog._set_shared_camera_row_visible(True)
    for name, col in (("Box", dialog.COL_BOX), ("Record", dialog.COL_SAVE),
                      ("Ready", dialog.COL_READY)):
        assert not dialog.table.isColumnHidden(col), f"{name} wrongly hidden"


def test_individual_mode_shows_every_per_camera_column(dialog):
    dialog._set_shared_camera_row_visible(False)
    for col in (dialog.COL_CAMERA_ID, dialog.COL_RES, dialog.COL_FPS,
                dialog.COL_FORMAT, dialog.COL_FLIP, dialog.COL_DETECT):
        assert not dialog.table.isColumnHidden(col)


def test_every_hidden_setting_is_reachable_in_the_shared_row(dialog):
    """The one that was actually broken.

    ``_page_options`` builds these controls and adds them to its own form, so
    moving them into the CCTV row before that call had them taken straight
    back: the row showed a label and nothing else, and because CCTV hides the
    per-row columns, resolution, rate, format, orientation and re-detect were
    unreachable in that mode.
    """
    for attr, _line, _stretch in dialog._SHARED_CAPTURE_WIDGETS:
        w = getattr(dialog, attr, None)
        assert w is not None, f"{attr} was never built"
        assert _in_shared_row(dialog, w), (
            f"{attr} is not in the CCTV row, so it cannot be reached at all "
            f"when the per-row columns are hidden")


def test_the_shared_row_is_not_just_a_label(dialog):
    """A cheap guard on the same failure, stated as a count."""
    real = []
    for line in dialog._shared_capture_lines:
        for i in range(line.count()):
            w = line.itemAt(i).widget()
            if w is not None and not isinstance(w, QtWidgets.QLabel):
                real.append(w)
    assert len(real) >= len(dialog._SHARED_CAPTURE_WIDGETS), (
        f"the CCTV row holds only {len(real)} control(s)")


def test_the_shared_controls_are_split_over_two_lines(dialog):
    """Six controls on one line leave each combo a few characters wide, and
    the resolution strings are the widest text in the dialog."""
    lines = dialog._shared_capture_lines
    assert len(lines) == 2, f"expected two lines, got {len(lines)}"
    counts = []
    for line in lines:
        n = sum(1 for i in range(line.count())
                if line.itemAt(i).widget() is not None
                and not isinstance(line.itemAt(i).widget(), QtWidgets.QLabel))
        counts.append(n)
    assert all(c > 0 for c in counts), f"a line is empty: {counts}"
    assert sum(counts) >= len(dialog._SHARED_CAPTURE_WIDGETS)


def test_each_line_says_what_it_is_for(dialog):
    """A row of bare combos with no label is a puzzle."""
    labels = []
    for line in dialog._shared_capture_lines:
        for i in range(line.count()):
            w = line.itemAt(i).widget()
            if isinstance(w, QtWidgets.QLabel) and w.text().strip():
                labels.append(w.text())
                break
    assert len(labels) == 2, f"a line has no label: {labels}"


def test_the_shared_row_appears_only_in_cctv(dialog):
    dialog._set_shared_camera_row_visible(False)
    assert dialog._shared_capture_row.isHidden()
    dialog._set_shared_camera_row_visible(True)
    assert not dialog._shared_capture_row.isHidden()


# ── the row must agree with the shared card about the same camera ────────

def test_an_unset_format_defaults_to_mjpeg(row_dialog):
    """The card defaults to mjpeg; the row fell through to "" and selected
    the "Any" entry, so one camera read Any here and MJPEG in CCTV. MJPEG is
    what lets a USB camera reach its rate at the higher resolutions.

    Asserted on the widget, not on the source: reading the code back only
    proves the code says what it says.
    """
    fmt = row_dialog.table.cellWidget(0, row_dialog.COL_FORMAT)
    assert fmt.currentText() == "MJPEG", fmt.currentText()


def test_the_rate_shown_is_one_the_resolution_can_deliver(qapp):
    """A rate not on this resolution's ladder must not stay selected: the
    row would show one number while the config held another."""
    modes = [(640, 480, 29.5, "dshow")]
    offered = _ladder(modes, (640, 480))
    assert 21 not in offered, "fixture assumption"
    # The chooser falls back to the default, never to the slowest rung.
    _res, fps = _row(modes, (640, 480))
    assert fps.currentData() in offered
    assert fps.currentData() != min(offered)


def test_the_label_and_the_ladder_never_disagree_about_the_ceiling(qapp):
    """Seen on the rig: a row read "1920×1080 (5)" and offered 4 fps. The
    label rounded the measured 4.9875 while the ladder truncated it, so the
    same number was printed two ways one cell apart. Below FPS_MIN the ladder
    is the single measured rate, and it has to be the rate the size claims."""
    for measured, want in ((4.9875, 5), (3.0, 3), (9.96, 10), (4.2, 4)):
        res, fps = _row([(1920, 1080, measured, "dshow")], (1920, 1080))
        label = res.itemText(0)
        offered = [fps.itemData(i) for i in range(fps.count())
                   if fps.itemData(i) is not None]
        assert offered == [want], f"{measured} offered {offered}"
        assert f"({want}" in label or f"({measured:.1f}" in label, label
        # and the two must name the same number
        assert str(want) in label, f"label {label!r} vs ladder {offered}"


# ── every door stays reachable from a table cell ─────────────────────────
#
# The compact list kept one entry per SIZE, which is one entry per size per
# fastest backend. On this rig that deleted Media Foundation from the per-box
# columns entirely while the shared card still offered it, and MF is the door
# that reaches 30 fps at 1080p where DirectShow reaches 5. An operator with a
# per-camera rig could not pick the faster door, and could not see which door
# the number beside a size had come from.

_TWO_DOORS = [(1920, 1080, 5.0, "dshow"), (1920, 1080, 30.4, "msmf"),
              (1280, 720, 10.0, "dshow"), (1280, 720, 29.9, "msmf"),
              (640, 480, 29.9, "dshow"), (640, 480, 29.9, "msmf")]


def _doors(combo):
    return {(combo.itemData(i) or {}).get("backend")
            for i in range(combo.count())
            if isinstance(combo.itemData(i), dict)}


def test_a_cell_offers_every_measured_size_and_door(qapp):
    """Both doors are real capability, and they differ enormously.

    On this rig 1920x1080 measures 17.6 fps through Media Foundation and 1.0
    through DirectShow, so which door the camera opens with is the operator's
    to choose and both belong on screen. What does NOT belong is a row for a
    door nobody has probed, which is why an untried door is one row rather
    than one per size.
    """
    res, _fps = _row(_TWO_DOORS)
    assert res.count() == len(_TWO_DOORS), (
        [res.itemText(i) for i in range(res.count())])
    assert _doors(res) == {"dshow", "msmf"}
    at_1080 = [res.itemData(i) for i in range(res.count())
               if res.itemData(i)["width"] == 1920]
    assert {d["backend"] for d in at_1080} == {"dshow", "msmf"}


def test_the_fastest_door_to_a_size_is_the_one_selected(qapp):
    """Both are listed; the faster one opens by default."""
    res, _fps = _row(_TWO_DOORS)
    assert res.currentData()["backend"] == "msmf"
    assert res.currentData()["fps"] > 29


def test_a_forced_door_is_always_drawn_even_when_slower(qapp):
    """The full list on the Camera options card is where a door is forced.

    A cell that could not draw the forced door would show a different one
    than the camera is about to open with, which is the contradiction this
    whole column exists to remove.
    """
    d = _Dlg.__new__(_Dlg)
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    d._populateResolutionCombo(_TWO_DOORS, res_combo=res, fps_combo=fps,
                               compact=True, preselect_wh=(1920, 1080),
                               preselect_backend="dshow")
    assert res.currentData()["backend"] == "dshow"
    assert res.currentData()["fps"] == 5.0, "and its own rate, not the other's"


def test_a_cell_names_the_door_when_there_is_a_choice(qapp):
    res, _fps = _row(_TWO_DOORS)
    labels = [res.itemText(i) for i in range(res.count())]
    assert any("MF" in t for t in labels) and any("DS" in t for t in labels), (
        f"two doors to the same size are indistinguishable: {labels}")


def test_one_door_needs_no_name(qapp):
    """The common case stays short: naming a choice nobody has is noise."""
    res, _fps = _row([(640, 480, 29.9, "dshow")])
    assert res.itemText(0) == "640\u00d7480 (30)"


def test_an_unmeasured_door_can_still_be_tried(qapp):
    """A backend whose probe FAILED carries rate 0.0.

    It is offered so it can be forced, which is the one time an operator has
    to choose a door by hand. Refusing it a rate ladder left the FPS cell
    empty and disabled, so picking it stored ``selected_fps=None`` and the
    camera opened at no rate at all.
    """
    res, fps = _row([(640, 480, 29.9, "dshow"), (640, 480, 0.0, "msmf")])
    i = next(i for i in range(res.count())
             if res.itemData(i)["backend"] == "msmf")
    res.setCurrentIndex(i)
    d = _Dlg.__new__(_Dlg)
    d._repopulate_fps_combo(res_combo=res, fps_combo=fps)
    rates = [fps.itemData(k) for k in range(fps.count())]
    assert rates, "an untried door must still offer a rate to try it at"
    assert fps.isEnabled()
    assert fps.currentData() == 30, "opens on the behavioural target"


def test_the_saved_door_is_the_one_re_selected(qapp):
    """Matching a saved pick on SIZE alone re-selected the fastest door.

    So a deliberate choice of the slower door was undone the moment the row
    redrew, and the config then took the door nobody picked.
    """
    d = _Dlg.__new__(_Dlg)
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    d._populateResolutionCombo(_TWO_DOORS, preselect_wh=(1920, 1080),
                               preselect_backend="dshow", res_combo=res,
                               fps_combo=fps, compact=True)
    assert res.currentData()["backend"] == "dshow"
    d._populateResolutionCombo(_TWO_DOORS, preselect_wh=(1920, 1080),
                               preselect_backend="msmf", res_combo=res,
                               fps_combo=fps, compact=True)
    assert res.currentData()["backend"] == "msmf"


def test_populating_restores_a_caller_s_signal_block(qapp):
    """``blockSignals`` is a flag, not a counter.

    ``_fill_row_capture`` blocks both combos for its critical section and
    calls the populator, which used to unblock unconditionally. Its next
    ``setCurrentIndex`` then fired ``_on_row_capture_changed`` re-entrantly,
    which wrote the config from a selection the populator had just replaced:
    picking the DirectShow door stored Media Foundation.
    """
    d = _Dlg.__new__(_Dlg)
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    res.blockSignals(True)
    fps.blockSignals(True)
    d._populateResolutionCombo(_TWO_DOORS, res_combo=res, fps_combo=fps,
                               compact=True)
    assert res.signalsBlocked(), "the caller's block was dropped"
    assert fps.signalsBlocked(), "the caller's block was dropped"


# ── the list stays short enough to use ───────────────────────────────────
#
# Offering an unmeasured door for EVERY size doubled the list: a camera with
# nine measured modes drew eighteen rows, half of them speculative, and the
# everyday job of picking a size was buried under a recovery action needed
# once. What makes the short list safe is that ``begin_capturing`` already
# falls back through every backend when the pinned one will not open, so an
# untried row pins a door rather than being the only way to reach it.

_NINE_SIZES = [(3840, 2160, 1.0, "dshow"), (1920, 1080, 1.0, "dshow"),
               (1600, 1200, 1.0, "dshow"), (1280, 960, 2.4, "dshow"),
               (1280, 720, 9.8, "dshow"), (1024, 768, 10.0, "dshow"),
               (800, 600, 20.0, "dshow"), (640, 480, 24.9, "dshow"),
               (320, 240, 24.6, "dshow")]


def _with_untried(measured):
    d = _Dlg.__new__(_Dlg)
    return measured + d._untried_backend_rows(measured)


def test_an_unmeasured_door_costs_one_row_not_one_per_size(qapp):
    rows = _with_untried(_NINE_SIZES)
    untried = [r for r in rows if r[2] == 0.0]
    assert len(untried) == 1, (
        f"{len(untried)} speculative rows for one unmeasured door")
    res, _fps = _row(rows)
    assert res.count() == len(_NINE_SIZES) + 1, (
        [res.itemText(i) for i in range(res.count())])


def test_the_untried_row_names_the_door_it_offers(qapp):
    """"(try)" on its own says nothing about what would be tried."""
    res, _fps = _row(_with_untried(_NINE_SIZES))
    untried = next(res.itemText(i) for i in range(res.count())
                   if (res.itemData(i) or {}).get("fps") == 0.0)
    assert "MF" in untried, untried


def test_one_measured_door_is_not_named_on_every_row(qapp):
    """A camera that has only ever had one door has no choice to label.

    An untried row is not a second measured door, and counting it as one put
    "DS" beside all nine sizes of a single-door camera.
    """
    res, _fps = _row(_with_untried(_NINE_SIZES))
    measured = [res.itemText(i) for i in range(res.count())
                if (res.itemData(i) or {}).get("fps", 0) > 0]
    assert measured and not any("DS" in t for t in measured), measured


def test_two_measured_doors_are_named_on_every_row(qapp):
    """When the choice is real it has to be visible on both."""
    res, _fps = _row(_TWO_DOORS)
    labels = [res.itemText(i) for i in range(res.count())]
    assert all(("DS" in t or "MF" in t) for t in labels), labels


# ── the label and the ladder read the same fact ──────────────────────────
#
# Seen on the rig, in a screenshot: a row read "640×480 DS (24.6)" while the
# FPS cell beside it was set to 20 and offered 20 and 30. 24.6 was the rate
# the camera had once been MEASURED delivering, which is a property of the
# room's lighting rather than of the camera, and it appeared nowhere else in
# the application. The two cells had drifted apart when the ladder moved to
# the camera's enumerated modes and the label stayed on the measurement.

def _row_with_enumeration(modes, offered, wh=None):
    """A row for a camera whose modes have been enumerated."""
    d = _Dlg.__new__(_Dlg)
    d._offered_rates_for = lambda mode_data, cam_id=None: tuple(offered)
    res, fps = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    d._populateResolutionCombo(modes, preselect_wh=wh, res_combo=res,
                               fps_combo=fps, compact=True)
    return res, fps


def test_the_label_prints_the_offered_ceiling_not_the_measured_one(qapp):
    """The camera offers 30; the room once gave 24.6. The cell says 30."""
    res, _fps = _row_with_enumeration(
        [(640, 480, 24.6, "dshow")], offered=(15.0, 20.0, 30.0), wh=(640, 480))
    label = res.itemText(0)
    assert "30" in label, label
    assert "24.6" not in label, (
        f"{label!r} still prints a measurement of the room; 24.6 is not a "
        "rate this camera has and appears nowhere else in the application")


def test_the_label_and_the_ladder_name_the_same_ceiling(qapp):
    """Whatever the cells say, they have to say it about the same number."""
    res, fps = _row_with_enumeration(
        [(640, 480, 24.6, "dshow")], offered=(15.0, 20.0, 30.0), wh=(640, 480))
    ladder = [fps.itemData(i) for i in range(fps.count())
              if fps.itemData(i) is not None]
    assert max(ladder) == 30
    assert str(max(ladder)) in res.itemText(0), (
        f"label {res.itemText(0)!r} against ladder {ladder}")


def test_a_camera_nobody_enumerated_still_shows_its_measurement(qapp):
    """The enumeration is the better answer, not the only one.

    An empty enumeration means the question could not be ASKED, so the
    measured rate is all there is and a cell with no number at all would be
    worse than one carrying the rate the camera was seen to reach.
    """
    res, _fps = _row_with_enumeration(
        [(640, 480, 24.6, "dshow")], offered=(), wh=(640, 480))
    # The measurement still reaches the operator, rounded to the value the
    # rate cell offers (24.6 -> a ladder ending at 25) so the two cannot
    # disagree. A cell with no number would be worse than either.
    assert "25" in res.itemText(0), res.itemText(0)
    assert "try" not in res.itemText(0), res.itemText(0)
