"""One list: the mode and the backend that serves it, best first.

Picking a resolution and picking the backend that can deliver it are one
decision, so they are one control. This rig's webcam reaches 1920x1080 at
~30 fps through Media Foundation and ~5 through DirectShow; a merged list
showed 30 without saying which door produced it, and a separate backend
picker asked the operator to make the same choice a second time, then hid
itself exactly when a backend failed to probe and the override was needed.

So the combo lists every measured mode per backend, sorted largest picture
then highest rate, and the row the operator picks carries the backend with it.
"""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets

from source.tests.qt_dispose import WidgetBin

pytest.importorskip("cv2")

_BIN = WidgetBin()


@pytest.fixture(autouse=True)
def _drain():
    yield
    _BIN.drain()


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def dialog(qapp):
    from source.gui.dialogs.camera_connect import CameraConnectDialog
    return _BIN.add(CameraConnectDialog(parent=None))


#: The real reading from this rig, as the probe returns it: per backend.
MEASURED = [
    (1920, 1080, 5.0, "dshow"),
    (1280, 720, 10.0, "dshow"),
    (640, 480, 30.4, "dshow"),
    (1920, 1080, 30.4, "msmf"),
    (1280, 720, 29.9, "msmf"),
    (640, 480, 29.9, "msmf"),
]


def _labels(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _data(combo):
    return [combo.itemData(i) for i in range(combo.count())]


def test_every_measured_mode_is_offered_once_per_backend(dialog):
    dialog._populateResolutionCombo(MEASURED)
    assert dialog.resolution_combo.count() == len(MEASURED)


def test_each_row_names_its_backend(dialog):
    dialog._populateResolutionCombo(MEASURED)
    labels = _labels(dialog.resolution_combo)
    assert any("Media Foundation" in x for x in labels), labels
    assert any("DirectShow" in x for x in labels), labels


def test_the_default_is_the_largest_picture_at_the_highest_rate(dialog):
    """"High res and high fps", and the two 1080p rows differ only by
    backend, so the default is the one that actually delivers."""
    dialog._populateResolutionCombo(MEASURED)
    first = dialog.resolution_combo.itemData(0)
    assert (first["width"], first["height"]) == (1920, 1080)
    assert first["fps"] == pytest.approx(30.4)
    assert first["backend"] == "msmf"
    assert dialog.resolution_combo.currentIndex() == 0


def test_rows_are_sorted_biggest_then_fastest(dialog):
    dialog._populateResolutionCombo(MEASURED)
    got = [(d["width"] * d["height"], d["fps"])
           for d in _data(dialog.resolution_combo)]
    assert got == sorted(got, key=lambda t: (-t[0], -t[1]))


def test_the_row_carries_the_backend_so_one_pick_sets_both(dialog):
    """The whole point of merging the two controls."""
    dialog._populateResolutionCombo(MEASURED)
    for d in _data(dialog.resolution_combo):
        assert set(d) >= {"width", "height", "fps", "backend"}


def test_a_saved_pick_beats_the_default(dialog):
    dialog._populateResolutionCombo(MEASURED, preselect_wh=(640, 480))
    d = dialog.resolution_combo.currentData()
    assert (d["width"], d["height"]) == (640, 480)


def test_a_camera_measured_before_backends_existed_still_populates(dialog):
    """Old projects and old machine caches hold flat ``(w, h, fps)``. They
    must still fill the list, just with no backend named."""
    dialog._populateResolutionCombo([(640, 480, 30.0), (1280, 720, 12.0)])
    assert dialog.resolution_combo.count() == 2
    assert all(d["backend"] == "" for d in _data(dialog.resolution_combo))


def test_no_modes_says_so_rather_than_offering_nothing(dialog):
    dialog._populateResolutionCombo([])
    assert dialog.resolution_combo.count() == 1
    assert dialog.resolution_combo.itemData(0) is None


def test_a_malformed_row_is_skipped_not_fatal(dialog):
    dialog._populateResolutionCombo(
        [(640, 480, 30.0, "dshow"), ("bad", None, None), (320, 240, 25.0)])
    assert dialog.resolution_combo.count() == 2


def test_an_unknown_backend_name_is_shown_not_hidden(dialog):
    """A backend nobody has given a pretty name is still one the camera
    measured; hiding it would make the row unexplainable."""
    dialog._populateResolutionCombo([(640, 480, 30.0, "someothersdk")])
    assert "someothersdk" in _labels(dialog.resolution_combo)[0]


def test_the_separate_backend_control_is_gone(dialog):
    """It asked for the same decision twice, and hid itself when a backend
    failed to probe, which is when an override matters."""
    assert not hasattr(dialog, "capture_backend_combo")


# ── what the probe hands the card ────────────────────────────────────────

def test_the_merged_result_is_a_plain_mode_list_not_a_probe_result():
    """``item.result`` changed shape from a ProbeResult to a merged list and
    a second consumer still did ``res.modes``; it took the dialog down the
    first time Detect was pressed."""
    from source.video.cameras.probe import Variant, merged_modes
    merged = merged_modes([Variant("dshow", [(640, 480, 30.0)])])
    assert isinstance(merged, list) and not hasattr(merged, "modes")
    assert merged and len(merged[0]) == 3


def test_a_backend_that_finds_nothing_is_retried_once():
    """An empty result is ambiguous: a backend that cannot serve this camera
    looks exactly like one that arrived while the previous still held it."""
    import source.video.cameras.probe as probe_mod

    calls = []

    class _R:
        def __init__(self, modes):
            self.modes, self.error, self.degraded = modes, None, False

    def _fake(camera_id, backend, target_fps, report=None, pixel_format="mjpeg"):
        calls.append(1)
        return _R([(640, 480, 30.0)] if len(calls) > 1 else [])

    orig, gap = probe_mod.probe_camera, probe_mod.BACKEND_RELEASE_S
    probe_mod.probe_camera, probe_mod.BACKEND_RELEASE_S = _fake, 0.0
    try:
        import cv2
        res = probe_mod._probe_one_backend("cam", "msmf", cv2.CAP_MSMF,
                                           30.0, None)
    finally:
        probe_mod.probe_camera, probe_mod.BACKEND_RELEASE_S = orig, gap
    assert len(calls) == 2
    assert res.modes == [(640, 480, 30.0)]


def test_every_offered_backend_resolves_to_something_openable():
    """A name the factory cannot translate would silently fall back to auto,
    so the row would look like it worked and change nothing."""
    from source.video.cameras.opencv import _uvc_backends, cv_backend_for
    for name in _uvc_backends():
        assert cv_backend_for(name) is not None
    assert cv_backend_for("") is None
    assert cv_backend_for("no-such-backend") is None


def test_a_webcam_reports_no_curated_features():
    """Everything a UVC camera reports is already a control on the card."""
    from source.video.cameras.builtin_backends import _opencv_features
    from source.video.cameras.features import TIER_CURATED

    class _Cam:
        def get_width(self):
            return 640

        def get_height(self):
            return 480

    feats = _opencv_features(_Cam())
    assert feats, "the features must still be reported, just not curated"
    assert not [f for f in feats if f.tier == TIER_CURATED]


# ── a backend whose probe failed must still be reachable ─────────────────

def _rows(measured):
    from source.gui.dialogs.camera_connect import CameraConnectDialog as D
    return measured + D._untried_backend_rows(measured)


def test_a_backend_that_never_measured_is_still_offered():
    """The reported case: Media Foundation could not open the camera during
    Detect, so it had no rows, so it could not be chosen, and the list
    offered only the backend that caps this camera at 5 fps at 1080p. The one
    time an operator must force a backend is when its probe failed."""
    measured = [(1920, 1080, 5.0, "dshow"), (640, 480, 29.9, "dshow")]
    rows = _rows(measured)
    backends = {b for _w, _h, _f, b in rows}
    assert "msmf" in backends, f"only {backends} offered"


def test_an_untried_row_carries_no_invented_rate():
    """Marked untried, not given a number nobody measured."""
    rows = _rows([(1920, 1080, 5.0, "dshow")])
    untried = [r for r in rows if r[3] == "msmf"]
    assert untried and all(r[2] == 0.0 for r in untried)


def test_untried_rows_say_so_in_the_label(qapp, dialog):
    dialog._populateResolutionCombo(_rows([(1920, 1080, 5.0, "dshow")]))
    labels = [dialog.resolution_combo.itemText(i)
              for i in range(dialog.resolution_combo.count())]
    tried = [x for x in labels if "not measured" not in x]
    untried = [x for x in labels if "not measured" in x]
    assert tried and untried, labels
    assert "Media Foundation" in untried[0]
    assert "fps" not in untried[0], "an unmeasured row must not print a rate"


def test_an_untried_row_never_becomes_the_default(qapp, dialog):
    """It is an escape hatch, not a recommendation, defaulting to a mode
    nobody has measured would be worse than the problem it solves."""
    dialog._populateResolutionCombo(
        _rows([(1920, 1080, 5.0, "dshow"), (640, 480, 29.9, "dshow")]))
    picked = dialog.resolution_combo.currentData()
    assert picked["fps"] > 0, f"defaulted to an unmeasured row: {picked}"


def test_nothing_is_added_when_every_backend_was_measured():
    """No clutter on a camera where the probe worked."""
    measured = [(640, 480, 30.0, "dshow"), (640, 480, 29.0, "msmf")]
    from source.gui.dialogs.camera_connect import CameraConnectDialog as D
    assert D._untried_backend_rows(measured) == []


def test_nothing_is_added_for_a_camera_with_no_backend_data_at_all():
    """An old flat cache has no backend names; inventing rows from nothing
    would offer modes never observed on any door."""
    from source.gui.dialogs.camera_connect import CameraConnectDialog as D
    assert D._untried_backend_rows([(640, 480, 30.0, "")]) == []
