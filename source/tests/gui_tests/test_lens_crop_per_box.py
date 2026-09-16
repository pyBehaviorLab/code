"""Under a shared (CCTV) camera, lens calibration is per box, and it has to
actually SEE that box's region.

The wizard was fed the whole camera frame regardless of which box it was
calibrating, so the checkerboard was solved against the full frame and then
filed under a per-box key. That stores N near-identical profiles that fit no
box: a crop has its own principal point and its own effective field of view.
"""
from types import SimpleNamespace
from typing import ClassVar

import pytest

from source.gui.dialogs.camera_connect import CameraConnectDialog
from source.tests.qt_dispose import dispose


class _FakeCfg:
    probed_modes: ClassVar[list] = []
    selected_resolution = (1920, 1080)
    selected_fps = 60
    frame_strategy = "accept"
    camera_backend = "opencv"


class _FakePipeline:
    def get_camera_config(self, cam_id):
        return _FakeCfg()

    def update_camera_config(self, cam_id, **kw):
        return _FakeCfg()

    def all_camera_configs(self):
        return {}


def _box(n, cam, roi=None):
    from PySide6 import QtWidgets
    w = QtWidgets.QWidget()
    w.setup_number = n
    e = QtWidgets.QLineEdit()
    e.setText(cam)
    w.camera_id_edit = e
    w._camera_backend = "OpenCV"
    w.save_video_enabled = True
    w.roi_normalized = roi
    return w



@pytest.fixture
def dlg(mock_qapplication):
    def _build(specs):
        d = CameraConnectDialog(parent=None)
        boxes = [_box(i + 1, c, roi) for i, (c, roi) in enumerate(specs)]
        d.main_window = SimpleNamespace(
            get_all_setup_widgets=lambda: list(boxes), pipeline=_FakePipeline(),
            _probed_resolution_modes={},
            video_manager=SimpleNamespace(box_camera_map={}))
        d._machine_cached_modes = lambda i: []
        d.populateBoxes()
        return d
    return _build


def test_dedicated_camera_calibrates_the_whole_frame(dlg):
    d = dlg([("0", None), ("1", None)])
    try:
        assert d._lens_crop_rect(0, None) is None
        # Even with a box number, a camera only one box uses is not cropped.
        assert d._lens_crop_rect(0, 1) is None
    finally:
        dispose(d)


def test_shared_camera_crops_to_the_boxs_own_region(dlg):
    d = dlg([("0", (0.0, 0.0, 0.5, 1.0)), ("0", (0.5, 0.0, 0.5, 1.0))])
    try:
        assert d._lens_crop_rect(0, 1) == (0.0, 0.0, 0.5, 1.0)
        assert d._lens_crop_rect(0, 2) == (0.5, 0.0, 1.0, 1.0)
    finally:
        dispose(d)


def test_shared_camera_without_a_region_falls_back_to_the_full_frame(dlg):
    """Nothing drawn yet is not an error, calibrate what we can see."""
    d = dlg([("0", None), ("0", None)])
    try:
        assert d._lens_crop_rect(0, 1) is None
    finally:
        dispose(d)


def test_each_shared_box_is_its_own_calibration_target(dlg):
    d = dlg([("0", (0.0, 0.0, 0.5, 1.0)), ("0", (0.5, 0.0, 0.5, 1.0))])
    try:
        targets = d._lens_targets()
        assert targets == [(0, 1), (0, 2)]
    finally:
        dispose(d)
