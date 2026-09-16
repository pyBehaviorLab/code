"""The live tile shows three different numbers, and they were two.

"capable" was read from ``selected_fps``, which is what the operator PICKED,
and then labelled "(configured, camera can deliver)". A box set to 25 fps on a
camera that only offers 30 therefore read:

    capable:    25.0 fps  (configured, camera can deliver)
    acquiring:  30.0 fps  (camera live delivery)

which says the camera is exceeding its own capability, and tells the reader
nothing about which of the two is wrong. Reported from the rig on 2026-09-10.

The pick, the measured ceiling, and the live rate are three separate facts.
"""
from __future__ import annotations

from types import SimpleNamespace

from source.gui.base import MainWindowBase
from source.video.framebus.types import CameraConfig


def _window(cfg, cam_id="camA", setup_id=1):
    """A window stub carrying just the two lookups the helper makes."""
    w = MainWindowBase.__new__(MainWindowBase)
    w.video_manager = SimpleNamespace(box_camera_map={setup_id: cam_id})
    w.pipeline = SimpleNamespace(get_camera_config=lambda c: cfg)
    return w


def _camera(selected_fps, ceiling_at_640):
    cfg = CameraConfig(camera_id="camA")
    cfg.selected_resolution = (640, 480)
    cfg.selected_fps = selected_fps
    if ceiling_at_640:
        cfg.probed_modes = [(640, 480, ceiling_at_640)]
    return cfg


def test_capable_is_the_measured_ceiling_not_the_pick():
    """The rig case: picked 25 on a camera measured at 29.9."""
    w = _window(_camera(selected_fps=25, ceiling_at_640=29.9))
    requested, capable = w._tile_fps_facts(1, target_fps=30)
    assert requested == 25.0, "the pick"
    assert capable == 29.9, (
        "the tile reported the operator's pick as the camera's capability")


def test_the_two_agree_when_the_pick_is_the_ceiling():
    w = _window(_camera(selected_fps=30, ceiling_at_640=29.9))
    requested, capable = w._tile_fps_facts(1, target_fps=30)
    assert (requested, capable) == (30.0, 29.9)


def test_an_uncalibrated_camera_falls_back_to_the_pick():
    """No measurement is not a reason to show nothing."""
    w = _window(_camera(selected_fps=20, ceiling_at_640=None))
    requested, capable = w._tile_fps_facts(1, target_fps=30)
    assert (requested, capable) == (20.0, 20.0)


def test_a_camera_with_nothing_set_falls_back_to_the_thread_target():
    w = _window(_camera(selected_fps=None, ceiling_at_640=None))
    requested, capable = w._tile_fps_facts(1, target_fps=30)
    assert requested == 0.0 and capable == 30.0


def test_a_box_with_no_camera_reports_nothing_and_does_not_raise():
    w = MainWindowBase.__new__(MainWindowBase)
    w.video_manager = SimpleNamespace(box_camera_map={})
    w.pipeline = SimpleNamespace(get_camera_config=lambda c: None)
    assert w._tile_fps_facts(1, target_fps=0) == (0.0, 0.0)


def test_the_ceiling_comes_from_per_door_measurements_too():
    """Detect records per door, so ``probed_modes`` is often empty."""
    cfg = CameraConfig(camera_id="camA")
    cfg.selected_resolution = (1920, 1080)
    cfg.selected_fps = 25
    cfg.probed_variants = {"dshow": [(1920, 1080, 5.0)],
                           "msmf": [(1920, 1080, 30.4)]}
    requested, capable = _window(cfg)._tile_fps_facts(1, target_fps=30)
    assert (requested, capable) == (25.0, 30.4)


# ── "capable" is what the camera OFFERS, not what a probe measured ───────
#
# The measured ceiling above is still the fallback, but it must not be the
# answer when the camera has been enumerated. Measured on this rig: the trial
# probe recorded 10.0 fps at 1920x1080 for a camera whose descriptors list
# 5/7/10/15/20/24/25/30/50/60, and 5.1 fps for one that lists up to 30. The
# tile called a 60 fps camera a 10 fps camera and then showed it "acquiring"
# 30, i.e. exceeding its own capability, which is the same nonsense this file
# was written to stop.

def _enumerated(monkeypatch, offered_max, size=(1920, 1080)):
    """A window whose camera reports ``offered_max`` at ``size``."""
    cfg = CameraConfig(camera_id="camA")
    cfg.selected_resolution = size
    cfg.selected_fps = 30
    cfg.probed_modes = [(size[0], size[1], 10.0)]     # the room's number
    w = _window(cfg)
    monkeypatch.setattr(MainWindowBase, "_offered_ceiling_for",
                        lambda self, cam, res: float(offered_max))
    return w


def test_capable_prefers_what_the_camera_offers(monkeypatch):
    """The regression: 60 is offered, 10 was measured in a dim room."""
    w = _enumerated(monkeypatch, offered_max=60.0)
    _requested, capable = w._tile_fps_facts(1, target_fps=30)
    assert capable == 60.0, (
        "the tile reported a room measurement as the camera's capability")


def test_capable_never_reads_below_what_the_camera_is_delivering(monkeypatch):
    """The visible symptom: acquiring greater than capable is impossible."""
    w = _enumerated(monkeypatch, offered_max=60.0)
    _requested, capable = w._tile_fps_facts(1, target_fps=30)
    acquiring = 30.0                      # what the camera actually delivers
    assert capable >= acquiring, (
        f"tile claims the camera exceeds its own capability: "
        f"capable {capable}, acquiring {acquiring}")


def test_a_camera_nobody_enumerated_still_shows_its_measurement(monkeypatch):
    """An empty enumeration means "not asked", not "no rates", so the
    measurement is still better than nothing."""
    w = _enumerated(monkeypatch, offered_max=0.0)
    _requested, capable = w._tile_fps_facts(1, target_fps=30)
    assert capable == 10.0, "fell through to no number at all"
