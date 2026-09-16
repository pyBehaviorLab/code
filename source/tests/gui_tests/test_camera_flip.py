"""A mirrored camera is corrected at capture, not at display.

Most webcams mirror their output, so the animal's left appears on the right.
Nothing in the pipeline flipped anything, so the operator's left/right was
simply wrong everywhere.

The fix has to land at capture. Flipping the display alone would make the
screen agree with the operator while the recording, the tracker, the zone
tests and the coordinates pushed to the board all kept the mirrored geometry,
the worst outcome, because the picture would look right and every recorded
number would be wrong. These tests pin the flip to the capture path, which is
upstream of all of them.
"""
from __future__ import annotations

import numpy as np
import pytest

from source.video.framebus.types import CameraConfig

pytestmark = pytest.mark.unit


def _thread(**cfg):
    """A CameraThread with its __init__ side effects avoided.

    Constructing one opens a device; only the batch transform is under test,
    so the flags it reads are set directly.
    """
    from source.video.cameras.capture import CameraThread
    t = object.__new__(CameraThread)
    t.camera_id = "0"
    t.flip_horizontal = cfg.get("flip_horizontal", False)
    t.flip_vertical = cfg.get("flip_vertical", False)
    return t


def _asym():
    """A frame whose corners are all different, so any flip is detectable."""
    f = np.zeros((4, 6, 3), dtype=np.uint8)
    f[0, 0] = (10, 10, 10)      # top-left
    f[0, -1] = (20, 20, 20)     # top-right
    f[-1, 0] = (30, 30, 30)     # bottom-left
    f[-1, -1] = (40, 40, 40)    # bottom-right
    return f


def _corners(f):
    return (tuple(f[0, 0]), tuple(f[0, -1]), tuple(f[-1, 0]), tuple(f[-1, -1]))


# --- the transform ----------------------------------------------------------

def test_mirror_swaps_left_and_right_only():
    t = _thread(flip_horizontal=True)
    tl, tr, bl, br = _corners(t._flip_batch([_asym()])[0])
    assert (tl, tr) == ((20, 20, 20), (10, 10, 10)), "top row not mirrored"
    assert (bl, br) == ((40, 40, 40), (30, 30, 30)), "bottom row not mirrored"


def test_vertical_flip_swaps_top_and_bottom_only():
    t = _thread(flip_vertical=True)
    tl, tr, bl, br = _corners(t._flip_batch([_asym()])[0])
    assert (tl, tr) == ((30, 30, 30), (40, 40, 40))
    assert (bl, br) == ((10, 10, 10), (20, 20, 20))


def test_both_flips_rotate_180():
    t = _thread(flip_horizontal=True, flip_vertical=True)
    tl, _tr, _bl, br = _corners(t._flip_batch([_asym()])[0])
    assert tl == (40, 40, 40) and br == (10, 10, 10)


def test_mirroring_twice_is_the_identity():
    t = _thread(flip_horizontal=True)
    once = t._flip_batch([_asym()])[0]
    twice = t._flip_batch([once])[0]
    assert np.array_equal(twice, _asym())


def test_the_whole_batch_is_flipped():
    t = _thread(flip_horizontal=True)
    out = t._flip_batch([_asym() for _ in range(4)])
    assert len(out) == 4
    for f in out:
        assert tuple(f[0, 0]) == (20, 20, 20)


def test_the_source_frame_is_not_modified_in_place():
    """Downstream keeps the reference without copying, and another thread may
    already be reading it, an in-place flip would corrupt that frame."""
    t = _thread(flip_horizontal=True)
    src = _asym()
    before = src.copy()
    out = t._flip_batch([src])[0]
    assert np.array_equal(src, before), "input frame was mutated"
    assert not np.shares_memory(out, src)


def test_a_grayscale_frame_flips_too():
    t = _thread(flip_horizontal=True)
    g = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = t._flip_batch([g])[0]
    assert out.shape == g.shape
    assert np.array_equal(out, g[:, ::-1])


def test_a_bad_frame_does_not_take_the_batch_down():
    """One unflippable frame must not stop the camera."""
    t = _thread(flip_horizontal=True)
    out = t._flip_batch([None, _asym()])
    assert len(out) == 2
    assert tuple(out[1][0, 0]) == (20, 20, 20)


# --- config plumbing --------------------------------------------------------

def test_the_flip_defaults_to_off():
    cfg = CameraConfig(camera_id="0")
    assert cfg.flip_horizontal is False
    assert cfg.flip_vertical is False


def test_the_flip_round_trips_through_the_saved_config():
    """It has to survive a save/load, or the rig is mirrored again tomorrow."""
    cfg = CameraConfig(camera_id="0", flip_horizontal=True, flip_vertical=True)
    back = CameraConfig.from_json(cfg.to_json())
    assert back.flip_horizontal is True
    assert back.flip_vertical is True


def test_an_older_config_without_the_key_loads_as_unflipped():
    back = CameraConfig.from_json({"camera_id": "0"})
    assert back.flip_horizontal is False
    assert back.flip_vertical is False


# --- it reaches the live camera, not just the stored config -----------------

def test_setting_the_flip_updates_the_running_camera():
    """The operator ticks the box mid-session; the very next frame must be
    corrected. Persisting alone would leave the preview mirrored until a
    reconnect."""
    from types import SimpleNamespace

    from source.video.framebus.controller import Pipeline

    pipe = Pipeline(target_fps=30)
    try:
        cam = SimpleNamespace(flip_horizontal=False, flip_vertical=False)
        pipe.video_manager.cameras["0"] = cam

        pipe.set_camera_flip("0", horizontal=True)

        assert cam.flip_horizontal is True, "live camera thread not updated"
        assert pipe.get_camera_config("0").flip_horizontal is True, (
            "flip not persisted; it would be lost on reload")
    finally:
        pipe.shutdown()


def test_setting_the_flip_for_an_unopened_camera_still_persists():
    """Configured before connecting is the normal case."""
    from source.video.framebus.controller import Pipeline

    pipe = Pipeline(target_fps=30)
    try:
        pipe.set_camera_flip("7", horizontal=True, vertical=True)
        cfg = pipe.get_camera_config("7")
        assert cfg.flip_horizontal is True and cfg.flip_vertical is True
    finally:
        pipe.shutdown()


def test_an_empty_update_changes_nothing():
    from source.video.framebus.controller import Pipeline

    pipe = Pipeline(target_fps=30)
    try:
        pipe.set_camera_flip("0", horizontal=True)
        pipe.set_camera_flip("0")          # neither argument given
        assert pipe.get_camera_config("0").flip_horizontal is True
    finally:
        pipe.shutdown()
