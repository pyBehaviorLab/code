"""The chosen backend and the measurements behind it must survive a save.

A capability probe that costs ~78 s and is forgotten on the next project load
is worse than no probe: the operator watched it run, saw the right rate, and
gets the wrong one tomorrow with no indication why.

Two things travel, and they are different in kind:

* the CHOICE (``capture_backend``): an operator setting like the resolution
  and the rate, and portable;
* the MEASUREMENTS (``probed_variants``): evidence, and NOT portable in the
  strict sense: achievable rate depends on the host's USB controller and the
  port. They are saved so a project can explain its own choice, while the
  machine-level calibration store stays authoritative on a host that has
  measured for itself.

Driven through the real serialisers rather than a restatement of them.
"""
from __future__ import annotations

import pytest

from source.video.framebus.types import CameraConfig

pytest.importorskip("cv2")

#: The real reading from this rig, the same camera through two doors.
VARIANTS = {
    "dshow": [(640, 480, 30.4), (1280, 720, 10.0), (1920, 1080, 5.0)],
    "msmf": [(640, 480, 29.9), (1280, 720, 29.9), (1920, 1080, 29.1)],
}


def test_camera_config_round_trips_the_choice_and_the_evidence():
    cfg = CameraConfig(camera_id="fp755b78f6-opencv",
                       selected_resolution=(1280, 720), selected_fps=30,
                       capture_backend="msmf", probed_variants=VARIANTS)
    back = CameraConfig.from_json(cfg.to_json())
    assert back.capture_backend == "msmf"
    assert back.selected_resolution == (1280, 720)
    assert back.probed_variants["msmf"] == VARIANTS["msmf"]
    assert back.probed_variants["dshow"] == VARIANTS["dshow"]


def test_a_config_saved_before_this_existed_still_loads():
    """Absent keys mean "auto", not a crash, every project on disk today was
    written without them."""
    raw = CameraConfig(camera_id="cam").to_json()
    raw.pop("capture_backend", None)
    raw.pop("probed_variants", None)
    back = CameraConfig.from_json(raw)
    assert back.capture_backend == ""
    assert back.probed_variants == {}


def test_the_project_layer_round_trips_both():
    """Through ``_camera_json_to_entry`` / ``_entry_to_camera_json``, the pair
    the project file actually goes through, which is a different shape from
    ``CameraConfig`` and was where a field could silently be dropped."""
    from source.config.experiment import (_camera_json_to_entry,
                                          _entry_to_camera_json)
    cfg = CameraConfig(camera_id="cam0", selected_resolution=(1280, 720),
                       selected_fps=30, capture_backend="msmf",
                       probed_variants=VARIANTS)
    entry = _camera_json_to_entry("cam0", cfg.to_json())
    assert entry.preference.capture_backend == "msmf"
    assert entry.capabilities.probed_variants["msmf"] == VARIANTS["msmf"]

    again = _entry_to_camera_json(entry)
    assert again["capture_backend"] == "msmf"
    assert again["probed_variants"]["dshow"] == VARIANTS["dshow"]

    final = CameraConfig.from_json(again)
    assert final.capture_backend == "msmf"
    assert final.probed_variants["msmf"] == VARIANTS["msmf"]


def test_the_machine_store_round_trips_variants(tmp_path, monkeypatch):
    """The store is per-machine on purpose, a rate measured on one host's USB
    controller is not a fact about the camera anywhere else."""
    from source.video.cameras import calibration_store as cs
    monkeypatch.setattr(cs, "_user_config_dir", lambda: tmp_path)
    assert cs.put("usb-1234:5678:ABC", "opencv", "Cam", VARIANTS["msmf"],
                  identity={"bus_speed": "5000"}, variants=VARIANTS)
    got = cs.get_variants("usb-1234:5678:ABC", {"bus_speed": "5000"})
    assert got["dshow"] == VARIANTS["dshow"]
    assert got["msmf"] == VARIANTS["msmf"]


def test_the_store_refuses_its_own_numbers_on_a_slower_port(tmp_path,
                                                            monkeypatch):
    """Moving the camera from USB3 to USB2 keeps the serial and destroys the
    ceiling, so the measurements must not be reused."""
    from source.video.cameras import calibration_store as cs
    monkeypatch.setattr(cs, "_user_config_dir", lambda: tmp_path)
    cs.put("usb-1:2:X", "opencv", "Cam", VARIANTS["msmf"],
           identity={"bus_speed": "5000"}, variants=VARIANTS)
    assert cs.get_variants("usb-1:2:X", {"bus_speed": "480"}) == {}


def test_the_pipeline_picks_the_faster_backend_from_a_loaded_config():
    """End of the line: a config loaded from disk, with no backend pinned,
    must resolve to the door that measured faster for ITS selected mode."""
    from source.video.framebus.controller import Pipeline
    pipe = Pipeline()
    try:
        cfg = CameraConfig(camera_id="cam0", selected_resolution=(1280, 720),
                           selected_fps=30, probed_variants=VARIANTS)
        assert pipe._fastest_backend_for(cfg) == "msmf"
        # At VGA the two tie, and a tie must not move the rig off its default.
        cfg_vga = CameraConfig(camera_id="cam0",
                               selected_resolution=(640, 480),
                               selected_fps=30, probed_variants=VARIANTS)
        assert pipe._fastest_backend_for(cfg_vga) == "dshow"
        # One backend measured is no contest, nothing to choose between.
        cfg_one = CameraConfig(camera_id="cam0",
                               selected_resolution=(1280, 720),
                               selected_fps=30,
                               probed_variants={"dshow": VARIANTS["dshow"]})
        assert pipe._fastest_backend_for(cfg_one) == ""
    finally:
        pipe.shutdown()
