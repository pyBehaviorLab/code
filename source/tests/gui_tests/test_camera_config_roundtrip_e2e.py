"""Camera settings must survive a project save/load, end to end.

The path a camera setting takes is long, and every hop is a chance to drop it:

    dialog -> pipeline CameraConfig -> CameraConfig.to_json()
           -> _camera_json_to_entry() -> CameraEntry (the saved shape)
           -> experiment_config.json on disk
           -> CameraEntry.from_dict() -> _entry_to_camera_json()
           -> CameraConfig.from_json() -> pipeline

A value dropped anywhere in that chain is silent: the rig comes back next
morning at a different frame rate or exposure and nothing says why. These
tests drive the real conversion functions and the real file, and assert the
values come back identical.
"""
import contextlib
import json
from dataclasses import asdict

import pytest

from source.config.experiment import (
    CameraEntry,
    _camera_json_to_entry,
    _entry_to_camera_json,
)
from source.video.framebus.types import CameraConfig


def _configured_camera() -> CameraConfig:
    """A camera set up the way an operator actually leaves one: a picked
    resolution and rate, a transport format, and sensor features read from
    and written back to the camera itself."""
    cfg = CameraConfig(camera_id="cam3cbb52d3")
    cfg.camera_backend = "spinnaker"
    cfg.selected_resolution = (1440, 1080)
    cfg.selected_fps = 60
    cfg.capture_format = "mjpeg"
    cfg.exposure_us = 4200
    cfg.gain_db = 3.5
    cfg.sync_role = "primary"
    cfg.probed_modes = [(1440, 1080, 60), (720, 540, 120)]
    cfg.features = {
        "exposure_auto": "Off",
        "pixel_format": "Mono8",
        "trigger_mode": "On",
        "gain_db": 3.5,
        "frame_rate": 60.0,
    }
    cfg.user_applied = True
    return cfg


def _round_trip(cfg: CameraConfig) -> CameraConfig:
    """Push one camera through the whole save/load chain, including the JSON
    text, so nothing survives merely by object identity."""
    entry = _camera_json_to_entry(cfg.camera_id, cfg.to_json())
    on_disk = json.loads(json.dumps(asdict(entry)))
    restored_entry = CameraEntry.from_dict(on_disk)
    return CameraConfig.from_json(_entry_to_camera_json(restored_entry))


# ── the identity a camera is found by ─────────────────────────────────────

def test_backend_and_resolution_survive():
    back = _round_trip(_configured_camera())
    assert back.camera_backend == "spinnaker"
    assert tuple(back.selected_resolution) == (1440, 1080)
    assert back.capture_format == "mjpeg"


def test_probed_modes_survive():
    back = _round_trip(_configured_camera())
    modes = {(int(w), int(h), int(f)) for w, h, f in back.probed_modes}
    assert (1440, 1080, 60) in modes
    assert (720, 540, 120) in modes


# ── the settings that actually change what is recorded ───────────────────

def test_selected_fps_survives():
    """The picked frame rate is the single value the whole recording chain
    keys on; losing it silently re-rates the rig."""
    back = _round_trip(_configured_camera())
    assert back.selected_fps == 60


def test_exposure_and_gain_survive():
    back = _round_trip(_configured_camera())
    assert back.exposure_us == 4200
    assert back.gain_db == pytest.approx(3.5)


def test_sync_role_survives():
    back = _round_trip(_configured_camera())
    assert back.sync_role == "primary"


def test_sdk_features_survive():
    """Everything set on the Camera options tab, read from the sensor and
    written back to it, has to come back, or a FLIR reverts to defaults on
    the next project load."""
    back = _round_trip(_configured_camera())
    assert back.features.get("exposure_auto") == "Off"
    assert back.features.get("pixel_format") == "Mono8"
    assert back.features.get("trigger_mode") == "On"


# ── through the real project file ─────────────────────────────────────────

def test_full_project_save_load_keeps_camera_settings(tmp_path):
    """Write a real experiment config to disk and read it back, so the test
    covers the file format and not just the in-memory converters."""
    from source.config.experiment import Config

    cfg = Config()
    cam = _configured_camera()
    cfg.cameras.registry = [_camera_json_to_entry(cam.camera_id, cam.to_json())]

    path = tmp_path / "experiment_config.json"
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    loaded = Config.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert len(loaded.cameras.registry) == 1
    back = CameraConfig.from_json(
        _entry_to_camera_json(loaded.cameras.registry[0]))

    assert back.camera_backend == "spinnaker"
    assert tuple(back.selected_resolution) == (1440, 1080)
    assert back.selected_fps == 60
    assert back.capture_format == "mjpeg"
    assert back.exposure_us == 4200
    assert back.features.get("pixel_format") == "Mono8"


def test_unconfigured_camera_stays_unconfigured(tmp_path):
    """A camera nobody has set up must not come back pretending it was:
    ``user_applied`` gates whether the recorder trusts these values."""
    cfg = CameraConfig(camera_id="cam5b4028a9")
    back = _round_trip(cfg)
    assert back.selected_resolution in (None, (None, None))
    assert not back.user_applied
    assert back.features == {}


# ── through the real host bridge (what save/load actually calls) ──────────

class _StubHost:
    """The surface the config bridge reads a rig's cameras through."""

    def __init__(self, pipeline, target_fps=30):
        self.pipeline = pipeline
        self.video_target_fps = target_fps
        self._active_config = None


@contextlib.contextmanager
def _pipeline_with(cams):
    """A live Pipeline, always shut down.

    Its sinks own daemon worker threads. Leaving one running and letting the
    cyclic GC free the Pipeline later frees memory the threads still touch,
    which corrupts the heap, far from the test that leaked it.
    """
    from source.video.framebus.controller import Pipeline
    pipe = Pipeline()
    try:
        for cam in cams:
            pipe.install_camera_configs({cam.camera_id: cam.to_json()})
        yield pipe
    finally:
        with contextlib.suppress(Exception):
            pipe.shutdown()


def test_host_bridge_round_trips_every_camera_setting(tmp_path):
    """pipeline -> cfg -> disk -> cfg -> pipeline, through the same two
    functions project save and project load actually call."""
    from source.config.experiment import (
        Config,
        _apply_camera_registry,
        _read_camera_registry,
    )

    with _pipeline_with([_configured_camera()]) as pipe:
        cfg = Config()
        _read_camera_registry(_StubHost(pipe), cfg)
        assert len(cfg.cameras.registry) == 1

        path = tmp_path / "experiment_config.json"
        path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        loaded = Config.from_dict(json.loads(path.read_text(encoding="utf-8")))

    with _pipeline_with([]) as fresh:
        _apply_camera_registry(loaded, _StubHost(fresh))
        back = fresh.all_camera_configs()["cam3cbb52d3"]

        assert back.camera_backend == "spinnaker"
        assert tuple(back.selected_resolution) == (1440, 1080)
        assert back.selected_fps == 60
        assert back.capture_format == "mjpeg"
        assert back.exposure_us == 4200
        assert back.sync_role == "primary"
        assert back.features.get("pixel_format") == "Mono8"
        assert back.features.get("trigger_mode") == "On"


def test_rig_wide_fps_only_fills_in_for_cameras_without_one(tmp_path):
    """A mixed-rate rig must not be flattened to one rate on load: the
    rig-wide default is a fallback, not an override."""
    from source.config.experiment import (
        Config,
        _apply_camera_registry,
        _read_camera_registry,
    )

    fast = _configured_camera()               # selected_fps = 60
    slow = CameraConfig(camera_id="cam5b4028a9")
    slow.camera_backend = "opencv"
    slow.selected_resolution = (640, 480)     # no rate picked

    with _pipeline_with([fast, slow]) as pipe:
        cfg = Config()
        _read_camera_registry(_StubHost(pipe), cfg)

    with _pipeline_with([]) as fresh:
        _apply_camera_registry(cfg, _StubHost(fresh, target_fps=30))
        cams = fresh.all_camera_configs()

        assert cams["cam3cbb52d3"].selected_fps == 60, "per-camera rate was overwritten"
        assert cams["cam5b4028a9"].selected_fps == 30, "rig default did not fill in"


# ── the registry stores identities, never indices ────────────────────────
#
# Measured on the rig, 2026-09-09: project files carried '0', '1' AND
# 'fp755b78f6' as separate registry entries for ONE physical camera, with
# contradictory settings (640x480@30 against 1280x720@20). Every box binds an
# identity, so everything written against the index entries silently never
# reached the rig.

class _FakePipe:
    def __init__(self, configs):
        self._configs = configs

    def all_camera_configs(self):
        return self._configs


class _Host:
    def __init__(self, configs):
        self.pipeline = _FakePipe(configs)


def _cam(cid):
    c = CameraConfig(camera_id=cid)
    c.selected_resolution = (1280, 720)
    c.selected_fps = 30
    return c


def _registry_ids(configs):
    from source.config.experiment import Config, _read_camera_registry
    cfg = Config()
    _read_camera_registry(_Host(configs), cfg)
    return [e.camera_id for e in cfg.cameras.registry]


def test_an_index_keyed_entry_is_never_saved():
    ids = _registry_ids({"0": _cam("0"), "cam22ecb311": _cam("cam22ecb311")})
    assert ids == ["cam22ecb311"], (
        "an index is a location, not an identity; it must not become a key")


def test_a_legacy_opencv_suffix_is_also_rejected():
    ids = _registry_ids({"3-opencv": _cam("3-opencv"),
                         "cam3cbb52d3": _cam("cam3cbb52d3")})
    assert ids == ["cam3cbb52d3"]


def test_dropping_an_index_entry_is_reported(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="source.config.experiment"):
        _registry_ids({"0": _cam("0")})
    assert any("not an identity" in r.getMessage() for r in caplog.records), (
        "silently dropping a saved entry would be its own bug")


def test_identity_keyed_entries_are_all_kept():
    ids = _registry_ids({"cam3cbb52d3": _cam("cam3cbb52d3"),
                         "cam5b4028a9": _cam("cam5b4028a9"),
                         "cam22ecb311": _cam("cam22ecb311")})
    assert sorted(ids) == ["cam22ecb311", "cam3cbb52d3", "cam5b4028a9"]


def test_old_fingerprint_entries_still_round_trip():
    """Projects saved before per-device ids must keep loading."""
    assert _registry_ids({"fp755b78f6": _cam("fp755b78f6")}) == ["fp755b78f6"]


def test_a_usb_id_survives_the_whole_save_load_chain(tmp_path):
    """The new id shape must pass through every hop unchanged."""
    cam = _cam("cam22ecb311")
    cam.capture_format = "mjpeg"
    entry = _camera_json_to_entry("cam22ecb311", cam.to_json())
    path = tmp_path / "cams.json"
    path.write_text(json.dumps(asdict(entry)), encoding="utf-8")
    back = CameraEntry.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert back.camera_id == "cam22ecb311"
    restored = CameraConfig.from_json(_entry_to_camera_json(back))
    assert restored.camera_id == "cam22ecb311"
    assert restored.selected_resolution == (1280, 720)
    assert restored.selected_fps == 30
    assert restored.capture_format == "mjpeg"


# ── nothing the pipeline holds may be dropped by the save ────────────────

def test_no_camera_field_is_lost_in_the_round_trip():
    """The contract, stated once. Measured 2026-09-09: flip_horizontal,
    flip_vertical, grayscale and frame_strategy were silently dropped, so an
    operator who flipped the image got it back the other way round after a
    reload with nothing reporting a change."""
    cam = CameraConfig(camera_id="cam22ecb311")
    src = cam.to_json()
    back = _entry_to_camera_json(_camera_json_to_entry("cam22ecb311", src))
    lost = sorted(set(src) - set(back))
    assert not lost, f"these settings never reach the project file: {lost}"


def test_orientation_and_grayscale_survive_save_and_load():
    cam = CameraConfig(camera_id="cam22ecb311")
    cam.flip_horizontal = True
    cam.flip_vertical = True
    cam.grayscale = True
    cam.frame_strategy = "latest"

    entry = _camera_json_to_entry("cam22ecb311", cam.to_json())
    # Through the saved dataclass, as a real project file does.
    restored = CameraConfig.from_json(
        _entry_to_camera_json(CameraEntry.from_dict(asdict(entry))))

    assert restored.flip_horizontal is True
    assert restored.flip_vertical is True
    assert restored.grayscale is True
    assert restored.frame_strategy == "latest"


def test_an_unflipped_camera_stays_unflipped():
    """The default must not become True by way of a truthy default."""
    cam = CameraConfig(camera_id="cam3cbb52d3")
    restored = CameraConfig.from_json(
        _entry_to_camera_json(_camera_json_to_entry("cam3cbb52d3",
                                                    cam.to_json())))
    assert restored.flip_horizontal is False
    assert restored.flip_vertical is False
    assert restored.grayscale is False
