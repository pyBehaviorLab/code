"""DLC / SLEAP config must survive a project save → load round-trip.

Reproduces the reported bug: after configuring DLC and saving the project,
reload shows an EMPTY tracking block (mode=none, dlc={}). Drives the real
save path (`_read_tracking` → `Config.to_dict` → `Config.from_dict` →
`_apply_tracking`) against a pipeline holding a TC exactly as
`_apply_dialog_config` would have installed it.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from source.config.experiment import (
    Config, BoxConfig, _read_tracking, _apply_tracking,
)
from source.video.framebus.controller import Pipeline
from source.video.framebus.types import TrackingConfig as PipeTC


def _pipe_with(tc_kwargs):
    pipe = Pipeline.__new__(Pipeline)
    pipe._tracking_configs = {}
    pipe._registered_boxes = set(); pipe._pycboards = {}; pipe._buses = {}
    pipe._box_unsubs = {}
    pipe._on_pose_result = []; pipe._on_tracker_result = []
    pipe._on_health = []; pipe._on_pose_failed = []
    pipe.recorder = MagicMock(); pipe.pose = MagicMock()
    pipe.tracker = MagicMock(); pipe.push = MagicMock()
    pipe._display_sinks = []
    pipe._tracking_configs[1] = PipeTC(setup_id=1, **tc_kwargs)
    host = MagicMock(); host.pipeline = pipe; host.tracking_zones = {}
    return host, pipe


def _save_then_load(host):
    """Run the real save (read live → cfg → on-disk dict) then load it back."""
    cfg = Config(); cfg.setup_config.boxes = [BoxConfig(setup_number=1)]
    _read_tracking(host, cfg)
    disk = cfg.to_dict()                       # what gets written to disk
    reloaded = Config.from_dict(disk)          # what comes back on load
    return cfg, disk, reloaded


# ── DLC: the operator picked a model and ticked the box ──────────────

def test_dlc_path_survives_when_box_enabled():
    host, _ = _pipe_with(dict(
        tracker_type="dlc", dlc_model_path="D:/models/Multimaze",
        keypoint_names=("Head", "Center", "Tailbase"),
        zone_change_body_part="Center",
        user_applied=True, online_tracking_enabled=True))
    cfg, disk, reloaded = _save_then_load(host)
    assert disk["tracking"]["mode"] == "dlc", disk["tracking"]
    assert disk["tracking"]["dlc"].get("model_path") == "D:/models/Multimaze"
    assert reloaded.tracking.dlc.model_path == "D:/models/Multimaze"
    assert reloaded.tracking.zone_change_body_part == "Center"


# ── DLC: model picked, but the per-box checkbox is OFF (rig-level) ────
# This is the suspected real case: pose configured at rig level, no box
# ticked → online_tracking_enabled False. The path must STILL persist.

def test_dlc_path_survives_when_box_not_enabled():
    host, _ = _pipe_with(dict(
        tracker_type="dlc", dlc_model_path="D:/models/Multimaze",
        keypoint_names=("Head", "Center", "Tailbase"),
        zone_change_body_part="Center",
        user_applied=True, online_tracking_enabled=False))
    cfg, disk, reloaded = _save_then_load(host)
    assert disk["tracking"]["mode"] == "dlc", disk["tracking"]
    assert reloaded.tracking.dlc.model_path == "D:/models/Multimaze"


# ── DLC: model loaded but user_applied somehow False (auto-init path) ─

def test_dlc_path_survives_without_user_applied_flag():
    host, _ = _pipe_with(dict(
        tracker_type="dlc", dlc_model_path="D:/models/Multimaze",
        keypoint_names=("Head", "Center", "Tailbase"),
        user_applied=False, online_tracking_enabled=False))
    cfg, disk, reloaded = _save_then_load(host)
    assert disk["tracking"]["mode"] == "dlc", disk["tracking"]
    assert reloaded.tracking.dlc.model_path == "D:/models/Multimaze"


# ── SLEAP round-trip ─────────────────────────────────────────────────

def test_sleap_path_survives():
    host, _ = _pipe_with(dict(
        tracker_type="sleap", dlc_model_path="D:/models/sleap.zip",
        keypoint_names=("Head", "Center"),
        user_applied=True, online_tracking_enabled=True))
    cfg, disk, reloaded = _save_then_load(host)
    assert disk["tracking"]["mode"] == "sleap", disk["tracking"]
    assert reloaded.tracking.sleap.model_path == "D:/models/sleap.zip"


# ── Full load installs the path back into the pipeline TC ────────────

def test_loaded_dlc_reinstalls_into_pipeline():
    host, _ = _pipe_with(dict(
        tracker_type="dlc", dlc_model_path="D:/models/Multimaze",
        keypoint_names=("Head", "Center", "Tailbase"),
        user_applied=True, online_tracking_enabled=True))
    _, _, reloaded = _save_then_load(host)
    # New session: empty pipeline, apply the reloaded cfg.
    host2, pipe2 = _pipe_with(dict(tracker_type="dlc"))
    pipe2._tracking_configs.clear()
    pipe2.install_tracking_configs = MagicMock(side_effect=lambda payload: [
        pipe2._tracking_configs.__setitem__(int(k), PipeTC.from_json(v, setup_id=int(k)))
        for k, v in payload.items()])
    reloaded.setup_config.boxes = [BoxConfig(setup_number=1)]
    _apply_tracking(reloaded, host2)
    tc = pipe2._tracking_configs.get(1)
    assert tc is not None and tc.dlc_model_path == "D:/models/Multimaze"
