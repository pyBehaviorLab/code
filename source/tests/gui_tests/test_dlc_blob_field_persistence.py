"""Every field that flows dialog → TC → project YAML must round-trip.

Pins the full save → load → save cycle for pose resize/instances and
blob smooth_tracking.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from source.config.experiment import (
    BoxConfig, Config, BlobConfig, DLCConfig, SleapConfig,
    TrackingConfig as ProjectTrackingConfig,
    _apply_tracking, _read_tracking,
)
from source.video.framebus.controller import Pipeline
from source.video.framebus.types import TrackingConfig as PipeTC


def _build_host_with_pipeline():
    pipe = Pipeline.__new__(Pipeline)
    pipe._tracking_configs = {}
    pipe._registered_boxes = set(); pipe._pycboards = {}; pipe._buses = {}
    pipe._box_unsubs = {}
    pipe._on_pose_result = []; pipe._on_tracker_result = []
    pipe._on_health = []; pipe._on_pose_failed = []
    pipe.recorder = MagicMock(); pipe.pose = MagicMock()
    pipe.tracker = MagicMock(); pipe.push = MagicMock()
    pipe._display_sinks = []
    host = MagicMock(); host.pipeline = pipe; host.tracking_zones = {}
    return host, pipe


# ─── DLC pose_resize_factor + pose_n_instances ──────────────────────


def test_dlc_resize_persists_through_save_load():
    """DLC resize/instances must survive save and reload, not reset to
    dataclass defaults."""
    host, pipe = _build_host_with_pipeline()
    cfg = Config()
    cfg.setup_config.boxes = [BoxConfig(setup_number=1)]
    cfg.tracking.enabled = True
    cfg.tracking.mode = "dlc"
    cfg.tracking.dlc.model_path = "/m.yml"
    cfg.tracking.dlc.resize = 0.5
    cfg.tracking.dlc.instances = 3

    # Step 1: install into live pipeline.
    _apply_tracking(cfg, host)
    tc = pipe._tracking_configs[1]
    assert tc.pose_resize_factor == 0.5
    assert tc.pose_n_instances == 3

    # Step 2: read back into a fresh cfg (the save path).
    cfg2 = Config()
    cfg2.setup_config.boxes = [BoxConfig(setup_number=1)]
    _read_tracking(host, cfg2)
    assert cfg2.tracking.dlc.resize == 0.5
    assert cfg2.tracking.dlc.instances == 3

    # Step 3: project save → YAML dict → reload → install.
    saved = cfg2.tracking.to_compact()
    assert saved["dlc"]["resize"] == 0.5
    assert saved["dlc"]["instances"] == 3

    # Roundtrip via from_dict.
    cfg3 = Config()
    cfg3.setup_config.boxes = [BoxConfig(setup_number=1)]
    cfg3.tracking = ProjectTrackingConfig.from_dict(saved)
    assert cfg3.tracking.dlc.resize == 0.5
    assert cfg3.tracking.dlc.instances == 3

    # Step 4: install again, confirm propagation.
    host2, pipe2 = _build_host_with_pipeline()
    _apply_tracking(cfg3, host2)
    tc3 = pipe2._tracking_configs[1]
    assert tc3.pose_resize_factor == 0.5
    assert tc3.pose_n_instances == 3


def test_sleap_resize_and_instances_persist():
    """SLEAP subconfig must also carry resize/instances."""
    host, pipe = _build_host_with_pipeline()
    cfg = Config()
    cfg.setup_config.boxes = [BoxConfig(setup_number=1)]
    cfg.tracking.enabled = True
    cfg.tracking.mode = "sleap"
    cfg.tracking.sleap.model_path = "/m.zip"
    cfg.tracking.sleap.resize = 0.75
    cfg.tracking.sleap.instances = 2

    _apply_tracking(cfg, host)
    tc = pipe._tracking_configs[1]
    assert tc.pose_resize_factor == 0.75
    assert tc.pose_n_instances == 2

    cfg2 = Config()
    cfg2.setup_config.boxes = [BoxConfig(setup_number=1)]
    _read_tracking(host, cfg2)
    assert cfg2.tracking.sleap.resize == 0.75
    assert cfg2.tracking.sleap.instances == 2


# ─── Blob smooth_tracking ───────────────────────────────────────────


def test_blob_smooth_tracking_persists():
    """User enables smooth_tracking in blob mode; reload must keep it on."""
    host, pipe = _build_host_with_pipeline()
    cfg = Config()
    cfg.setup_config.boxes = [BoxConfig(setup_number=1)]
    cfg.tracking.enabled = True
    cfg.tracking.mode = "blob"
    cfg.tracking.blob.smooth_tracking = True

    _apply_tracking(cfg, host)
    tc = pipe._tracking_configs[1]
    assert tc.blob_smooth_tracking is True

    cfg2 = Config()
    cfg2.setup_config.boxes = [BoxConfig(setup_number=1)]
    _read_tracking(host, cfg2)
    assert cfg2.tracking.blob.smooth_tracking is True

    saved = cfg2.tracking.to_compact()
    assert saved["blob"]["smooth_tracking"] is True

    cfg3 = Config()
    cfg3.tracking = ProjectTrackingConfig.from_dict(saved)
    assert cfg3.tracking.blob.smooth_tracking is True


# ─── DLCConfig / SleapConfig dataclass defaults ─────────────────────


def test_dlc_config_defaults():
    cfg = DLCConfig()
    assert cfg.resize == 1.0
    assert cfg.instances == 1


def test_sleap_config_defaults():
    cfg = SleapConfig()
    assert cfg.resize == 1.0
    assert cfg.instances == 1


def test_blob_config_smooth_tracking_default():
    """ON, matching the dialog checkbox, which has always been checked.

    A stored default of False meant a project saved before the key existed
    loaded with smoothing off while the UI showed it on, and the operator
    had no way to tell which one the run actually used.
    """
    cfg = BlobConfig()
    assert cfg.smooth_tracking is True


def test_blob_config_smooth_tracking_can_still_be_turned_off():
    """On by DEFAULT is not on regardless."""
    assert BlobConfig.from_dict({"smooth_tracking": False}).smooth_tracking is False


# ─── DLCConfig.from_dict back-compat ────────────────────────────────


def test_dlc_config_from_dict_uses_defaults_for_old_yaml():
    """Old project YAML without resize/instances keys still loads
    cleanly with dataclass defaults."""
    cfg = DLCConfig.from_dict({"model_path": "/m.yml", "confidence": 0.7})
    assert cfg.resize == 1.0
    assert cfg.instances == 1
    assert cfg.confidence == 0.7


def test_dlc_config_from_dict_reads_resize_and_instances():
    cfg = DLCConfig.from_dict({"resize": 0.5, "instances": 4})
    assert cfg.resize == 0.5
    assert cfg.instances == 4
