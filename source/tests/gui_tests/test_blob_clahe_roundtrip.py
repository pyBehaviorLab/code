"""CLAHE / adaptive-threshold blob settings must reach the live tracker.

``use_clahe`` / ``clahe_clip_limit`` / ``clahe_tile_size`` /
``use_adaptive_threshold`` are user-settable in the tracking dialog, persisted
in ``BlobConfig`` (project YAML via ``asdict``), and consumed by ``blob.py``.
Every hop between those two ends has to carry them: a framebus
``TrackingConfig`` without the fields drops them silently, CLAHE
worked in the dialog preview but vanished at Record. This pins the full path
cfg.tracking.blob → framebus TC → settings dict.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from source.config.experiment import (
    BoxConfig, Config,
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


def test_clahe_survives_config_to_pipeline():
    """Operator enables CLAHE + adaptive threshold in blob mode; the live
    framebus TC must carry them across the hop."""
    host, pipe = _build_host_with_pipeline()
    cfg = Config()
    cfg.setup_config.boxes = [BoxConfig(setup_number=1)]
    cfg.tracking.enabled = True
    cfg.tracking.mode = "blob"
    cfg.tracking.blob.use_clahe = True
    cfg.tracking.blob.clahe_clip_limit = 4.5
    cfg.tracking.blob.clahe_tile_size = 16
    cfg.tracking.blob.use_adaptive_threshold = True

    _apply_tracking(cfg, host)
    tc = pipe._tracking_configs[1]
    assert tc.blob_use_clahe is True
    assert tc.blob_clahe_clip_limit == 4.5
    assert tc.blob_clahe_tile_size == 16
    assert tc.blob_use_adaptive_threshold is True


def test_clahe_survives_pipeline_to_config():
    """Read-back (save path) must preserve CLAHE settings."""
    host, pipe = _build_host_with_pipeline()
    cfg = Config()
    cfg.setup_config.boxes = [BoxConfig(setup_number=1)]
    cfg.tracking.enabled = True
    cfg.tracking.mode = "blob"
    cfg.tracking.blob.use_clahe = True
    cfg.tracking.blob.clahe_clip_limit = 2.0
    cfg.tracking.blob.clahe_tile_size = 4
    cfg.tracking.blob.use_adaptive_threshold = True

    _apply_tracking(cfg, host)
    cfg2 = Config()
    cfg2.setup_config.boxes = [BoxConfig(setup_number=1)]
    _read_tracking(host, cfg2)
    assert cfg2.tracking.blob.use_clahe is True
    assert cfg2.tracking.blob.clahe_clip_limit == 2.0
    assert cfg2.tracking.blob.clahe_tile_size == 4
    assert cfg2.tracking.blob.use_adaptive_threshold is True

    # And through the project-YAML compaction + reload.
    saved = cfg2.tracking.to_compact()
    assert saved["blob"]["use_clahe"] is True
    assert saved["blob"]["use_adaptive_threshold"] is True
    cfg3 = Config()
    cfg3.tracking = ProjectTrackingConfig.from_dict(saved)
    assert cfg3.tracking.blob.use_clahe is True
    assert cfg3.tracking.blob.clahe_clip_limit == 2.0


def test_framebus_tc_clahe_json_roundtrip():
    """The framebus TC itself must serialise + restore the CLAHE fields."""
    tc = PipeTC(setup_id=1, tracker_type="blob",
                blob_use_clahe=True, blob_clahe_clip_limit=5.0,
                blob_clahe_tile_size=12, blob_use_adaptive_threshold=True)
    restored = PipeTC.from_json(tc.to_json())
    assert restored.blob_use_clahe is True
    assert restored.blob_clahe_clip_limit == 5.0
    assert restored.blob_clahe_tile_size == 12
    assert restored.blob_use_adaptive_threshold is True


def test_framebus_tc_clahe_defaults():
    """Old TC JSON without CLAHE keys loads with safe defaults (off)."""
    restored = PipeTC.from_json({"box_id": 1, "tracker_type": "blob"})
    assert restored.blob_use_clahe is False
    assert restored.blob_clahe_clip_limit == 3.0
    assert restored.blob_clahe_tile_size == 8
    assert restored.blob_use_adaptive_threshold is False
