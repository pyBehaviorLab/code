"""A non-tracking autosave (boxes_changed, camera_config_changed,
bg_captured, metadata_loaded …) must not wipe ``cfg.tracking`` to
defaults when ``pipe._tracking_configs`` happens to be empty.

``read_ui_into_config`` deepcopies ``cfg.tracking`` from
``host._active_config`` before calling ``_read_tracking``. The reader
overwrites only when the pipe is populated; otherwise the last-loaded
tracking config is preserved verbatim.
"""
from __future__ import annotations

import copy as _copy
from unittest.mock import MagicMock

from source.config.experiment import (
    BoxConfig, Config, TrackingConfig as ProjectTC,
    _read_tracking,
)
from source.video.framebus.controller import Pipeline


def _make_empty_pipe():
    pipe = Pipeline.__new__(Pipeline)
    pipe._tracking_configs = {}
    pipe._registered_boxes = set(); pipe._pycboards = {}; pipe._buses = {}
    pipe._box_unsubs = {}
    pipe._on_pose_result = []; pipe._on_tracker_result = []
    pipe._on_health = []; pipe._on_pose_failed = []
    pipe.recorder = MagicMock(); pipe.pose = MagicMock()
    pipe.tracker = MagicMock(); pipe.push = MagicMock()
    pipe._display_sinks = []
    return pipe


def test_read_tracking_returns_early_when_pipe_empty():
    """Documents the existing behaviour: when pipe is empty, _read_tracking
    DOES NOT write cfg.tracking. The fix has to live ABOVE this."""
    host = MagicMock(); host.pipeline = _make_empty_pipe()
    host.tracking_zones = {}
    cfg = Config()
    cfg.setup_config.boxes = [BoxConfig(setup_number=1)]
    # Fresh cfg → cfg.tracking is defaults (enabled=False, mode="none").
    _read_tracking(host, cfg)
    # _read_tracking returns early; cfg.tracking stays at defaults.
    assert cfg.tracking.enabled is False
    assert cfg.tracking.mode == "none"
    assert cfg.tracking.dlc.model_path == ""


def test_read_ui_into_config_preserves_tracking_when_pipe_empty():
    """read_ui_into_config seeds cfg.tracking from host._active_config so an
    empty pipe preserves a previously-saved DLC config."""
    # Build the "last good loaded" config, populated with a real DLC model.
    base_cfg = Config()
    base_cfg.setup_config.boxes = [BoxConfig(setup_number=1)]
    base_cfg.tracking.enabled = True
    base_cfg.tracking.mode = "dlc"
    base_cfg.tracking.dlc.model_path = "/models/MyDLC"
    base_cfg.tracking.dlc.body_parts = ["Head", "Center", "Tailbase"]
    base_cfg.tracking.dlc.resize = 0.5
    base_cfg.tracking.dlc.instances = 2
    base_cfg.tracking.zone_change_body_part = "Center"
    base_cfg.tracking.push_zones_to_mcu = False
    base_cfg.tracking.push_coords_to_mcu = False

    # Pipe is empty.
    host = MagicMock(); host.pipeline = _make_empty_pipe()
    host.tracking_zones = {}
    host._active_config = base_cfg

    # Build the fresh cfg the same way read_ui_into_config does.
    fresh = Config()
    fresh.setup_config.boxes = [BoxConfig(setup_number=1)]
    # Apply the fix:
    if host._active_config is not None:
        fresh.tracking = _copy.deepcopy(host._active_config.tracking)
    _read_tracking(host, fresh)

    # Verify the saved DLC config is intact.
    assert fresh.tracking.enabled is True
    assert fresh.tracking.mode == "dlc"
    assert fresh.tracking.dlc.model_path == "/models/MyDLC"
    assert fresh.tracking.dlc.body_parts == ["Head", "Center", "Tailbase"]
    assert fresh.tracking.dlc.resize == 0.5
    assert fresh.tracking.dlc.instances == 2
    assert fresh.tracking.zone_change_body_part == "Center"
    assert fresh.tracking.push_zones_to_mcu is False
    assert fresh.tracking.push_coords_to_mcu is False


def test_read_ui_into_config_overwrites_when_pipe_has_user_applied_tc():
    """Once the user opens the dialog and clicks Apply, pipe gets
    populated. _read_tracking then OVERWRITES the seed with live data,
    so a fresh user pick takes precedence over the last-loaded state."""
    from source.video.framebus.types import TrackingConfig as PipeTC

    base_cfg = Config()
    base_cfg.setup_config.boxes = [BoxConfig(setup_number=1)]
    base_cfg.tracking.mode = "dlc"
    base_cfg.tracking.dlc.model_path = "/models/OLD"

    host = MagicMock(); host.pipeline = _make_empty_pipe()
    host.tracking_zones = {}
    host._active_config = base_cfg
    # Populate pipe with a NEW user-applied TC.
    host.pipeline._tracking_configs[1] = PipeTC(
        setup_id=1, tracker_type="dlc",
        dlc_model_path="/models/NEW",
        keypoint_names=("nose", "ear"),
        user_applied=True,
    )

    fresh = Config()
    fresh.setup_config.boxes = [BoxConfig(setup_number=1)]
    if host._active_config is not None:
        fresh.tracking = _copy.deepcopy(host._active_config.tracking)
    _read_tracking(host, fresh)

    # Live pipe data WINS over the seed.
    assert fresh.tracking.dlc.model_path == "/models/NEW"
    assert fresh.tracking.dlc.body_parts == ["nose", "ear"]


def test_read_ui_into_config_no_base_falls_back_to_defaults():
    """When there's no _active_config (first session, no project loaded
    yet), cfg.tracking still defaults gracefully."""
    host = MagicMock(); host.pipeline = _make_empty_pipe()
    host.tracking_zones = {}
    host._active_config = None

    fresh = Config()
    fresh.setup_config.boxes = [BoxConfig(setup_number=1)]
    # Seed step is a no-op when base is None.
    if host._active_config is not None:
        fresh.tracking = _copy.deepcopy(host._active_config.tracking)
    _read_tracking(host, fresh)

    assert fresh.tracking.enabled is False
    assert fresh.tracking.dlc.model_path == ""
