"""DLC configured AFTER project creation must survive save → reload.

The recurring complaint: create a project without tracking, configure DLC
later, reload, and the model path is gone / not initialized. These exercise
the real persistence functions (``_read_tracking`` on save, ``to_compact`` /
``from_dict`` on disk, ``_apply_tracking`` on reload, plus the durable
``sync_active_config_tracking`` mirror) with a real Pipeline so the
round-trip is proven without a camera or Qt event loop.
"""
from __future__ import annotations

import source.config.experiment as E
from source.config.experiment import (
    Config, BoxConfig, TrackingConfig, sync_active_config_tracking,
)
from source.video.framebus.controller import Pipeline


class _Host:
    pass


def _host_with_pipeline():
    h = _Host()
    pipe = Pipeline.__new__(Pipeline)
    pipe._tracking_configs = {}
    h.pipeline = pipe
    h.tracking_zones = {}
    return h, pipe


def _configure_dlc(pipe, box=1, model=r"D:/models/mymodel"):
    """Mirror what _apply_dialog_config does when the user applies DLC."""
    pipe.update_tracking_config(
        box, tracker_type="dlc", user_applied=True,
        online_tracking_enabled=False, dlc_model_path=model,
        keypoint_names=("nose", "tail"), confidence_threshold=0.6,
        pose_resize_factor=0.5, pose_n_instances=1)


def test_configure_later_saves_and_reloads_dlc(tmp_path):
    h, pipe = _host_with_pipeline()

    # Project created WITHOUT dlc.
    active = Config()
    active.setup_config.boxes = [BoxConfig(setup_number=1)]
    assert active.tracking.mode == "none"

    # User configures DLC later, then applies → the durable mirror updates.
    _configure_dlc(pipe)
    h._active_config = active
    sync_active_config_tracking(h)
    assert h._active_config.tracking.mode == "dlc"
    assert h._active_config.tracking.dlc.model_path == r"D:/models/mymodel"

    # Autosave: fresh Config seeded from _active_config, then _read_tracking.
    save = Config()
    save.setup_config.boxes = [BoxConfig(setup_number=1)]
    save.tracking = h._active_config.tracking
    E._read_tracking(h, save)
    assert save.tracking.enabled and save.tracking.mode == "dlc"

    # Disk round-trip.
    disk = save.tracking.to_compact()
    assert disk["dlc"]["model_path"] == r"D:/models/mymodel"
    reloaded = TrackingConfig.from_dict(disk)
    assert reloaded.enabled and reloaded.mode == "dlc"
    assert reloaded.dlc.model_path == r"D:/models/mymodel"

    # Reload: _apply_tracking installs the DLC TC so the dialog + auto-init
    # can find the model path.
    h2, pipe2 = _host_with_pipeline()
    cfg2 = Config()
    cfg2.setup_config.boxes = [BoxConfig(setup_number=1)]
    cfg2.tracking = reloaded
    E._apply_tracking(cfg2, h2)
    tc = pipe2.get_tracking_config(1)
    assert tc.has_dlc()
    assert tc.dlc_model_path == r"D:/models/mymodel"


def test_transient_empty_pipeline_does_not_drop_saved_dlc():
    """Once mirrored into _active_config, a save while the pipeline is
    momentarily empty (e.g. rebuilt on camera reconnect) keeps the DLC."""
    h, pipe = _host_with_pipeline()
    active = Config()
    active.setup_config.boxes = [BoxConfig(setup_number=1)]
    _configure_dlc(pipe)
    h._active_config = active
    sync_active_config_tracking(h)          # DLC now durable

    # Pipeline gets wiped (reconnect rebuild), TC registry empty.
    pipe._tracking_configs = {}

    save = Config()
    save.setup_config.boxes = [BoxConfig(setup_number=1)]
    save.tracking = h._active_config.tracking   # seed carries DLC
    E._read_tracking(h, save)                    # empty pipe must not clear it
    assert save.tracking.mode == "dlc"
    assert save.tracking.dlc.model_path == r"D:/models/mymodel"
