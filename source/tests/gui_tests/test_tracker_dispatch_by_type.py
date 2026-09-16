"""Wave 3: Test-Tracking dispatches on the operator's tracker_type (T1/T4),
surfaces failures (T3), and never silently downgrades DLC to blob.

Drives ``MainWindowBase._start_tracking_for_box_impl`` directly with a stub
pipeline / TrackingConfig, no Qt, no real models.
"""
from types import SimpleNamespace

from source.gui.base import MainWindowBase
from source.video.framebus.types import TrackingConfig


class _TC:
    def __init__(self, tracker_type="blob", model="", zones=False, bg=False):
        self.tracker_type = tracker_type
        self.dlc_model_path = model
        self.keypoint_names = ("head", "tail")
        self.confidence_threshold = 0.5
        self.pose_resize_factor = 1.0
        self.online_tracking_enabled = True
        self._zones = zones
        # Real field the dispatch reads for the pose-no-model→blob fallback.
        self.blob_background_path = "/bg/box1.png" if bg else ""
        # SLEAP option fields the dispatch threads through for pose mode.
        self.sleap_model_type = "auto"
        self.sleap_centroid_path = "/models/centroid"
        self.sleap_runtime = "auto"
        self.sleap_device = "auto"
        self.sleap_fp16 = False
        self.sleap_compile = False
        self.sleap_peak_threshold = 0.2
        # DLC engine options + the pose-input group, which the one builder
        # carries and the old per-branch dicts dropped.
        self.dlc_model_type = "auto"
        self.dlc_precision = "FP32"
        self.dlc_device = "auto"
        self.dlc_dynamic_threshold = 0.5
        self.dlc_dynamic_margin = 10
        self.pose_colour_mode = "auto"
        self.pose_input_mode = "auto"
        self.pose_input_w = 0
        self.pose_input_h = 0
        self.pose_crop_conf_min = 0.20
        self.pose_crop_good_min = 3
        self.pose_crop_reacquire = True
        self.pose_gap_frames = 5
        self.pose_n_instances = 1
        self.skeleton = ()
        self.blob_smooth_tracking = True
        self.setup_id = 1

    # The REAL derivations, not a copy: the sink's cache key is built from
    # these same two methods, so a stub with its own version here would be
    # the very drift the one builder exists to remove.
    derived_sleap_opts = TrackingConfig.derived_sleap_opts
    derived_dlc_opts = TrackingConfig.derived_dlc_opts

    def has_dlc(self):
        return bool(self.dlc_model_path)

    def has_blob(self):
        return bool(self.blob_background_path) or bool(self._zones)


def _host(tc):
    calls = {"pose": [], "blob": [], "fail": []}
    # The dispatch takes the box's settings from the ONE builder now, rather
    # than hand-building a dict per branch. Each branch built a different one
    # and each left out a different set of fields, which is how the model the
    # run started with stopped matching the model that was prepared.
    from source.gui.pose_subsystem import pose_settings_from_config
    h = SimpleNamespace(
        pipeline=SimpleNamespace(
            get_tracking_config=lambda sid: tc,
            enable_blob_tracking=lambda sid: calls["blob"].append(sid),
            has_pose_model=lambda: False,
            pose_fingerprint=lambda: None,
        ),
        pose_settings_for_box=lambda sid: pose_settings_from_config(tc),
        tracking_enabled={},
        tracking_zones={},
        _enable_pose_for_box=lambda sid, cfg: (
            calls["pose"].append((sid, cfg)) or True),
        _note_tracking_start_failure=lambda sid, why: calls["fail"].append((sid, why)),
        _setup_blob_tracker_for_box=lambda *a, **k: None,
        _tracking_config_to_settings_dict=lambda sid: {},
        _rebuild_zone_manager=lambda sid, z: None,
        _apply_push_policy=lambda sid: None,
    )
    h._calls = calls
    return h


def _run(tc):
    h = _host(tc)
    ok = MainWindowBase._start_tracking_for_box_impl(h, 1, force=True)
    return ok, h._calls


def test_dlc_with_model_runs_pose():
    ok, calls = _run(_TC("dlc", model="/m/model.yaml", zones=True))
    assert ok is True
    assert calls["pose"] and calls["pose"][0][1]["mode"] == "dlc"
    assert not calls["blob"]


def test_sleap_with_model_runs_pose_as_sleap():
    # T4: SLEAP must reach the pose path with mode "sleap", not forced to dlc,
    # AND carry its SLEAP options (centroid path etc.) through.
    ok, calls = _run(_TC("sleap", model="/m/sleap_dir", zones=True))
    assert ok is True
    cfg = calls["pose"][0][1]
    assert cfg["mode"] == "sleap"
    assert cfg["sleap_opts"]["centroid_path"] == "/models/centroid"  # not dropped


def test_dlc_choice_with_zones_but_no_model_does_not_run_blob():
    # T1: the exact reported bug, DLC selected, zones drawn, model path lost.
    # Must NOT silently run blob; must surface a failure.
    ok, calls = _run(_TC("dlc", model="", zones=True))
    assert ok is False
    assert not calls["blob"]          # never downgraded to blob
    assert not calls["pose"]
    assert calls["fail"] and "no model" in calls["fail"][0][1].lower()


def test_blob_choice_runs_blob():
    ok, calls = _run(_TC("blob", zones=True))
    assert ok is True
    assert calls["blob"] == [1]
    assert not calls["pose"]


def test_dlc_choice_no_model_but_blob_configured_falls_back_to_blob():
    # A DLC box with no model but a blob background is still usable as blob.
    ok, calls = _run(_TC("dlc", model="", bg=True))
    assert ok is True
    assert calls["blob"] == [1]


def test_legacy_tc_bare_model_no_tracker_type_runs_pose():
    ok, calls = _run(_TC("", model="/m/model.yaml"))
    assert ok is True
    assert calls["pose"] and calls["pose"][0][1]["mode"] == "dlc"


def test_nothing_configured_surfaces_failure():
    ok, calls = _run(_TC("blob", zones=False, bg=False))
    assert ok is False
    assert calls["fail"] and "nothing configured" in calls["fail"][0][1].lower()
