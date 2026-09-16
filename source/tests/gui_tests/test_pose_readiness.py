"""A box that was prepared must report ready, and a run start must not load.

The fault this pins: the host DESCRIBED the loaded model in its own second
implementation, which left out the DeepLabCut engine options and the pose-input
group, so the description could never equal the sink's. Measured on the real
classes before the fix:

    host  ('D:/models/m','dlc',1.0,0.5,('snout','tail'), '')
    sink  ('D:/models/m','dlc',1.0,0.5,('snout','tail'),
           'dlc_model_type=auto|precision=FP32|device=auto')

Every DLC box therefore answered "needs init" on every Record click, and the
model was rebuilt with the animal already in the box.

There is now ONE function, :func:`pose_fingerprint`. The sink stores what it
was called with; the host builds the same tuple from the box config. These
tests hold that arrangement in place: if a future option reaches the model
without reaching the fingerprint, the round trip below stops matching.
"""
from __future__ import annotations

import numpy as np
import pytest

from source.gui.pose_subsystem import (PoseSubsystemMixin, pose_ready_for,
                                       pose_settings_differ,
                                       pose_settings_from_config,
                                       settings_fingerprint)
from source.video.framebus.pose_sink import PoseSink
from source.video.framebus.types import TrackingConfig


class _Model:
    def get_body_parts(self):
        return ["snout", "tail"]


class _Handle:
    def __init__(self, key):
        self.key = key
        self.model = _Model()


class _Backend:
    """Stands in for the inference backend: no torch, no GPU, no model file."""

    def __init__(self):
        self.calls = []

    def get_or_create(self, key, **kw):
        self.calls.append((key, kw))
        return _Handle(key)


def _sink():
    return PoseSink(backend=_Backend())


def _load(sink, settings, *, n_boxes=1):
    """Load exactly the way ``_configure_pose_for_box`` does."""
    p = PoseSubsystemMixin._pose_model_params(settings)
    return sink.configure_model(
        tracker_type=p["mode"], model_path=settings["model_path"],
        probe_frame=np.zeros((176, 315, 3), np.uint8),
        resize_factor=p["resize"], body_parts=settings["body_parts"],
        confidence=p["confidence"], sleap_opts=p["sleap_opts"],
        dlc_opts=p["dlc_opts"], colour_mode=p["colour_mode"],
        input_mode=p["input_mode"], input_wh=p["input_wh"],
        crop_opts=p["crop_opts"], n_boxes=n_boxes)


def _dlc_config(**over):
    fields = dict(setup_id=1, tracker_type="dlc", dlc_model_path="D:/models/m",
                  keypoint_names=("snout", "tail"))
    fields.update(over)
    return TrackingConfig(**fields)


def _sleap_config(**over):
    fields = dict(setup_id=2, tracker_type="sleap",
                  dlc_model_path="D:/models/s",
                  keypoint_names=("snout", "tail"))
    fields.update(over)
    return TrackingConfig(**fields)


class _Pipeline:
    """Only the two calls the readiness check makes."""

    def __init__(self, sink):
        self.pose = sink

    def has_pose_model(self):
        return self.pose.fingerprint() is not None

    def pose_fingerprint(self):
        return self.pose.fingerprint()


# ── the round trip ────────────────────────────────────────────────────


@pytest.mark.parametrize("config", [_dlc_config(), _sleap_config()])
def test_a_loaded_model_reports_ready_for_the_config_it_came_from(config):
    """The one that was broken. Both backends, because each failed for its own
    reason: DLC over the engine options, SLEAP over the injected batch size."""
    sink = _sink()
    settings = pose_settings_from_config(config)
    _load(sink, settings)
    assert sink.fingerprint() == settings_fingerprint(settings)
    assert pose_ready_for(_Pipeline(sink), settings) is True


def test_the_batch_size_does_not_decide_readiness():
    """It comes from how many boxes the rig has, not from anything the operator
    set. Folding it in made a 16-box rig disagree with its own model."""
    sink = _sink()
    settings = pose_settings_from_config(_sleap_config())
    _load(sink, settings, n_boxes=16)
    assert pose_ready_for(_Pipeline(sink), settings) is True


def test_no_model_loaded_is_not_ready():
    assert pose_ready_for(_Pipeline(_sink()),
                          pose_settings_from_config(_dlc_config())) is False


def test_a_config_with_no_model_path_is_never_ready():
    settings = pose_settings_from_config(_dlc_config(dlc_model_path=None))
    assert settings_fingerprint(settings) is None
    assert pose_ready_for(_Pipeline(_sink()), settings) is False


# ── what must and must not force a re-load ────────────────────────────


@pytest.mark.parametrize("field,value,named", [
    ("dlc_model_path", "D:/models/other", "model_path"),
    ("confidence_threshold", 0.9, "confidence"),
    ("pose_resize_factor", 0.5, "resize"),
    ("keypoint_names", ("snout", "tail", "ear"), "body_parts"),
    ("dlc_precision", "FP16", "dlc_opts"),
    ("dlc_device", "cuda:1", "dlc_opts"),
    ("dlc_model_type", "pytorch", "dlc_opts"),
    ("pose_colour_mode", "grayscale", "colour_mode"),
    ("pose_input_mode", "crop_track", "input_mode"),
    ("pose_input_w", 320, "input_size"),
    ("pose_crop_conf_min", 0.5, "crop_opts"),
])
def test_changing_a_model_setting_needs_a_reload(field, value, named):
    """Each of these changes what the network is or how it is built, so each
    must make the box not-ready and be NAMED in the log line. The four DLC
    engine rows are the ones the old comparison could not see at all."""
    sink = _sink()
    _load(sink, pose_settings_from_config(_dlc_config()))
    after = pose_settings_from_config(_dlc_config(**{field: value}))
    assert pose_ready_for(_Pipeline(sink), after) is False
    assert named in pose_settings_differ(sink.fingerprint(), after)


@pytest.mark.parametrize("field,value", [
    ("pose_n_instances", 4),
    ("push_zones_to_mcu", False),
    ("push_coords_to_mcu", False),
    ("online_tracking_enabled", True),
    ("zone_change_body_part", "snout"),
    ("annotation_enabled", True),
])
def test_a_runtime_knob_does_not_need_a_reload(field, value):
    """Asking for an init after a zone or push change would be the same fault
    in the other direction: a modal for something the model does not care
    about."""
    sink = _sink()
    _load(sink, pose_settings_from_config(_dlc_config()))
    after = pose_settings_from_config(_dlc_config(**{field: value}))
    assert pose_ready_for(_Pipeline(sink), after) is True


def test_the_loader_reuses_the_model_when_nothing_changed():
    """The cache key and the fingerprint must agree about "same model", or a
    ready box still pays for a rebuild. Two loads, one backend call."""
    sink = _sink()
    settings = pose_settings_from_config(_dlc_config())
    _load(sink, settings)
    _load(sink, settings)
    assert len(sink._backend.calls) == 2          # asked twice
    assert sink._backend.calls[0][0] == sink._backend.calls[1][0]   # same key


# ── the settings builder ──────────────────────────────────────────────


def test_the_builder_carries_the_engine_options():
    """They reached the model from none of the five old builders, so a project
    saved as FP16 ran FP32 in silence."""
    settings = pose_settings_from_config(
        _dlc_config(dlc_precision="FP16", dlc_device="cuda:1"))
    assert settings["dlc_opts"]["precision"] == "FP16"
    assert settings["dlc_opts"]["device"] == "cuda:1"
    assert settings["sleap_opts"] == {}          # never handed to DLC


def test_the_builder_carries_the_sleap_options():
    settings = pose_settings_from_config(
        _sleap_config(sleap_centroid_path="D:/models/c", sleap_fp16=True))
    assert settings["sleap_opts"]["centroid_path"] == "D:/models/c"
    assert settings["sleap_opts"]["fp16"] is True
    assert settings["dlc_opts"] == {}            # never handed to SLEAP


def test_the_builder_takes_the_options_from_the_config_itself():
    """One derivation. The sink's cache key is built from these same two
    methods, so a hand-written copy here is exactly how the two came to
    describe one model differently."""
    tc = _dlc_config(dlc_precision="FP16")
    settings = pose_settings_from_config(tc)
    assert settings["dlc_opts"] == tc.derived_dlc_opts()
    assert settings["sleap_opts"] == tc.derived_sleap_opts()


def test_the_resolved_model_path_is_what_gets_fingerprinted():
    """Operant turns a ``.yml`` pick into its parent folder before loading. The
    check has to see the same string or every operant box looks uninitialised."""
    tc = _dlc_config(dlc_model_path="D:/models/m/pose_cfg.yml")
    settings = pose_settings_from_config(tc, model_path="D:/models/m")
    assert settings["model_path"] == "D:/models/m"
    assert settings_fingerprint(settings)[1] == "D:/models/m"
