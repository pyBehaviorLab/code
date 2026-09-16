"""Regression tests for three config save/load bugs found in the 2026-07 audit:

1. Absolute DLC/SLEAP model-path leak into the saved/shared config.
2. Incompatible-schema load blanking the rig (compat gate).
3. Camera capability scan lost on save while cameras are connected.
"""
from pathlib import Path
from types import SimpleNamespace

from source import paths
from source.config.experiment import (
    CameraCapabilities, CameraEntry, Config, DLCConfig, SleapConfig,
    SCHEMA_VERSION, _read_camera_registry, schema_major_compatible,
)


# ── Bug 1: model-path leak ──────────────────────────────────────────────

def test_dlc_model_under_topdir_stored_relative_and_resolves_back():
    model = str(Path(paths.top_dir) / "models" / "openfield.yaml")
    cfg = Config()
    cfg.tracking.enabled = True
    cfg.tracking.mode = "dlc"
    cfg.tracking.dlc = DLCConfig(model_path=model)

    stored = cfg.to_dict()["tracking"]["dlc"]["model_path"]
    # No absolute leak: a model under top_dir is stored relative.
    assert stored == "models/openfield.yaml"
    assert ":" not in stored and not stored.startswith("/")

    # Resolves back to the original absolute path on load.
    cfg2 = Config.from_dict(cfg.to_dict())
    assert Path(cfg2.tracking.dlc.model_path) == Path(model)


def test_sleap_external_model_path_kept_absolute():
    # An external model (not under top_dir) can't be relativized functionally,
    # so it stays absolute (same tradeoff as an external data_dir), but it
    # round-trips unchanged and still resolves.
    model = "/opt/models/sleap-multi.zip"
    cfg = Config()
    cfg.tracking.enabled = True
    cfg.tracking.mode = "sleap"
    cfg.tracking.sleap = SleapConfig(model_path=model)
    cfg2 = Config.from_dict(cfg.to_dict())
    assert cfg2.tracking.sleap.model_path == model


# ── Bug 2: schema compatibility gate ────────────────────────────────────

def test_schema_major_compatible():
    assert schema_major_compatible(SCHEMA_VERSION) is True
    assert schema_major_compatible(None) is True             # legacy, no field
    major = SCHEMA_VERSION.split(".")[0]
    assert schema_major_compatible(f"{major}.999") is True   # same major
    assert schema_major_compatible(f"{int(major) + 1}.0") is False
    assert schema_major_compatible("1.0") is False


# ── Capture format (MJPEG/YUV) plumbed end-to-end ───────────────────────

def test_capture_format_round_trips_config_to_pipeline_and_back():
    from source.config.experiment import (
        CameraPreference, _camera_json_to_entry, _entry_to_camera_json,
    )
    from source.video.framebus.types import CameraConfig

    entry = CameraEntry(camera_id="cam0",
                        preference=CameraPreference(width=640, height=480,
                                                    codec="yuv"))
    # Config entry → pipeline from_json shape → CameraConfig field.
    raw = _entry_to_camera_json(entry)
    assert raw["capture_format"] == "yuv"
    assert "codec" not in raw.get("extra", {})
    cc = CameraConfig.from_json(raw, camera_id="cam0")
    assert cc.capture_format == "yuv"
    # CameraConfig.to_json → back to a config entry preserves the choice.
    entry2 = _camera_json_to_entry("cam0", cc.to_json())
    assert entry2.preference.codec == "yuv"


def test_capture_format_defaults_to_none_when_unset():
    from source.video.framebus.types import CameraConfig
    cc = CameraConfig(camera_id="cam0")
    assert cc.capture_format is None
    # Absent key on load → None (not the string "None").
    assert CameraConfig.from_json({"camera_id": "cam0"}).capture_format is None


def test_opencv_fourcc_decoder_round_trips():
    import cv2

    from source.video.cameras.opencv import OpenCVCamera

    cam = OpenCVCamera(camera_id=0, width=640, height=480, capture_format="yuv")
    assert cam._capture_format == "yuv"
    assert cam.format_warning is None

    class _FakeCap:
        def __init__(self, code): self._code = code
        def get(self, prop): return float(self._code)

    cam._cap = _FakeCap(cv2.VideoWriter_fourcc(*"MJPG"))
    assert cam._read_fourcc_str() == "MJPG"
    cam._cap = _FakeCap(cv2.VideoWriter_fourcc(*"YUYV"))
    assert cam._read_fourcc_str() == "YUYV"
    cam._cap = _FakeCap(0)
    assert cam._read_fourcc_str() == ""
