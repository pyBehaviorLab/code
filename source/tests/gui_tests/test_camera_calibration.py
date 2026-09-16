"""Host-only tests for the multi-camera calibration feature.

No hardware: OpenCV probe primitives are monkeypatched, USB enumeration is
faked (sysfs tree / DirectShow monikers), the machine store is redirected to a
temp file, and the plan decision matrix runs against fakes.
"""

from __future__ import annotations

import pytest

from source.video.cameras import calibration_store as store
from source.video.cameras import usb_identity as uid
from source.video.cameras.probe import ProbeResult, probe_camera

# ---------------------------------------------------------------------------
# probe_camera
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_opencv(monkeypatch):
    from source.video.cameras.opencv import OpenCVCamera
    calls = {"measure": []}

    def fake_probe(camera_id, candidates=None, report=None):
        return ([(640, 480), (1280, 720)], (1280, 720))

    def fake_measure(camera_id, w, h, target_fps=30.0, report=None):
        calls["measure"].append((w, h, target_fps))
        return 30.0

    # The fast path: probe_camera prefers the batched single-open measure.
    # Emit the same per-mode events the real one does so callers still see
    # live progress, and record the target_fps for the assertion.
    def fake_measure_batch(camera_id, modes, target_fps=30.0, report=None):
        out = []
        total = len(modes)
        for i, (w, h) in enumerate(modes, start=1):
            calls["measure"].append((w, h, target_fps))
            if report:
                report("measuring", w=w, h=h, index=i, total=total)
                report("mode_done", w=w, h=h, fps=30.0)
            out.append((w, h, 30.0))
        return out

    monkeypatch.setattr(OpenCVCamera, "probe_supported_resolutions",
                        staticmethod(fake_probe))
    monkeypatch.setattr(OpenCVCamera, "measure_fps_at",
                        staticmethod(fake_measure))
    monkeypatch.setattr(OpenCVCamera, "measure_fps_for_modes",
                        staticmethod(fake_measure_batch))
    return calls


def test_probe_camera_returns_modes_and_reports_in_order(patched_opencv):
    events = []
    res = probe_camera(0, "opencv", target_fps=25.0,
                       report=lambda kind, **d: events.append((kind, d)))

    assert isinstance(res, ProbeResult)
    assert res.error is None and res.degraded is False
    assert res.modes == [(640, 480, 30.0), (1280, 720, 30.0)]
    assert [k for (k, _d) in events] == [
        "resolutions", "measuring", "mode_done", "measuring", "mode_done"]
    assert events[0][1]["modes"] == [(640, 480), (1280, 720)]
    assert events[1][1]["index"] == 1 and events[1][1]["total"] == 2
    assert patched_opencv["measure"][0][2] == 25.0


def test_probe_camera_error_is_captured(monkeypatch):
    from source.video.cameras.opencv import OpenCVCamera
    monkeypatch.setattr(
        OpenCVCamera, "probe_supported_resolutions",
        staticmethod(lambda cid, candidates=None, report=None:
                     (_ for _ in ()).throw(RuntimeError("device busy"))))
    res = probe_camera(0, "opencv", target_fps=30.0)
    assert res.error is not None and "device busy" in res.error
    assert res.modes == []


def test_probe_camera_degrades_to_native_mode(monkeypatch):
    from source.video.cameras.opencv import OpenCVCamera
    monkeypatch.setattr(OpenCVCamera, "probe_supported_resolutions",
                        staticmethod(lambda cid, candidates=None, report=None: ([], None)))
    monkeypatch.setattr(OpenCVCamera, "native_mode",
                        classmethod(lambda cls, cid: (640, 480)))
    monkeypatch.setattr(OpenCVCamera, "measure_fps_at",
                        staticmethod(lambda cid, w, h, target_fps=30.0, report=None: 15.0))
    res = probe_camera(0, "opencv", target_fps=30.0)
    assert res.degraded is True
    assert res.modes == [(640, 480, 15.0)]
    assert res.error is None


def test_probe_camera_unknown_backend():
    res = probe_camera("x", "nonsense", target_fps=30.0)
    assert res.error is not None and res.modes == []


# ---------------------------------------------------------------------------
# usb_identity resolver (mocked sysfs / DirectShow)
# ---------------------------------------------------------------------------

def _make_sysfs(tmp_path, index, *, vid, pid, serial=None, speed=None,
                product="Acme Cam"):
    """Build a fake /sys tree; the video node's ``device`` IS the USB node."""
    dev = tmp_path / "class" / "video4linux" / f"video{index}" / "device"
    dev.mkdir(parents=True)
    (dev / "idVendor").write_text(vid)
    (dev / "idProduct").write_text(pid)
    if serial is not None:
        (dev / "serial").write_text(serial)
    if speed is not None:
        (dev / "speed").write_text(speed)
    (dev / "product").write_text(product)
    return str(tmp_path)


def test_linux_identity_reads_sysfs(tmp_path):
    root = _make_sysfs(tmp_path, 0, vid="046D", pid="0825",
                       serial="ABC123", speed="5000")
    ident = uid.usb_identity(0, sysfs_root=root, platform="linux")
    assert ident["vid"] == "046d" and ident["pid"] == "0825"
    assert ident["serial"] == "ABC123"
    assert ident["bus_speed"] == "5000"


def test_resolve_identity_serial_key(tmp_path):
    root = _make_sysfs(tmp_path, 0, vid="046d", pid="0825", serial="S1")
    r = uid.resolve_identity(0, "opencv", sysfs_root=root, platform="linux")
    assert r["unique_id"] == "usb-046d:0825:S1"
    assert r["weak"] is False and r["ambiguous"] is False


def test_resolve_identity_no_serial_is_port_scoped(tmp_path):
    root = _make_sysfs(tmp_path, 0, vid="046d", pid="0825", serial=None)
    r = uid.resolve_identity(0, "opencv", sysfs_root=root, platform="linux")
    assert r["unique_id"].startswith("usb-046d:0825@")
    assert r["ambiguous"] is True and r["weak"] is False


def test_resolve_identity_weak_when_no_usb(tmp_path):
    # Empty sysfs → no descriptor → weak index fallback (Jetson CSI case).
    r = uid.resolve_identity(0, "opencv", sysfs_root=str(tmp_path),
                             platform="linux")
    assert r["unique_id"] == "0-opencv"
    assert r["weak"] is True


def test_resolve_identity_vendor_serial():
    r = uid.resolve_identity("SER99", "spinnaker")
    assert r["unique_id"] == "SER99-spinnaker"
    assert r["weak"] is False


def test_windows_moniker_parsing(monkeypatch):
    monkeypatch.setattr(uid, "_windows_video_device_paths",
                        lambda: [r"\\?\usb#vid_046d&pid_0825#ABC123#{guid}"])
    ident = uid.usb_identity(0, platform="win32")
    assert ident["vid"] == "046d" and ident["serial"] == "ABC123"

    monkeypatch.setattr(uid, "_windows_video_device_paths",
                        lambda: [r"\\?\usb#vid_046d&pid_0825#7&1a2b&0&0000#{guid}"])
    ident2 = uid.usb_identity(0, platform="win32")
    assert ident2["serial"] is None
    assert ident2["port_path"] is not None


# ---------------------------------------------------------------------------
# calibration_store (USB identity keying + bus-speed staleness)
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "store_path",
                        lambda: tmp_path / "camera_calibrations.json")
    return tmp_path


def test_store_put_get_roundtrip(temp_store):
    ident = {"vid": "046d", "pid": "0825", "serial": "S1", "bus_speed": "5000"}
    modes = [(1920, 1080, 29.9), (1280, 720, 30.0)]
    assert store.put("usb-046d:0825:S1", "opencv", "Camera 0", modes, ident) is True
    # Scenario 1: replug/reboot, new index, same serial → hit.
    assert store.get("usb-046d:0825:S1", ident) == [(1920, 1080, 29.9),
                                                     (1280, 720, 30.0)]


def test_store_scenario2_identical_models_distinct_serials(temp_store):
    modes_a = [(1280, 720, 30.0)]
    modes_b = [(640, 480, 60.0)]
    store.put("usb-046d:0825:AAA", "opencv", "A", modes_a,
              {"vid": "046d", "pid": "0825", "serial": "AAA"})
    store.put("usb-046d:0825:BBB", "opencv", "B", modes_b,
              {"vid": "046d", "pid": "0825", "serial": "BBB"})
    assert store.get("usb-046d:0825:AAA", None) == [(1280, 720, 30.0)]
    assert store.get("usb-046d:0825:BBB", None) == [(640, 480, 60.0)]


def test_store_scenario4_bus_speed_change_is_stale(temp_store):
    ident_usb3 = {"vid": "046d", "pid": "0825", "serial": "S1", "bus_speed": "5000"}
    store.put("usb-046d:0825:S1", "opencv", "c", [(1920, 1080, 60.0)], ident_usb3)
    # Same serial, now on a USB2 port (480) → stale → re-probe.
    ident_usb2 = {"vid": "046d", "pid": "0825", "serial": "S1", "bus_speed": "480"}
    assert store.get("usb-046d:0825:S1", ident_usb2) is None
    # Same speed → hit.
    assert store.get("usb-046d:0825:S1", ident_usb3) == [(1920, 1080, 60.0)]


def test_store_weak_entry_is_written_and_read(temp_store):
    # Weak (index-only) cameras now persist under a per-PC fallback key so
    # calibration is not silently lost; the UI warns about port moves.
    assert store.put("0-opencv", "opencv", "weak", [(640, 480, 30.0)],
                     {}, weak=True) is True
    assert store.get("0-opencv", None) == [(640, 480, 30.0)]


def test_store_missing_and_corrupt_file_are_safe(temp_store):
    assert store.load() == {}
    assert store.get("usb-x", None) is None
    store.store_path().write_text("{ not json", encoding="utf-8")
    assert store.load() == {}
    assert store.get("usb-x", None) is None
