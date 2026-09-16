"""Detect must read a Linux camera's offered modes.

The device lookup only knew DirectShow, which lists nothing off Windows, so on
Linux every Detect logged "offered modes NOT read" and the FPS list was built
from a measured ceiling instead of the camera's real intervals. On the Jetson
rig that offered 30 fps for a 640x480 MJPEG mode that runs at 120 fps only.
On Linux the OpenCV index is the V4L2 node, and ``enumerate_modes`` reads
``/dev/videoN`` exactly with v4l2-ctl.
"""
from source.gui.dialogs import camera_connect as cc
from source.video.cameras import calibration_store as store
from source.video.cameras import enumerate_modes as em
from source.video.cameras import identity as ident


class _FakeDialog:
    """Just enough of the dialog to call ``_enumerate_offered`` unbound."""

    @staticmethod
    def _append_cal_log(log_view, text):
        if log_view is not None:
            log_view.append(str(text))


class _Log:
    def __init__(self):
        self.lines = []

    def append(self, text):
        self.lines.append(text)


def _enumerate(camera_id, log_view=None):
    return cc.CameraConnectDialog._enumerate_offered(
        _FakeDialog(), camera_id, log_view)


def test_a_linux_camera_is_enumerated_from_its_v4l2_node(monkeypatch):
    monkeypatch.setattr(ident, "list_dshow_devices", lambda refresh=False: [])
    monkeypatch.setattr(ident, "address_of", lambda cid: 0)
    monkeypatch.setattr(cc.os.path, "exists", lambda p: p == "/dev/video0")
    asked = {}

    def fake_enumerate(device):
        asked["device"] = device
        return [em.OfferedMode(640, 480, "mjpeg", (120.101,))]

    monkeypatch.setattr(em, "enumerate_modes", fake_enumerate)
    monkeypatch.setattr(store, "put_offered", lambda uid, modes: True)

    assert _enumerate("cam8ad01c2e") == 1
    assert asked["device"] == "/dev/video0"


def test_without_a_device_node_it_still_says_it_read_nothing(monkeypatch):
    monkeypatch.setattr(ident, "list_dshow_devices", lambda refresh=False: [])
    monkeypatch.setattr(ident, "address_of", lambda cid: 5)
    monkeypatch.setattr(cc.os.path, "exists", lambda p: False)
    log = _Log()
    assert _enumerate("cam00000000", log) == 0
    assert "NOT read" in " ".join(log.lines)
