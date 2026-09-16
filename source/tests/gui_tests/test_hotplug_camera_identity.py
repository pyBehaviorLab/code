"""A camera plugged in after launch must get its real identity.

The DirectShow device list is cached because listing it spawns ffmpeg, and it
is the ONLY source of the ``cam<hash>`` identity. When a camera was plugged in
after that cache was built, the bus walk found the new camera while the device
list still predated it, so:

  * ``device_path_for_index`` returned None for the new index,
  * the enumerator fell back to the capability fingerprint and issued
    ``fp<hash>#<ordinal>``,
  * and Detect, which finds a camera's device by matching ``cam`` ids, could
    never match that id, so it read no offered modes, stored nothing, and
    reported success.

On this rig three cameras all fingerprint alike, so the fourth came out as
``fp755b78f6#3`` and its Resolution/FPS/Format pickers stayed empty.
"""
import pytest

from source.video.cameras import factory
from source.video.cameras import identity as ident

# Two of these deliberately share a friendly name: identical models report the
# same name, which is why a name can never identify a device.
DEVICES = [
    {"name": "HD USB Camera", "path": r"@device_pnp_\\?\usb#vid_32e4&pid_2210#a"},
    {"name": "USB Video", "path": r"@device_pnp_\\?\usb#vid_345f&pid_2109#b"},
    {"name": "HD USB Camera", "path": r"@device_pnp_\\?\usb#vid_32e4&pid_9230#c"},
]


@pytest.fixture
def stale_cache(monkeypatch):
    """A device list built when only the first two cameras were attached.

    The third is on the bus; the cache does not know about it. That is exactly
    the state the GUI is in after a hot-plug.
    """
    # The Windows branch is the one under test, and these targets include
    # Linux and Jetson, so the platform is faked on the module rather than on
    # the real ``sys``, which every other import shares.
    monkeypatch.setattr(ident, "sys", type("_S", (), {"platform": "win32"}))
    listed = {"calls": 0}

    def fake_list(refresh=False):
        if ident._dshow_cache is not None and not refresh:
            return list(ident._dshow_cache)
        listed["calls"] += 1
        ident._dshow_cache = list(DEVICES)
        return list(DEVICES)

    monkeypatch.setattr(ident, "list_dshow_devices", fake_list)
    ident._dshow_cache = list(DEVICES[:2])
    yield listed
    ident._dshow_cache = None


def test_an_index_past_the_cache_re_lists_rather_than_returning_none(stale_cache):
    """An index beyond the list is PROOF the list is stale, not proof the
    camera has no device path: the caller got that index from OpenCV."""
    assert ident.device_path_for_index(2) == DEVICES[2]["path"]
    assert stale_cache["calls"] == 1, "the stale list was never refreshed"


def test_a_hotplugged_camera_gets_a_cam_id_not_a_fingerprint(stale_cache):
    """The regression. ``fp...#N`` here means Detect can never find it."""
    cam_id = ident.device_id_for_index(2)
    assert cam_id is not None, "hot-plugged camera had no device identity"
    assert cam_id.startswith("cam"), f"fell back to a fingerprint id: {cam_id}"


def test_identical_models_still_get_distinct_ids(stale_cache):
    """Indices 0 and 2 report the same friendly name and differ only by path."""
    assert ident.device_id_for_index(0) != ident.device_id_for_index(2)


def test_invalidating_the_enumeration_drops_the_device_list_too(monkeypatch):
    """Dropping a subset of the caches is worse than dropping none: the bus
    walk then finds a camera the device list cannot describe."""
    ident._dshow_cache = list(DEVICES[:2])
    try:
        factory.invalidate_camera_enumeration()
        assert ident._dshow_cache is None, (
            "invalidate_camera_enumeration left the DirectShow list stale")
    finally:
        ident._dshow_cache = None


# ── Detect's own device lookup ───────────────────────────────────────────

class _FakeDialog:
    """Just enough of the dialog to call ``_enumerate_offered`` unbound."""

    @staticmethod
    def _append_cal_log(log_view, text):
        if log_view is not None:
            log_view.append(str(text))


def _enumerate_offered(camera_id, log_view=None):
    from source.gui.dialogs.camera_connect import CameraConnectDialog
    return CameraConnectDialog._enumerate_offered(
        _FakeDialog(), camera_id, log_view)


def test_detect_enumerates_by_path_not_by_friendly_name(monkeypatch):
    """Two cameras of the same model share a name, so a name-based lookup
    reads the FIRST one's modes and files them under the second one's id."""
    from source.video.cameras import enumerate_modes as em

    monkeypatch.setattr(ident, "list_dshow_devices", lambda refresh=False: list(DEVICES))
    monkeypatch.setattr(ident, "device_id_for_index",
                        lambda i: f"cam{i}" if 0 <= i < len(DEVICES) else None)
    asked = {}

    def fake_enumerate(device):
        asked["device"] = device
        return [em.OfferedMode(640, 480, "mjpeg", (30.0,))]

    monkeypatch.setattr(em, "enumerate_modes", fake_enumerate)
    from source.video.cameras import calibration_store as store
    monkeypatch.setattr(store, "put_offered", lambda uid, modes: True)

    assert _enumerate_offered("cam2") == 1
    assert asked["device"] == DEVICES[2]["path"], (
        f"enumerated by name, not path: {asked['device']!r}")


def test_a_fingerprint_id_is_resolved_to_an_index_rather_than_passed_to_ffmpeg(
        monkeypatch):
    """An id ffmpeg has never heard of used to be handed to ffmpeg as a device
    name, which could only fail, and failed silently."""
    from source.video.cameras import calibration_store as store
    from source.video.cameras import enumerate_modes as em

    monkeypatch.setattr(ident, "list_dshow_devices", lambda refresh=False: list(DEVICES))
    monkeypatch.setattr(ident, "device_id_for_index",
                        lambda i: f"cam{i}" if 0 <= i < len(DEVICES) else None)
    monkeypatch.setattr(ident, "address_of", lambda cid: 2)
    asked = {}

    def fake_enumerate(device):
        asked["device"] = device
        return [em.OfferedMode(640, 480, "mjpeg", (30.0,))]

    monkeypatch.setattr(em, "enumerate_modes", fake_enumerate)
    monkeypatch.setattr(store, "put_offered", lambda uid, modes: True)

    assert _enumerate_offered("fp755b78f6#3") == 1
    assert asked["device"] == DEVICES[2]["path"]


def test_an_unmatchable_id_says_so_instead_of_returning_silently(monkeypatch):
    """No silent failure: a Detect that read nothing must say it read nothing."""
    monkeypatch.setattr(ident, "list_dshow_devices", lambda refresh=False: list(DEVICES))
    monkeypatch.setattr(ident, "device_id_for_index", lambda i: None)
    monkeypatch.setattr(ident, "address_of", lambda cid: None)

    class _Log:
        def __init__(self):
            self.lines = []

        def append(self, text):
            self.lines.append(text)

    log = _Log()
    assert _enumerate_offered("nonsense", log) == 0
    assert log.lines, "Detect read no modes and reported nothing at all"
    assert "NOT read" in " ".join(log.lines)
