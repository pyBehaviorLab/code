"""A camera's identity must not be its index.

``cv2.VideoCapture(n)`` addresses cameras by enumeration order, and the OS
assigns that order as devices appear. Measured on a two-camera machine:
plugging in an external webcam gave it index 0 and moved the built-in one to
1, so a project that had recorded "camera 0" silently opened different
hardware with a different view.

An enumerator that builds identity FROM the index
(``unique_id = f"{idx}-opencv"``) records the symptom as the identity.
"""
import pytest

from source.video.cameras import identity as ident


def _fp(index, default, modes):
    return {"index": index, "default": list(default),
            "modes": [list(m) for m in modes]}


HD = _fp(0, (640, 480), [(1920, 1080), (1280, 720), (640, 480)])
SD = _fp(1, (640, 480), [(1280, 720), (640, 480), (320, 240)])


# ── identity ─────────────────────────────────────────────────────────────

@pytest.fixture
def no_usb_identity(monkeypatch):
    """Exercise the FALLBACK path deliberately.

    ``stable_unique_id`` now prefers the USB device path for the index, which
    on a machine with cameras attached answers from real hardware. These
    tests are about what happens when that is unavailable (Linux without
    by-id, ffmpeg missing), so the branch is turned off explicitly rather
    than left to depend on what is plugged into the machine running them.

    BOTH hardware lookups have to go. Stubbing only the device path left
    ``fingerprint_index`` free to open the camera at that index and answer
    from it, so a test asserting that nothing could identify the camera
    failed on a machine where something could.
    """
    monkeypatch.setattr(ident, "device_id_for_index", lambda _i: None)
    monkeypatch.setattr(ident, "fingerprint_index", lambda *a, **k: None)


def test_the_id_does_not_contain_the_index(no_usb_identity):
    """The whole point: an id that moves with the index is not an identity."""
    uid = ident.stable_unique_id(0, os_cameras=[], fingerprint=HD)
    assert "idx" not in uid, f"identity still index-derived: {uid}"
    assert uid == ident.stable_unique_id(7, os_cameras=[], fingerprint=HD), (
        "the same camera got a different id from a different index")


def test_different_cameras_get_different_ids(no_usb_identity):
    assert (ident.stable_unique_id(0, [], HD)
            != ident.stable_unique_id(1, [], SD))


def test_a_lone_camera_uses_its_os_device_id(no_usb_identity):
    """With one camera attached there is no ambiguity, so the real USB id,
    which survives re-indexing AND identifies the physical port, is best."""
    os_cams = [{"name": "webcam", "device_id": r"USB\VID_5843&PID_E819\X",
                "vid": "5843", "pid": "E819"}]
    assert ident.stable_unique_id(0, os_cams, HD) == r"USB\VID_5843&PID_E819\X"


def test_several_cameras_do_not_get_os_ids_by_position():
    """The OS order is NOT OpenCV's order, measured, PnP listed the built-in
    first while OpenCV index 0 was the external. Indexing the OS list by the
    OpenCV index labels each camera with the OTHER one's USB id."""
    os_cams = [{"name": "built-in", "device_id": r"USB\A", "vid": "", "pid": ""},
               {"name": "external", "device_id": r"USB\B", "vid": "", "pid": ""}]
    uid = ident.stable_unique_id(0, os_cams, HD)
    assert uid not in (r"USB\A", r"USB\B"), (
        "claimed an OS device id it cannot possibly have verified")


def test_unknown_camera_says_it_is_unstable(no_usb_identity):
    """When nothing can identify it, the value must LOOK wrong, an id that
    is really just an index should never pass for identity.

    Without ``no_usb_identity`` this asked the machine running it: with four
    or more cameras plugged in, index 3 resolved to a real USB path and the
    test failed for having found an identity, which is the opposite of what
    it is checking."""
    assert "unstable" in ident.stable_unique_id(3, os_cameras=[],
                                                fingerprint=None)


# ── resolution after a shuffle ───────────────────────────────────────────

def test_a_moved_camera_is_found_at_its_new_index(monkeypatch):
    monkeypatch.setattr(ident, "scan_indices", lambda *a, **k: [HD, SD])
    r = ident.resolve_index(SD, stored_index=0)     # project recorded index 0
    assert r["index"] == 1, "did not follow the camera to its new index"
    assert r["moved"] is True
    assert r["confident"] is True


def test_an_unmoved_camera_reports_not_moved(monkeypatch):
    monkeypatch.setattr(ident, "scan_indices", lambda *a, **k: [HD, SD])
    r = ident.resolve_index(HD, stored_index=0)
    assert r["index"] == 0 and r["moved"] is False


def test_identical_cameras_are_reported_as_not_confident(monkeypatch):
    """Two of the same model fingerprint alike and no capability probe can
    separate them. Saying so is the honest answer, opening the wrong one of
    two identical cameras is just as wrong, and harder to notice."""
    twin = _fp(1, HD["default"], HD["modes"])
    monkeypatch.setattr(ident, "scan_indices", lambda *a, **k: [HD, twin])
    assert ident.resolve_index(HD, stored_index=0)["confident"] is False


def test_a_missing_camera_resolves_to_nothing(monkeypatch):
    monkeypatch.setattr(ident, "scan_indices", lambda *a, **k: [])
    assert ident.resolve_index(HD, stored_index=0)["index"] is None


def test_scoring_prefers_an_exact_mode_match():
    assert ident.match_score(HD, HD) > ident.match_score(HD, SD)


# ── identity -> current index ────────────────────────────────────────
#
# The id is what is stored; the index is where that camera happens to be
# right now, and it moves whenever the set of connected cameras changes.

def _table(*entries):
    """Build a live table like ``live_cameras`` returns."""
    out, seen = [], {}
    for index, fid in entries:
        ordinal = seen.get(fid, 0)
        seen[fid] = ordinal + 1
        out.append({"index": index, "fp": fid, "ordinal": ordinal,
                    "fingerprint": {"index": index},
                    "id": fid if ordinal == 0 else f"{fid}#{ordinal}"})
    return out


def test_identity_resolves_to_the_index_it_is_on_now():
    t = _table((0, "fpAAA"), (1, "fpBBB"))
    assert ident.resolve_id("fpBBB", table=t)["index"] == 1


def test_the_same_identity_follows_the_camera_when_indices_shift():
    """Another camera appearing must not repoint a stored id."""
    before = _table((0, "fpAAA"))
    after = _table((0, "fpNEW"), (1, "fpAAA"))     # new camera took index 0
    assert ident.resolve_id("fpAAA", table=before)["index"] == 0
    assert ident.resolve_id("fpAAA", table=after)["index"] == 1


def test_the_unique_id_carries_the_backend_suffix_harmlessly():
    t = _table((2, "fpAAA"))
    assert ident.resolve_id("fpAAA-opencv", table=t)["index"] == 2


def test_identical_cameras_are_told_apart_by_order_and_said_to_be_uncertain():
    t = _table((0, "fpSAME"), (3, "fpSAME"))
    first = ident.resolve_id("fpSAME", table=t)
    second = ident.resolve_id("fpSAME#1", table=t)
    assert (first["index"], second["index"]) == (0, 3)
    assert first["confident"] is False and second["confident"] is False


def test_a_disconnected_camera_resolves_to_nothing():
    r = ident.resolve_id("fpGONE", table=_table((0, "fpAAA")))
    assert r["index"] is None and "not connected" in r["reason"]


def test_a_bare_index_is_honoured_but_not_called_identity():
    for raw in ("3", "3-opencv", "idx3-unstable"):
        r = ident.resolve_id(raw, table=_table((0, "fpAAA")))
        assert r["index"] == 3, raw
        assert r["confident"] is False, raw


def test_an_os_device_id_works_only_while_it_is_the_only_camera():
    lone = ident.resolve_id(r"USB\VID_5843&PID_E819\X", table=_table((0, "fpAAA")))
    assert lone["index"] == 0 and lone["confident"] is True
    crowd = ident.resolve_id(r"USB\VID_5843&PID_E819\X",
                             table=_table((0, "fpAAA"), (1, "fpBBB")))
    assert crowd["index"] is None


def test_opencv_opens_the_address_not_the_identity(monkeypatch):
    """The whole point: cv2 must never be handed an identity string."""
    from source.video.cameras import opencv as ocv
    monkeypatch.setattr(ocv, "_coerce_cam_id", ocv._coerce_cam_id)  # real one
    monkeypatch.setattr("source.video.cameras.identity.index_for_id",
                        lambda s, **kw: 5 if s == "fpAAA" else None)
    monkeypatch.setattr("source.video.cameras.identity.live_cameras",
                        lambda **kw: [])
    assert ocv._coerce_cam_id("fpAAA") == 5
    assert ocv._coerce_cam_id("2") == 2          # legacy project file
    assert ocv._coerce_cam_id(2) == 2
    assert ocv._coerce_cam_id("/dev/video0") == "/dev/video0"
    assert ocv._coerce_cam_id("fpUNKNOWN") == "fpUNKNOWN"   # unresolved: as-is


# ── one camera, two spellings ────────────────────────────────────────

def test_matching_key_finds_the_same_camera_under_either_name(monkeypatch):
    """A dict keyed by the index a project saved must be findable by the
    identity the picker offers, and vice versa.

    Missing this is not a cosmetic lookup failure: the ROI dialog and the
    zone editor fall back to opening a SECOND handle on a device the capture
    thread already owns, which stalls on Windows and then yields no frame.
    """
    monkeypatch.setattr(ident, "live_cameras",
                        lambda **kw: _table((0, "fpAAA"), (1, "fpBBB")))
    assert ident.matching_key("fpAAA", [0, 1]) == 0
    assert ident.matching_key("fpBBB", [0, 1]) == 1
    assert ident.matching_key(0, ["fpAAA", "fpBBB"]) == "fpAAA"
    assert ident.matching_key("0", [0]) == 0            # str vs int
    assert ident.matching_key("fpAAA-opencv", [0]) == 0  # with the suffix
    assert ident.matching_key("fpGONE", [0, 1]) is None


def test_address_of_needs_no_scan(monkeypatch):
    """Lookups run on the GUI thread, so resolution must use what is already
    known and never walk the bus."""
    called = {"n": 0}

    def _boom(**kw):
        called["n"] += 1
        assert kw.get("cached_only"), "a lookup must not trigger a bus scan"
        return _table((3, "fpAAA"))

    monkeypatch.setattr(ident, "live_cameras", _boom)
    assert ident.address_of("fpAAA") == 3
    assert ident.address_of("7") == 7 and ident.address_of(7) == 7
    assert called["n"] == 1, "only the identity form needs the table"


# ── identity from the USB device, not from what the camera can do ────────
#
# Measured on the rig, 2026-09-09, with three cameras attached: a Logitech
# C920 and a no-name "HD USB Camera" both accept 1920x1080 and 640x480, so
# both hashed to the SAME capability fingerprint, ``fp755b78f6``. Every
# project on that machine had bound every box to that one id, so per-camera
# settings written for one landed on the other. The capability hash also
# drifts for a single camera, because which modes negotiate depends on USB
# bandwidth and on what else is plugged in.

_DSHOW = [
    {"name": "HD USB Camera",
     "path": r"@device_pnp_\?\usb#vid_32e4&pid_2210&mi_00#6&13191d6a&0&0000#{g}"},
    {"name": "USB Video",
     "path": r"@device_pnp_\?\usb#vid_345f&pid_2109&mi_00#6&162b5ad0&0&0000#{g}"},
    {"name": "HD Pro Webcam C920",
     "path": r"@device_pnp_\?\usb#vid_046d&pid_08e5&mi_00#7&2fcc25a4&0&0000#{g}"},
]


@pytest.fixture
def dshow(monkeypatch):
    """Pin the DirectShow listing, so these run without hardware."""
    monkeypatch.setattr(ident, "list_dshow_devices",
                        lambda refresh=False: list(_DSHOW))
    monkeypatch.setattr(ident.sys, "platform", "win32")
    return _DSHOW


def test_each_index_maps_to_its_own_device(dshow):
    """OpenCV only offers 0..n, so the mapping must go index -> device."""
    ids = [ident.device_id_for_index(i) for i in range(3)]
    assert all(ids), ids
    assert len(set(ids)) == 3, f"two cameras share an id: {ids}"


def test_two_cameras_of_different_models_that_share_modes_stay_distinct(dshow):
    """The exact collision seen on the rig: same modes, different hardware."""
    same_modes = _fp(0, (640, 480), [(1920, 1080), (640, 480)])
    a = ident.stable_unique_id(0, os_cameras=[], fingerprint=same_modes)
    c = ident.stable_unique_id(2, os_cameras=[], fingerprint=same_modes)
    assert a != c, (
        "identical capability fingerprints must not produce one id when the "
        "USB paths differ")


def test_the_id_survives_reindexing(dshow, monkeypatch):
    """Unplug the first camera: the others shift down an index, and each must
    keep the id it had."""
    before = {ident.device_id_for_index(i): _DSHOW[i]["path"]
              for i in range(3)}
    monkeypatch.setattr(ident, "list_dshow_devices",
                        lambda refresh=False: list(_DSHOW[1:]))
    after = {ident.device_id_for_index(i): _DSHOW[1:][i]["path"]
             for i in range(2)}
    for cid, path in after.items():
        assert before.get(cid) == path, (
            f"{cid} changed device when the indices shifted")


def test_an_id_resolves_back_to_the_index_it_is_on_now(dshow, monkeypatch):
    monkeypatch.setattr(ident, "list_dshow_devices",
                        lambda refresh=False: list(_DSHOW[1:]))
    c920 = ident.device_id_for_index(1)          # was index 2, now 1
    r = ident.resolve_id(c920)
    assert r["index"] == 1 and r["confident"], r


def test_an_absent_camera_is_reported_absent_not_guessed(dshow):
    r = ident.resolve_id("cam00000000")
    assert r["index"] is None
    assert not r["confident"]
    assert "not connected" in r["reason"]


def test_usb_identity_beats_the_capability_fingerprint(dshow):
    """Both are available; the one that cannot mismatch must win."""
    uid = ident.stable_unique_id(2, os_cameras=[], fingerprint=HD)
    assert uid.startswith("cam"), uid
    assert uid != ident.fingerprint_id(HD)


def test_the_fingerprint_still_answers_when_usb_identity_is_unavailable(
        monkeypatch):
    """Linux without by-id, or ffmpeg missing: degrade, do not crash."""
    monkeypatch.setattr(ident, "device_id_for_index", lambda _i: None)
    uid = ident.stable_unique_id(0, os_cameras=[], fingerprint=HD)
    assert uid == ident.fingerprint_id(HD)


def test_old_fingerprint_ids_still_resolve(dshow, monkeypatch):
    """Projects saved before this change carry ``fp...``; they must keep
    opening rather than being rejected."""
    monkeypatch.setattr(ident, "live_cameras",
                        lambda refresh=False, cached_only=False: [
                            {"index": 1, "fp": "fp755b78f6", "ordinal": 0,
                             "id": "fp755b78f6"}])
    r = ident.resolve_id("fp755b78f6")
    assert r["index"] == 1


def test_the_device_list_is_parsed_from_real_ffmpeg_output(monkeypatch):
    """Guards the parser against an ffmpeg output change."""
    sample = (
        '[in#0 @ 000] "HD USB Camera" (video)\n'
        r'[in#0 @ 000]   Alternative name "@device_pnp_\\?\usb#vid_32e4"'
        '\n'
        '[in#0 @ 000] "Microphone (C920)" (audio)\n'
        '[in#0 @ 000]   Alternative name "@device_cm_33D9"\n')

    class _Proc:
        stderr = sample

    monkeypatch.setattr(ident.sys, "platform", "win32")
    monkeypatch.setattr(ident.subprocess, "run", lambda *a, **k: _Proc())
    ident._dshow_cache = None
    devices = ident.list_dshow_devices(refresh=True)
    ident._dshow_cache = None
    assert len(devices) == 1, "audio devices must not be counted as cameras"
    assert devices[0]["name"] == "HD USB Camera"
