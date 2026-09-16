"""Stable identity for OpenCV cameras, because the index is not one.

``cv2.VideoCapture(0)`` addresses cameras by enumeration order, and that order
is assigned by the OS as devices appear. Plug in a second webcam and it can
take index 0, silently displacing the camera that was there: a project that
recorded "camera 0" now opens a different physical device, with a different
view, and nothing reports a problem. Measured on a two-camera machine, the
newly-plugged external camera took index 0 and moved the built-in one to 1.

An enumerator that builds its identity FROM that index
(``unique_id = f"{idx}-opencv"``) has a "unique id" that changes whenever the
index does, it records the symptom as the identity.

What is actually stable is the OS device id. On Windows:

    USB\\VID_5843&PID_E819&MI_00\\6&218045C&0&0000

``VID``/``PID`` name the model; the instance suffix encodes the USB port path,
so two identical cameras in different ports stay distinct and one camera in
one port keeps its id across reboots. Linux exposes the same thing as the
``/dev/v4l/by-id`` symlink names.

Mapping that identity back to an index is the remaining problem, because the
OS enumeration order and OpenCV's are not guaranteed to agree. So identity is
resolved by *fingerprint*: what resolutions a given index actually offers.
Cameras of different models differ there; identical models fall back to the
enumeration order, which is the best available and is reported as uncertain
rather than assumed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional

from source.log import get_logger

logger = get_logger()

_MAX_INDEX = 8            # how far to probe when scanning for cameras
# Two probes, not four: each set+get costs ~2 s on DSHOW, and the pair below
# already separates the camera models on a rig (a 1080p-capable device from a
# 720p one). Every extra mode is another two seconds of the operator waiting.
_PROBE_MODES = ((1920, 1080), (640, 480))

# The live table is expensive to build (every index is opened and probed), and
# every camera open needs it, so it is cached until something says otherwise.
_live_lock = threading.Lock()
_live_cache: Optional[List[Dict[str, Any]]] = None


# ── OS-level device enumeration ──────────────────────────────────────────

def list_os_cameras() -> List[Dict[str, str]]:
    """Camera devices as the OS knows them, with stable ids.

    Returns ``[{"name": ..., "device_id": ..., "vid": ..., "pid": ...}, ...]``
    in OS enumeration order. Empty when the platform cannot be queried, the
    caller then falls back to index-only identity, which is what the whole
    module exists to avoid, so the emptiness is logged.
    """
    if os.name == "nt":
        return _list_windows()
    if sys.platform.startswith("linux"):
        return _list_linux()
    return []


def _list_windows() -> List[Dict[str, str]]:
    """Windows: Win32_PnPEntity, filtered to camera/imaging classes.

    PowerShell rather than a COM binding so this needs no new dependency;
    the call is only made when enumerating, not per frame.
    """
    ps = ("Get-CimInstance Win32_PnPEntity | "
          "Where-Object {$_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image'} | "
          "ForEach-Object { $_.Name + '|' + $_.DeviceID }")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=20)
    except Exception as e:
        logger.debug("camera identity: PnP query failed: %s", e)
        return []
    cams = []
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        name, dev = line.split("|", 1)
        vid = pid = ""
        m = re.search(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})", dev)
        if m:
            vid, pid = m.group(1).upper(), m.group(2).upper()
        cams.append({"name": name.strip(), "device_id": dev.strip(),
                     "vid": vid, "pid": pid})
    return cams


def _list_linux() -> List[Dict[str, str]]:
    """Linux: /dev/v4l/by-id symlinks carry vendor, model and serial."""
    by_id = "/dev/v4l/by-id"
    if not os.path.isdir(by_id):
        return []
    cams = []
    for entry in sorted(os.listdir(by_id)):
        if not entry.endswith("-index0"):
            continue          # each camera exposes several nodes; take the first
        try:
            target = os.path.realpath(os.path.join(by_id, entry))
        except OSError:
            continue
        cams.append({"name": entry, "device_id": entry, "vid": "", "pid": "",
                     "node": target})
    return cams


# ── DirectShow order, which IS OpenCV's index order (Windows) ────────────

#: One entry per video device, in DirectShow enumeration order. Cached
#: because listing spawns ffmpeg; cleared by ``refresh_dshow_devices``.
_dshow_cache: Optional[List[Dict[str, str]]] = None
_dshow_lock = threading.Lock()

_DSHOW_NAME = re.compile(r'^\[[^\]]*\]\s+"(.+)"\s+\(video\)\s*$')
_DSHOW_ALT = re.compile(r'^\[[^\]]*\]\s+Alternative name\s+"(.+)"\s*$')


def list_dshow_devices(refresh: bool = False) -> List[Dict[str, str]]:
    """Video devices in DirectShow order, with their USB instance paths.

    This is the piece that was missing. The OS enumeration (PnP) lists
    cameras in an order that has nothing to do with OpenCV's indices, which
    is why identity fell back to guessing from capabilities. DirectShow's
    order, however, IS the order OpenCV's DSHOW backend indexes, so position
    in this list maps directly to ``cv2.VideoCapture(index)``.

    ffmpeg is already a hard requirement of this application (it encodes
    every recording), and its dshow lister reports each device's
    "Alternative name", the full USB instance path::

        @device_pnp_\\\\?\\usb#vid_046d&pid_08e5&mi_00#7&2fcc25a4&0&0000#{...}

    ``vid``/``pid`` name the model and the instance suffix encodes the port
    path, so two identical cameras in different ports stay distinct and one
    camera keeps its identity across reboots and re-indexing.

    Returns [] on anything that is not Windows, or when ffmpeg is absent.
    """
    global _dshow_cache
    if sys.platform != "win32":
        return []
    with _dshow_lock:
        if _dshow_cache is not None and not refresh:
            return list(_dshow_cache)
    devices: List[Dict[str, str]] = []
    try:
        # ffmpeg exits non-zero here by design: it lists the devices and then
        # fails to open the dummy input, so the return code says nothing.
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true",
             "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        pending = None
        for line in (proc.stderr or "").splitlines():
            m = _DSHOW_NAME.match(line.strip())
            if m:
                pending = m.group(1)
                continue
            m = _DSHOW_ALT.match(line.strip())
            if m and pending is not None:
                devices.append({"name": pending, "path": m.group(1)})
                pending = None
    except FileNotFoundError:
        logger.warning("camera identity: ffmpeg not found, so cameras cannot "
                       "be told apart by USB identity and fall back to their "
                       "capabilities, which two same-model cameras share.")
        return []
    except Exception as e:
        logger.warning("camera identity: listing DirectShow devices failed "
                       "(%s); falling back to capability fingerprints.", e)
        return []
    with _dshow_lock:
        _dshow_cache = list(devices)
    return devices


def refresh_dshow_devices() -> List[Dict[str, str]]:
    """Re-enumerate; call after a camera is plugged in or removed."""
    return list_dshow_devices(refresh=True)


def invalidate_dshow_devices() -> None:
    """Drop the cached device list, call after plugging or unplugging.

    Separate from ``refresh_dshow_devices`` because clearing is free and
    re-listing spawns ffmpeg: an invalidation should not pay for an
    enumeration that the next caller may never ask for.
    """
    global _dshow_cache
    with _dshow_lock:
        _dshow_cache = None


def device_path_for_index(index: int) -> Optional[str]:
    """The USB instance path of the camera OpenCV addresses as ``index``."""
    if sys.platform == "win32":
        devices = list_dshow_devices()
        # An index past the end of the list is PROOF the list is stale: the
        # caller is holding an index that OpenCV answered to, so a camera
        # exists that this list does not know about. That happens whenever a
        # camera is plugged in after the list was first built, and the result
        # used to be silent: no path, so no ``cam`` id, so the caller fell
        # back to the capability fingerprint and the camera came out as
        # ``fp<hash>#<n>``, an id that nothing which matches on device path
        # can ever resolve. Re-list once and answer correctly instead.
        if int(index) >= len(devices):
            devices = list_dshow_devices(refresh=True)
        if 0 <= int(index) < len(devices):
            return devices[int(index)]["path"]
        return None
    # Linux: /dev/videoN is the index, and by-id carries vendor+model+serial.
    node = f"/dev/video{int(index)}"
    by_id = "/dev/v4l/by-id"
    try:
        if os.path.isdir(by_id):
            for entry in sorted(os.listdir(by_id)):
                if os.path.realpath(os.path.join(by_id, entry)) == node:
                    return entry
    except OSError as e:
        logger.debug("camera identity: by-id lookup for %s failed: %s", node, e)
    return None


def device_id_for_index(index: int) -> Optional[str]:
    """A stable id for the camera at ``index``, from its USB identity.

    ``cam`` + djb2 of the device path, so it is short enough to read in a
    config file and carries no separators that would upset a filename.
    Returns None when the platform cannot say, and the caller falls back to
    the capability fingerprint.
    """
    path = device_path_for_index(index)
    if not path:
        return None
    from source.config.hashing import djb2_hex_from_text
    return f"cam{djb2_hex_from_text(path.strip().lower())}"


# ── index fingerprinting ─────────────────────────────────────────────────

def fingerprint_index(index: int, backend: Optional[int] = None
                      ) -> Optional[Dict[str, Any]]:
    """What does the camera at ``index`` actually offer?

    Opens briefly and records the resolutions it accepts. This is what allows
    a stored camera to be found again after the indices shuffle: a 1080p
    external and a 720p built-in are told apart by what they can do, without
    needing the OS enumeration order to match OpenCV's.

    Returns None when nothing is at that index.
    """
    import cv2
    if backend is None:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = None
    try:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            return None
        ok, _frame = cap.read()
        if not ok:
            return None
        default = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        modes = []
        for w, h in _PROBE_MODES:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (w, h):
                modes.append([w, h])
        return {"index": int(index), "default": list(default), "modes": modes}
    except Exception as e:
        logger.debug("camera identity: fingerprint index %s failed: %s",
                     index, e)
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def scan_indices(max_index: int = _MAX_INDEX) -> List[Dict[str, Any]]:
    """Fingerprint every index that has a camera on it."""
    out = []
    for idx in range(max_index):
        fp = fingerprint_index(idx)
        if fp is not None:
            out.append(fp)
    return out


# ── binding identity to an index ─────────────────────────────────────────

def match_score(stored: Dict[str, Any], candidate: Dict[str, Any]) -> int:
    """How well does a fingerprint match a stored one? Higher is better."""
    if not stored or not candidate:
        return 0
    score = 0
    if list(stored.get("default") or []) == list(candidate.get("default") or []):
        score += 2
    smodes = {tuple(m) for m in (stored.get("modes") or [])}
    cmodes = {tuple(m) for m in (candidate.get("modes") or [])}
    if smodes and smodes == cmodes:
        score += 5
    elif smodes and cmodes:
        score += len(smodes & cmodes)
    return score


def resolve_index(stored_fingerprint: Optional[Dict[str, Any]],
                  stored_index: Optional[int] = None,
                  max_index: int = _MAX_INDEX) -> Dict[str, Any]:
    """Find the index the stored camera is on NOW.

    Returns ``{"index": int|None, "moved": bool, "confident": bool}``.

    ``confident`` is False when several cameras fingerprint alike, identical
    models, which no capability probe can separate. The caller should say so
    rather than pretend: opening the wrong one of two identical cameras is
    exactly as wrong as opening a different model, it is just harder to see.
    """
    if not stored_fingerprint:
        return {"index": stored_index, "moved": False, "confident": False}
    live = scan_indices(max_index)
    if not live:
        return {"index": None, "moved": False, "confident": False}
    scored = sorted(((match_score(stored_fingerprint, c), c) for c in live),
                    key=lambda t: t[0], reverse=True)
    best_score, best = scored[0]
    if best_score <= 0:
        return {"index": None, "moved": False, "confident": False}
    tied = sum(1 for s, _ in scored if s == best_score) > 1
    idx = int(best["index"])
    return {"index": idx,
            "moved": stored_index is not None and idx != int(stored_index),
            "confident": not tied}


def invalidate_live_cameras() -> None:
    """Drop the cached live table, call after plugging or unplugging."""
    global _live_cache
    with _live_lock:
        _live_cache = None


def live_cameras(refresh: bool = False,
                 cached_only: bool = False) -> List[Dict[str, Any]]:
    """Every index that currently has a camera, with the identity it answers to.

    ``[{"index": 0, "fingerprint": {...}, "fp": "fp1234", "ordinal": 0,
        "id": "fp1234"}, ...]`` in index order.

    The ordinal exists because a fingerprint describes a MODEL: two identical
    cameras produce the same ``fp``, and the only thing left to tell them
    apart is the order they enumerate in. Scoping the ordinal to the group of
    same-model cameras is what makes it survive an unrelated camera appearing
    and shifting every index, which is exactly what a bare index does not.
    """
    global _live_cache
    with _live_lock:
        if _live_cache is not None and not refresh:
            return [dict(e) for e in _live_cache]
        if cached_only:
            # Building the table opens every index and takes seconds. A GUI
            # path asks for what is already known and does without when it
            # isn't, rather than freezing while the bus is walked.
            return []
        table: List[Dict[str, Any]] = []
        seen: Dict[str, int] = {}
        for fp in scan_indices():
            idx = int(fp["index"])
            fid = fingerprint_id(fp)
            ordinal = seen.get(fid, 0)
            seen[fid] = ordinal + 1
            # The id the rest of the app stores and binds to. The USB device
            # path is preferred because it is the only one that separates two
            # cameras of the same model: on this rig a C920 and a no-name HD
            # camera both fingerprinted to fp755b78f6, so every project bound
            # every box to one id that two different cameras answered to.
            # ``fp`` is still carried, unchanged, so ids stored by earlier
            # versions keep resolving.
            dev_id = device_id_for_index(idx)
            table.append({
                "index": idx,
                "fingerprint": fp,
                "fp": fid,
                "ordinal": 0 if dev_id else ordinal,
                "id": dev_id or (fid if ordinal == 0 else f"{fid}#{ordinal}"),
            })
        _live_cache = table
        return [dict(e) for e in table]


def resolve_id(stable_id: Any, table: Optional[List[Dict[str, Any]]] = None,
               refresh: bool = False) -> Dict[str, Any]:
    """Which index is the camera with this id on RIGHT NOW?

    Returns ``{"index": int|None, "confident": bool, "reason": str}``.

    The id is the thing that is stored and carried; the index is a location
    that changes whenever the set of connected cameras changes, so it is
    looked up here on every open rather than remembered. Accepted forms:

    ``fp1234`` / ``fp1234#2``  fingerprint identity, the enumerator's output
    ``USB\\VID_...``           an OS device id, resolvable only while it is
                               the sole camera, which is when it was issued
    ``3`` / ``3-opencv``       a legacy index from an older project file
    ``idx3-unstable``          an id that admits it is really an index
    """
    s = str(stable_id).strip()
    if s.endswith("-opencv"):
        s = s[: -len("-opencv")]
    if s.isdigit():
        return {"index": int(s), "confident": False, "reason": "index, not identity"}
    m = re.fullmatch(r"idx(\d+)-unstable", s)
    if m:
        return {"index": int(m.group(1)), "confident": False,
                "reason": "index, not identity"}

    # A USB-identity id: ask each live index what device it is and compare.
    # This is the only form that separates two cameras of the same model, so
    # it is answered before anything that can only guess.
    if s.startswith("cam"):
        if refresh:
            refresh_dshow_devices()
        for idx in range(_MAX_INDEX):
            if device_id_for_index(idx) == s:
                return {"index": idx, "confident": True,
                        "reason": "matched by USB device path"}
        return {"index": None, "confident": False,
                "reason": "this camera is not connected"}

    live = table if table is not None else live_cameras(refresh=refresh)
    if not live:
        return {"index": None, "confident": False, "reason": "no cameras found"}

    if not s.startswith("fp"):
        # An OS device id: it was only ever issued when one camera was
        # attached, so it can only be honoured while that is still true.
        if len(live) == 1:
            return {"index": live[0]["index"], "confident": True,
                    "reason": "sole camera"}
        return {"index": None, "confident": False,
                "reason": "OS device id cannot be tied to an index with "
                          f"{len(live)} cameras attached"}

    fid, _, ord_txt = s.partition("#")
    want_ordinal = int(ord_txt) if ord_txt.isdigit() else 0
    group = [e for e in live if e["fp"] == fid]
    if not group:
        return {"index": None, "confident": False, "reason": "camera not connected"}
    for e in group:
        if e["ordinal"] == want_ordinal:
            return {"index": e["index"], "confident": len(group) == 1,
                    "reason": "matched" if len(group) == 1
                              else "identical models, matched by order"}
    # The ordinal is gone but the model is here: one of a pair was unplugged.
    # Naming the uncertainty beats silently opening the survivor as if it were
    # the one that was asked for.
    return {"index": group[0]["index"], "confident": False,
            "reason": "ordinal no longer present, using the remaining "
                      "camera of that model"}


def address_of(cam_id: Any) -> Optional[int]:
    """What ``cam_id`` opens as, without opening anything.

    Numeric ids are already addresses; identities resolve through the cached
    live table only, so this is cheap enough to call from a lookup.
    """
    if isinstance(cam_id, bool) or cam_id is None:
        return None
    if isinstance(cam_id, int):
        return cam_id
    s = str(cam_id).strip()
    if s.endswith("-opencv"):
        s = s[: -len("-opencv")]
    if s.isdigit():
        return int(s)
    r = resolve_id(s, table=live_cameras(cached_only=True))
    return r["index"]


def matching_key(target: Any, keys) -> Any:
    """The key in ``keys`` that names the same camera as ``target``.

    Two forms of the same camera's id, the index a project saved and the
    identity the enumerator now issues, must find each other, or a dialog
    looks up a camera that is streaming, misses, and opens a SECOND handle on
    a device the capture thread already owns. On Windows that stalls for
    seconds and then yields nothing, which is indistinguishable from "the
    camera is broken".
    """
    keys = list(keys)
    for k in keys:
        if k == target or str(k) == str(target):
            return k
    addr = address_of(target)
    if addr is None:
        return None
    for k in keys:
        if address_of(k) == addr:
            return k
    return None


def index_for_id(stable_id: Any, *, allow_scan: bool = False) -> Optional[int]:
    """The current index for a stored camera id.

    ``allow_scan`` splits the two kinds of caller, and the split matters:

    * LOOKUPS (the picker, a keystroke, matching a dict key) must never walk
      the bus, ~10 s on the GUI thread is a frozen app, and a lookup that
      cannot answer yet is harmless.
    * OPENING a camera must resolve or the identity reaches ``cv2`` verbatim
      and the connect fails outright with "could not open camera fp…". A
      pause on a button press is worth far more than that, and the startup
      prewarm means the table is normally already there.
    """
    r = resolve_id(stable_id, table=live_cameras(cached_only=not allow_scan))
    if r["index"] is None and allow_scan:
        # The table may predate a replug; one rescan before giving up.
        r = resolve_id(stable_id, refresh=True)
    if r["index"] is None:
        logger.warning("camera identity: %s is not connected (%s)",
                       stable_id, r["reason"])
        return None
    if not r["confident"]:
        logger.info("camera identity: %s -> index %s (%s)",
                    stable_id, r["index"], r["reason"])
    return int(r["index"])


def fingerprint_id(fingerprint: Dict[str, Any]) -> str:
    """A stable id derived from what the camera can DO.

    djb2 (the project's only hash) over the accepted resolutions. It survives
    re-indexing because it never mentions the index, and it needs no OS
    query, but it identifies a *model*, so two identical cameras share it.

    ``_PROBE_MODES`` IS PART OF THIS ID. Change that list and every camera on
    every rig is issued a new id: stored project bindings and calibration keys
    stop matching, and each camera has to be picked again. Treat the list as
    frozen unless you intend exactly that.
    """
    from source.config.hashing import djb2_hex_from_text
    modes = ",".join(f"{w}x{h}" for w, h in (fingerprint.get("modes") or []))
    default = "x".join(str(v) for v in (fingerprint.get("default") or []))
    return f"fp{djb2_hex_from_text(f'{default}|{modes}')}"


def stable_unique_id(index: int, os_cameras: Optional[List[Dict[str, str]]] = None,
                     fingerprint: Optional[Dict[str, Any]] = None) -> str:
    """Best available stable id for the camera at ``index``.

    **The OS enumeration order is NOT OpenCV's index order.** Measured on a
    two-camera machine: PnP listed the built-in camera first, while OpenCV
    index 0 was the external one. So indexing the OS list by the OpenCV index
    labels each camera with the other one's USB id, the same confusion this
    module exists to remove, just harder to notice.

    That is why ``device_id_for_index`` exists: DirectShow's order IS
    OpenCV's index order, so a device's USB instance path can be looked up
    BY INDEX and cannot be mismatched. It is tried first, and it is the only
    branch that distinguishes two cameras of the same model.

    The capability fingerprint is the fallback for platforms or setups where
    that lookup is unavailable. It is genuinely weak, and measured to fail on
    this rig: a Logitech C920 and a no-name HD camera both accept 1920x1080
    and 640x480, so both hashed to ``fp755b78f6`` and every per-camera
    setting written for one landed on the other. It also drifts for a single
    camera, because which modes negotiate depends on USB bandwidth and on
    what else is plugged in.

    The OS device id is used only when exactly one camera is attached, where
    it cannot be assigned to the wrong one. When nothing is available the
    value says ``-unstable`` outright, so an id that is really just an index
    is recognisable as one rather than passing for identity.
    """
    # Identity from the device itself, matched to this exact index.
    dev_id = device_id_for_index(index)
    if dev_id:
        return dev_id

    cams = os_cameras if os_cameras is not None else list_os_cameras()
    if len(cams) == 1:
        dev = cams[0].get("device_id") or ""
        if dev:
            return dev
    fp = fingerprint if fingerprint is not None else fingerprint_index(index)
    if fp and fp.get("modes"):
        return fingerprint_id(fp)
    return f"idx{int(index)}-unstable"
