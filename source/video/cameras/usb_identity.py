"""Cross-platform USB camera identity resolver.

OpenCV hides the real device behind an integer index, but every USB camera
exposes an OS-level identity (VID/PID/serial/topology). Calibration is
inherently per-PC, the achievable FPS ceiling depends on THIS machine's USB
controller and the port's negotiated speed, so a stable identity lets the
machine-level calibration cache survive replug / reboot index shuffles while
correctly re-probing when a camera moves to a slower port.

Everything here is best-effort and exception-safe. When the OS won't reveal a
descriptor (Jetson CSI / libcamera nodes, missing enumeration), the resolver
returns a *weak* identity that still probes fine but is never cached.

All OS access is funnelled through small, monkeypatchable helpers so the
resolver is unit-testable against fake sysfs trees / DirectShow monikers.

Identity dict keys (any may be ``None`` / absent):
    vid, pid       lowercase 4-hex USB vendor / product id
    serial         USB iSerial string (the stable, port-independent id)
    port_path      USB topology path (e.g. Linux "1-1.2"); identifies the port
    bus_speed      negotiated link speed (Linux sysfs "480"/"5000"); the
                   FPS ceiling depends on it, so a change invalidates a cache
    name           human-readable product string
"""

from __future__ import annotations

import os
import re
import sys

from source.log import get_logger

logger = get_logger()

# \\?\usb#vid_XXXX&pid_YYYY#SERIAL#{guid}
_WIN_MONIKER_RE = re.compile(
    r"usb#vid_([0-9a-f]{4})&pid_([0-9a-f]{4})#([^#]+)#", re.IGNORECASE)


def _coerce_index(camera_id) -> int | None:
    """The index this camera is on, from whatever form of id was given.

    USB info is read per index, but what the app carries is an identity, so a
    non-numeric id is resolved the same way an open resolves it. Without this
    every enumerated camera looked like it had no USB information at all, and
    fell back to the weak index-based key it was supposed to replace.
    """
    if isinstance(camera_id, bool):
        return None
    if isinstance(camera_id, int):
        return camera_id
    if isinstance(camera_id, str) and camera_id.isdigit():
        return int(camera_id)
    if isinstance(camera_id, str) and camera_id.strip():
        # Cached table only. This runs on every Camera-ID keystroke, so it may
        # not walk the bus, an unknown id simply has no USB info yet.
        try:
            from source.video.cameras.identity import address_of
            return address_of(camera_id)
        except Exception:
            return None
    return None


def _read_sysfs(path: str) -> str | None:
    """Read + strip a sysfs attribute file, or ``None`` on any error."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return None


# --- Linux (sysfs) -----------------------------------------------------------

def _linux_identity(camera_id, sysfs_root: str = "/sys") -> dict:
    """Resolve ``/dev/videoN`` to its parent USB device via sysfs.

    Walks up from ``video4linux/videoN/device`` until a node exposing
    ``idVendor`` is found, that node is the USB device carrying the
    descriptor + topology.
    """
    idx = _coerce_index(camera_id)
    if idx is None:
        return {}
    dev_link = os.path.join(sysfs_root, "class", "video4linux",
                            f"video{idx}", "device")
    try:
        node = os.path.realpath(dev_link)
    except Exception:
        return {}

    usb_node = None
    for _ in range(10):
        if _read_sysfs(os.path.join(node, "idVendor")) is not None:
            usb_node = node
            break
        parent = os.path.dirname(node)
        if not parent or parent == node:
            break
        node = parent
    if usb_node is None:
        return {}

    vid = _read_sysfs(os.path.join(usb_node, "idVendor"))
    if not vid:
        return {}
    pid = _read_sysfs(os.path.join(usb_node, "idProduct"))
    serial = _read_sysfs(os.path.join(usb_node, "serial"))
    speed = _read_sysfs(os.path.join(usb_node, "speed"))
    product = _read_sysfs(os.path.join(usb_node, "product"))
    return {
        "vid": vid.lower(),
        "pid": pid.lower() if pid else None,
        "serial": serial or None,
        "port_path": os.path.basename(usb_node) or None,
        "bus_speed": speed or None,
        "name": product or "USB Camera",
    }


# --- Windows (DirectShow moniker) --------------------------------------------

def _windows_video_device_paths() -> list:
    """DirectShow video device paths in cv2 index order (best-effort).

    rig-verify: OpenCV's CAP_DSHOW indexes devices in the order the
    DirectShow ``CLSID_VideoInputDeviceCategory`` enumerator returns them.
    ``pygrabber`` (optional) walks the same enumerator; we read each device's
    ``DevicePath`` property so the moniker VID/PID/serial can be parsed. Any
    failure (package missing, COM error) yields ``[]`` and the caller falls
    back to an index identity. Confirm the order matches cv2 on the rig.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph  # optional dependency
    except Exception:
        return []
    try:
        graph = FilterGraph()
        # get_input_devices() returns friendly names; the device path lives on
        # the underlying moniker. pygrabber exposes it via the system enum.
        return [str(dev) for dev in graph.get_input_devices()]
    except Exception as e:
        logger.debug("DirectShow enumeration failed: %s", e)
        return []


def _parse_win_moniker(path: str) -> dict:
    r"""Parse ``\\?\usb#vid_XXXX&pid_YYYY#SERIAL#...`` into an identity dict.

    The third ``#``-delimited field is the instance id: a real iSerial when
    the descriptor carries one, otherwise a bus-position token beginning with
    ``&`` (not globally unique): treated as a port path, not a serial.
    """
    if not path:
        return {}
    m = _WIN_MONIKER_RE.search(str(path))
    if not m:
        return {}
    vid, pid, inst = m.group(1).lower(), m.group(2).lower(), m.group(3)
    serial: str | None = None
    port_path: str | None = None
    # A synthesised instance id (no real serial) looks like "7&1a2b3c&0&0000".
    if inst.startswith("&") or inst.count("&") >= 2:
        port_path = inst
    else:
        serial = inst
    return {
        "vid": vid,
        "pid": pid,
        "serial": serial,
        "port_path": port_path,
        "bus_speed": None,  # not exposed on the moniker; omit
        "name": "USB Camera",
    }


def _windows_identity(camera_id) -> dict:
    idx = _coerce_index(camera_id)
    if idx is None:
        return {}
    paths = _windows_video_device_paths()
    if idx < 0 or idx >= len(paths):
        return {}
    return _parse_win_moniker(paths[idx])


# --- Public API --------------------------------------------------------------

def usb_identity(camera_id, *, sysfs_root: str = "/sys",
                 platform: str | None = None) -> dict:
    """Best-effort USB identity for an OpenCV camera index. Never raises."""
    plat = platform or sys.platform
    try:
        if plat.startswith("linux"):
            return _linux_identity(camera_id, sysfs_root)
        if plat.startswith("win"):
            return _windows_identity(camera_id)
    except Exception as e:
        logger.debug("usb_identity(%s) failed: %s", camera_id, e)
    return {}  # macOS / unknown / unreadable → empty (weak)


def resolve_identity(camera_id, backend: str = "opencv", **kw) -> dict:
    """Return ``{unique_id, identity, weak, ambiguous}`` for a camera.

    Vendor cameras (Spinnaker/Ximea) already carry a globally-unique serial in
    ``camera_id``, so their unique_id is the familiar ``SERIAL-backend``.

    OpenCV cameras resolve to a USB identity:
      * serial present     → ``usb-{vid}:{pid}:{serial}`` (survives replug)
      * no serial, has port → ``usb-{vid}:{pid}@{port}``  (port-scoped;
                              moving ports reads as a new camera, ``ambiguous``)
      * no USB info        → ``{index}-opencv``, ``weak=True`` (never cached)
    """
    backend = (backend or "opencv").lower()
    if backend != "opencv":
        serial = str(camera_id)
        return {
            "unique_id": f"{camera_id}-{backend}",
            "identity": {"serial": serial, "name": serial},
            "weak": False,
            "ambiguous": False,
        }

    ident = usb_identity(camera_id, **kw)
    vid, pid = ident.get("vid"), ident.get("pid")
    serial, port = ident.get("serial"), ident.get("port_path")
    if vid and pid and serial:
        return {"unique_id": f"usb-{vid}:{pid}:{serial}", "identity": ident,
                "weak": False, "ambiguous": False}
    if vid and pid and port:
        # No serial: identical no-serial units are told apart only by port,
        # and moving the camera makes it look new. Cacheable, but ambiguous.
        return {"unique_id": f"usb-{vid}:{pid}@{port}", "identity": ident,
                "weak": False, "ambiguous": True}
    # Jetson CSI / libcamera / unreadable → index fallback, non-cacheable.
    return {"unique_id": f"{camera_id}-opencv", "identity": ident or {},
            "weak": True, "ambiguous": True}
