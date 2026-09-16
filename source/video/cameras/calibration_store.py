"""Machine-level camera calibration cache.

Probing a camera's resolution ladder + realistic FPS costs ~5 s per mode, so
the result is worth caching. Unlike a project's ``probed_modes`` (which travel
with the project), this cache is keyed to the *machine*, the achievable FPS
ceiling depends on THIS PC's USB controller and the port the camera is on, so
calibrations are shared across every project on this machine and never across
machines (each PC has its own store file).

Entries are keyed by the camera's USB identity (see ``usb_identity.py``):
``usb-{vid}:{pid}:{serial}`` when a serial exists (survives replug/reboot),
else ``usb-{vid}:{pid}@{port}``. A cache hit additionally requires bus-speed
compatibility, a camera moved from a USB3 port to a USB2 port keeps its
serial but can no longer sustain the USB3 ceiling, so its cache is treated as
stale and re-probed.

Stored file: ``<user config>/pybehaviorlab/camera_calibrations.json``.

Every function is exception-safe: a missing/corrupt file yields an empty store
and never raises into the GUI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from source.log import get_logger

logger = get_logger()

_STORE_FILENAME = "camera_calibrations.json"

# Identity fields copied into each entry for diagnostics + match/staleness.
_IDENTITY_KEYS = ("vid", "pid", "serial", "port_path", "bus_speed", "name")


def _user_config_dir() -> Path:
    """Return the per-user config dir for pybehaviorlab (created on demand)."""
    import sys
    base: str | None = None
    try:
        if os.name == "nt":
            base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        elif sys.platform == "darwin":
            base = str(Path.home() / "Library" / "Application Support")
        else:
            base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    except Exception:
        base = None
    root = Path(base) if base else Path.home()
    d = root / "pybehaviorlab"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.debug("calibration_store: mkdir %s failed: %s", d, e)
    return d


def store_path() -> Path:
    """Absolute path to the calibration JSON file."""
    return _user_config_dir() / _STORE_FILENAME


def load() -> dict:
    """Return the whole store as a dict; ``{}`` if absent or unreadable."""
    p = store_path()
    if not p.is_file():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("calibration_store: could not read %s: %s", p, e)
        return {}


def save(data: dict) -> bool:
    """Write the whole store. True on success, False (logged) on error."""
    p = store_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data or {}, f, indent=2)
        return True
    except Exception as e:
        logger.warning("calibration_store: could not write %s: %s", p, e)
        return False


def _speed_compatible(stored: dict, context: dict | None) -> bool:
    """False when the port's negotiated bus speed changed (USB3 -> USB2).

    Only decides "incompatible" when BOTH sides report a speed and they
    differ; unknown speeds don't invalidate a hit (best-effort).
    """
    ctx = context or {}
    a = stored.get("bus_speed")
    b = ctx.get("bus_speed")
    return not (a and b and str(a) != str(b))


def get(unique_id: str, context: dict | None = None) -> list | None:
    """Return cached ``[(w, h, fps), ...]`` for ``unique_id``, or ``None``.

    A hit requires the id to be present AND the USB context to be compatible
    (same bus speed, a slower port re-probes). ``context`` is the identity
    dict from ``usb_identity.resolve_identity``. ``None`` means "miss, probe".
    """
    if not unique_id:
        return None
    entry = load().get(str(unique_id))
    if not isinstance(entry, dict):
        return None
    # Index-only (weak) entries ARE honoured: they are keyed by
    # ``<camera_id>-<backend>`` (per-PC), so they only mismatch if the same
    # index is later a different physical camera (port reorder). Accepted
    # trade-off, the user was warned at calibration time.
    if not _speed_compatible(entry, context):
        logger.info("calibration_store: %s stale (bus speed changed), re-probe",
                    unique_id)
        return None
    modes: list = []
    for m in (entry.get("modes") or []):
        try:
            modes.append((int(m[0]), int(m[1]), float(m[2])))
        except (TypeError, ValueError, IndexError):
            continue
    return modes or None


def put_offered(unique_id: str, modes) -> bool:
    """Record what a camera OFFERS: its sizes, formats and rates.

    Kept apart from ``variants``, which is what the camera was measured
    DELIVERING. They answer different questions and neither may overwrite the
    other. Offered is a property of the hardware and the link and changes only
    when the camera is moved; delivered is a property of the room and the load
    and changes minute to minute, which is exactly why measuring could never
    produce a correct list of offered rates.

    ``modes`` is a sequence of ``OfferedMode``. Storing an empty list is a
    no-op, because empty means the question could not be ASKED and must not be
    written over an answer that was.
    """
    if not unique_id or not modes:
        return False
    data = load()
    entry = data.get(str(unique_id))
    if not isinstance(entry, dict):
        entry = {"backend": "opencv", "label": "", "modes": [],
                 "variants": {}, "weak": True}
        data[str(unique_id)] = entry
    entry["offered"] = [
        {"width": int(m.width), "height": int(m.height),
         "pixel_format": str(m.pixel_format),
         "rates": [float(r) for r in (m.rates or ())]}
        for m in modes
    ]
    entry["offered_at"] = datetime.now().isoformat(timespec="seconds")
    logger.info("calibration %s: recorded %d offered (size, format) "
                "combination(s).", unique_id, len(entry["offered"]))
    return save(data)


def get_offered(unique_id: str) -> list:
    """What ``unique_id`` offers, as ``OfferedMode``, or ``[]`` if unknown.

    ``[]`` means nobody has enumerated this camera, not that it has no modes.
    """
    if not unique_id:
        return []
    entry = load().get(str(unique_id))
    if not isinstance(entry, dict):
        return []
    from source.video.cameras.enumerate_modes import OfferedMode
    out = []
    for row in (entry.get("offered") or []):
        try:
            out.append(OfferedMode(
                int(row["width"]), int(row["height"]),
                str(row["pixel_format"]),
                tuple(float(r) for r in (row.get("rates") or ()))))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _rejection_key(size, pixel_format) -> str:
    """Where a rejection is filed: door, SIZE and FORMAT all change the answer.

    The format is not optional detail. On this rig one camera advertises
    640x480 at 120 fps in MJPEG and at 30 in YUY2, so "30 was not honoured at
    640x480" is only true of one of them. Filed without the format, a
    rejection learned in YUY2 deleted 30 fps from the MJPEG list, which is the
    one rate a behavioural rig actually wants.
    """
    fmt = str(pixel_format or "any").lower() or "any"
    return f"{int(size[0])}x{int(size[1])}/{fmt}"


def note_rate_rejected(unique_id: str, backend: str, size, asked, got,
                       pixel_format=None) -> None:
    """Remember that this camera does not offer ``asked`` fps at ``size``.

    A UVC camera exposes a handful of discrete rates and silently rounds to
    the nearest one it has, reporting the request back unchanged. The picker
    offers a ladder that is a superset of them, so a rate the camera turns out
    not to have is learned here rather than probed for: probing every rate at
    every size would multiply an already slow calibration.

    Recorded per door, per size AND per format, because all three change the
    answer, and stamped, because a rejection is only good until the camera is
    asked what it offers (see ``get_rejected_rates``).
    """
    if not unique_id or not size:
        return
    try:
        key = _rejection_key(size, pixel_format)
        asked = round(float(asked))
    except (TypeError, ValueError, IndexError):
        return
    data = load()
    entry = data.get(str(unique_id))
    if not isinstance(entry, dict):
        return
    rejected = entry.setdefault("rejected_fps", {})
    per_door = rejected.setdefault(str(backend or "opencv"), {})
    at_size = per_door.setdefault(key, [])
    if asked in at_size:
        return
    at_size.append(asked)
    at_size.sort()
    stamp = datetime.now().isoformat(timespec="seconds")
    entry["updated"] = stamp
    entry["rejected_at"] = stamp
    logger.info(
        "calibration %s: %s does not offer %d fps at %s (it delivered %.1f); "
        "it will not be offered for this camera again.",
        unique_id, backend, asked, key, float(got or 0.0))
    save(data)


def get_rejected_rates(unique_id: str, backend: str, size,
                       pixel_format=None) -> set:
    """Rates this camera has been SEEN not to offer at ``size``, or empty.

    A rejection filed without a format (the old key shape) is NOT honoured: it
    was recorded against whatever format happened to be open and then applied
    to all of them, which is the bug itself. There is no way to tell after the
    fact which format it belonged to, so it is dropped rather than guessed at.

    Everything here is cleared by a fresh Detect; see ``put(replace=True)``.
    """
    if not unique_id or not size:
        return set()
    try:
        key = _rejection_key(size, pixel_format)
        any_key = _rejection_key(size, None)
    except (TypeError, ValueError, IndexError):
        return set()
    entry = load().get(str(unique_id))
    if not isinstance(entry, dict):
        return set()
    per_door = (entry.get("rejected_fps") or {}).get(str(backend or "opencv"))
    if not isinstance(per_door, dict):
        return set()
    out: set = set()
    for k in ({key, any_key} if key != any_key else {key}):
        for v in (per_door.get(k) or []):
            if isinstance(v, (int, float)):
                out.add(int(v))
    return out


def get_variants(unique_id: str, context: dict | None = None) -> dict:
    """``{backend_name: [(w, h, fps), ...]}`` for ``unique_id``, or ``{}``.

    What each OS capture backend can do with this camera, on this machine.
    Kept here rather than in the project file for the reason this whole store
    exists: the achievable rate is a property of THIS PC's USB controller and
    the port, not of the experiment, so it must not travel with a project to
    a different rig and be believed there.
    """
    if not unique_id:
        return {}
    entry = load().get(str(unique_id))
    if not isinstance(entry, dict) or not _speed_compatible(entry, context):
        return {}
    out: dict = {}
    for name, modes in (entry.get("variants") or {}).items():
        clean = []
        for m in (modes or []):
            try:
                clean.append((int(m[0]), int(m[1]), float(m[2])))
            except (TypeError, ValueError, IndexError):
                continue
        # An EMPTY list is kept, and means "this door was measured and could
        # not serve this camera". Dropping it made that indistinguishable from
        # a door never tried, so the picker kept offering it, every time.
        if True:
            out[str(name)] = clean
    return out


def get_variants_for_format(unique_id: str, pixel_format: str) -> dict:
    """``{backend: [(w, h, fps)]}`` measured IN ``pixel_format``, or ``{}``.

    ``{}`` means this camera has not been measured in that format, which is
    different from a camera that cannot serve it, and the caller must offer
    Detect rather than an empty list.
    """
    if not unique_id or not pixel_format:
        return {}
    entry = load().get(str(unique_id))
    if not isinstance(entry, dict):
        return {}
    per_fmt = (entry.get("by_format") or {}).get(str(pixel_format).lower())
    if not isinstance(per_fmt, dict):
        return {}
    out: dict = {}
    for name, modes in per_fmt.items():
        clean = []
        for m in (modes or []):
            try:
                clean.append((int(m[0]), int(m[1]), float(m[2])))
            except (TypeError, ValueError, IndexError):
                continue
        out[str(name)] = clean
    return out


def put(unique_id: str, backend: str, label: str,
        modes: list, identity: dict | None = None,
        weak: bool = False, variants: dict | None = None,
        replace: bool = False, pixel_format: str | None = None) -> bool:
    """Insert/replace the calibration for ``unique_id``.

    ``replace`` is what pressing Detect means: the operator asked the camera
    again, so the answer they just got IS the record. Nothing measured before
    is merged in and nothing learned before survives, because the reason to
    press Detect is that the stored numbers are not trusted. Without it a
    camera whose rates were all measured wrong stayed wrong, since the merge
    below keeps whichever reading was HIGHER and the rejections kept deleting
    rates no matter what the new measurement said.

    ``weak`` entries (index-only, no USB descriptor) ARE written now, under a
    ``<camera_id>-<backend>`` fallback key scoped to this PC. They only
    mismatch if the same index later maps to a different physical camera
    (port reorder), the calibration UI warns about this. Returns save success.
    """
    if not unique_id:
        return False
    data = load()
    clean_modes = []
    for m in (modes or []):
        try:
            clean_modes.append([int(m[0]), int(m[1]), float(m[2])])
        except (TypeError, ValueError, IndexError):
            continue
    ident = identity or {}
    entry: dict[str, Any] = {
        "backend": str(backend or "opencv"),
        "label": str(label or ""),
        "modes": clean_modes,
        # Per-backend measurements. ``modes`` above stays the CHOSEN
        # backend's, so every existing reader is unaffected.
        "variants": {
            str(name): [[int(m[0]), int(m[1]), float(m[2])]
                        for m in (mlist or [])
                        if len(m) >= 3]
            for name, mlist in (variants or {}).items()
        },
        "weak": bool(weak),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    # The flat list is the best of every door, so it cannot disagree with the
    # per-door lists it is drawn from.
    flat: dict = {}
    for mlist in entry["variants"].values():
        for w, h, f in (tuple(m[:3]) for m in (mlist or [])):
            key = (int(w), int(h))
            if key not in flat or float(f) > flat[key]:
                flat[key] = float(f)
    if flat:
        entry["modes"] = sorted(([w, h, f] for (w, h), f in flat.items()),
                                key=lambda m: -(m[0] * m[1]))

    for k in _IDENTITY_KEYS:
        entry[k] = ident.get(k)
    # Anything this function does not own is CARRIED FORWARD. ``put`` builds a
    # fresh entry and replaces the old one, so every field written by another
    # writer was silently discarded by the next probe: the enumeration
    # (``offered``) and the rates a camera was seen to round away
    # (``rejected_fps``) both come from elsewhere and both answer questions
    # this function is not answering.
    previous_entry = data.get(str(unique_id))
    if isinstance(previous_entry, dict):
        for key, value in previous_entry.items():
            if key not in entry:
                entry[key] = value
    # Measurements filed under the FORMAT they were taken in. The rate a
    # size reaches depends on the format, so one flat table per backend cannot
    # hold both answers: a YUY2 Detect silently overwrote the MJPEG numbers
    # and the combo then showed rates from a format the operator had not
    # chosen. ``by_format`` is the authoritative table; the flat ``variants``
    # above stays as the merged view every older reader still uses.
    if pixel_format:
        # COPIED, not aliased. The carry-forward above hands over the previous
        # entry's own nested dict, so writing this format into it mutated the
        # record being compared against and a door measured last time vanished.
        fmt_key = str(pixel_format).lower()
        by_format = {k: dict(v) for k, v in (entry.get("by_format") or {}).items()
                     if isinstance(v, dict)}
        # Merged WITHIN this format only. A door carries a number for the
        # format it was measured in and for no other: this camera's MJPEG and
        # YUY2 rates at one size are 120 and 30, so letting an MJPEG figure
        # stand in for a door's YUY2 entry states a rate nobody measured.
        merged = dict(by_format.get(fmt_key) or {})
        merged.update(dict(entry["variants"]))
        by_format[fmt_key] = merged
        entry["by_format"] = by_format
    if replace:
        # A re-measurement supersedes what was learned from the old one, for
        # the doors it actually measured. A door that was NOT part of this
        # Detect keeps what it had: measuring Media Foundation is not a
        # statement about DirectShow, exactly as measuring MJPEG is not a
        # statement about YUY2.
        was = (previous_entry or {}).get("variants") or {}
        for name, modes in was.items():
            if name not in (variants or {}):
                entry["variants"][name] = modes
        # ``by_format`` was already merged within its own format above; the
        # flat ``variants`` must NOT be copied over it, because that table
        # spans formats and would import one format's rate into another.
        entry.pop("rejected_fps", None)
        entry.pop("rejected_at", None)
        data[str(unique_id)] = entry
        logger.info("calibration %s: replaced by a fresh Detect (%d mode(s)); "
                    "previous measurements and learned rejections discarded.",
                    unique_id, len(entry.get("modes") or []))
        return save(data)
    # A door that measured NOTHING this time does not erase a door that
    # measured something last time. A probe can fail for reasons that have
    # nothing to do with the camera: another process holding it, or the OS
    # still tearing down the previous handle. Overwriting a good measurement
    # with an empty one threw away what the rig had learned and, worse, made
    # a working door look permanently dead.
    # MERGED with what this door measured before, keeping the HIGHER rate for
    # each size. A rate is a CEILING, and measuring one is noisy: the same
    # 1920x1080 mode read 26.7 fps on a quiet device and 17.2 fps while the
    # camera was being probed repeatedly. Replacing outright meant one bad
    # reading permanently downgraded a mode, and the rig then offered a rate
    # below what the camera can actually hold. The highest reading on THIS PC
    # is the best estimate of the ceiling, which is what the whole per-machine
    # store exists to record.
    previous = (data.get(str(unique_id)) or {}).get("variants") or {}
    for name, was in previous.items():
        if not was:
            continue
        now = entry["variants"].get(name) or []
        if not now:
            entry["variants"][name] = was
            logger.info(
                "calibration %s: %s measured nothing this time; keeping the "
                "%d mode(s) it measured before rather than recording it as a "
                "door that cannot serve this camera.",
                unique_id, name, len(was))
            continue
        best: dict = {}
        raised = 0
        for w, h, f in ([tuple(m[:3]) for m in was]
                        + [tuple(m[:3]) for m in now]):
            key = (int(w), int(h))
            if key not in best or float(f) > best[key]:
                best[key] = float(f)
        for w, h, f in (tuple(m[:3]) for m in now):
            if best[(int(w), int(h))] > float(f):
                raised += 1
        entry["variants"][name] = [[w, h, f] for (w, h), f in best.items()]
        if raised:
            logger.info(
                "calibration %s: %s measured %d mode(s) slower than a previous "
                "run; kept the faster reading, a rate is a ceiling and "
                "measuring one is noisy.", unique_id, name, raised)
    data[str(unique_id)] = entry
    return save(data)
