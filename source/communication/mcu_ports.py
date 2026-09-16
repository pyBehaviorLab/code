"""USB-serial-number ↔ live device path wrapper for MCU pyboards.

The host OS assigns ``/dev/ttyACMn`` (Linux) and ``COMn`` (Windows)
based on enumeration order, replug a cable or reboot and those
numbers reshuffle, but the physical MCU ↔ Box mapping stays fixed.

This module exposes pyboards by their stable USB **serial number**
(e.g. ``"315535563234"``) and maps to the live device path at connect
time. The GUI shows operators serial numbers only; this module is the
one place that knows about ``ttyACMn`` / ``COMn``.

Matched on the MicroPython VID ``0xF055`` alone, the PID changes with the
board's USB mode (VCP+MSC vs VCP-only after Disable Flash Drive), so filtering
on PID would lose the board the moment the flash drive is toggled. Devices
without a serial-number USB descriptor are excluded; the fix is a firmware
update.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

# MicroPython VID. Match on VID only, the PID is not stable across USB modes.
_PYBOARD_VID = 0xF055

# Display mode, how an MCU is labelled for the operator. Binding identity is
# always the USB serial; this only changes what's shown. Process-wide setting
# mirrored from the global app default + the loaded project's Meta.mcu_display_mode.
#   "hashed" (default), djb2(serial), an 8-hex id identical on every PC/OS
#   "native", the live OS port (COMn / /dev/ttyACMn)
#   "serial", the raw USB serial number
DISPLAY_MODES = ("hashed", "native", "serial")
_DISPLAY_MODE = "hashed"


def set_display_mode(mode: str) -> None:
    """Set the process-wide MCU label mode (one of ``DISPLAY_MODES``).
    Unknown values fall back to ``hashed``."""
    global _DISPLAY_MODE
    _DISPLAY_MODE = mode if mode in DISPLAY_MODES else "hashed"


def hashed_label(mcu_serial: str) -> str:
    """Stable 8-hex djb2 of the USB serial, same on every PC/OS for the
    same physical board. Empty when there's no serial to hash."""
    serial = (mcu_serial or "").strip()
    if not serial:
        return ""
    from source.config.hashing import djb2_hex_from_text
    return djb2_hex_from_text(serial)


def list_mcu_serials() -> List[Tuple[str, str]]:
    """Return ``[(serial_number, device_path), ...]`` for every live
    pyboard. Sorted by serial number for stable dropdown ordering."""
    from serial.tools import list_ports
    out: List[Tuple[str, str]] = []
    for p in list_ports.comports():
        if p.vid == _PYBOARD_VID:
            sn = (p.serial_number or "").strip()
            if sn:
                out.append((sn, p.device))
    out.sort(key=lambda x: x[0])
    return out


def device_for_serial(mcu_serial: str) -> Optional[str]:
    """Map a saved ``mcu_serial`` → current device path. ``None`` when
    that MCU isn't currently plugged in. (Reverse of
    ``serial_for_device``.)"""
    if not mcu_serial:
        return None
    for sn, dev in list_mcu_serials():
        if sn == mcu_serial:
            return dev
    return None


def label_for(mcu_serial: str, device: Optional[str] = None) -> str:
    """Operator-facing label for an MCU under the current display mode.

    Purely cosmetic, the binding identity is always the USB serial
    (``mcu_serial`` / widget ``_mcu_serial``), never this label. ``device``
    is the live port if the caller already resolved it (saves a lookup).

    Falls back gracefully: a board with no serial can't be hashed, so it
    shows its native port; ``native`` mode with the board unplugged falls
    back to the hashed id so the field is never blank.
    """
    serial = (mcu_serial or "").strip()
    if not serial:
        return device or ""
    if _DISPLAY_MODE == "serial":
        return serial
    if _DISPLAY_MODE == "native":
        dev = device if device is not None else device_for_serial(serial)
        return dev or hashed_label(serial)
    return hashed_label(serial)


def tooltip_for(mcu_serial: str, device: Optional[str] = None) -> str:
    """Full identity tooltip, shown regardless of label mode so the
    operator can always see the stable id, the live port, and the raw
    serial at a glance (e.g. when tracing a physical cable)."""
    serial = (mcu_serial or "").strip()
    if not serial:
        return f"currently at {device}" if device else ""
    dev = device if device is not None else device_for_serial(serial)
    return (f"id {hashed_label(serial)}  ·  "
            + (f"{dev}" if dev else "(not plugged in)")
            + f"  ·  serial {serial}")


def serial_for_device(device_path: str) -> Optional[str]:
    """Reverse lookup: device path → serial number. Used to auto-upgrade a
    project that has only ``com_port`` set: the first connect captures the
    serial so the next autosave persists it."""
    if not device_path:
        return None
    for sn, dev in list_mcu_serials():
        if dev == device_path:
            return sn
    return None


__all__ = ["list_mcu_serials", "device_for_serial", "serial_for_device",
           "label_for", "tooltip_for", "hashed_label",
           "set_display_mode", "DISPLAY_MODES"]
