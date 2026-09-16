"""DJB2 hash helpers, the project's sole hash function.

Two flavours:

* :func:`djb2_int_from_file` / :func:`djb2_hex_from_file` read the file in
  4-byte little-endian chunks. This matches the algorithm pyControl writes
  into the MCU TSV header as ``task_file_hash`` / ``hardware_def_hash`` so
  analysis tooling can join across log surfaces by hash.

* :func:`djb2_hex_from_bytes` / :func:`djb2_hex_from_text` apply the
  textbook byte-by-byte djb2 to arbitrary content. Used for config
  self-hash, DLC manifests, snapshot keys, etc.

``source.communication.pycboard`` carries its own ``_djb2_file`` that
mirrors :func:`djb2_int_from_file` byte-for-byte. That duplicate is kept
intentionally so the MCU layer stays self-contained (the
``feedback_mcu_layer_compact`` rule).

That justification only holds while the duplicate stays *inside* the MCU
layer. Host code must import from here, never from ``pycboard``,
importing the private mirror from the GUI makes the MCU layer a
dependency of the GUI's hashing, which is the coupling the duplicate
exists to avoid.
"""
from __future__ import annotations

from pathlib import Path


def djb2_int_from_file(path: str | Path) -> int:
    """4-byte-LE-chunked djb2 of a file. Matches pyControl MCU upload hash.

    Returns 0 on OSError so callers can treat a missing file as a stable
    "no hash" sentinel without an extra try/except at every site.
    """
    h = 5381
    try:
        with open(path, "rb") as f:
            while True:
                c = f.read(4)
                if not c:
                    break
                h = ((h << 5) + h + int.from_bytes(c, "little")) & 0xFFFFFFFF
    except OSError:
        return 0
    return h


def djb2_hex_from_file(path: str | Path) -> str:
    """8-char zero-padded hex of :func:`djb2_int_from_file`. Used as the
    on-disk filename inside ``<project>/source/<djb2>.py``."""
    return f"{djb2_int_from_file(path):08x}"


def djb2_hex_from_bytes(data: bytes) -> str:
    """Byte-wise djb2 of ``data``. Distinct from the file form: this is
    the textbook algorithm applied to arbitrary content."""
    h = 5381
    for b in data:
        h = ((h << 5) + h + b) & 0xFFFFFFFF
    return f"{h:08x}"


def djb2_hex_from_text(text: str) -> str:
    """UTF-8 encode + :func:`djb2_hex_from_bytes`. Convenience for
    canonical-json hashing and other text inputs."""
    return djb2_hex_from_bytes(text.encode("utf-8"))


__all__ = [
    "djb2_int_from_file",
    "djb2_hex_from_file",
    "djb2_hex_from_bytes",
    "djb2_hex_from_text",
]
