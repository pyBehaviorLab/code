"""Process-wide datetime format constants, single source of truth.

Every on-disk timestamp in the system uses one of three formats below.
DO NOT bake format strings inline at call sites; import these so the
convention can be changed in exactly one place.

  * ``DATE_DIR_FMT``     ``YYYY-MM-DD``           daily folder names.
                                                  Sortable lexicographically.
  * ``FILE_STEM_TS_FMT`` ``YYYY-MM-DD-HHMMSS``    file-stem timestamp.
                                                  Filename-safe (no
                                                  colons), sortable.
  * ``HEADER_TS_FMT``    ``YYYY-MM-DD HH:MM:SS``  text headers, JSON
                                                  values, and human-
                                                  readable UI labels.
                                                  Space separator (no
                                                  ``T``). Sortable
                                                  lexicographically.

Banned formats (DO NOT use anywhere):
  * ``%d%m%y``: not sortable, locale-confusable, 2-digit year.
  * ``%Y%m%d_%H%M%S``: replaced by ``FILE_STEM_TS_FMT`` for
    consistency with the session-stem convention.
  * ``%Y-%m-%dT%H:%M:%S``: the ``T`` separator is intentionally
    avoided; use the space-separated ``HEADER_TS_FMT`` everywhere.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional


DATE_DIR_FMT     = "%Y-%m-%d"
FILE_STEM_TS_FMT = "%Y-%m-%d-%H%M%S"
HEADER_TS_FMT    = "%Y-%m-%d %H:%M:%S"


def format_date_dir(dt: Optional[datetime] = None) -> str:
    """``YYYY-MM-DD``, daily data folder name."""
    return (dt or datetime.now()).strftime(DATE_DIR_FMT)


def format_header_ts(dt: datetime) -> str:
    """``YYYY-MM-DD HH:MM:SS``, header / JSON / UI label timestamp.

    Space-separated, no ``T``. Sortable lexicographically and
    immediately readable by humans without needing to mentally split
    on the ``T``.
    """
    return dt.strftime(HEADER_TS_FMT)


def format_header_ts_ms(dt: datetime) -> str:
    """``YYYY-MM-DD HH:MM:SS.mmm``, header timestamp with millisecond
    precision. Same space-separated, no-``T`` convention as
    :func:`format_header_ts`; used by writers that need sub-second
    resolution (frame log session_start, drop log row, etc.)."""
    return f"{dt.strftime(HEADER_TS_FMT)}.{dt.microsecond // 1000:03d}"


def format_run_clock(ms) -> str:
    """``HH:MM:SS`` from milliseconds since framework run start.

    Single source of truth for every *session* elapsed clock, the box-card
    timer, the LiveStatus mirror, the plot's on-canvas clock, and the
    frozen value at stop. The input is the MCU framework time
    (``pycboard.get_timestamp()`` / ``pycboard.timestamp``), so the display
    always matches the TSV timestamps. Clamped at 0 so a pre-first-message
    read never shows a negative clock.
    """
    s = max(0, int(ms) // 1000)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


__all__ = [
    "DATE_DIR_FMT",
    "FILE_STEM_TS_FMT",
    "HEADER_TS_FMT",
    "format_date_dir",
    "format_header_ts",
    "format_header_ts_ms",
    "format_run_clock",
]
