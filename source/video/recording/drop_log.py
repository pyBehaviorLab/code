"""Per-session drop log: every dropped frame, every drop site.

The single ``drop_log`` instance exposed at module level is the live drop
recorder the pipeline calls into. Always on (cheap; only writes when a drop
happens); set its session path via ``drop_log.set_session_path(path)``.

Output: a TSV (``_drops.tsv``) with columns
    wall_iso  monotonic_ms  stream  frame_idx  capture_ts  reason
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional


class DropLog:
    """Append-only per-session TSV of dropped frames.

    Columns: wall_iso, monotonic_ms, stream, frame_idx, capture_ts, reason
      stream     -- 'recorder' | 'tracker' | 'camera_ring' | 'mcu_push' | other
      frame_idx  -- monotonic counter from the dropping subsystem (or 'na')
      capture_ts -- camera capture timestamp seconds (or 'na')
      reason     -- short tag, e.g. 'queue_full', 'inflight_box3'

    The file is opened lazily on the first drop after set_session_path() is
    called. Until a path is set, drops are buffered in memory (bounded ring,
    1024 entries) so they aren't lost between app start and the first session.
    """

    _RING_LEN = 1024

    # Batched flush: 64 KB buffer with a 500 ms periodic flush, so a drop
    # cascade (multi-box buffer overflow) doesn't fsync on every line.
    _BUFFER_SIZE = 64 * 1024
    _FLUSH_INTERVAL_S = 0.5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path: Optional[Path] = None
        self._fp = None
        self._ring: deque[str] = deque(maxlen=self._RING_LEN)
        # Per-stream live counters for cheap UI surface (queryable any time).
        self._counts: dict[str, int] = {}
        self._t0 = time.monotonic()
        self._last_flush = self._t0

    # ---- configuration -------------------------------------------------

    def set_session_path(self, path: str | Path) -> None:
        """Point the log at this session's _drops.tsv. Idempotent for the
        same path; rotates the file when called with a different path."""
        new_path = Path(path)
        with self._lock:
            if self._path == new_path and self._fp is not None:
                return
            self._close_locked()
            self._path = new_path
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                self._fp = new_path.open(
                    "a", encoding="utf-8", buffering=self._BUFFER_SIZE)
                self._last_flush = time.monotonic()
                if self._fp.tell() == 0:
                    self._fp.write(
                        "wall_iso\tmonotonic_ms\tstream\tframe_idx\tcapture_ts\treason\n"
                    )
                # Drain anything buffered before a path was known.
                while self._ring:
                    self._fp.write(self._ring.popleft())
            except Exception:
                self._fp = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._fp is not None:
            try:
                self._fp.flush()
                self._fp.close()
            except Exception:
                pass
        self._fp = None

    # ---- record --------------------------------------------------------

    def record(self,
               stream: str,
               frame_idx: Optional[int] = None,
               capture_ts: Optional[float] = None,
               reason: str = "") -> None:
        """Log one drop. Cheap and safe to call from any thread."""
        # HEADER_TS_FMT (space-separated, no T), same convention every
        # other writer in the project uses.
        wall_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        mono_ms = (time.monotonic() - self._t0) * 1000.0
        idx_s = "na" if frame_idx is None else str(frame_idx)
        ts_s = "na" if capture_ts is None else f"{capture_ts:.6f}"
        line = f"{wall_iso}\t{mono_ms:.3f}\t{stream}\t{idx_s}\t{ts_s}\t{reason}\n"

        with self._lock:
            self._counts[stream] = self._counts.get(stream, 0) + 1
            if self._fp is not None:
                try:
                    self._fp.write(line)
                except Exception:
                    self._ring.append(line)
                # Periodic flush, keeps drop-log readable in
                # near-real-time without an fsync per drop in cascades.
                now = time.monotonic()
                if now - self._last_flush >= self._FLUSH_INTERVAL_S:
                    try:
                        self._fp.flush()
                    except Exception:
                        pass
                    self._last_flush = now
            else:
                self._ring.append(line)

    # ---- query ---------------------------------------------------------

    def reset_counts(self) -> None:
        with self._lock:
            self._counts.clear()


# Module-level singleton, what every drop site imports.
drop_log = DropLog()


__all__ = ["drop_log", "DropLog"]
