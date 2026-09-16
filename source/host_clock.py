"""The one clock the host measures itself by.

Every instant that is ever *compared with another instant*, a frame's capture
time, a recording's start, an inference's completion, a push to the board, the
anchor that maps host time onto MCU framework time, comes from here. That is
the whole point of the module: those values share an epoch by construction
rather than by coincidence, so no future edit can put two of them on different
clocks.

**Why not ``time.monotonic()``.** On Windows it is ``GetTickCount64()``, whose
resolution is **15.625 ms**, measured on this rig and unchanged by
``timeBeginPeriod(1)`` (Windows 11 no longer lets a process raise it). A frame
interval at 20 fps is 50 ms, so capture instants landed on one of three values
per interval: every latency figure derived from them was quantised to a tick,
and ``frame_ts_ms`` in a recording was quantised with them.
``time.perf_counter()`` is ``QueryPerformanceCounter()`` at 100 ns, some
156,000 times finer, and is monotonic and non-adjustable, which are the only
other properties this needs.

**Why this is safe on Linux and the Jetson.** There, CPython implements BOTH
``time.monotonic()`` and ``time.perf_counter()`` as
``clock_gettime(CLOCK_MONOTONIC)``: same clock, same epoch, nanosecond
resolution. The change is a no-op on those platforms and the entire fix on
Windows. It is *checked* rather than assumed, see ``describe()`` and the
warning below, because a clock that silently got worse is exactly the kind of
thing that shows up months later as unexplainable timing.

**Why the two halves could not be migrated separately.** Both clocks tick at
the same rate, so it is tempting to measure the offset once and convert at the
boundary. Measured here, the offset between them varies by **15.4 ms** across
samples, the coarse clock's own quantisation, which is precisely the error
being removed. There is no stable offset to convert through.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

#: Coarser than this and the clock cannot resolve a frame interval, which is
#: the job. 1 ms is comfortably finer than the fastest camera the rig runs
#: (200 fps → 5 ms) and comfortably coarser than any real implementation.
COARSE_NS = 1_000_000


def host_ns() -> int:
    """Now, in nanoseconds, on the host's shared monotonic epoch."""
    return time.perf_counter_ns()


def host_s() -> float:
    """Now, in seconds, on the same epoch as :func:`host_ns`."""
    return time.perf_counter()


def resolution_ns() -> int:
    """The clock's resolution, in nanoseconds, as the platform reports it."""
    return round(time.get_clock_info("perf_counter").resolution * 1e9)


def describe() -> str:
    """One line naming the clock and what it can resolve."""
    info = time.get_clock_info("perf_counter")
    return (f"{info.implementation}, resolution "
            f"{info.resolution * 1e3:.6f} ms, monotonic={info.monotonic}")


def check() -> bool:
    """Warn if this platform's clock cannot do the job. True when it can.

    Called once at start-up. A clock too coarse to separate two frames does
    not stop the rig, recording and tracking are unaffected, but every
    latency number and every ``frame_ts_ms`` it produces is then quantised,
    and that has to be said out loud rather than discovered later in the data.
    """
    info = time.get_clock_info("perf_counter")
    if not info.monotonic:
        logger.warning(
            "the host clock (%s) is not monotonic on this platform, so frame "
            "ordering and every elapsed time derived from it can go backwards",
            info.implementation)
        return False
    if resolution_ns() > COARSE_NS:
        logger.warning(
            "the host clock (%s) resolves only %.3f ms. Timestamps and "
            "latency measurements will be quantised to that step, which is "
            "coarser than the %.1f ms this build expects.",
            info.implementation, resolution_ns() / 1e6, COARSE_NS / 1e6)
        return False
    logger.info("host clock: %s", describe())
    return True


__all__ = ["COARSE_NS", "check", "describe", "host_ns", "host_s",
           "resolution_ns"]
