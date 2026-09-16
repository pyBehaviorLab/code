"""One clock for every instant that is compared with another instant.

A frame's capture time, the recording's start, the moment inference finished,
the last push to the board, and the anchor that maps host time onto MCU
framework time are all subtracted from one another. They are only meaningful
on a shared epoch, and nothing in the type system says so, which is how they
came to be spread across two clocks with 15.6 ms between them.

So it is asserted here instead: the properties the clock must have, and a
guard that fails if a shared-epoch module goes back to reading the wall on its
own.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

import pytest

from source import host_clock

_SOURCE = Path(__file__).resolve().parents[2]

#: Every module that produces or consumes an instant on the shared epoch.
#: Timeouts are NOT in this list and are none of this test's business, a
#: ``deadline = time.monotonic() + 2.0`` is compared only with itself, so its
#: clock can be whatever is cheapest.
SHARED_EPOCH = (
    "video/cameras/capture.py",
    "video/cameras/opencv.py",
    "video/framebus/frame_bus.py",
    "video/framebus/latency.py",
    "video/framebus/mcu_pusher.py",
    "video/framebus/pose_sink.py",
    "video/framebus/recorder_sink.py",
    "video/framebus/sink_base.py",
    "video/recording/recorder.py",
    "communication/pycboard.py",
)


# ── the clock itself ─────────────────────────────────────────────────────

def test_the_clock_is_monotonic():
    """Frame ordering and every elapsed time depend on it."""
    assert time.get_clock_info("perf_counter").monotonic
    readings = [host_clock.host_ns() for _ in range(2000)]
    assert readings == sorted(readings)


def test_the_clock_can_resolve_a_frame():
    """The failure this exists to prevent: a clock coarser than the interval
    it is measuring. On Windows ``time.monotonic()`` is GetTickCount64 at
    15.625 ms, which cannot separate two frames at 20 fps."""
    assert host_clock.resolution_ns() <= host_clock.COARSE_NS
    assert host_clock.check() is True


def test_seconds_and_nanoseconds_share_an_epoch():
    """``host_s`` and ``host_ns`` are both used against ``capture_host_ns``,
    ``pycboard`` anchors in seconds, the cameras stamp in nanoseconds."""
    a = host_clock.host_ns()
    s = host_clock.host_s()
    b = host_clock.host_ns()
    assert a / 1e9 <= s <= b / 1e9


def test_the_clock_actually_advances_at_the_rate_of_time():
    start = host_clock.host_ns()
    time.sleep(0.05)
    elapsed_ms = (host_clock.host_ns() - start) / 1e6
    assert 45.0 <= elapsed_ms <= 200.0


def test_it_is_finer_than_the_clock_it_replaced():
    """Not a tautology on Windows, and a no-op on Linux and the Jetson, where
    both calls are ``clock_gettime(CLOCK_MONOTONIC)``. Stated as >= so the
    platforms where they are the same clock still pass."""
    coarse = time.get_clock_info("monotonic").resolution
    assert time.get_clock_info("perf_counter").resolution <= coarse


# ── the guard ────────────────────────────────────────────────────────────

def _clock_calls(path: Path):
    """(lineno, dotted name) for every ``time.monotonic*`` call in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "time"
                and func.attr in ("monotonic", "monotonic_ns")):
            yield node.lineno, f"time.{func.attr}()"


def _is_a_shared_instant(path: Path, lineno: int) -> bool:
    """Whether this reading meets another clock, or only ever itself.

    Two mechanical rules, both true of this codebase:

    * ``time.monotonic_ns()`` is always an instant here, every nanosecond
      reading in these modules ends up in a ``*_host_ns`` / ``*_done_ns``
      field that something else subtracts from.
    * ``time.monotonic()`` in seconds is an anchor when it is stored on
      ``self``, and a local timeout otherwise (``deadline = ... + 2.0``,
      ``while ... < t_end``).

    What it deliberately does NOT catch: a seconds reading kept in a local and
    passed to a collaborator that differences it, the trigger feature
    tracker's ``now_s`` was one, and had to be found by reading. A guard that
    tried to chase those would be guessing; this one states a rule it can
    actually enforce.
    """
    line = path.read_text(encoding="utf-8").splitlines()[lineno - 1]
    if "monotonic_ns" in line:
        return True
    left = line.split("=")[0] if "=" in line else ""
    return left.strip().startswith("self.")


@pytest.mark.parametrize("rel", SHARED_EPOCH)
def test_shared_epoch_modules_do_not_read_their_own_clock(rel):
    offenders = []
    path = _SOURCE / rel
    for lineno, name in _clock_calls(path):
        if _is_a_shared_instant(path, lineno):
            offenders.append(f"{rel}:{lineno} calls {name}")
    assert not offenders, (
        "these instants are compared against frame capture times and the MCU "
        "anchor, so they must come from source.host_clock, a second clock "
        "here is a silent 15.6 ms error on Windows:\n  "
        + "\n  ".join(offenders))


def test_the_mcu_anchor_uses_the_shared_clock():
    """The one that makes ``mcu_ts_ms`` mean anything: ``fw_ms_at`` is handed
    a camera's ``capture_host_ns``, so the anchor it subtracts must be on the
    same epoch as the cameras."""
    text = (_SOURCE / "communication" / "pycboard.py").read_text(
        encoding="utf-8")
    assert "from source import host_clock" in text
    for line in text.splitlines():
        # The anchor may be set from a local that was itself read from the
        # shared clock a line earlier; what must never appear is the coarse
        # clock feeding it directly.
        if "_last_message_mono =" in line:
            assert "time.monotonic" not in line, (
                f"the MCU anchor is set from the coarse clock, so every "
                f"mcu_ts_ms it produces is quantised: {line.strip()}")
