"""A calibration is accumulated, not replaced.

Measuring a frame rate is noisy. The same 1920x1080 mode on this rig read
26.7 fps on a quiet device and 17.2 fps while the camera was being probed
repeatedly, and a probe can fail outright for reasons that say nothing about
the camera: another process holding it, or the OS still tearing down the
previous handle.

Replacing outright meant one bad run permanently downgraded a mode, and one
failed run recorded a working door as a door that cannot serve the camera.
Both happened on the rig, in the same evening, and both are silent: the
picker simply offers less than the camera can do.

A rate is a CEILING, so the highest reading on THIS machine is the best
estimate of it, which is exactly what a per-machine store exists to hold.
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from source.video.cameras import calibration_store as store

KEY = "unit-test-camera"


@pytest.fixture(autouse=True)
def _clean():
    yield
    p = (pathlib.Path(os.environ.get("LOCALAPPDATA") or pathlib.Path.home())
         / "pybehaviorlab" / "camera_calibrations.json")
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.pop(KEY, None) is not None:
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")


def _put(variants):
    modes = [m for ms in variants.values() for m in ms]
    store.put(KEY, "opencv", "t", modes, {}, variants=variants)
    return store.get_variants(KEY)


def test_a_slower_reading_does_not_downgrade_a_mode():
    _put({"msmf": [[1920, 1080, 26.7]]})
    got = _put({"msmf": [[1920, 1080, 17.2]]})
    assert got["msmf"] == [(1920, 1080, 26.7)], (
        "a noisy re-measurement replaced a rate the camera had already held")


def test_a_faster_reading_raises_the_ceiling():
    _put({"dshow": [[640, 480, 29.9]]})
    got = _put({"dshow": [[640, 480, 30.4]]})
    assert got["dshow"] == [(640, 480, 30.4)]


def test_a_door_that_measured_nothing_keeps_what_it_had():
    """A failed probe is not evidence that the door cannot serve the camera."""
    _put({"msmf": [[1920, 1080, 26.7]], "dshow": [[640, 480, 29.9]]})
    got = _put({"msmf": [], "dshow": []})
    assert got["msmf"] == [(1920, 1080, 26.7)]
    assert got["dshow"] == [(640, 480, 29.9)]


def test_a_door_that_has_never_measured_anything_stays_empty():
    """Empty is still meaningful: it says this door WAS tried and failed.

    Without that, a door tried and found useless is indistinguishable from
    one never tried, and the picker keeps offering it for ever.
    """
    got = _put({"dshow": [[640, 480, 29.9]], "msmf": []})
    assert got.get("msmf") == [], got


def test_sizes_from_different_runs_accumulate():
    """Two runs that each saw part of the ladder end up with all of it."""
    _put({"dshow": [[1920, 1080, 5.0]]})
    got = _put({"dshow": [[640, 480, 29.9]]})
    assert dict(((w, h), f) for w, h, f in got["dshow"]) == {
        (1920, 1080): 5.0, (640, 480): 29.9}


def test_the_flat_list_is_the_best_of_every_door():
    """``modes`` is what older readers use, so it must not contradict the
    per-door lists it is drawn from."""
    _put({"dshow": [[1920, 1080, 5.0]], "msmf": [[1920, 1080, 30.4]]})
    flat = store.get(KEY) or []
    assert (1920, 1080, 30.4) in [tuple(m) for m in flat], flat
