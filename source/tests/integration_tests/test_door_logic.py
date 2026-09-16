"""End-to-end logic test for the Door device (devices/door.py).

door.py is MCU MicroPython code, so we mock the modules it imports
(``micropython``/``pyb``/``pyControl.hardware``), load it directly by path
(bypassing the devices package __init__ that would pull in every driver), and
drive the move by calling the step-timer ISR ``_tick`` by hand while toggling
the limit pin, exercising the real ramp / limit / step-count / position logic.

Focus: the bottom/open limit is MANDATORY; the top/close limit is OPTIONAL,
a door with no top switch must close by step count (not run away) and report
is_closed() from the tracked position.
"""

import os
import sys
import types
import importlib.util

import pytest


def _install_mcu_mocks():
    mp = types.ModuleType("micropython")
    mp.native = lambda f: f

    pyb = types.ModuleType("pyb")

    class _Pin:
        OUT = 0
        IN = 1
        PULL_UP = 2
        PULL_DOWN = 3
        PULL_NONE = 0

        def __init__(self, *a, **k):
            self._v = 0

        def value(self, *a):
            if a:
                self._v = a[0]
                return None
            return self._v

    class _Timer:
        def __init__(self, *a, **k):
            self._cb = None

        def init(self, *a, **k):
            pass

        def callback(self, cb=None):
            self._cb = cb

        def deinit(self):
            pass

        def period(self, *a):
            pass

        def source_freq(self):
            return 84000000

    pyb.Pin = _Pin
    pyb.Timer = _Timer

    pc = types.ModuleType("pyControl")
    hw = types.ModuleType("pyControl.hardware")
    hw.available_timers = list(range(2, 80))   # plenty for many test doors

    class Digital_output:
        def __init__(self, pin, inverted=False):
            self.pin = _Pin(pin)
            self.inverted = inverted
            self.timer = False
            self.state = False

        def on(self):
            self.pin.value(0 if self.inverted else 1)
            self.state = True

        def off(self):
            self.pin.value(1 if self.inverted else 0)
            self.state = False

    class Digital_input:
        def __init__(self, pin, pull=None, rising_event=None,
                     falling_event=None, debounce=5):
            self.pin = _Pin(pin)

    hw.Digital_output = Digital_output
    hw.Digital_input = Digital_input
    pc.hardware = hw

    for name, mod in (("micropython", mp), ("pyb", pyb),
                      ("pyControl", pc), ("pyControl.hardware", hw)):
        sys.modules[name] = mod


def _load_door():
    _install_mcu_mocks()
    path = os.path.join(os.path.dirname(__file__),
                        "..", "..", "..", "devices", "door.py")
    spec = importlib.util.spec_from_file_location("door_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Door


Door = _load_door()


def _new(**kw):
    base = dict(direction_pin="DIR", step_pin="STEP",
                limit_open_pin="BOT", max_steps=1000, step_rate=1800)
    base.update(kw)
    return Door(**base)


def _run(door, trip_after=None, cap=500000):
    """Pump the ISR until the move ends. If trip_after is given, press the
    active watch pin once that many steps have been issued. Returns step count."""
    steps = 0
    prev = door._steps_left
    for _ in range(cap):
        if not door.is_moving():
            break
        door._tick(None)
        if door._steps_left != prev:          # a step (rising edge) happened
            steps += 1
            prev = door._steps_left
            if (trip_after is not None and steps >= trip_after
                    and door._watch_pin is not None):
                door._watch_pin.value(door._on)   # simulate the limit pressed
    return steps


def test_open_stops_at_bottom_limit_and_learns_travel():
    d = _new(limit_close_pin="TOP", max_steps=4000)
    assert d.open() is True
    assert d._has_limit is True
    steps = _run(d, trip_after=300)
    assert d.is_moving() is False
    assert d.position == d.max_steps         # at the bottom/open end
    assert d.is_open() is True               # mandatory bottom pin pressed
    assert d._travel is not None             # travel learned from the bottom limit
    assert 300 <= steps <= 305               # stopped right at the trip


def test_close_with_top_limit_stops_on_limit():
    d = _new(limit_close_pin="TOP", max_steps=4000)
    assert d.close() is True
    assert d._has_limit is True
    steps = _run(d, trip_after=250)
    assert d.is_moving() is False
    assert d.position == 0                    # at the top/closed end
    assert d.is_closed() is True
    assert 250 <= steps <= 255


def test_close_with_no_top_limit_stops_by_count_not_runaway():
    """The key flexible case: no top switch → close() stops at the step count
    (planned = max_steps), NOT at the -max_steps overrun backstop."""
    d = _new(max_steps=1000)                  # no limit_close_pin
    assert d.close() is True
    assert d._has_limit is False
    steps = _run(d)                           # no trip, must stop itself
    assert d.is_moving() is False
    assert 995 <= steps <= 1005               # stopped at the planned count
    assert steps < 1100                       # did NOT run to the 2*max_steps backstop
    assert d.position == 0                    # tracked to the closed end
    assert d.is_closed() is True              # position-based (no top pin)


def test_bottom_only_full_cycle_learns_and_reuses_travel():
    d = _new(max_steps=1000)                  # bottom-only door
    # First close from unknown → raises max_steps by count.
    d.close()
    _run(d)
    assert d.position == 0
    # Open → onto the bottom limit; trip at 800 → learns travel ~800.
    d.open()
    _run(d, trip_after=800)
    assert d.position == d.max_steps
    assert d._travel is not None and abs(d._travel - 800) <= 5
    # Next close now plans for the learned travel, not max_steps.
    s2 = _run_move(d, d.close)
    assert abs(s2 - d._travel) <= 15
    assert d.position == 0


def _run_move(door, fn):
    fn()
    return _run(door)


def test_home_drives_to_bottom_limit():
    d = _new(max_steps=1000)                  # no top limit
    assert d.is_moving() is False
    d.home()
    assert d.is_moving() is True
    assert d._watch_at_open is True           # homing toward the bottom/open end
    _run(d, trip_after=200)
    assert d.position == d.max_steps          # homed onto the bottom limit


def test_open_pre_check_refuses_when_already_at_bottom():
    d = _new(max_steps=1000)
    d._lmt_open_pin_ref.value(d._on)          # already pressed
    assert d.open() is False
    assert d.is_moving() is False
    assert d.position == d.max_steps


def test_close_n_steps_override_stops_at_override_not_max():
    d = _new(max_steps=4000)                  # bottom-only door, big max_steps
    assert d.close(n_steps=500) is True        # override the travel for this call
    steps = _run(d)
    assert 495 <= steps <= 505                 # stopped at the override, not 4000
    assert d.position == 0


def test_open_accepts_step_rate_and_min_speed_overrides():
    d = _new(max_steps=1000, limit_close_pin="TOP")
    # Overrides just change the ramp; the move still runs and stops at the limit.
    assert d.open(step_rate=3000, min_speed=500) is True
    _run(d, trip_after=300)
    assert d.is_moving() is False
    assert d.position == d.max_steps


def test_defaults_unchanged_after_an_override_move():
    d = _new(max_steps=4000)
    d.close(n_steps=300); _run(d)              # override move
    # A subsequent default close must use the full default travel, not 300.
    d.close(); steps = _run(d)
    assert steps > 1000                        # default planned = max_steps (4000)
