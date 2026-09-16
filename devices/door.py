import micropython
import pyb
import pyControl.hardware as _h


class Door():
    """Stepper door, ramped motion, software pin-value end stops (no IRQ).
    Speed takes two numbers: min_speed (start/end steps/s, slow enough to
    pull away from rest without stalling; default step_rate // 4) and
    step_rate (cruise steps/s). The acceleration curve is derived per move
    from those two and the planned travel, no ramp knobs to tune.

    HD examples:
        # full (both limits):
        Door(direction_pin='PG9', step_pin='PG13',
             limit_close_pin='PD6', limit_open_pin='PD5',
             max_steps=4000, step_rate=1800, min_speed=800, doorup_dir='backward')
        # bottom-only (no top switch); close() stops by max_steps:
        Door(direction_pin='PG9', step_pin='PG13',
             limit_open_pin='PD5',
             max_steps=4000, step_rate=1800, doorup_dir='backward')
    """

    def __init__(self, motor_port=None, motor=None,
                 direction_pin=None, step_pin=None,
                 limit_close_pin=None, limit_open_pin=None,
                 max_steps=4000, step_rate=1500, min_speed=None,
                 doorup_dir='backward', limit_active_high=True):
        if motor:
            direction_pin = motor.DIR
            step_pin = motor.STEP
        elif motor_port:
            direction_pin = motor_port.DIO_A
            step_pin = motor_port.DIO_B
        self._direction = _h.Digital_output(direction_pin)
        self._step = _h.Digital_output(step_pin)
        self._step_inv = 1 if self._step.inverted else 0

        self._max_steps = max_steps
        self._position = None
        self._moving = False
        self._up_forward = (doorup_dir == 'forward')

        # Speed knobs (integer steps/s). The ramp itself is derived in
        # _begin() from min/cruise and the move's planned travel.
        self._cruise = int(step_rate)
        self._min = int(min_speed) if min_speed else max(1, self._cruise // 4)
        self._travel = None                     # learned step count to a limit

        # Limit inputs, plain reads (no ExtInt). _on = "at limit" pin value.
        self._on = 1 if limit_active_high else 0
        pull = 'down' if limit_active_high else 'up'
        self._lmt_close_pin_ref = None
        self._lmt_open_pin_ref = None
        if limit_close_pin:
            self._lmt_close_pin_ref = _h.Digital_input(limit_close_pin, pull=pull).pin
        if limit_open_pin:
            self._lmt_open_pin_ref = _h.Digital_input(limit_open_pin, pull=pull).pin

        # Engine state. _psc/_tick_rate fixed on the first move → the ISR only
        # writes the period register (no freq() recompute).
        self._timer = None
        self._psc = 0
        self._tick_rate = 1000000
        self._watch_pin = None
        self._has_limit = False
        self._watch_at_open = False
        self._planned = 0
        self._steps_left = 0
        self._decel_at = 0
        self._f = self._min
        self._level = 0
        # This move's speeds + ramp multipliers (set by _begin each move).
        self._m_cruise = self._cruise
        self._m_min = self._min
        self._m_up = 100
        self._m_dn = 100

    # ── Direction ──────────────────────────────────────────────

    def _dir_up(self):
        if self._up_forward:
            self._direction.off()
        else:
            self._direction.on()

    def _dir_down(self):
        if self._up_forward:
            self._direction.on()
        else:
            self._direction.off()

    # ── Step engine (one timer; its frequency IS the ramp) ─────
    def _begin(self, watch_pin, at_open, planned, cruise, min_speed):
        self._watch_pin = watch_pin
        # No limit pin → stop by step count instead of a limit.
        self._has_limit = watch_pin is not None
        self._watch_at_open = at_open
        self._planned = planned
        self._steps_left = planned
        # Derived ramp: accelerate/decelerate over ~1% of the travel
        # (bounded 10–100 steps); the per-step % change is whatever spans
        # min→cruise across that window. Short moves ramp over half each way.
        ramp = planned // 100
        if ramp < 10:
            ramp = 10
        elif ramp > 100:
            ramp = 100
        self._decel_at = ramp if planned > 2 * ramp else planned // 2
        if min_speed >= cruise:
            min_speed = cruise
            a = 1
        else:
            a = int(100.0 * ((cruise / min_speed) ** (1.0 / ramp) - 1.0)) + 1
            if a > 30:
                a = 30
        self._m_cruise = cruise      # this move's speeds + ramp multipliers
        self._m_min = min_speed
        self._m_up = 100 + a
        self._m_dn = 100 - a
        self._f = min_speed
        self._level = 0
        self._moving = True
        if self._timer is None:
            self._timer = pyb.Timer(_h.available_timers.pop())
            # Fix prescaler once (~1 MHz tick) → per-step is an ARR write, no float.
            src = self._timer.source_freq()
            self._psc = src // 1000000 - 1
            if self._psc < 0:
                self._psc = 0
            self._tick_rate = src // (self._psc + 1)
        p = self._tick_rate // (self._f + self._f) - 1   # toggle at 2*f
        if p < 1:
            p = 1
        self._timer.init(prescaler=self._psc, period=p)
        self._timer.callback(self._tick)

    @micropython.native
    def _tick(self, t):
        wp = self._watch_pin
        if wp is not None and wp.value() == self._on:
            self._finish(True)
            return
        lvl = self._level ^ 1
        self._level = lvl
        self._step.pin.value(lvl ^ self._step_inv)
        if not lvl:                              # falling half-period
            return
        sl = self._steps_left - 1                # rising edge = one step
        self._steps_left = sl
        if self._has_limit:
            if sl <= -self._max_steps:           # backstop: limit never came
                self._finish(False)
                return
        elif sl <= 0:                            # no limit this way → stop by count
            self._finish(False)
            return
        f = self._f
        if sl <= self._decel_at:
            f = f * self._m_dn // 100
            if f < self._m_min:
                f = self._m_min
        elif f < self._m_cruise:
            f = f * self._m_up // 100
            if f > self._m_cruise:
                f = self._m_cruise
        if f != self._f:                         # only re-tune while ramping
            self._f = f
            p = self._tick_rate // (f + f) - 1   # ARR write only, no alloc
            if p < 1:
                p = 1
            self._timer.period(p)

    def _finish(self, at_limit):
        if self._timer is not None:
            self._timer.deinit()
        self._step.pin.value(self._step_inv)
        self._level = 0
        self._moving = False
        # Learn travel from the bottom limit (the reliable one) for later ramps.
        if at_limit and self._watch_at_open and self._travel is None:
            issued = self._planned - self._steps_left
            if issued > 2 * self._decel_at:
                self._travel = issued
        # Position on every stop → is_open()/is_closed() work with no top switch.
        self._position = self._max_steps if self._watch_at_open else 0
        self._watch_pin = None

    # ── Public API ─────────────────────────────────────────────
    # open/close/home per-call overrides (else HD defaults): n_steps (travel),
    # step_rate (cruise steps/s), min_speed (start/end steps/s).

    def _move_params(self, n_steps, step_rate, min_speed):
        planned = int(n_steps) if n_steps else (self._travel or self._max_steps)
        cruise = int(step_rate) if step_rate else self._cruise
        mn = int(min_speed) if min_speed else self._min
        return planned, cruise, mn

    def open(self, n_steps=None, step_rate=None, min_speed=None):
        """Lower toward the open/bottom limit. False if already there."""
        if self._lmt_open_pin_ref and self._lmt_open_pin_ref.value() == self._on:
            self._position = self._max_steps
            return False
        self._dir_down()
        planned, cruise, mn = self._move_params(n_steps, step_rate, min_speed)
        self._begin(self._lmt_open_pin_ref, True, planned, cruise, mn)
        return True

    def close(self, n_steps=None, step_rate=None, min_speed=None):
        """Raise toward the closed/top limit. False if already there."""
        if self._lmt_close_pin_ref and self._lmt_close_pin_ref.value() == self._on:
            self._position = 0
            return False
        self._dir_up()
        planned, cruise, mn = self._move_params(n_steps, step_rate, min_speed)
        self._begin(self._lmt_close_pin_ref, False, planned, cruise, mn)
        return True

    def stop(self):
        """Emergency stop."""
        if self._timer is not None:
            self._timer.deinit()
        self._step.pin.value(self._step_inv)
        self._moving = False
        self._watch_pin = None

    def home(self, n_steps=None, step_rate=None, min_speed=None):
        """Drive to the bottom/open limit to calibrate position."""
        if self._lmt_open_pin_ref and self._lmt_open_pin_ref.value() == self._on:
            self._position = self._max_steps
            return
        self._dir_down()
        planned, cruise, mn = self._move_params(n_steps, step_rate, min_speed)
        self._begin(self._lmt_open_pin_ref, True, planned, cruise, mn)

    def is_open(self):
        if self._lmt_open_pin_ref:
            return self._lmt_open_pin_ref.value() == self._on
        return self._position is not None and self._position >= self._max_steps

    def is_closed(self):
        if self._lmt_close_pin_ref:
            return self._lmt_close_pin_ref.value() == self._on
        return self._position is not None and self._position <= 0

    def is_moving(self):
        return self._moving

    @property
    def position(self):
        return self._position

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def step_rate(self):
        return self._cruise
