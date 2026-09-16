import pyb
from pyControl.hardware import Digital_output


class TMC_motor:
    """TMC stepper wrapper.

    ``doorUp`` records which raw direction lifts the door (= opens the
    passage) for this wiring. Tasks should call ``up()`` / ``down()`` so
    the same task code runs whether a particular maze is wired forward-up
    or backward-up, only the hardware_definition flips ``doorUp``.
    """

    def __init__(self, motor=None, direction_pin=None, step_pin=None,
                 enable_pin=None, doorUp='forward'):
        if motor:
            direction_pin = motor.DIR
            step_pin = motor.STEP
            enable_pin = motor.EN
        self._direction = Digital_output(direction_pin)
        self._step = Digital_output(step_pin)

        if enable_pin:
            self._enable = pyb.Pin(enable_pin, pyb.Pin.OUT)
            self._enable.high()  # default: disabled until enable_motor()
        else:
            self._enable = None

        if doorUp not in ('forward', 'backward'):
            raise ValueError("doorUp must be 'forward' or 'backward'")
        self._doorUp = doorUp

    def forward(self, step_rate, n_steps=False):
        self._direction.off()  # set direction forward
        self._step.pulse(step_rate, n_pulses=n_steps)

    def backward(self, step_rate, n_steps=False):
        self._direction.on()  # set direction back
        self._step.pulse(step_rate, n_pulses=n_steps)

    def up(self, step_rate, n_steps=False):
        """Lift the door (open the passage). Resolves to forward/backward
        per the ``doorUp`` setting in hardware_definition."""
        if self._doorUp == 'forward':
            self.forward(step_rate, n_steps)
        else:
            self.backward(step_rate, n_steps)

    def down(self, step_rate, n_steps=False):
        """Drop the door (close the passage). Inverse of ``up``."""
        if self._doorUp == 'forward':
            self.backward(step_rate, n_steps)
        else:
            self.forward(step_rate, n_steps)

    def stop(self):
        self._step.off()

    def enable_motor(self):
        if self._enable:
            self._enable.low()  # Active-low: pin LOW = motor enabled

    def disable_motor(self):
        if self._enable:
            self._enable.high()  # Active-low: pin HIGH = motor disabled
