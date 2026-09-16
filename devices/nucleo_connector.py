"""Hardware connector / board descriptors used by the breakout drivers.

Pin/connector containers + board helpers (peers of pyControl's ``Port``) that the
pyBehaviorLab breakout and hardware-definition files instantiate. Kept HERE in
``devices/`` (not in ``pyControl/``) so the framework folder stays identical to
upstream pyControl; these live alongside the breakout drivers that use them.

``Port`` itself is upstream (``pyControl.hardware``) and re-exported here so a
breakout can pull the whole connector vocabulary from one import.
"""

from pyControl.hardware import Port 


# Default pullup/pulldown registry. A Mainboard subclass seeds it via
# set_pull_updown(), board-level pull-resistor configuration.
default_pull = {}


class Mainboard:
    # Parent class for devboard and breakout boards.
    def set_pull_updown(self, pull):
        default_pull.update(pull)


class Digital_output_group:
    # Grouping of Digital_output objects with methods for turning on or off together.
    def __init__(self, digital_outputs):
        self.digital_outputs = digital_outputs

    def on(self):
        for digital_output in self.digital_outputs:
            digital_output.on()

    def off(self):
        for digital_output in self.digital_outputs:
            digital_output.off()


class Motor:
    # Class representing motor driver pins (direction, step, enable).
    def __init__(self, DIR, STEP, EN=None):
        self.DIR = DIR
        self.STEP = STEP
        self.EN = EN


class Photometry_port:
    # Pin container for pyPhotometry board connection (ADC signals, DAC LED control, digital sync).
    def __init__(self, DIGITAL1, DIGITAL2, SIGNAL1, SIGNAL2, LED1CON, LED2CON):
        self.DIGITAL1 = DIGITAL1  # Digital sync output pin.
        self.DIGITAL2 = DIGITAL2  # Digital sync output pin.
        self.SIGNAL1 = SIGNAL1    # Analog input pin (photodetector).
        self.SIGNAL2 = SIGNAL2    # Analog input pin (photodetector).
        self.LED1CON = LED1CON    # DAC output pin (LED driver control).
        self.LED2CON = LED2CON    # DAC output pin (LED driver control).


class ESP_UART_port:
    # Pin container for ESP32-S3 UART mic interface with enable and trigger GPIOs.
    def __init__(self, TX, RX, GPIO1, GPIO2, UART=None):
        self.TX = TX        # UART TX pin.
        self.RX = RX        # UART RX pin.
        self.GPIO1 = GPIO1  # Enable ESP pin (output).
        self.GPIO2 = GPIO2  # Trigger from ESP pin (input).
        self.UART = UART    # UART peripheral number.
