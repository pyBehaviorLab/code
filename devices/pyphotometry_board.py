import pyb
from pyControl.hardware import Digital_output, Digital_input, Analog_input


class PyPhotometry_board:
    # pyPhotometry acquisition board interface.
    # Provides analog signal inputs (photodetector), DAC LED control, and digital sync outputs.

    def __init__(self, photometry_port, sampling_rate=1000):
        # Digital sync/trigger outputs.
        self.digital1 = Digital_output(photometry_port.DIGITAL1)
        self.digital2 = Digital_output(photometry_port.DIGITAL2)
        # Analog signal inputs from photodetector.
        self.signal1 = Analog_input(
            photometry_port.SIGNAL1, name='photometry_signal1', sampling_rate=sampling_rate
        )
        self.signal2 = Analog_input(
            photometry_port.SIGNAL2, name='photometry_signal2', sampling_rate=sampling_rate
        )
        # DAC outputs for LED driver control.
        self.led1_dac = pyb.DAC(pyb.Pin(photometry_port.LED1CON))
        self.led2_dac = pyb.DAC(pyb.Pin(photometry_port.LED2CON))

    def set_led1(self, value):
        # Set LED1 driver voltage (0-255 for 8-bit, 0-4095 for 12-bit).
        self.led1_dac.write(value)

    def set_led2(self, value):
        # Set LED2 driver voltage (0-255 for 8-bit, 0-4095 for 12-bit).
        self.led2_dac.write(value)

    def leds_off(self):
        self.led1_dac.write(0)
        self.led2_dac.write(0)
