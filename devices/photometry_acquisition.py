import pyb
from array import array
from pyControl.hardware import Analog_channel, IO_object, assign_ID, available_timers


class Photometry_acquisition(IO_object):
    # Pulsed / time-multiplexed photometry acquisition.
    #
    # A single hardware timer cycles the excitation LEDs in lock-step with the
    # detector ADC: per phase it turns one LED on, waits a settle delay, reads
    # the detector, turns the LED off, and pushes the sample into that phase's
    # pyControl Analog_channel. Each excitation therefore streams as its own
    # analog channel, so the host records (.npy), plots and FW-clock aligns it
    # through the standard analog pipeline with no host-side changes.
    #
    # Use case: 470/405 isosbestic on one detector ('2EX_1EM_pulsed'), which a
    # free-running Analog_input cannot do (its sampling is not synchronised with
    # LED switching).
    #
    # Function-first: a single ADC.read() per phase after the settle delay.
    # Oversampling (read_timed averaging) and LED-off background subtraction are
    # deferred quality steps, not required for correct demultiplexing.

    def __init__(self, photometry_port, mode='2EX_1EM_pulsed',
                 sampling_rate=130, settle_us=300, dac_bits=12):
        self.mode = mode
        self.sampling_rate = sampling_rate
        self.settle_us = settle_us

        # Excitation LED DACs (LED1=DAC1/PA4, LED2=DAC2/PA5).
        self.led1 = pyb.DAC(pyb.Pin(photometry_port.LED1CON), bits=dac_bits)
        self.led2 = pyb.DAC(pyb.Pin(photometry_port.LED2CON), bits=dac_bits)
        self.led1.write(0)
        self.led2.write(0)
        self.led_level = array('H', [0, 0])  # DAC code per LED, set by set_led1/2.

        # Detector ADCs.
        adc1 = pyb.ADC(pyb.Pin(photometry_port.SIGNAL1))
        adc2 = pyb.ADC(pyb.Pin(photometry_port.SIGNAL2))

        # LED3 (3EX modes) is driven on/off via the DIGITAL2 pin (no current control).
        self._led3 = None

        # Phase table for the chosen mode: each entry is (led_index, adc, label),
        # executed once per timer tick. led_index 0/1 = DAC LEDs, 2 = LED3 digital.
        if mode == '2EX_1EM_pulsed':
            phases = [(0, adc1, '470'), (1, adc1, '405')]
        elif mode == '2EX_2EM_pulsed':
            phases = [(0, adc1, '470'), (1, adc2, '560')]
        elif mode == '3EX_2EM_pulsed':
            self._led3 = pyb.Pin(photometry_port.DIGITAL2, pyb.Pin.OUT)
            self._led3.value(0)
            phases = [(0, adc1, '470'), (1, adc2, '560'), (2, adc1, '415')]
        else:
            raise ValueError('Unknown photometry mode: ' + mode)

        self.n_phases = len(phases)
        self._led_idx = array('b', [p[0] for p in phases])
        self._adc = [p[1] for p in phases]
        # Each excitation samples once per cycle -> per-channel rate = sampling_rate.
        self.channels = [Analog_channel('photometry_' + p[2], sampling_rate, 'H')
                         for p in phases]
        self._phase = 0

        # Dedicated sampling timer (framework clock=1, rotary=2, audio=4, DAC=6).
        self.timer = pyb.Timer(available_timers.pop())

        assign_ID(self)  # After channels, so they reset before our _run_start starts the timer.

    # LED current control (same API as PyPhotometry_board so tasks are interchangeable).
    def set_led1(self, value):
        self.led_level[0] = value

    def set_led2(self, value):
        self.led_level[1] = value

    def leds_off(self):
        self.led1.write(0)
        self.led2.write(0)
        if self._led3 is not None:
            self._led3.value(0)

    def _set_led(self, idx, on):
        if idx == 0:
            self.led1.write(self.led_level[0] if on else 0)
        elif idx == 1:
            self.led2.write(self.led_level[1] if on else 0)
        else:  # LED3 digital on/off.
            self._led3.value(1 if on else 0)

    def _run_start(self):
        self._phase = 0
        self.timer.init(freq=self.sampling_rate * self.n_phases)
        self.timer.callback(self._ISR)

    def _run_stop(self):
        self.timer.deinit()
        self.leds_off()

    def off(self):
        self.leds_off()

    @micropython.native
    def _ISR(self, t):
        i = self._phase
        idx = self._led_idx[i]
        self._set_led(idx, True)        # LED on.
        pyb.udelay(self.settle_us)      # Settle.
        sample = self._adc[i].read()    # Read detector (12-bit).
        self._set_led(idx, False)       # LED off.
        self.channels[i].put(sample)    # Stream via standard pyControl path.
        self._phase = (i + 1) % self.n_phases
