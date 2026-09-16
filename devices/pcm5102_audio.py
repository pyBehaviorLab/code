"""PCM5102 / PCM5102A I2S DAC driver (pyControl / MicroPython, STM32).

Amp layer over ``I2S_Audio`` (blocking-write model). The PCM5102 is a strap-pin
DAC: NO I2C, no registers, no hardware volume. The base runs the I2S clocks for
the whole run (continuous-clock model) and deinits at run end. This subclass adds
only the optional XSMT (soft-mute) pin if it is wired to an MCU GPIO:
XSMT low = muted, high = unmuted (DAC soft-ramp).

If XSMT is strapped high on the board, leave ``xsmt_pin=None``.

Hardware (defaults): I2S2 WS/LRCK=B12 SCK/BCK=B13 SD/DIN=B15. Note: per-stimulus
``level`` sets loudness (the DAC has no volume register); ``set_volume`` is a
no-op here.
"""

from devices.i2s import I2S_Audio


class PCM5102Audio(I2S_Audio):
    """PCM5102-specific amp layer (no I2C; optional XSMT mute)."""

    def __init__(
        self,
        xsmt_pin=None,
        i2s_id=2,
        i2s_ws="B12",
        i2s_sck="B13",
        i2s_sd="B15",
        sample_rate=48000,
        sync_pin=None,
        i2s_ibuf=8192,
        settle_ms=25,
    ):
        super().__init__(
            i2s_id=i2s_id, i2s_ws=i2s_ws, i2s_sck=i2s_sck, i2s_sd=i2s_sd,
            sample_rate=sample_rate, sync_pin=sync_pin,
            i2s_ibuf=i2s_ibuf, settle_ms=settle_ms,
        )
        # XSMT held LOW (muted) until start(), if wired.
        self._xsmt = None
        if xsmt_pin is not None:
            self._xsmt = self._pin_class(xsmt_pin, self._pin_class.OUT)
            self._xsmt.value(0)

    def _amp_start(self):
        # Called by the base AFTER the I2S clocks are running. Unmute for the whole
        # run: the continuous-clock model streams silence between sounds, so XSMT
        # stays high until stop() (there is no per-sound HiZ hook in the engine).
        if self._xsmt is not None:
            self._xsmt.value(1)          # unmute (DAC soft-ramps)

    def _amp_stop(self):
        if self._xsmt is not None:
            self._xsmt.value(0)          # mute at run end

    def status(self):
        return {"xsmt": (None if self._xsmt is None else self._xsmt.value())}
