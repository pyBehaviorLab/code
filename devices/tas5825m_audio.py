"""TAS5825M I2S class-D amplifier driver (pyControl / MicroPython, STM32).

Amp layer over ``I2S_Audio`` (continuous-clocks + blocking-write model). The base
keeps the I2S running (clocks live) and plays each stimulus with one blocking
write; this subclass configures the amp ONCE and leaves it in PLAY:

  _amp_power_on : PDN high + 150 ms settle.
  _amp_start    : I2C init + page/book, analog gain, DIG_VOL, then FAULT_CLEAR +
                  PLAY retried until POWER_STATE reads PLAY. Called by the base
                  AFTER the I2S clocks are running. Register 0x02 (FSW/modulation)
                  is left at the chip default (BD), exactly like WorkingMMN.py.
  _amp_stop     : HiZ + PDN low (session end).

Hardware (defaults, matching WorkingMMN.py): I2C1 SCL=PB8 SDA=PB9 @100 kHz,
addr 0x4C, PDN PB6, I2S2 WS=PB12 SCK=PB13 SD=PB15.
"""

import math
import time

from devices.i2s import I2S_Audio

try:
    from machine import I2C as _MachineI2C
    from machine import SoftI2C as _SoftI2C
except ImportError:  # pragma: no cover - MicroPython only
    _MachineI2C = None
    _SoftI2C = None
try:
    import pyb as _pyb
except ImportError:  # pragma: no cover - MicroPython only
    _pyb = None


class TAS5825MAudio(I2S_Audio):
    """TAS5825M amp layer over the continuous-clocks blocking-write engine."""

    DEFAULT_I2C_ADDR = 0x4C
    VOLUME_REG = 0x4C
    ANALOG_GAIN_REG = 0x54

    # Hard output ceiling: level=100 maps to 90% of full scale (never 1.0), to
    # keep margin below the rails and avoid clipping. The DIG_VOL byte
    # is clamped so it can never go louder than this fraction.
    MAX_OUTPUT_FRAC = 0.90
    DIGVOL_0DB = 0x30        # DIG_VOL register byte for 0 dB (full scale = 1.0)
    DIGVOL_MUTE = 0xFF       # DIG_VOL byte for mute

    def __init__(
        self,
        i2c_addr=DEFAULT_I2C_ADDR,
        i2c_id=1,
        i2c_scl="PB8",
        i2c_sda="PB9",
        i2c_freq=100000,
        pdn_pin="PB6",
        level=50,                 # DIG_VOL ~-30 dB = WorkingMMN's clean setting;
                                  #   raise carefully (80 ~= -12 dB may clip/distort)
        analog_gain=0x00,
        i2s_id=2,
        i2s_ws="PB12",
        i2s_sck="PB13",
        i2s_sd="PB15",
        sample_rate=48000,
        sync_pin=None,
        i2s_ibuf=32768,           # larger ring = more DMA-refill slack (see base)
        settle_ms=50,
    ):
        super().__init__(
            i2s_id=i2s_id, i2s_ws=i2s_ws, i2s_sck=i2s_sck, i2s_sd=i2s_sd,
            sample_rate=sample_rate, sync_pin=sync_pin,
            i2s_ibuf=i2s_ibuf, settle_ms=settle_ms,
        )
        self._i2c_addr = int(i2c_addr)
        self._i2c_id = i2c_id
        self._i2c_scl = i2c_scl
        self._i2c_sda = i2c_sda
        self._i2c_freq = int(i2c_freq)
        self._level = max(0, min(100, int(level)))
        self._analog_gain = int(analog_gain) & 0xFF
        self._i2c = None
        self._saved_level = None

        self._pdn = self._pin_class(pdn_pin, self._pin_class.OUT)
        self._pdn.value(0)

    # ------------------------------------------------------------------
    # Amp hooks
    # ------------------------------------------------------------------
    def _amp_power_on(self):
        self._pdn.value(1)
        time.sleep_ms(150)

    def _amp_start(self):
        # Clocks are already running (base started the I2S + silence feed).
        self._i2c = self._init_i2c()
        self._i2c_write_reg(0x00, 0x00)              # page 0
        self._i2c_write_reg(0x7F, 0x00)              # book 0
        self._i2c_write_reg(self.ANALOG_GAIN_REG, self._analog_gain)
        self._i2c_write_reg(self.VOLUME_REG, self._level_to_digvol(self._level))
        state = 0
        for _ in range(8):
            self._i2c_write_reg(0x78, 0x80)          # FAULT_CLEAR
            time.sleep_ms(10)
            self._i2c_write_reg(0x03, 0x03)          # PLAY
            time.sleep_ms(30)
            state = self._i2c_read_reg(0x68) & 0x03
            fault1 = self._i2c_read_reg(0x71)
            if state == 0x03 and (fault1 & 0x04) == 0:
                break
            time.sleep_ms(40)
        if state != 0x03:
            print("TAS5825M not in PLAY (0x%02X) - check clocks/wiring/RATE" % state)

    def _amp_stop(self):
        try:
            self._i2c_write_reg(0x03, 0x02)          # HiZ
        except Exception:
            pass
        self._pdn.value(0)

    # ------------------------------------------------------------------
    # I2C plumbing
    # ------------------------------------------------------------------
    def _init_i2c(self):
        if _MachineI2C is not None:
            try:
                return _MachineI2C(self._i2c_id, freq=self._i2c_freq)
            except (ValueError, OSError):
                if _SoftI2C is not None:
                    return _SoftI2C(
                        scl=self._pin_class(self._i2c_scl),
                        sda=self._pin_class(self._i2c_sda),
                        freq=self._i2c_freq,
                    )
                raise
        if _pyb and hasattr(_pyb, "I2C"):
            return _pyb.I2C(self._i2c_id, _pyb.I2C.MASTER, baudrate=self._i2c_freq)
        raise RuntimeError("No I2C class available (machine.I2C or pyb.I2C required).")

    def _i2c_write_reg(self, reg, value):
        reg = int(reg) & 0xFF
        value = int(value) & 0xFF
        if hasattr(self._i2c, "writeto_mem"):
            self._i2c.writeto_mem(self._i2c_addr, reg, bytes([value]))
        else:
            self._i2c.mem_write(value, self._i2c_addr, reg)

    def _i2c_read_reg(self, reg):
        reg = int(reg) & 0xFF
        if hasattr(self._i2c, "readfrom_mem"):
            return self._i2c.readfrom_mem(self._i2c_addr, reg, 1)[0]
        return self._i2c.mem_read(1, self._i2c_addr, reg)[0]

    # ------------------------------------------------------------------
    # Volume (hardware DIG_VOL)
    # ------------------------------------------------------------------
    @classmethod
    def _frac_to_digvol(cls, frac):
        # Output fraction (0..MAX_OUTPUT_FRAC) -> DIG_VOL register byte via the
        # chip's 0.5 dB/step law: byte = DIGVOL_0DB(0 dB, full) + attenuation steps.
        # frac is clamped to the 0.90 ceiling first so it can never exceed it.
        frac = min(cls.MAX_OUTPUT_FRAC, max(0.0, float(frac)))
        if frac <= 0.0:
            return cls.DIGVOL_MUTE
        db = 20.0 * math.log10(frac)                 # <= 20*log10(0.90) (negative)
        byte = cls.DIGVOL_0DB + int(round(-db / 0.5))  # more attenuation = larger byte
        # Never louder than the ceiling; never past mute.
        floor = cls.DIGVOL_0DB + int(round(-20.0 * math.log10(cls.MAX_OUTPUT_FRAC) / 0.5))
        if byte < floor:
            byte = floor
        if byte > 0xFF:
            byte = 0xFF
        return byte

    @classmethod
    def _level_to_digvol(cls, level):
        # level 0..100 -> output fraction (100 -> MAX_OUTPUT_FRAC = 0.90).
        level = max(0, min(100, int(level)))
        if level == 0:
            return cls.DIGVOL_MUTE
        return cls._frac_to_digvol((level / 100.0) * cls.MAX_OUTPUT_FRAC)

    def set_volume(self, level0_100):
        try:
            lvl = max(0, min(100, int(level0_100)))
        except (TypeError, ValueError):
            return
        self._level = lvl
        if self._i2c is not None:
            self._i2c_write_reg(self.VOLUME_REG, self._level_to_digvol(lvl))

    def mute(self, on=True):
        if on:
            self._saved_level = self._level
            if self._i2c is not None:
                self._i2c_write_reg(self.VOLUME_REG, self.DIGVOL_MUTE)
        else:
            self.set_volume(self._saved_level if self._saved_level is not None else self._level)

    def status(self):
        if self._i2c is None:
            return {"powered": False}
        try:
            return {"powered": True, "power_state": self._i2c_read_reg(0x68),
                    "fault1": self._i2c_read_reg(0x71), "level": self._level}
        except Exception as e:
            return {"powered": True, "error": str(e)}
