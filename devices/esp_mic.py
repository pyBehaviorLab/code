import pyb
from pyControl.hardware import Digital_output, Digital_input


class ESP_mic:
    # ESP32-S3 two-mic sound-level meter, read over UART with two optional
    # handshake GPIOs for near-instant access. The ESP (ESP32S3/main.py) measures
    # A-weighted dB(A) continuously; this board fetches the latest value on demand.
    #
    # In the hardware definition, give it the UART number, the wired UART pins,
    # and -- carefully, by DIRECTION -- the two handshake pins:
    #
    #     sound_sensor = ESP_mic(
    #         uart=7, tx='PF7', rx='PF6',
    #         req_pin='PA13',           # OUTPUT: this board -> ESP  (we pulse it to ask)
    #         alert_pin='PA14',         # INPUT : ESP -> this board  (ESP raises it when loud)
    #         alert_event='loud_sound') # framework event fired on the ALERT rising edge
    #
    #   * req_pin   is a Digital_OUTPUT here (the ESP side is an input with an ISR).
    #   * alert_pin is a Digital_INPUT here (the ESP side is an output).
    #     Wire them the right way round or the handshake does nothing.
    #   * On STM32 the UART *number* fixes tx/rx pins (UART7 = PF7/PF6); tx/rx here
    #     just document the wiring.
    #
    # read_levels(): if req_pin is set, pulse it HIGH -> the ESP ISR replies
    # instantly; otherwise fall back to the UART "?\n" poll. Either way it parses
    # "SPL,<L>,<R>". A short read timeout keeps it from ever hanging the task.
    #
    # Protocol (ASCII, newline-terminated):
    #     REQ pulse  or  "?\n"     -> "SPL,<dBL>,<dBR>\n"
    #     "P\n"                    -> "PK,<dBL>,<dBR>\n"   (peak-hold)
    #     "C,<offset>\n"           -> set calibration offset
    #     "T,<hi>,<lo>\n"          -> set ALERT thresholds (dB(A), with hysteresis)
    #     "R,<ms>\n"               -> arm a fixed-duration record (fallback)
    #     "G\n"                    -> "REC,<n>,<rate>\n" + n x int16 LE samples
    #
    # Waveform capture: the recording window is driven by
    # a hardware REC/rec_trig TTL wired to the ESP -- the task raises it just
    # before playing a stimulus and lowers it after, so the ESP records the
    # ACTUAL acoustic output at the real sample rate. ``get_recording()`` then
    # streams that waveform back for comparison with the commanded stimulus.
    # ``rec_trig`` itself is a plain Digital_output owned by the hardware
    # definition (not this driver) -- see hardware_definitions/pyBehLab_auditory.py.

    def __init__(self, uart, tx, rx, req_pin=None, alert_pin=None,
                 alert_event=None, alert_off_event=None, enable_pin=None,
                 baudrate=115200, timeout=15):
        self.tx = tx                 # wiring record (UART number fixes the pins)
        self.rx = rx
        self.uart = pyb.UART(uart, baudrate)
        try:
            self.uart.init(baudrate, bits=8, parity=None, stop=1,
                           timeout=timeout, read_buf_len=128)
        except TypeError:
            try:
                self.uart.init(baudrate, bits=8, parity=None, stop=1, timeout=timeout)
            except TypeError:
                pass
        # REQ: OUTPUT (this board -> ESP). Pulsed HIGH to request a reading.
        self.req = Digital_output(req_pin) if req_pin else None
        if self.req:
            self.req.off()
        # ALERT: INPUT (ESP -> this board). Rising edge = level crossed threshold;
        # fires a framework event the task can react to. Falling edge optional.
        self.alert = (Digital_input(alert_pin, rising_event=alert_event,
                                    falling_event=alert_off_event)
                      if alert_pin else None)
        # Optional separate enable line (some wirings gate ESP power).
        self.enable = Digital_output(enable_pin) if enable_pin else None
        self.last_rate = 0        # sample rate (Hz) of the last recording fetched

    # ---- power / raw UART ---------------------------------------------------

    def enable_esp(self):
        if self.enable:
            self.enable.on()

    def disable_esp(self):
        if self.enable:
            self.enable.off()

    def send(self, data):
        self.uart.write(data)

    def read(self, n_bytes=None):
        if n_bytes:
            return self.uart.read(n_bytes)
        return self.uart.readline()

    def any(self):
        return self.uart.any()

    # ---- dB(A) sound level --------------------------------------------------

    def _parse(self, line, tag):
        if not line:
            return None
        try:
            parts = line.strip().split(b',')
            if len(parts) == 3 and parts[0] == tag:
                return (float(parts[1]), float(parts[2]))
        except (ValueError, IndexError):
            pass
        return None

    def read_levels(self):
        """Latest (dBL, dBR), or None. Uses the REQ GPIO (instant) if wired, else
        the UART "?\n" poll. Stale bytes are flushed first so we read THIS reply."""
        while self.uart.any():
            self.uart.read()
        if self.req:
            self.req.on()                       # rising edge -> ESP ISR replies
            line = self.uart.readline()
            self.req.off()
        else:
            self.uart.write(b'?\n')
            line = self.uart.readline()
        return self._parse(line, b'SPL')

    def read_peak(self):
        """Latest peak-hold (dBL, dBR), or None. (UART command; not GPIO-driven.)"""
        while self.uart.any():
            self.uart.read()
        self.uart.write(b'P\n')
        return self._parse(self.uart.readline(), b'PK')

    def set_calibration(self, offset_db):
        """Update the ESP's dB calibration offset at runtime."""
        self.uart.write('C,{}\n'.format(offset_db))

    def set_alert_thresholds(self, hi_db, lo_db):
        """Set the ESP's ALERT thresholds (dB(A)); lo_db < hi_db gives hysteresis."""
        self.uart.write('T,{},{}\n'.format(hi_db, lo_db))

    # ---- recorded waveform ---------------------------------------------------

    def arm_record(self, ms=None):
        """Start recording. Default (no argument): recording follows the hardware
        rec_trig TTL, raise it before playing a stimulus and lower it after, and
        the ESP captures the mic for exactly that window. With ``ms`` set, ask the
        ESP to record a fixed duration over UART (fallback when no TTL is wired)."""
        if ms is not None:
            self.uart.write('R,{}\n'.format(int(ms)))

    def get_recording(self):
        """Fetch the last recorded STEREO waveform. Returns ``(left, right, rate)``:
        two lists of int16 mic samples and the sample rate (Hz). Empty lists on
        failure or when nothing was recorded. Reads are bounded so a partial UART
        transfer can never hang the task."""
        while self.uart.any():
            self.uart.read()
        self.uart.write(b'G\n')
        header = self.uart.readline()                 # b"REC,<n>,<rate>,<ch>\n"
        if not header or not header.startswith(b'REC,'):
            return ([], [], 0)
        try:
            parts = header.strip().split(b',')
            n = int(parts[1])
            rate = int(parts[2])
            ch = int(parts[3]) if len(parts) > 3 else 1
        except (ValueError, IndexError):
            return ([], [], 0)
        self.last_rate = rate
        if n <= 0:
            return ([], [], rate)
        need = n * 2                                  # bytes (int16 samples)
        buf = bytearray()
        deadline = pyb.millis() + 3000
        while len(buf) < need and pyb.millis() < deadline:
            chunk = self.uart.read(need - len(buf))
            if chunk:
                buf += chunk
        if len(buf) < need:
            return ([], [], rate)
        left = []
        right = []
        for i in range(n):
            v = buf[2 * i] | (buf[2 * i + 1] << 8)
            if v >= 32768:
                v -= 65536
            if ch == 2 and (i & 1):
                right.append(v)
            else:
                left.append(v)
        return (left, right, rate)
