"""Amp-agnostic I2S audio for pyControl / MicroPython (STM32).

Lives in ``devices/`` so it only ships to a board whose hardware-definition uses
an audio amp; the amp drivers subclass ``I2S_Audio``.

Usage modes:
  1. Synthesised tones, pre-rendered into a 32-bit-stereo buffer, played with one
     blocking gapless write (``play``).
  2. Recorded audio from SD, ``play_wav`` streams a pre-converted blob via the
     non-blocking IRQ feed: no RAM ceiling, framework stays responsive.

Playback model for clean audio: the I2S peripheral inits ONCE and stays running,
an irq callback streams SILENCE between sounds (clocks stay locked), and play()
does ONE blocking write of the whole pre-rendered tone (the C driver DMA-streams
it gaplessly). No per-chunk feed and no per-sound clock-halt / amp toggling, which
would distort the sound.

The amp is put in PLAY once at start and stays there; only DIG_VOL changes at
runtime. Lazy lifecycle bounds idle heat: a task that never touches audio never
inits, and ``stop()`` at run end deinits the I2S and reclaims the SCK/WS/SD pins
so the clocks go silent.

Amp hooks a subclass implements: _amp_power_on (PDN), _amp_start (I2C config +
PLAY), _amp_stop (HiZ + PDN low), set_volume.
"""

import math
import struct

import pyControl.hardware as hw

try:
    import urandom as _urandom
except ImportError:  # pragma: no cover - MicroPython only
    _urandom = None
try:
    from machine import I2S as _MachineI2S
    from machine import Pin as _MachinePin
except ImportError:  # pragma: no cover - MicroPython only
    _MachineI2S = None
    _MachinePin = None
try:
    import pyb as _pyb
except ImportError:  # pragma: no cover - MicroPython only
    _pyb = None
try:
    import time as _time
except ImportError:  # pragma: no cover
    _time = None
try:
    import gc as _gc
except ImportError:  # pragma: no cover
    _gc = None


# Bytes per output frame: 32-bit slot x stereo.
_BYTES_PER_FRAME = const(8)


class I2S_Audio(hw.IO_object):
    """Shared I2S audio engine, continuous clocks + blocking-write (WorkingMMN)."""

    def __init__(
        self,
        i2s_id=2,
        i2s_ws="B12",
        i2s_sck="B13",
        i2s_sd="B15",
        sample_rate=48000,
        sync_pin=None,
        i2s_ibuf=32768,           # ~85 ms ring slack @48k/32-bit-stereo; guards
                                  #   the DMA refill against framework/USB jitter.
        settle_ms=50,
        sd_mount_point="/sd",
        stream_chunk=4096,        # SD read / IRQ-feed chunk (mult. of 8); 4 KB ~= 10.7 ms
    ):
        self._i2s_id = i2s_id
        self._i2s_ws = i2s_ws
        self._i2s_sck = i2s_sck
        self._i2s_sd = i2s_sd
        sr = int(sample_rate)
        self._sample_rate = sr if sr in (48000, 44100, 32000) else 48000
        self._i2s_ibuf = int(i2s_ibuf)
        self._settle_ms = int(settle_ms)

        self._pin_class = _MachinePin if _MachinePin else (_pyb.Pin if _pyb else None)
        if self._pin_class is None:
            raise RuntimeError("No Pin class (machine.Pin or pyb.Pin required).")

        self._sync = None
        if sync_pin is not None:
            self._sync = self._pin_class(sync_pin, self._pin_class.OUT)
            self._sync.value(0)

        # Silence buffer streamed by the irq between sounds (32-bit stereo).
        self._silence = bytearray(512 * _BYTES_PER_FRAME)

        self._stims = {}        # name -> ("buf", bytearray_32bit_stereo, n_frames)
        self._pack_ch = "both"
        self._i2s = None
        self._irq_mode = False
        self._started = False

        # Continuous-masker / cue-in-noise feed state (selected in the IRQ).
        self._noise_on = False
        self._noise_buf = None
        self._noise_level = 0.25     # level of the running masker (for cue-in-noise mix)
        self._cue_mix = None

        # Mode 2, SD WAV streaming. The IRQ feed reads the next chunk from an open
        # file and writes it, so arbitrarily long audio plays without a RAM ceiling.
        # Buffers allocated lazily on first play_wav (synth-only tasks pay nothing).
        self._sd_mount_point = sd_mount_point
        self._sd_mounted = False
        self._stream_file = None
        self._stream_buf = None
        self._stream_mv = None
        self._stream_loop = False
        self._stream_chunk = int(stream_chunk)
        self._stream_sync_pending = False

        hw.assign_ID(self)

    # ------------------------------------------------------------------
    # Lifecycle (clocks run continuously; amp PLAY continuously).
    # ------------------------------------------------------------------
    def start(self):
        if self._started:
            return
        self._amp_power_on()                       # PDN high + settle (no clocks yet)
        self._i2s = self._init_i2s(self._sample_rate)
        self._irq_mode = hasattr(self._i2s, "irq")
        if self._irq_mode:
            self._i2s.irq(self._feed)              # stream silence between sounds
        self._i2s_write(self._silence)             # prime -> clocks start + settle
        if _time is not None:
            _time.sleep_ms(self._settle_ms)
        self._amp_start()                          # I2C config + PLAY (clocks present)
        self._started = True

    def stop(self):
        # Idempotent teardown: silence the amp and deinit the I2S so the clocks
        # stop toggling and the DMA + ring buffer are freed. Each step is guarded.
        self._sync_low()
        if self._stream_file is not None:          # stop any Mode-2 stream first
            try:
                self._stream_file.close()
            except Exception:
                pass
            self._stream_file = None
        if self._i2s is not None and self._irq_mode and hasattr(self._i2s, "irq"):
            try:
                self._i2s.irq(None)
            except Exception:
                pass
        try:
            self._amp_stop()                       # HiZ + PDN low
        finally:
            self._i2s_stop()                       # i2s.deinit() + drop reference
        self._started = False

    def _ensure_started(self):
        # Lazy init: the first load_*/play_* in a run brings up the I2S + amp.
        if not self._started:
            self.start()

    def _run_start(self):
        # Bring the engine up at run start so it's ready before the first sound.
        self._ensure_started()

    def _run_stop(self):
        self.stop()

    def off(self):
        self._sync_low()

    def _timer_callback(self):
        pass

    @micropython.native
    def _feed(self, arg):
        # Scheduler-context (soft) callback, file I/O is legal here. Pick ONE
        # source and write it whole; steady state does no allocation (only the EOF
        # tail uses a memoryview slice). Priority: one-shot cue-mix -> SD stream ->
        # running noise -> silence. Runs at the DMA-refill rate, so cache lookups.
        i2s = self._i2s
        try:
            cm = self._cue_mix
            if cm is not None:
                self._cue_mix = None          # one-shot splice
                i2s.write(cm)
                return
            f = self._stream_file
            if f is not None:                 # Mode 2: stream next chunk from SD.
                buf = self._stream_buf
                n = f.readinto(buf)
                if n:
                    n -= n % _BYTES_PER_FRAME  # whole 32-bit-stereo frames only
                if not n:                     # EOF (or sub-frame tail)
                    if self._stream_loop:
                        f.seek(0)
                        n = f.readinto(buf)
                        n -= n % _BYTES_PER_FRAME
                    if not n:                 # empty / not looping -> stop
                        self._stream_file = None
                        try:
                            f.close()
                        except Exception:
                            pass
                        self._sync_low()
                        i2s.write(self._silence)
                        return
                if self._stream_sync_pending:  # sync TTL marks true first-sample onset
                    self._stream_sync_pending = False
                    self._sync_high()
                if n == len(buf):
                    i2s.write(buf)            # no alloc
                else:
                    i2s.write(self._stream_mv[:n])  # short tail only
                return
            if self._noise_on and self._noise_buf is not None:
                i2s.write(self._noise_buf)
            else:
                i2s.write(self._silence)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Playback: ONE blocking write of the whole pre-rendered tone (clean).
    # ------------------------------------------------------------------
    def play(self, name, sync=True):
        """Play a pre-loaded stimulus with a single blocking write (gapless).
        Pauses the silence irq, writes the whole buffer, resumes the silence irq.
        Blocks the caller for ~the stimulus duration.

        GC is collected then disabled across the write: a GC pause mid-write would
        let the DMA ring drain and the driver zero-fill -> audible grit. Re-enabled
        in finally."""
        self._ensure_started()
        d = self._stims.get(name)
        if d is None or d[0] != "buf" or self._i2s is None:
            return
        buf = d[1]
        if _gc is not None:
            _gc.collect()                          # compact now; no GC needed mid-tone
            _gc.disable()                          # no GC pause can starve the DMA ring
        try:
            if sync:
                self._sync_high()
            if self._irq_mode:
                try:
                    self._i2s.irq(None)            # pause the silence feed
                except Exception:
                    pass
            self._i2s_write(buf)                   # whole stimulus, blocking, gapless
            if self._irq_mode:
                try:
                    self._i2s.irq(self._feed)      # resume the silence feed
                except Exception:
                    pass
                self._i2s_write(self._silence)
            if sync:
                self._sync_low()
        finally:
            if _gc is not None:
                _gc.enable()

    def play_sine(self, freq, ms, ramp_ms=5, level=0.5, ch="both", sync=True):
        self.load_sine(freq, ms, ramp_ms=ramp_ms, level=level, name="_t", ch=ch)
        self.play("_t", sync=sync)

    def play_square(self, freq, ms, duty=0.5, ramp_ms=5, level=0.5, ch="both",
                    sync=True):
        self.load_square(freq, ms, duty=duty, ramp_ms=ramp_ms, level=level,
                         name="_t", ch=ch)
        self.play("_t", sync=sync)

    def play_noise(self, ms, level=0.25, ramp_ms=3, ch="both", sync=True):
        self.load_noise(ms, level=level, ramp_ms=ramp_ms, name="_t", ch=ch)
        self.play("_t", sync=sync)

    def play_click(self, ms=1, level=0.8, ch="both", sync=True):
        self.load_click(ms=ms, level=level, name="_t", ch=ch)
        self.play("_t", sync=sync)

    def play_sweep(self, f0, f1, ms=None, rate_oct_s=None, log=False,
                   ramp_ms=5, level=0.4, ch="both", sync=True):
        self.load_sweep(f0, f1, ms=ms, rate_oct_s=rate_oct_s, log=log,
                        ramp_ms=ramp_ms, level=level, name="_t", ch=ch)
        self.play("_t", sync=sync)

    def play_chord(self, frequencies, ms, amplitudes=None, ramp_ms=5,
                   level=0.9, ch="both", sync=True):
        self.load_chord(frequencies, ms, amplitudes=amplitudes, ramp_ms=ramp_ms,
                        level=level, name="_t", ch=ch)
        self.play("_t", sync=sync)

    def play_pulsed_sine(self, freq, ms, pulse_rate, pulse_duty=0.5, ramp_ms=2,
                         level=0.5, ch="both", sync=True):
        self.load_pulsed_sine(freq, ms, pulse_rate, pulse_duty=pulse_duty,
                              ramp_ms=ramp_ms, level=level, name="_t", ch=ch)
        self.play("_t", sync=sync)

    def play_stepped_sine(self, f_start, f_end, n_steps, step_ms, log=True,
                          ramp_ms=5, level=0.5, ch="both", sync=True):
        self.load_stepped_sine(f_start, f_end, n_steps, step_ms, log=log,
                               ramp_ms=ramp_ms, level=level, name="_t", ch=ch)
        self.play("_t", sync=sync)

    # ------------------------------------------------------------------
    # Continuous masking noise + cue-in-noise (IRQ-fed, no per-sample work)
    # ------------------------------------------------------------------
    def start_noise(self, ms=80, level=0.25, ch="both", name="_noise"):
        """Start a continuous white-noise masker. The IRQ loops one pre-built
        buffer, so the loop period = ``ms`` and a short buffer repeats audibly
        (RAM limit, a longer, less-periodic loop needs a larger ibuf). ``ms`` is
        capped so the buffer (8 bytes/frame) fits ibuf, else it won't fully play.
        """
        max_ms = int(self._i2s_ibuf / _BYTES_PER_FRAME / self._sample_rate * 1000)
        if ms > max_ms:
            print("start_noise: ms capped to %d (ibuf limit)" % max_ms)
            ms = max_ms
        self.load_noise(ms, level=level, name=name, ch=ch)   # builds buf; ensures started
        self._noise_buf = self._stims[name][1]
        self._noise_level = max(0.0, min(0.9, float(level)))
        self._noise_on = True

    def stop_noise(self):
        """Stop the masker; the feed reverts to silence. Buffer kept for reuse."""
        self._noise_on = False

    def play_cue_in_noise(self, freq, ms, level=0.5, ramp_ms=2, ch="both"):
        """Play a ramped sine cue MIXED into the running noise. The mix is built
        in TASK context (not the IRQ) and spliced as one feed write; noise then
        resumes. With no noise running this falls back to a plain cue (play_sine).
        ``ms`` is capped to fit ibuf (same as the noise buffer)."""
        max_ms = int(self._i2s_ibuf / _BYTES_PER_FRAME / self._sample_rate * 1000)
        if ms > max_ms:
            ms = max_ms
        if not (self._noise_on and self._noise_buf is not None):
            return self.play_sine(freq, ms, level=level, ramp_ms=ramp_ms, ch=ch)
        n = int(self._sample_rate * ms / 1000)
        ramp_n = int(self._sample_rate * ramp_ms / 1000)
        self._pack_ch = ch
        if _gc is not None:
            _gc.collect()
        buf = bytearray(n * _BYTES_PER_FRAME)        # fresh mix buffer (task context)
        step = 2.0 * math.pi * self._clamp_freq(freq) / self._sample_rate
        # Same 90% ceiling as everything else: cap cue amplitude at 0.9*full.
        cue_amp = max(0.0, min(0.9, float(level))) * 32767.0
        noise_level = self._noise_level              # masker level baked into the mix
        ph = 0.0
        for i in range(n):
            cue = math.sin(ph) * cue_amp * self._ramp(i, n, ramp_n)
            if _urandom:
                noise = (_urandom.getrandbits(16) - 32768) * noise_level
            else:
                noise = math.sin(i * 0.123) * 32767.0 * noise_level
            self._pack(buf, i, cue + noise)          # _pack clips to +-32767
            ph += step
            if ph >= 2.0 * math.pi:
                ph -= 2.0 * math.pi
        self._cue_mix = buf                          # spliced once by the next _feed

    # ------------------------------------------------------------------
    # Mode 2, recorded audio streamed from SD (SDMMC) via the IRQ feed
    # ------------------------------------------------------------------
    def mount_sd(self, card, mount_point=None):
        """Mount an already-constructed ``machine.SDCard`` so play_wav paths
        resolve. Pin/slot wiring is the hardware-definition's job, the engine
        stays pin-agnostic. Idempotent; skip it if the card is mounted at boot."""
        if mount_point is None:
            mount_point = self._sd_mount_point
        try:
            try:
                import vfs as _vfs
                _vfs.mount(card, mount_point)
            except ImportError:
                import os as _os
                _os.mount(card, mount_point)         # older MicroPython
            self._sd_mount_point = mount_point
            self._sd_mounted = True
        except Exception as e:
            print("mount_sd failed: %s" % e)
        return self._sd_mounted

    def play_wav(self, path, loop=False, sync=True):
        """Stream a pre-converted blob from SD. ``path`` must be a HEADERLESS raw
        file in the engine's playback format (16-bit sample left-justified in a
        32-bit LE slot, stereo interleaved, at sample_rate), build it with
        tools/wav_to_i2s_blob.py. ``loop=True`` repeats until stop_wav (a
        background). Returns True if streaming started.

        Plays without the RAM ceiling and WITHOUT freezing the framework: the
        scheduler-context feed pulls chunks between bytecodes. GC stays enabled
        (the feed is allocation-free), unlike the blocking play(). The sync TTL
        rises on the first sample actually written, so it marks true onset."""
        self._ensure_started()
        if not self._irq_mode or self._i2s is None:
            print("play_wav needs non-blocking I2S (irq), unavailable")
            return False
        if self._stream_buf is None:                 # lazy: synth-only tasks pay nothing
            self._stream_buf = bytearray(self._stream_chunk)
            self._stream_mv = memoryview(self._stream_buf)
        self.stop_wav()                              # replace any current stream
        try:
            f = open(path, "rb")
        except Exception as e:
            print("play_wav open failed: %s" % e)
            return False
        self._stream_loop = bool(loop)
        self._stream_sync_pending = bool(sync)
        self._stream_file = f                        # feed streams it from here on
        return True

    def stop_wav(self):
        """Stop SD streaming; the feed reverts to running noise (if any) or
        silence. Safe to call when nothing is streaming."""
        f = self._stream_file
        self._stream_file = None                     # feed stops reading at once
        self._stream_loop = False
        self._stream_sync_pending = False
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._sync_low()

    # ------------------------------------------------------------------
    # I2S engine
    # ------------------------------------------------------------------
    def _i2s_stop(self):
        if self._i2s is not None and hasattr(self._i2s, "deinit"):
            try:
                self._i2s.deinit()              # tears down DMA, gates the I2S
            except Exception:
                pass
        self._i2s = None
        # deinit() leaves SCK/WS/SD in alternate-function mode, so the bit clock
        # keeps appearing and the amp keeps switching. Re-claim them and drive them
        # low (no residual edges, no floating noise). start() re-inits them as I2S.
        if self._pin_class is not None:
            out_mode = getattr(self._pin_class, "OUT", None)
            if out_mode is None:                       # pyb.Pin uses OUT_PP, not OUT
                out_mode = getattr(self._pin_class, "OUT_PP", None)
            for _p in (self._i2s_sck, self._i2s_ws, self._i2s_sd):
                try:
                    if out_mode is not None:
                        _pin = self._pin_class(_p, out_mode)
                        _pin.value(0)                  # hold the line low
                    else:
                        self._pin_class(_p, self._pin_class.IN)
                except Exception:
                    pass

    def _init_i2s(self, sample_rate):
        # 32-bit slots, STEREO (16-bit sample in the high half), F767 + TAS5825M.
        if _MachineI2S:
            return _MachineI2S(
                self._i2s_id,
                sck=self._pin_class(self._i2s_sck),
                ws=self._pin_class(self._i2s_ws),
                sd=self._pin_class(self._i2s_sd),
                mode=_MachineI2S.TX,
                bits=32,
                format=_MachineI2S.STEREO,
                rate=int(sample_rate),
                ibuf=self._i2s_ibuf,
            )
        if _pyb and hasattr(_pyb, "I2S"):
            return _pyb.I2S(
                self._i2s_id, _pyb.I2S.MASTER_TX, bits=32,
                format=_pyb.I2S.STEREO, rate=int(sample_rate),
            )
        raise RuntimeError("I2S support missing (machine.I2S or pyb.I2S required).")

    def _i2s_write(self, buf):
        if hasattr(self._i2s, "write"):
            self._i2s.write(buf)
        elif hasattr(self._i2s, "send"):
            self._i2s.send(buf)
        else:
            raise RuntimeError("I2S object has no write/send method.")

    # ------------------------------------------------------------------
    # Sync TTL
    # ------------------------------------------------------------------
    def _sync_high(self):
        if self._sync is not None:
            self._sync.value(1)

    def _sync_low(self):
        if self._sync is not None:
            self._sync.value(0)

    # ------------------------------------------------------------------
    # Stimulus builders, pre-render into final 32-bit-stereo format once.
    # RAM = 8 bytes/frame (e.g. 100 ms = 38 KB). Keep cues short and/or load on
    # demand; the MCU heap is small (a 300 ms tone = 115 KB will MemoryError).
    # ------------------------------------------------------------------
    def _new_buf(self, n_frames, name):
        self._stims.pop(name, None)
        if _gc is not None:
            _gc.collect()
        return bytearray(n_frames * _BYTES_PER_FRAME)   # 32-bit stereo, one compact alloc

    def _clamp_freq(self, f):
        # Keep every frequency below Nyquist: 0.45*sample_rate ~= 21.6 kHz @48k.
        return min(float(f), 0.45 * self._sample_rate)

    @micropython.native
    def _ramp(self, i, n, ramp_n):
        if ramp_n <= 0:
            return 1.0
        if i < ramp_n:
            return 0.5 * (1.0 - math.cos(math.pi * i / ramp_n))
        if i >= n - ramp_n:
            k = n - 1 - i
            return 0.5 * (1.0 - math.cos(math.pi * k / ramp_n))
        return 1.0

    @micropython.native
    def _pack(self, buf, i, value):
        v = int(value)
        if v > 32767:
            v = 32767
        elif v < -32768:
            v = -32768
        v <<= 16
        ch = self._pack_ch
        l = v if ch != "right" else 0
        r = v if ch != "left" else 0
        struct.pack_into("<ii", buf, i * _BYTES_PER_FRAME, l, r)

    def load_sine(self, freq, ms, ramp_ms=5, level=0.5, name="sine", ch="both"):
        self._ensure_started()
        n = int(self._sample_rate * ms / 1000)
        ramp_n = int(self._sample_rate * ramp_ms / 1000)
        self._pack_ch = ch
        buf = self._new_buf(n, name)
        step = 2.0 * math.pi * self._clamp_freq(freq) / self._sample_rate
        amp = max(0.0, min(1.0, float(level))) * 32767.0
        ph = 0.0
        for i in range(n):
            self._pack(buf, i, math.sin(ph) * amp * self._ramp(i, n, ramp_n))
            ph += step
            if ph >= 2.0 * math.pi:
                ph -= 2.0 * math.pi
        self._stims[name] = ("buf", buf, n)
        return name

    def load_square(self, freq, ms, duty=0.5, ramp_ms=5, level=0.5,
                    name="square", ch="both"):
        """Raw bipolar square wave. Only the overall on/off envelope is ramped
        (a per-edge ramp would round the edges and un-square it). ``duty`` = the
        high fraction of each cycle (0..1)."""
        self._ensure_started()
        n = int(self._sample_rate * ms / 1000)
        ramp_n = int(self._sample_rate * ramp_ms / 1000)
        self._pack_ch = ch
        buf = self._new_buf(n, name)
        amp = max(0.0, min(1.0, float(level))) * 32767.0
        period = self._sample_rate / self._clamp_freq(freq)   # frames per cycle
        duty = max(0.0, min(1.0, float(duty)))
        for i in range(n):
            phase = (i % period) / period             # 0..1 within the cycle
            s = amp if phase < duty else -amp
            self._pack(buf, i, s * self._ramp(i, n, ramp_n))
        self._stims[name] = ("buf", buf, n)
        return name

    def load_pulsed_sine(self, freq, ms, pulse_rate, pulse_duty=0.5, ramp_ms=2,
                         level=0.5, name="pulsed", ch="both"):
        """Sine carrier gated ON/OFF at ``pulse_rate`` (continuous phase). Each
        ON burst gets its own cosine in/out ramp (``ramp_ms``) so the pulses
        don't click, PLUS the overall on/off envelope across the whole cue."""
        self._ensure_started()
        n = int(self._sample_rate * ms / 1000)
        ramp_n = int(self._sample_rate * ramp_ms / 1000)
        self._pack_ch = ch
        buf = self._new_buf(n, name)
        amp = max(0.0, min(1.0, float(level))) * 32767.0
        pulse_period = int(self._sample_rate / float(pulse_rate))   # frames/pulse
        on_n = max(1, int(pulse_period * max(0.0, min(1.0, float(pulse_duty)))))
        pramp_n = int(self._sample_rate * ramp_ms / 1000)           # per-pulse edge
        step = 2.0 * math.pi * self._clamp_freq(freq) / self._sample_rate
        ph = 0.0
        for i in range(n):
            pos = i % pulse_period
            gate = self._ramp(pos, on_n, pramp_n) if pos < on_n else 0.0
            self._pack(buf, i, math.sin(ph) * amp * gate * self._ramp(i, n, ramp_n))
            ph += step
            if ph >= 2.0 * math.pi:
                ph -= 2.0 * math.pi
        self._stims[name] = ("buf", buf, n)
        return name

    def load_stepped_sine(self, f_start, f_end, n_steps, step_ms, log=True,
                          ramp_ms=5, level=0.5, name="stepped", ch="both"):
        """``n_steps`` discrete frequency segments stepping f_start->f_end, phase
        CONTINUOUS across steps (no boundary click). Geometric (log) spacing by
        default; pass log=False for linear. Only the overall on/off envelope is
        ramped."""
        self._ensure_started()
        n_steps = max(2, int(n_steps))
        step_n = int(self._sample_rate * step_ms / 1000)
        n = step_n * n_steps
        self._pack_ch = ch
        buf = self._new_buf(n, name)
        amp = max(0.0, min(1.0, float(level))) * 32767.0
        f_start = self._clamp_freq(f_start)
        f_end = self._clamp_freq(f_end)
        if log and f_start > 0 and f_end > 0:
            ratio = (f_end / f_start) ** (1.0 / (n_steps - 1))
            freqs = [f_start * (ratio ** k) for k in range(n_steps)]
        else:
            freqs = [f_start + (f_end - f_start) * k / (n_steps - 1)
                     for k in range(n_steps)]
        ramp_n = int(self._sample_rate * ramp_ms / 1000)
        ph = 0.0
        for i in range(n):
            k = i // step_n
            if k >= n_steps:
                k = n_steps - 1
            step = 2.0 * math.pi * freqs[k] / self._sample_rate
            self._pack(buf, i, math.sin(ph) * amp * self._ramp(i, n, ramp_n))
            ph += step
            if ph >= 2.0 * math.pi:
                ph -= 2.0 * math.pi
        self._stims[name] = ("buf", buf, n)
        return name

    def load_noise(self, ms, level=0.25, ramp_ms=3, name="noise", ch="both"):
        self._ensure_started()
        n = int(self._sample_rate * ms / 1000)
        ramp_n = int(self._sample_rate * ramp_ms / 1000)
        self._pack_ch = ch
        buf = self._new_buf(n, name)
        amp = max(0.0, min(1.0, float(level)))
        for i in range(n):
            if _urandom:
                s = (_urandom.getrandbits(16) - 32768) * amp
            else:
                s = math.sin(i * 0.123) * 32767.0 * amp
            self._pack(buf, i, s * self._ramp(i, n, ramp_n))
        self._stims[name] = ("buf", buf, n)
        return name

    def load_click(self, ms=1, level=0.8, name="click", ch="both"):
        self._ensure_started()
        n = int(self._sample_rate * ms / 1000) or 1
        self._pack_ch = ch
        buf = self._new_buf(n, name)
        amp = max(0.0, min(1.0, float(level))) * 32767.0
        for i in range(n):
            self._pack(buf, i, amp if (i & 1) == 0 else -amp)
        self._stims[name] = ("buf", buf, n)
        return name

    def load_sweep(self, f0, f1, ms=None, rate_oct_s=None, log=False,
                   ramp_ms=5, level=0.4, name="sweep", ch="both"):
        """FM sweep f0 -> f1 (Hz); f1 < f0 = downward. LINEAR by default (matches
        the HALIP linear up-sweep, 10->15 kHz); pass log=True for a logarithmic
        chirp, or give RATE in octaves/second -> dur = |log2(f1/f0)|/rate. 5 ms
        cosine (cos^2) on/off ramps; closed-form chirp phase (no float drift)."""
        self._ensure_started()
        f0 = self._clamp_freq(f0)
        f1 = self._clamp_freq(f1)
        if ms is None:
            if not rate_oct_s:
                raise ValueError("load_sweep needs ms or rate_oct_s")
            octaves = abs(math.log(f1 / f0) / math.log(2.0))
            ms = 1000.0 * octaves / float(rate_oct_s)
        T = ms / 1000.0
        n = int(self._sample_rate * ms / 1000)
        ramp_n = int(self._sample_rate * ramp_ms / 1000)
        self._pack_ch = ch
        buf = self._new_buf(n, name)
        amp = max(0.0, min(1.0, float(level))) * 32767.0
        if log and f0 > 0.0 and f1 > 0.0 and f1 != f0:
            lnK = math.log(f1 / f0)
            coef = 2.0 * math.pi * f0 * T / lnK
            for i in range(n):
                tn = i / n if n else 0.0
                ph = coef * (math.exp(tn * lnK) - 1.0)
                self._pack(buf, i, math.sin(ph) * amp * self._ramp(i, n, ramp_n))
        else:
            k = (f1 - f0) / T if T else 0.0
            for i in range(n):
                t = i / self._sample_rate
                ph = 2.0 * math.pi * (f0 * t + 0.5 * k * t * t)
                self._pack(buf, i, math.sin(ph) * amp * self._ramp(i, n, ramp_n))
        self._stims[name] = ("buf", buf, n)
        return name

    def load_chord(self, frequencies, ms, amplitudes=None, ramp_ms=5,
                   level=0.9, name="chord", ch="both"):
        """Sum-of-sines chord (Bpod GenerateSineChord). ``amplitudes`` default to
        an equal 1/N split; ``level`` scales the unity-peak chord to leave margin before clipping."""
        self._ensure_started()
        freqs = [self._clamp_freq(f) for f in frequencies]
        nf = len(freqs)
        if nf == 0:
            raise ValueError("load_chord needs at least one frequency")
        if amplitudes is None:
            amps = [1.0 / nf] * nf
        else:
            amps = [float(a) for a in amplitudes]
            if len(amps) != nf:
                raise ValueError("amplitudes length must match frequencies")
        n = int(self._sample_rate * ms / 1000)
        ramp_n = int(self._sample_rate * ramp_ms / 1000)
        self._pack_ch = ch
        buf = self._new_buf(n, name)
        scale = max(0.0, min(1.0, float(level))) * 32767.0
        steps = [2.0 * math.pi * f / self._sample_rate for f in freqs]
        phases = [0.0] * nf
        for i in range(n):
            s = 0.0
            for c in range(nf):
                s += math.sin(phases[c]) * amps[c]
                phases[c] += steps[c]
                if phases[c] >= 2.0 * math.pi:
                    phases[c] -= 2.0 * math.pi
            self._pack(buf, i, s * scale * self._ramp(i, n, ramp_n))
        self._stims[name] = ("buf", buf, n)
        return name

    # ------------------------------------------------------------------
    # Amp hooks, defaults are no-ops. Subclasses override.
    # ------------------------------------------------------------------
    def _amp_power_on(self):
        pass

    def _amp_start(self):
        pass

    def _amp_stop(self):
        pass

    def set_volume(self, level0_100):
        pass

    def mute(self, on=True):
        pass

    def status(self):
        return {}

    def check_ok(self):
        return True
