"""GPU capability probe + NVENC admission-control (CPU fallback) tests.

Covers:
  * source/video/gpu.py, platform + inference + cv2.cuda probes are safe,
    cached, and never raise; startup probe does not import torch/TF.
  * ffmpeg hardware-session counter (inc/dec/read).
  * VideoWriterFactory.create_writer, when a hardware encoder is present
    but its writer won't open (NVENC session cap), it falls through to CPU
    libx264 by default so the box still records, and refuses only when
    allow_cpu_fallback=False.
"""
import sys
import types

from source.video import gpu
from source.video.recording import ffmpeg as ff

# ── gpu.py probe ────────────────────────────────────────────────────────

def test_platform_kind_is_known():
    assert gpu.platform_kind() in (
        "jetson", "windows", "macos", "linux-x86_64", "linux-arm64")


def test_inference_probe_no_import_when_frameworks_absent(monkeypatch):
    # With allow_import=False and no torch/TF already imported, the probe
    # must NOT import them (that would cost multi-second boot time).
    gpu._inference_probe.cache_clear()
    monkeypatch.setitem(sys.modules, "torch", None)  # force "absent" branch
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    ok, desc = gpu.inference_device(allow_import=False)
    assert ok is False
    assert isinstance(desc, str)
    gpu._inference_probe.cache_clear()


def test_inference_probe_reads_already_imported_torch(monkeypatch):
    gpu._inference_probe.cache_clear()
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda i: "FakeRTX",
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    ok, desc = gpu.inference_device(allow_import=False)
    assert ok is True
    assert "FakeRTX" in desc
    gpu._inference_probe.cache_clear()


def test_cv2_cuda_available_returns_bool():
    assert isinstance(gpu.cv2_cuda_available(), bool)


def test_summary_and_log_never_raise():
    s = gpu.summary(allow_inference_import=False)
    assert set(s) >= {"platform", "encode", "inference_cuda", "cv2_cuda"}
    gpu.log_summary(allow_inference_import=False)  # must not raise


# ── hardware-session counter ────────────────────────────────────────────

def test_hw_session_counter_inc_dec():
    start = ff.hw_encoder_sessions()
    ff._inc_hw_sessions()
    ff._inc_hw_sessions()
    assert ff.hw_encoder_sessions() == start + 2
    ff._dec_hw_sessions()
    assert ff.hw_encoder_sessions() == start + 1
    ff._dec_hw_sessions()
    assert ff.hw_encoder_sessions() == start
    # never goes negative
    ff._dec_hw_sessions()
    assert ff.hw_encoder_sessions() >= 0


# ── create_writer admission control ─────────────────────────────────────

class _FakeCaps:
    """Hardware encoder present, but the HW writer will refuse to open.

    Mirrors the real ``EncoderCapabilities`` detector surface. Every hardware
    detector the factory consults must exist here, a stub that lags the real
    interface turns a fallback test into an AttributeError from inside the
    code under test, which proves nothing about fallback.
    """
    def gpu_force_required(self): return True
    def detect_hevc_nvenc(self): return False
    def detect_nvenc(self): return True
    def detect_qsv(self): return False
    def detect_amf(self): return False
    def detect_mf(self): return False
    def detect_nvmpi(self): return False
    def detect_v4l2m2m(self): return False
    def detect_libx264(self): return True
    def detect_libx265(self): return False
    def is_jetson(self): return False


class _FakeWriter:
    def __init__(self, encoder): self.encoder = encoder
    def isOpened(self): return True


def _patch_hw_fails_cpu_ok(monkeypatch):
    monkeypatch.setattr(ff.EncoderCapabilities, "get_instance",
                        staticmethod(lambda: _FakeCaps()))

    calls = []

    def fake_try(path_obj, fps, size, encoder, crf, grayscale, label,
                 is_hardware=False):
        calls.append((encoder, is_hardware))
        if is_hardware:
            return None                    # NVENC session cap → won't open
        return _FakeWriter(encoder)        # CPU libx264 opens fine

    monkeypatch.setattr(ff.VideoWriterFactory, "_try_ffmpeg",
                        staticmethod(fake_try))
    return calls


def test_create_writer_falls_back_to_cpu_when_nvenc_busy(monkeypatch):
    calls = _patch_hw_fails_cpu_ok(monkeypatch)
    w = ff.VideoWriterFactory.create_writer(
        "x.mp4", 20, (320, 240), use_gpu=True, allow_cpu_fallback=True)
    assert w is not None
    assert w.encoder == "libx264"          # recorded on CPU, not lost
    # It tried the hardware encoder first, then fell through to CPU.
    assert ("h264_nvenc", True) in calls
    assert ("libx264", False) in calls


def test_create_writer_strict_refuses_when_nvenc_busy(monkeypatch):
    _patch_hw_fails_cpu_ok(monkeypatch)
    w = ff.VideoWriterFactory.create_writer(
        "x.mp4", 20, (320, 240), use_gpu=True, allow_cpu_fallback=False)
    assert w is None                        # strict mode: refuses, no CPU


def test_create_writer_hw_success_marks_hardware_and_counts(monkeypatch):
    monkeypatch.setattr(ff.EncoderCapabilities, "get_instance",
                        staticmethod(lambda: _FakeCaps()))
    start = ff.hw_encoder_sessions()

    # Use the REAL _try_ffmpeg accounting path but with a fake writer.
    def fake_ctor(path, fps, size, encoder='h264_nvenc', crf=23,
                  preset='fast', grayscale=False):
        return _FakeWriter(encoder)
    monkeypatch.setattr(ff, "FFmpegVideoWriter", fake_ctor)

    w = ff.VideoWriterFactory._try_ffmpeg(
        __import__("pathlib").Path("x.mp4"), 20, (320, 240),
        "h264_nvenc", 23, False, "GPU H.264", is_hardware=True)
    assert w is not None
    assert getattr(w, "_is_hardware", False) is True
    assert ff.hw_encoder_sessions() == start + 1
    # cleanup accounting
    ff._dec_hw_sessions()


# ── every hardware encoder, not just NVIDIA's ────────────────────────

class _IntelOnlyCaps:
    """The common lab machine: no NVIDIA, but an Intel iGPU that encodes."""
    def gpu_force_required(self): return True
    def detect_hevc_nvenc(self): return False
    def detect_nvenc(self): return False
    def detect_qsv(self): return True
    def detect_amf(self): return False
    def detect_mf(self): return False
    def detect_nvmpi(self): return False
    def detect_v4l2m2m(self): return False
    def detect_libx264(self): return True
    def detect_libx265(self): return False
    def is_jetson(self): return False


def test_quick_sync_is_used_when_there_is_no_nvidia(monkeypatch, tmp_path):
    """The defect this pins: a machine with two hardware encoders recorded on
    a 15 W CPU because only NVENC was ever probed, and lost a third of the
    session to encoder backpressure."""
    monkeypatch.setattr(ff.EncoderCapabilities, "get_instance",
                        staticmethod(lambda: _IntelOnlyCaps()))
    tried = []

    def fake_try(path_obj, fps, size, encoder, crf, grayscale, label,
                 is_hardware=False):
        tried.append(encoder)
        return _FakeWriter(encoder)

    monkeypatch.setattr(ff.VideoWriterFactory, "_try_ffmpeg",
                        staticmethod(fake_try))
    w = ff.VideoWriterFactory.create_writer(
        str(tmp_path / "v.mp4"), 20.0, (720, 700), use_gpu=True)
    assert w.encoder == "h264_qsv", f"picked {w.encoder} with a working iGPU"
    assert "libx264" not in tried, "the CPU must not be reached at all"


def test_a_hardware_encoded_session_is_not_reported_as_cpu(monkeypatch, tmp_path):
    """``_using_gpu`` is what the session metadata and the operator believe the
    video was encoded with. Counting only NVENC labelled a Quick Sync
    recording "CPU"."""
    from source.video.recording.recorder import VideoRecorder

    rec = VideoRecorder.__new__(VideoRecorder)
    rec.video_path = str(tmp_path / "v.mp4")
    rec.fps, rec.use_gpu, rec._crf = 20.0, True, 23
    rec._prefer_hevc, rec._grayscale, rec.allow_cpu_fallback = False, False, True
    rec.writer = None
    rec._using_gpu, rec._actual_encoder = False, None

    monkeypatch.setattr(
        "source.video.recording.recorder.VideoWriterFactory.create_writer",
        staticmethod(lambda *a, **k: _FakeWriter("h264_qsv")))
    rec._try_ffmpeg_writer((720, 700), gpu_forced=True)

    assert rec._actual_encoder == "h264_qsv"
    assert rec._using_gpu is True, "a Quick Sync recording reported as CPU"


def test_the_cpu_encoder_uses_realtime_flags(tmp_path):
    """A capture pipe is not a transcode. Measured on this rig, -preset fast
    stalled the pipe 843 ms repeatedly; ultrafast+zerolatency capped it at
    19.6 ms."""
    w = ff.FFmpegVideoWriter.__new__(ff.FFmpegVideoWriter)
    w.encoder, w.crf, w.preset, w.fps = "libx264", 23, "fast", 20.0
    w.width, w.height, w.grayscale, w.path = 720, 700, False, str(tmp_path / "x.mp4")
    cmd = w._build_ffmpeg_command() if hasattr(w, "_build_ffmpeg_command") else None
    if cmd is None:
        import inspect
        cmd = inspect.getsource(ff.FFmpegVideoWriter)
        assert "'-tune', 'zerolatency'" in cmd and "'-preset', 'ultrafast'" in cmd
        assert "'-bf', '0'" in cmd
    else:
        assert "zerolatency" in cmd and "ultrafast" in cmd


# ── Jetson: h264_nvmpi, not h264_v4l2m2m ─────────────────────────────

class _JetsonCaps:
    """JetPack 6: h264_v4l2m2m is compiled in but finds no device; the
    jetson-ffmpeg build's h264_nvmpi is what reaches the encoder."""
    def gpu_force_required(self): return True
    def detect_hevc_nvenc(self): return False
    def detect_nvenc(self): return False
    def detect_qsv(self): return False
    def detect_amf(self): return False
    def detect_mf(self): return False
    def detect_nvmpi(self): return True
    def detect_v4l2m2m(self): return False
    def detect_libx264(self): return True
    def detect_libx265(self): return False
    def is_jetson(self): return True


def test_a_jetson_records_on_nvmpi_not_the_cpu(monkeypatch, tmp_path):
    """Both 4-box Jetson runs of 2026-09-15 were x264-encoded because the only
    Jetson codec known here was h264_v4l2m2m, which cannot open on JetPack 6."""
    monkeypatch.setattr(ff.EncoderCapabilities, "get_instance",
                        staticmethod(lambda: _JetsonCaps()))
    tried = []

    def fake_try(path_obj, fps, size, encoder, crf, grayscale, label,
                 is_hardware=False):
        tried.append((encoder, is_hardware))
        return _FakeWriter(encoder)

    monkeypatch.setattr(ff.VideoWriterFactory, "_try_ffmpeg",
                        staticmethod(fake_try))
    w = ff.VideoWriterFactory.create_writer(
        str(tmp_path / "v.mp4"), 30.0, (240, 240), use_gpu=True)
    assert w.encoder == "h264_nvmpi"
    assert tried == [("h264_nvmpi", True)], "the CPU must not be reached at all"


def test_nvmpi_writer_command_is_hardware_h264(tmp_path):
    w = ff.FFmpegVideoWriter.__new__(ff.FFmpegVideoWriter)
    w.encoder, w.crf, w.preset, w.fps = "h264_nvmpi", 23, "fast", 30.0
    w.width, w.height, w.grayscale, w.path = 240, 240, False, str(tmp_path / "x.mp4")
    cmd = w._build_command()
    assert cmd[cmd.index("-c:v") + 1] == "h264_nvmpi"
    assert cmd[cmd.index("-pix_fmt", cmd.index("-c:v")) + 1] == "yuv420p"
    # The quantiser floor is what keeps one noisy frame from wedging the
    # encoder for the rest of the session.
    assert int(cmd[cmd.index("-qmin") + 1]) >= 20


def test_an_nvmpi_session_is_not_reported_as_cpu(monkeypatch, tmp_path):
    from source.video.recording.recorder import VideoRecorder

    rec = VideoRecorder.__new__(VideoRecorder)
    rec.video_path = str(tmp_path / "v.mp4")
    rec.fps, rec.use_gpu, rec._crf = 30.0, True, 23
    rec._prefer_hevc, rec._grayscale, rec.allow_cpu_fallback = False, False, True
    rec.writer = None
    rec._using_gpu, rec._actual_encoder = False, None

    monkeypatch.setattr(
        "source.video.recording.recorder.VideoWriterFactory.create_writer",
        staticmethod(lambda *a, **k: _FakeWriter("h264_nvmpi")))
    rec._try_ffmpeg_writer((240, 240), gpu_forced=True)

    assert rec._actual_encoder == "h264_nvmpi"
    assert rec._using_gpu is True


def test_a_box_below_the_jetson_encoders_minimum_records_on_the_cpu(monkeypatch, tmp_path):
    """Given a 144x144 frame the Orin encoder rejects it and the jetson-ffmpeg
    wrapper spins forever rather than failing, so such a box must never be
    handed to h264_nvmpi."""
    monkeypatch.setattr(ff.EncoderCapabilities, "get_instance",
                        staticmethod(lambda: _JetsonCaps()))
    tried = []

    def fake_try(path_obj, fps, size, encoder, crf, grayscale, label,
                 is_hardware=False):
        tried.append(encoder)
        return _FakeWriter(encoder)

    monkeypatch.setattr(ff.VideoWriterFactory, "_try_ffmpeg",
                        staticmethod(fake_try))
    w = ff.VideoWriterFactory.create_writer(
        str(tmp_path / "v.mp4"), 30.0, (144, 144), use_gpu=True)
    assert "h264_nvmpi" not in tried
    assert w.encoder == "libx264"


def test_the_nvmpi_probe_uses_a_frame_the_encoder_accepts():
    """The shared probe is 64x64, which hangs the Jetson encoder until the
    probe timeout kills it, so a working encoder read as absent."""
    seen = []
    fake = types.SimpleNamespace(
        _nvmpi_available=None, detect_ffmpeg=lambda: True, is_jetson=lambda: True,
        _probe_encoder=lambda codec, size='64x64': seen.append((codec, size)) or True)
    assert ff.EncoderCapabilities.detect_nvmpi(fake) is True
    codec, size = seen[0]
    w, h = (int(v) for v in size.split("x"))
    assert codec == "h264_nvmpi"
    assert min(w, h) >= ff.EncoderCapabilities.NVMPI_MIN_SIDE
