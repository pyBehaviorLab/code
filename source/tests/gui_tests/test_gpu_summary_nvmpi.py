"""The startup GPU summary must say what the Jetson actually records with.

With h264_nvmpi working, the summary still probed h264_v4l2m2m (which logs
"recording will fall back to CPU libx264") and then printed ``encode=CPU``,
while every box was being encoded on the hardware.
"""
from source.video import gpu
from source.video.recording import ffmpeg as ff


class _JetsonOnNvmpi:
    def __init__(self):
        self.asked = []

    def detect_all(self):
        pass

    def is_jetson(self):
        return True

    def detect_ffmpeg(self):
        return True

    def detect_nvenc(self):
        return False

    def detect_hevc_nvenc(self):
        return False

    def detect_nvmpi(self):
        return True

    def detect_v4l2m2m(self):
        self.asked.append("v4l2m2m")
        return False

    def detect_libx264(self):
        return True

    def gpu_force_required(self):
        return True


class _NotAJetson(_JetsonOnNvmpi):
    def is_jetson(self):
        return False

    def detect_nvmpi(self):
        raise AssertionError("nvmpi probed on a machine that is not a Jetson")


def test_a_jetson_on_nvmpi_is_summarised_as_nvmpi(monkeypatch):
    caps = _JetsonOnNvmpi()
    monkeypatch.setattr(ff.EncoderCapabilities, "get_instance",
                        staticmethod(lambda: caps))
    enc = gpu.encode_summary()
    assert enc["nvmpi"] is True
    assert enc["v4l2m2m"] is False
    assert "v4l2m2m" not in caps.asked, (
        "probed h264_v4l2m2m, which logs a CPU fallback that is not happening")

    lines = []
    monkeypatch.setattr(gpu.logger, "info", lambda msg, *a: lines.append(msg % a))
    gpu.log_summary(allow_inference_import=False)
    assert any("encode=nvmpi" in line for line in lines), lines


def test_other_machines_are_summarised_as_before(monkeypatch):
    monkeypatch.setattr(ff.EncoderCapabilities, "get_instance",
                        staticmethod(lambda: _NotAJetson()))
    enc = gpu.encode_summary()
    assert enc["nvmpi"] is False
    assert enc["v4l2m2m"] is False
