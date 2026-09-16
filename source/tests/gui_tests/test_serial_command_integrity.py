"""Host→MCU command path: a command must either arrive or be reported.

Two failures this pins:

  V1  ``send_serial_data`` computed its checksum without the 16-bit mask the
      board compares against, so a large ``set_variable`` raised OverflowError
      instead of sending.
  V3  The coordinate and intrinsic-event handlers dropped bad frames silently.
      Those channels run at pose rate, so a systematic fault produced a session
      that looked perfectly healthy while the task never saw a coordinate.
"""
from pathlib import Path

import pytest

from source.communication.pycboard import Pycboard

FRAMEWORK = Path(__file__).resolve().parents[2] / "pyControl" / "framework.py"


class FakeSerial:
    """Captures what would go on the wire."""

    def __init__(self):
        self.written = b""

    def write(self, data):
        self.written += data


def _board():
    board = Pycboard.__new__(Pycboard)
    board.serial = FakeSerial()
    return board


def _parse(frame):
    """Split a framed command back into (command, payload, checksum)."""
    command = frame[:1]
    data_len = int.from_bytes(frame[1:3], "little")
    payload = frame[3:3 + data_len]
    checksum = int.from_bytes(frame[3 + data_len:5 + data_len], "little")
    return command, payload, checksum


# ── V1: the checksum mask ────────────────────────────────────────────────

@pytest.mark.parametrize("value", [
    600000,                                   # ordinary scalar
    list(range(120)),                         # byte sum well under 65535
    list(range(400)),                         # byte sum ~89k, over 16 bits
    "x" * 700,                                # byte sum ~85k, over 16 bits
    {f"zone_{i}": [i * 1.0, i * 2.0] for i in range(60)},
])
def test_large_payloads_are_sent_not_raised(value):
    board = _board()
    board.send_serial_data(repr(("v", value)), "V", "s")
    _cmd, payload, checksum = _parse(board.serial.written)
    # The board validates with this exact expression; the host must agree.
    assert checksum == (sum(payload) & 0xFFFF)


def test_mask_does_not_change_payloads_that_already_worked():
    """Below the threshold, masked and unmasked are the same 2 bytes, so no
    command that worked before this fix goes on the wire differently."""
    board = _board()
    board.send_serial_data(repr(("run_dur_ms", 600000)), "V", "s")
    _cmd, payload, checksum = _parse(board.serial.written)
    assert sum(payload) <= 0xFFFF, "this case must be below the mask threshold"
    assert checksum == sum(payload)


def test_the_unmasked_form_really_would_have_raised():
    """Guards the premise: without the mask these payloads are unsendable."""
    payload = ("s" + repr(("v", list(range(400))))).encode()
    assert sum(payload) > 0xFFFF
    with pytest.raises(OverflowError):
        sum(payload).to_bytes(2, "little")


# ── V3: no silent drops on the pose-rate channels ────────────────────────

def _framework_source():
    return FRAMEWORK.read_text(encoding="utf-8", errors="replace")


def test_no_silent_drop_remains_in_the_host_command_handlers():
    """MCU code cannot be imported on the host, so this reads the source.

    The two bare swallows and the two silent returns were the whole defect;
    if any comes back, a pose-rate channel can fail invisibly again.
    """
    src = _framework_source()
    assert "pass  # Malformed payload" not in src
    assert "pass  # Unknown event ID" not in src
    assert src.count("return  # Bad checksum.") == 0


def test_every_failure_path_reports_once():
    src = _framework_source()
    for key in ("c_sum", "c_bad", "z_sum", "z_bad"):
        assert f'_warn_once("{key}"' in src, f"{key} failure is not reported"


def test_reports_are_rate_limited_not_per_occurrence():
    """These handlers sit on the pose-rate path. Warning on every occurrence
    would put tens of rows a second on the queue the science data shares."""
    src = _framework_source()
    assert "_reported_faults = set()" in src
    assert "def _warn_once(" in src
    # The guard must come before the emit, or it is not a limiter at all.
    body = src.split("def _warn_once(", 1)[1].split("def ", 1)[0]
    assert body.index("in _reported_faults") < body.index("data_output_queue.put")


def test_faults_reset_each_run():
    """Otherwise run 2 inherits run 1's suppression and reports nothing."""
    src = _framework_source()
    run_body = src.split("def run():", 1)[1]
    assert "_reported_faults.clear()" in run_body.split("while running", 1)[0]


def test_intrinsic_event_failure_still_does_not_abort_the_run():
    """Reporting must not become raising, aborting the framework mid-session
    would cost the rest of the data, which is worse than the fault."""
    src = _framework_source()
    handler = src.split('elif new_byte == b"Z":', 1)[1].split("def run()", 1)[0]
    assert "except Exception:" in handler
    # A raise STATEMENT, not the word in a comment or in "handler raised".
    statements = [ln.strip() for ln in handler.splitlines()
                  if not ln.strip().startswith("#")]
    assert not [ln for ln in statements if ln.startswith("raise ") or ln == "raise"]
