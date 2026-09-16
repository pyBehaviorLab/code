"""Serial read robustness: bounded reads + stream resync.

Pins the structural invariants, a port read timeout bounds every read so a
silent board can't block the GUI thread, and a bad/short frame resyncs to the
next message-start byte instead of cascading. Real-world behaviour is only
confirmable on a rig (pull USB mid-run, inject a bad byte).
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


def _body(src: str, name: str) -> str:
    # Stop at the next method, the next top-level class, or end-of-file so the
    # last method in a class is matched too.
    m = re.search(rf"\n    def {name}\(.*?(?=\n    def |\nclass |\Z)", src, re.DOTALL)
    assert m is not None, f"could not locate {name}"
    return m.group(0)


def _body_func(src: str, name: str) -> str:
    """Body of a module-level (column-0) function."""
    m = re.search(rf"\ndef {name}\(.*?(?=\ndef |\nclass |\Z)", src, re.DOTALL)
    assert m is not None, f"could not locate top-level {name}"
    return m.group(0)


# ── Every serial.read(n) is bounded by a port timeout ──────────────────────


def test_serial_port_has_read_timeout():
    src = _read("source/communication/pyboard.py")
    init = _body(src, "__init__")
    # A `timeout=` on the port bounds the bare read(2) ack and the
    # process_data header/body reads, no read can block the GUI forever.
    assert "timeout=" in init
    assert re.search(r"serial\.Serial\([^)]*timeout=\d", init), "no read timeout on the port"


# ── Short / bad / malformed frames resync instead of cascading ──────────────


def test_process_data_resyncs_on_bad_frame():
    src = _read("source/communication/pycboard.py")
    # The resync helper exists and discards to the next 0x07 message-start byte.
    resync = _body(src, "_resync_to_message_start")
    assert 'b"\\x07"' in resync
    assert "in_waiting" in resync

    body = _body(src, "process_data")
    # Short header / short body are detected (not blindly sliced).
    assert "len(header) < 4" in body
    assert "len(message) < message_len" in body
    # A malformed payload (bad analog ID, bad JSON, short msg_len) is caught
    # and downgraded to one lost row rather than aborting the whole tick.
    assert "except Exception" in body
    # Any bad frame triggers a resync.
    assert "_resync_to_message_start()" in body


# ── Variable REPL evals degrade gracefully (no crash on garble) ─────────────


def test_variable_evals_are_guarded():
    src = _read("source/communication/pycboard.py")
    # get_variables: a garbled response returns {} instead of raising out of
    # the run-stop capture flow.
    gv = _body(src, "get_variables")
    assert "try:" in gv and "return {}" in gv
    # set_variable REPL path: parse failure → False, not a cryptic SyntaxError.
    sv = _body(src, "set_variable")
    assert "except Exception" in sv and "return False" in sv
    # set_variables batch: on parse failure, fall back to per-variable sets so
    # an Upload still applies the task's variables.
    svs = _body(src, "set_variables")
    assert "retrying per-variable" in svs


# ── Dropped host control command surfaces a WARNG (not silent) ──────────────


def test_dropped_host_command_is_surfaced():
    src = _read("source/pyControl/framework.py")
    body = _body_func(src, "receive_data")
    # Bad-checksum VARBL/EVENT path emits a WARNG row instead of silently
    # returning. Coord/zone ('c'/'Z') stay best-effort (not asserted here).
    assert "WARNG_TYP" in body
    assert "Host command dropped" in body
