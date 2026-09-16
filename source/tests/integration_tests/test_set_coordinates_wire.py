"""Wire format of pycboard.set_coordinates ↔ MCU b'c' handler.

The host serializes a (c_name, c_value) push; the MCU framework evals it back
and `setattr`s it onto the task's `c` namespace. The value must arrive as its
NATIVE type so task-side comparisons work:

  - a zone name string ``"RightArm"`` must compare equal to ``"RightArm"`` and
    satisfy ``c.loc_center in ['RightArm']``, NOT arrive as ``"'RightArm'"``.
  - numeric pushes (speed, x, y) must arrive as float/int, not strings.

Regression guard for the double-repr bug that wrapped strings in literal quotes
and stringified numbers, stalling maze tasks at zone-gated transitions.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from source.communication.pycboard import Pycboard


def _capture_payload(c_name, c_value):
    """Run set_coordinates against a fake serial, return the bytes written."""
    pyc = Pycboard.__new__(Pycboard)
    pyc.framework_running = True
    pyc.serial = MagicMock()
    pyc.set_coordinates(c_name, c_value)
    assert pyc.serial.write.called, "nothing written"
    return pyc.serial.write.call_args[0][0]


def _decode_like_mcu(payload):
    """Mirror framework.py's b'c' handler: strip framing, eval, return value."""
    assert payload[:1] == b"c"
    data_len = int.from_bytes(payload[1:3], "little")
    data = payload[3:3 + data_len]
    checksum = int.from_bytes(payload[3 + data_len:3 + data_len + 2], "little")
    assert checksum == (sum(data) & 0xFFFF), "bad checksum"
    assert data[-1:] == b"c", "missing sanity tag"
    c_name, c_value = eval(data[:-1])
    return c_name, c_value


def test_zone_name_arrives_as_plain_string():
    payload = _capture_payload("loc_center", "RightArm")
    name, value = _decode_like_mcu(payload)
    assert name == "loc_center"
    assert value == "RightArm"            # NOT "'RightArm'"
    assert value in ["RightArm"]          # the exact task-side test


def test_empty_zone_arrives_as_empty_string():
    _, value = _decode_like_mcu(_capture_payload("loc_center", ""))
    assert value == ""
    assert value not in ["RightArm"]


def test_speed_arrives_as_float():
    _, value = _decode_like_mcu(_capture_payload("speed", 12.5))
    assert value == 12.5
    assert isinstance(value, float)


def test_int_and_bool_preserve_type():
    _, iv = _decode_like_mcu(_capture_payload("inter_state_idx", 4))
    assert iv == 4 and isinstance(iv, int)
    _, bv = _decode_like_mcu(_capture_payload("change_state", True))
    assert bv is True


def test_string_with_quotes_or_specials_roundtrips():
    """Zone names with apostrophes still round-trip exactly (repr handles it)."""
    _, value = _decode_like_mcu(_capture_payload("loc_center", "Arm's_End"))
    assert value == "Arm's_End"
