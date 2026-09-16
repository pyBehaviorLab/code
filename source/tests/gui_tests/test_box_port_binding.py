"""What a box saves as its port, and what its Connect button opens.

Boxes are bound to boards by USB serial number, which is identical on Windows
and Linux; ``com_port`` is only the fallback for a box whose serial is unknown.
Two places quietly broke that:

* Saving took the per-box COM field verbatim. That field shows a LABEL, and the
  default display mode labels a board with an 8-hex hash of its serial, so
  projects ended up with ``com_port: "7dcefce1"`` (and under "serial" mode, the
  serial itself). Neither can be opened on any machine.
* The operant Connect button passed the hidden per-box port picker's text,
  which only ever names a local port, so it ignored the serial the box is
  bound to.
"""
from __future__ import annotations

from source.config.experiment import BoxConfig, _box_port_value, _looks_like_port


class _Field:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _Box:
    """Just the attributes the save path reads off a box widget."""

    def __init__(self, shown, serial=""):
        self.com_id_edit = _Field(shown)
        self._mcu_serial = serial


def _saved(shown, serial="", previous="", live=None, monkeypatch=None):
    import source.communication.mcu_ports as mcu_ports
    if monkeypatch is not None:
        monkeypatch.setattr(mcu_ports, "device_for_serial", lambda sn: live)
    base = BoxConfig(setup_number=1, mcu_serial=serial, com_port=previous)
    return _box_port_value(_Box(shown, serial), base)


def test_a_real_port_in_the_field_is_saved(monkeypatch):
    assert _saved("/dev/ttyACM1", monkeypatch=monkeypatch) == "/dev/ttyACM1"
    assert _saved("COM7", monkeypatch=monkeypatch) == "COM7"


def test_a_hashed_label_is_not_saved_as_a_port(monkeypatch):
    """The defect: projects carried com_port values like "7dcefce1"."""
    got = _saved("7dcefce1", serial="AAA", previous="COM7",
                 live="/dev/ttyACM2", monkeypatch=monkeypatch)
    assert got == "/dev/ttyACM2"


def test_a_serial_label_is_not_saved_as_a_port(monkeypatch):
    got = _saved("316135633234", serial="316135633234", previous="COM28",
                 live="/dev/ttyACM1", monkeypatch=monkeypatch)
    assert got == "/dev/ttyACM1"


def test_an_unplugged_board_keeps_the_port_saved_before(monkeypatch):
    got = _saved("7dcefce1", serial="AAA", previous="COM7", live=None,
                 monkeypatch=monkeypatch)
    assert got == "COM7", "a board that is not plugged in must not lose its port"


def test_an_empty_field_keeps_the_port_saved_before(monkeypatch):
    assert _saved("", previous="COM7", monkeypatch=monkeypatch) == "COM7"


def test_a_box_with_nothing_saves_nothing(monkeypatch):
    assert _saved("--- Select COM ---", monkeypatch=monkeypatch) == ""


def test_port_shapes():
    assert _looks_like_port("COM12") and _looks_like_port("/dev/ttyACM0")
    assert not _looks_like_port("7dcefce1")
    assert not _looks_like_port("316135633234")
    assert not _looks_like_port("")


# ── the operant Connect button ──────────────────────────────────────────

class _Stub:
    """Only what onConnectClicked touches."""
    is_connected = False
    setup_number = 1

    def __init__(self, serial, combo_text):
        self._mcu_serial = serial
        self.serial_combo = _Combo(combo_text)
        self.opened = []

    def connect_mcu(self, mcu_id):
        self.opened.append(mcu_id)

    def disconnect_mcu(self):
        self.opened.append("disconnect")

    def _set_error(self, msg):
        raise AssertionError(msg)


class _Combo:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text


def _click(stub):
    from source.gui.widgets.box_control import BoxControlWidget
    BoxControlWidget.onConnectClicked(stub)


def test_connect_uses_the_bound_serial_not_the_local_port():
    stub = _Stub("316135633234", "/dev/ttyACM4")
    _click(stub)
    assert stub.opened == ["316135633234"], (
        "the button opened a local port instead of the board this box is bound to")


def test_connect_falls_back_to_the_picker_when_no_serial_is_bound():
    stub = _Stub("", "/dev/ttyACM4")
    _click(stub)
    assert stub.opened == ["/dev/ttyACM4"]
