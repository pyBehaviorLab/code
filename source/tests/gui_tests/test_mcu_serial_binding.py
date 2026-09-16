"""Box ↔ MCU binding via stable USB serial number.

Locks down:
  * ``BoxConfig.mcu_serial`` round-trips through to_compact / from_dict.
  * Back-compat: old project files without ``mcu_serial`` load cleanly
    with ``""`` and the legacy ``com_port`` still works.
  * ``mcu_ports.list_mcu_serials`` filters by pyboard VID/PID and
    only returns devices with a USB serial-number descriptor.
  * ``mcu_ports.device_for_serial(sn)`` looks up the live device path.
  * ``mcu_ports.serial_for_device(dev)`` is the reverse path used
    for legacy auto-upgrade.
  * On project load, an old ``com_port``-only box auto-upgrades to
    ``mcu_serial`` once a matching live device is found.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from source.config.experiment import BoxConfig


class _FakePort:
    """Stand-in for a pyserial ListPortInfo."""
    def __init__(self, device, serial_number, vid=0xF055, pid=0x9800,
                 description="Pyboard"):
        self.device = device
        self.serial_number = serial_number
        self.vid = vid
        self.pid = pid
        self.description = description


# ─── schema: round-trip + back-compat ─────────────────────────────────


def test_box_config_mcu_serial_round_trips():
    box = BoxConfig(setup_number=1, mcu_serial="315535563234",
                    com_port="/dev/ttyACM2")
    raw = box.to_compact()
    assert raw["mcu_serial"] == "315535563234"
    assert raw["com_port"] == "/dev/ttyACM2"
    rebuilt = BoxConfig.from_dict(raw)
    assert rebuilt.mcu_serial == "315535563234"
    assert rebuilt.com_port == "/dev/ttyACM2"


def test_box_config_legacy_project_loads_with_empty_serial():
    """Old project files without mcu_serial load cleanly, ``mcu_serial``
    defaults to ``""`` and ``com_port`` stays."""
    legacy = {
        "box_number": 1,
        "com_port": "/dev/ttyACM0",
        # NO mcu_serial key
    }
    box = BoxConfig.from_dict(legacy)
    assert box.mcu_serial == ""
    assert box.com_port == "/dev/ttyACM0"


def test_box_config_mcu_serial_default_is_empty():
    box = BoxConfig(setup_number=1)
    assert box.mcu_serial == ""


# ─── mcu_ports: filter + device_for_serial + reverse ──────────────────


def _patch_comports(ports):
    return patch(
        "serial.tools.list_ports.comports", return_value=ports)


def test_list_mcu_serials_filters_by_vid_pid():
    from source.communication.mcu_ports import list_mcu_serials
    fake = [
        _FakePort("/dev/ttyACM0", "AAA", vid=0xF055, pid=0x9800),
        _FakePort("/dev/ttyUSB0", "BBB", vid=0x1234, pid=0x5678),  # not pyboard
        _FakePort("/dev/ttyACM1", "CCC", vid=0xF055, pid=0x9801),
    ]
    with _patch_comports(fake):
        out = list_mcu_serials()
    serials = [sn for sn, _ in out]
    assert serials == sorted(["AAA", "CCC"])
    # Sort is by serial, verify devices are paired correctly.
    devmap = dict(out)
    assert devmap["AAA"] == "/dev/ttyACM0"
    assert devmap["CCC"] == "/dev/ttyACM1"


def test_list_mcu_serials_excludes_ports_without_serial_descriptor():
    """MicroPython firmware variants that don't ship a USB serial
    string set serial_number=None; those MCUs are excluded from the
    dropdown rather than appearing as ambiguous entries."""
    from source.communication.mcu_ports import list_mcu_serials
    fake = [
        _FakePort("/dev/ttyACM0", None, vid=0xF055, pid=0x9800),
        _FakePort("/dev/ttyACM1", "XYZ", vid=0xF055, pid=0x9800),
    ]
    with _patch_comports(fake):
        out = list_mcu_serials()
    assert [sn for sn, _ in out] == ["XYZ"]


def test_device_for_serial_returns_live_device_for_saved_serial():
    """Even when the OS gives a pyboard a different ttyACMn than last
    session, device_for_serial() finds it by serial."""
    from source.communication.mcu_ports import device_for_serial
    # Same MCU (serial AAA) now enumerated at /dev/ttyACM3.
    fake = [_FakePort("/dev/ttyACM3", "AAA")]
    with _patch_comports(fake):
        assert device_for_serial("AAA") == "/dev/ttyACM3"


def test_device_for_serial_returns_none_when_mcu_not_plugged_in():
    from source.communication.mcu_ports import device_for_serial
    with _patch_comports([_FakePort("/dev/ttyACM0", "BBB")]):
        assert device_for_serial("AAA") is None


def test_device_for_serial_returns_none_for_empty_serial():
    from source.communication.mcu_ports import device_for_serial
    assert device_for_serial("") is None
    assert device_for_serial(None) is None  # type: ignore[arg-type]


def test_serial_for_device_reverse_lookup():
    """Used for legacy auto-upgrade, old project has only com_port,
    first connect captures the serial via this reverse lookup."""
    from source.communication.mcu_ports import serial_for_device
    fake = [
        _FakePort("/dev/ttyACM0", "AAA"),
        _FakePort("/dev/ttyACM1", "BBB"),
    ]
    with _patch_comports(fake):
        assert serial_for_device("/dev/ttyACM1") == "BBB"
        assert serial_for_device("/dev/ttyACMnope") is None
        assert serial_for_device("") is None


# ─── _apply_per_box: legacy auto-upgrade ──────────────────────────────


def test_apply_per_box_upgrades_legacy_com_port_to_serial():
    """When loading an old project that has com_port but no mcu_serial,
    if the live device at that path has a USB serial, capture it onto
    the widget AND patch cfg.box.mcu_serial so the next autosave
    persists the upgrade."""
    from source.config.experiment import _apply_per_box, Config, BoxConfig

    # Old project: com_port='/dev/ttyACM2', mcu_serial=''
    cfg = Config()
    cfg.setup_config.boxes = [BoxConfig(setup_number=1, com_port="/dev/ttyACM2")]

    # Stand-in widget the bridge can write to.
    bw = type("W", (), {})()
    bw._mcu_serial = ""
    host = MagicMock()
    host.iter_box_widgets = lambda: [(1, bw)]
    # Avoid heavy widget setattr paths, stub the helpers
    host._tracking_dialog_globals = {}
    host.tracking_zones = {}

    # Fake live USB enumeration: /dev/ttyACM2 has serial 'STABLE_777'
    fake = [_FakePort("/dev/ttyACM2", "STABLE_777")]
    with _patch_comports(fake), \
         patch("source.config.experiment._set_widget_text"):
        _apply_per_box(cfg, host)

    # Upgrade landed on widget + cfg.
    assert bw._mcu_serial == "STABLE_777"
    assert cfg.setup_config.boxes[0].mcu_serial == "STABLE_777"


def test_apply_per_box_no_upgrade_when_device_not_present():
    """If the legacy com_port refers to a device that isn't currently
    plugged in, leave mcu_serial empty, operator will pick from the
    dialog when they connect."""
    from source.config.experiment import _apply_per_box, Config, BoxConfig

    cfg = Config()
    cfg.setup_config.boxes = [BoxConfig(setup_number=1, com_port="/dev/ttyACM7")]
    bw = type("W", (), {})()
    bw._mcu_serial = ""
    host = MagicMock()
    host.iter_box_widgets = lambda: [(1, bw)]

    # No pyboards online.
    with _patch_comports([]), \
         patch("source.config.experiment._set_widget_text"):
        _apply_per_box(cfg, host)

    assert bw._mcu_serial == ""
    assert cfg.setup_config.boxes[0].mcu_serial == ""


def test_apply_per_box_uses_saved_serial_over_com_port():
    """When mcu_serial is already saved, that's what the widget gets.
    com_port is ignored as the binding (only kept for legacy display)."""
    from source.config.experiment import _apply_per_box, Config, BoxConfig

    cfg = Config()
    cfg.setup_config.boxes = [BoxConfig(
        setup_number=1,
        mcu_serial="STABLE_AAA",
        com_port="/dev/ttyACM0_OLD_FROM_LAST_SESSION",
    )]
    bw = type("W", (), {})()
    bw._mcu_serial = ""
    host = MagicMock()
    host.iter_box_widgets = lambda: [(1, bw)]

    # Live enumeration doesn't matter, saved serial wins.
    with _patch_comports([_FakePort("/dev/ttyACM3", "STABLE_AAA")]), \
         patch("source.config.experiment._set_widget_text"):
        _apply_per_box(cfg, host)

    assert bw._mcu_serial == "STABLE_AAA"


# ─── _read_per_box: widget._mcu_serial → cfg.box.mcu_serial ──────────


def test_read_per_box_captures_widget_mcu_serial():
    from source.config.experiment import _read_per_box, Config

    bw = MagicMock()
    bw._mcu_serial = "FRESH_SERIAL"
    # Stub fields _read_per_box reads
    bw.init_hw_def = None
    bw.action_config = None
    bw.api_class = None
    bw.zones = []
    bw.tracking_enabled = False
    bw.save_tracking = False
    bw.save_video_enabled = True
    bw.bg_captured_at = ""
    bw.roi_normalized = None
    bw.roi_segment = None
    bw.camera_id_edit = None

    host = MagicMock()
    host.iter_box_widgets = lambda: [(1, bw)]
    cfg = Config()

    with patch("source.config.experiment._widget_text", return_value=""):
        _read_per_box(host, cfg)

    assert cfg.setup_config.boxes[0].mcu_serial == "FRESH_SERIAL"


def test_read_per_box_falls_back_to_base_serial_when_widget_unset():
    """If the autosave fires BEFORE the widget has been connected
    (so ``bw._mcu_serial`` is empty), keep the previously-saved serial.

    The previously-saved value lives on ``host._active_config``, the fresh
    ``cfg`` that ``read_ui_into_config`` hands to ``_read_per_box`` is empty,
    so the fallback must read the last-saved boxes from the active config."""
    from source.config.experiment import _read_per_box, Config, BoxConfig

    bw = MagicMock()
    bw._mcu_serial = ""
    bw.init_hw_def = None; bw.action_config = None; bw.api_class = None
    bw.zones = []; bw.tracking_enabled = False; bw.save_tracking = False
    bw.save_video_enabled = True; bw.bg_captured_at = ""
    bw.roi_normalized = None; bw.roi_segment = None
    bw.camera_id_edit = None

    host = MagicMock()
    host.iter_box_widgets = lambda: [(1, bw)]
    # The last-saved config is where the fallback serial actually lives.
    host._active_config = Config()
    host._active_config.setup_config.boxes = [
        BoxConfig(setup_number=1, mcu_serial="PRESERVED")]
    cfg = Config()          # fresh, exactly as read_ui_into_config builds it

    with patch("source.config.experiment._widget_text", return_value=""):
        _read_per_box(host, cfg)

    assert cfg.setup_config.boxes[0].mcu_serial == "PRESERVED"
