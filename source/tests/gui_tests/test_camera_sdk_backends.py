"""Drive the Spinnaker and Ximea feature paths against faithful fake SDKs.

Neither SDK is installed on every dev machine (and FLIR ships no PySpin wheel
for every Python the app supports), so these paths otherwise ship unexecuted.
The fakes here mimic the shapes the real APIs present, a GenICam node map of
typed nodes behind ``C*Ptr`` casts for PySpin, a parameter table reached by
``get_param``/``set_param`` for xiAPI, so the walk, the type mapping, the
clamping and the write-back are all genuinely exercised.

They are deliberately strict: a wrong-typed cast raises, exactly as an invalid
pointer does in PySpin. Code that guesses at a node's type fails here.
"""
import sys
import types

import pytest

from source.video.cameras.features import (
    KIND_BOOL,
    KIND_ENUM,
    KIND_FLOAT,
    KIND_INT,
    KIND_STRING,
)

# ── fake PySpin ───────────────────────────────────────────────────────────

INTF = {name: i for i, name in enumerate(
    ["intfIValue", "intfIBase", "intfIInteger", "intfIBoolean", "intfICommand",
     "intfIFloat", "intfIString", "intfIRegister", "intfICategory",
     "intfIEnumeration", "intfIEnumEntry", "intfIPort"])}


class _Node:
    def __init__(self, name, iface, value=None, minimum=None, maximum=None,
                 inc=None, entries=(), readable=True, writable=True,
                 tooltip=""):
        self.name = name
        self.iface = iface
        self.value = value
        self.minimum = minimum
        self.maximum = maximum
        self.inc = inc
        self.entries = list(entries)
        self.readable = readable
        self.writable = writable
        self.tooltip = tooltip

    def GetPrincipalInterfaceType(self):
        return self.iface

    def GetName(self):
        return self.name

    def GetToolTip(self):
        return self.tooltip

    def GetDescription(self):
        return self.tooltip


class _Category(_Node):
    def __init__(self, name, children):
        super().__init__(name, INTF["intfICategory"])
        self.children = children


class _EnumEntry(_Node):
    def __init__(self, symbolic, intval):
        super().__init__(symbolic, INTF["intfIEnumEntry"])
        self.symbolic = symbolic
        self.intval = intval

    def GetSymbolic(self):
        return self.symbolic

    def GetValue(self):
        return self.intval


def _typed_ptr(expected):
    """A C*Ptr cast that raises on the wrong node type, like an invalid
    pointer dereference does in PySpin."""
    class _Ptr:
        def __init__(self, node):
            if node.iface != INTF[expected]:
                raise RuntimeError(f"{expected} cast on {node.name}")
            self._n = node

        def GetValue(self):
            return self._n.value

        def SetValue(self, v):
            self._n.value = v

        def GetMin(self):
            return self._n.minimum

        def GetMax(self):
            return self._n.maximum

        def GetInc(self):
            return self._n.inc

        def GetEntries(self):
            return self._n.entries

        def GetCurrentEntry(self):
            return next(e for e in self._n.entries
                        if e.intval == self._n.value)

        def GetEntryByName(self, name):
            return next((e for e in self._n.entries if e.symbolic == name),
                        None)

        def SetIntValue(self, v):
            self._n.value = v

        def GetName(self):
            return self._n.name

        def GetFeatures(self):
            return self._n.children

        def Execute(self):
            self._n.value = "executed"
    return _Ptr


def _make_pyspin():
    m = types.ModuleType("PySpin")
    for k, v in INTF.items():
        setattr(m, k, v)
    m.CFloatPtr = _typed_ptr("intfIFloat")
    m.CIntegerPtr = _typed_ptr("intfIInteger")
    m.CBooleanPtr = _typed_ptr("intfIBoolean")
    m.CStringPtr = _typed_ptr("intfIString")
    m.CEnumerationPtr = _typed_ptr("intfIEnumeration")
    m.CEnumEntryPtr = lambda n: n
    m.CCommandPtr = _typed_ptr("intfICommand")
    m.CCategoryPtr = _typed_ptr("intfICategory")
    m.IsAvailable = lambda n: True
    m.IsReadable = lambda n: n.readable
    m.IsWritable = lambda n: n.writable
    return m


def _nodemap():
    exposure = _Node("ExposureTime", INTF["intfIFloat"], 5000.0, 14.0, 29998.0,
                     tooltip="Exposure time in microseconds.")
    width = _Node("Width", INTF["intfIInteger"], 1440, 16, 1440, inc=4)
    model = _Node("DeviceModelName", INTF["intfIString"], "Blackfly S BFS-U3",
                  writable=False)
    serial = _Node("DeviceSerialNumber", INTF["intfIString"], "21398712",
                   writable=False)
    fmt = _Node("PixelFormat", INTF["intfIEnumeration"], 1,
                entries=[_EnumEntry("Mono8", 1), _EnumEntry("Mono16", 2)])
    rev = _Node("ReverseX", INTF["intfIBoolean"], False)
    nodes = {n.name: n for n in (exposure, width, model, serial, fmt, rev)}
    root = _Category("Root", [
        _Category("AnalogControl", [exposure, rev]),
        _Category("ImageFormatControl", [width, fmt]),
        _Category("DeviceControl", [model, serial]),
    ])

    class _Map:
        def GetNode(self, name):
            return root if name == "Root" else nodes.get(name)
    return _Map(), nodes


@pytest.fixture
def pyspin(monkeypatch):
    m = _make_pyspin()
    monkeypatch.setitem(sys.modules, "PySpin", m)
    return m


class _Cam:
    def __init__(self, nodemap):
        self._nodemap = nodemap


# ── Spinnaker: describe ───────────────────────────────────────────────────

def test_spinnaker_walk_maps_every_node_type(pyspin):
    from source.video.cameras.builtin_backends import _spinnaker_features
    nm, _ = _nodemap()
    feats = {f.key: f for f in _spinnaker_features(_Cam(nm))}

    assert feats["exposure_us"].kind == KIND_FLOAT
    assert (feats["exposure_us"].minimum, feats["exposure_us"].maximum) == (
        14.0, 29998.0)
    assert feats["exposure_us"].unit == "us"
    assert feats["exposure_us"].live is True        # applies without reopening

    assert feats["width"].kind == KIND_INT
    assert feats["width"].increment == 4            # sensor step, not 1

    assert feats["pixel_format"].kind == KIND_ENUM
    assert [v for v, _ in feats["pixel_format"].options] == ["Mono8", "Mono16"]

    assert feats["model"].kind == KIND_STRING
    assert feats["reverse_x"].kind == KIND_BOOL


def test_spinnaker_walk_carries_access_and_help(pyspin):
    from source.video.cameras.builtin_backends import _spinnaker_features
    nm, _ = _nodemap()
    feats = {f.key: f for f in _spinnaker_features(_Cam(nm))}
    # A read-only node must not be offered as editable.
    assert feats["model"].access == "ro"
    assert feats["exposure_us"].access == "rw"
    assert "microseconds" in feats["exposure_us"].help


def test_spinnaker_walk_survives_a_missing_nodemap(pyspin):
    from source.video.cameras.builtin_backends import _spinnaker_features
    assert _spinnaker_features(object()) == []


# ── Spinnaker: read ───────────────────────────────────────────────────────

@pytest.mark.parametrize("key,expected", [
    ("exposure_us", 5000.0),
    ("width", 1440),
    ("model", "Blackfly S BFS-U3"),
    ("serial", "21398712"),
    ("pixel_format", "Mono8"),
])
def test_spinnaker_get_reads_each_type(pyspin, key, expected):
    from source.video.cameras.builtin_backends import _spinnaker_get
    nm, _ = _nodemap()
    assert _spinnaker_get(_Cam(nm), key) == expected


def test_spinnaker_get_returns_none_for_unreadable(pyspin):
    from source.video.cameras.builtin_backends import _spinnaker_get
    nm, nodes = _nodemap()
    nodes["ExposureTime"].readable = False
    assert _spinnaker_get(_Cam(nm), "exposure_us") is None


# ── Spinnaker: write ──────────────────────────────────────────────────────

def test_spinnaker_set_clamps_to_the_sensor_range(pyspin):
    from source.video.cameras.builtin_backends import _spinnaker_set
    nm, nodes = _nodemap()
    cam = _Cam(nm)
    assert _spinnaker_set(cam, "exposure_us", 999999) is True
    assert nodes["ExposureTime"].value == 29998.0      # clamped to max
    assert _spinnaker_set(cam, "exposure_us", 1) is True
    assert nodes["ExposureTime"].value == 14.0         # clamped to min


def test_spinnaker_set_enum_uses_the_symbolic_name(pyspin):
    from source.video.cameras.builtin_backends import _spinnaker_set
    nm, nodes = _nodemap()
    assert _spinnaker_set(_Cam(nm), "pixel_format", "Mono16") is True
    assert nodes["PixelFormat"].value == 2
    # An entry this camera does not offer is refused, not silently ignored.
    assert _spinnaker_set(_Cam(nm), "pixel_format", "BayerRG8") is False


def test_spinnaker_set_refuses_a_read_only_node(pyspin):
    from source.video.cameras.builtin_backends import _spinnaker_set
    nm, nodes = _nodemap()
    assert _spinnaker_set(_Cam(nm), "model", "nope") is False
    assert nodes["DeviceModelName"].value == "Blackfly S BFS-U3"


# ── Ximea ─────────────────────────────────────────────────────────────────

class _FakeXiCam:
    """xiAPI surface: typed get_param/set_param plus _maximum/_minimum."""

    def __init__(self):
        self.params = {
            "device_name": "MQ022RG-CM", "device_sn": "XECAS1930018",
            "exposure": 5000, "gain": 0.0, "framerate": 60.0,
            "width": 2048, "height": 1088,
            "image_data_format": "XI_MONO8", "trg_source": "XI_TRG_OFF",
        }
        self.bounds = {"exposure": (14, 1000000), "gain": (0.0, 24.0),
                       "framerate": (1.0, 170.5), "width": (32, 2048)}

    def get_param(self, name):
        if name.endswith(":max"):
            return self.bounds[name[:-4]][1]
        if name.endswith(":min"):
            return self.bounds[name[:-4]][0]
        if name not in self.params:
            raise RuntimeError(f"unknown param {name}")
        return self.params[name]

    def set_param(self, name, value):
        if name not in self.params:
            raise RuntimeError(f"unknown param {name}")
        self.params[name] = value


class _XiWrapper:
    def __init__(self):
        self._cam = _FakeXiCam()


def test_ximea_describes_its_parameter_table():
    from source.video.cameras.builtin_backends import _ximea_features
    feats = {f.key: f for f in _ximea_features(_XiWrapper())}
    assert feats["exposure_us"].kind == KIND_INT
    assert feats["exposure_us"].unit == "us"
    assert feats["gain_db"].kind == KIND_FLOAT
    assert feats["pixel_format"].kind == KIND_ENUM
    assert "XI_MONO8" in [v for v, _ in feats["pixel_format"].options]
    # Identity is reported, never offered as an editable control.
    assert feats["model"].access == "ro"


def test_ximea_reads_and_writes_through_the_wrapper():
    from source.video.cameras.builtin_backends import _ximea_get, _ximea_set
    cam = _XiWrapper()
    assert _ximea_get(cam, "exposure_us") == 5000
    assert _ximea_get(cam, "model") == "MQ022RG-CM"
    assert _ximea_set(cam, "exposure_us", 8000) is True
    assert cam._cam.params["exposure"] == 8000
    assert _ximea_set(cam, "pixel_format", "XI_MONO16") is True
    assert cam._cam.params["image_data_format"] == "XI_MONO16"


def test_ximea_unknown_parameter_is_reported_not_raised():
    from source.video.cameras.builtin_backends import _ximea_get, _ximea_set
    cam = _XiWrapper()
    assert _ximea_get(cam, "no_such_key") is None
    assert _ximea_set(cam, "no_such_key", 1) is False


# ── the registry routes to the right backend ──────────────────────────────

def test_registry_describes_through_the_backend_spec(pyspin):
    from source.video.cameras import registry
    nm, _ = _nodemap()
    feats = registry.describe_features("spinnaker", _Cam(nm))
    assert any(f.key == "exposure_us" for f in feats)


def test_registry_is_quiet_about_an_unknown_backend():
    from source.video.cameras import registry
    assert registry.describe_features("no_such_backend", object()) == []
    assert registry.get_feature("no_such_backend", object(), "k") is None
    assert registry.set_feature("no_such_backend", object(), "k", 1) is False
