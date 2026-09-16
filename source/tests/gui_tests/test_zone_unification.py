"""There is one Zone class, and a zone's authored policy survives the trip.

Two of them, an 18-field persisted one and a 9-field runtime one, cannot
hold the same zone: the runtime class cannot represent eight of the persisted
fields, the two disagree on the on-disk key name, and a bridge that names
fields explicitly drops whatever it was not told about. The visible result is
every zone running on the DEFAULT transmit policy rather than the authored
one.
"""
import json
from pathlib import Path

import pytest

from source.config.experiment import Zone as ConfigZone
from source.config.experiment import zone_to_dict
from source.video.framebus.types import _zone_from_json, _zone_to_json
from source.video.zones.schema import Zone
from source.video.zones.triggering import Zone as TriggerZone

REPO = Path(__file__).resolve().parents[3]


def _poly(**kw):
    base = {"name": "Arm", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            "type": "polygon"}
    base.update(kw)
    return base


def test_there_is_exactly_one_zone_class():
    assert ConfigZone is Zone
    assert TriggerZone is Zone


def test_on_disk_key_is_type_not_zone_type():
    """Every project file, zone file and tracking config in the repo uses
    "type". The field is named zone_type only because `type` is a builtin."""
    d = zone_to_dict(Zone.from_dict(_poly()))
    assert d["type"] == "polygon"
    assert "zone_type" not in d


def test_authored_transmit_policy_survives_a_round_trip():
    z = Zone.from_dict(_poly(transmit_mode="event_enter_exit",
                             event_on_enter="arm_in",
                             event_on_exit="arm_out",
                             coord_var="loc_arm"))
    z2 = Zone.from_dict(zone_to_dict(z))
    assert z2.transmit_mode == "event_enter_exit"
    assert z2.event_on_enter == "arm_in"
    assert z2.event_on_exit == "arm_out"
    assert z2.coord_var == "loc_arm"


def test_scale_and_ellipse_fields_survive():
    """These are the fields the runtime class could not hold at all."""
    z = Zone.from_dict(_poly(type="scale", scale_length=12.5, scale_unit="cm",
                             scale_length_mm=125.0, coord_space="normalized",
                             validated=True))
    z2 = Zone.from_dict(zone_to_dict(z))
    assert (z2.scale_length, z2.scale_unit, z2.scale_length_mm) == (12.5, "cm", 125.0)
    assert z2.coord_space == "normalized"
    assert z2.validated is True

    e = Zone.from_dict(_poly(type="circle", center=[0.5, 0.5],
                             semi_axes=[0.2, 0.1], radius_norm=0.15))
    e2 = Zone.from_dict(zone_to_dict(e))
    assert e2.center == [0.5, 0.5]
    assert e2.semi_axes == [0.2, 0.1]
    assert e2.radius_norm == 0.15


@pytest.mark.parametrize("value", [None, "", 0])
def test_null_transmit_policy_resolves_to_the_working_default(value):
    """Consumers read `getattr(z, "transmit_mode", "coord")`, whose default
    only fires when the attribute is ABSENT. A present-but-None value flows
    through and matches no branch, silently disabling the zone."""
    z = Zone.from_dict(_poly(transmit_mode=value, coord_var=value))
    assert z.transmit_mode == "coord"
    assert z.coord_var == "loc_center"


def test_framebus_and_config_serialisers_agree():
    """framebus wrote/read "zone_type" while everything on disk uses "type",
    so loading a real file through it gave every zone the default shape."""
    z = Zone.from_dict(_poly(type="circle", transmit_mode="event_enter"))
    assert _zone_to_json(z) == zone_to_dict(z)
    assert _zone_from_json(zone_to_dict(z)) == z


def test_a_real_file_keeps_its_shapes():
    """Shapes must survive the round trip, not all become the default."""
    f = REPO / "experiments/config/zones/Multimaze_with_motors_Zones.json"
    if not f.exists():
        pytest.skip("sample zone file not present")
    raw = json.loads(f.read_text(encoding="utf-8"))
    zones = raw.get("zones") if isinstance(raw, dict) else raw
    if isinstance(zones, dict):
        zones = list(zones.values())
    zones = [z for z in zones if isinstance(z, dict) and "points" in z]
    assert zones, "sample file had no zones"
    for zd in zones:
        loaded = _zone_from_json(zd)
        assert loaded.zone_type == zd.get("type", "rectangle"), (
            f"{zd.get('name')} loaded as {loaded.zone_type}, "
            f"file says {zd.get('type')}")


def test_serialisation_is_json_encodable_with_a_live_shapely_cache():
    """asdict() on a Zone holding a Shapely polygon fails at encode time."""
    z = Zone.from_dict(_poly(points=[[0, 0], [1, 0], [1, 1], [0, 1]]))
    json.dumps(zone_to_dict(z))


def test_zone_survives_deepcopy_and_pickle():
    """A prepared Shapely geometry refuses to pickle outright, which surfaced
    as a PicklingError from the autosave/reload path."""
    import copy
    import pickle

    z = Zone.from_dict(_poly(points=[[0, 0], [1, 0], [1, 1], [0, 1]],
                             transmit_mode="event_enter",
                             event_on_enter="in_arm"))
    assert copy.deepcopy(z) == z
    revived = pickle.loads(pickle.dumps(z))
    assert revived == z
    assert revived.transmit_mode == "event_enter"
    # The cache is derived, so it must come back usable rather than stale.
    assert revived.contains(0.5, 0.5)


def test_shapely_cache_is_not_part_of_equality_or_serialisation():
    a = Zone.from_dict(_poly())
    b = Zone.from_dict(_poly())
    assert a == b
    assert "_polygon" not in zone_to_dict(a)
    assert "_prepared" not in zone_to_dict(a)
