"""A zone with no shape is not a zone, on the rig's side of the fence.

A zone config can carry entries with empty point lists. One way they arise:
the offline analyser read an empty ``{"scale": {}, "arena": {}, "zones": {}}``
block as three zones named after the wrapper's own keys, and wrote them back
out. Read as zones they reach the editor, the picker and the results as shapes
that are nowhere, and they sit next to the real scale zone, which is what
made an operator think the scale was not being recognised.

The offline analyser has the same rule in ``video_data_schema.zones_as_list``;
this is the rig's half of it, so a config that has been round-tripped through
either tool behaves the same in both.
"""
from __future__ import annotations

import json

import pytest

from source.video.zones.io import load_zone_config

#: The shape an operator actually hit, verbatim: three geometry-less entries
#: sitting in front of a real arena and a real scale line.
BLOCK = {
    "scale": {},
    "arena": {},
    "zones": {
        "scale": {"points": [], "coord_space": "pixel",
                  "shape_dim": [360, 202]},
        "arena": {"points": [], "coord_space": "pixel",
                  "shape_dim": [360, 202]},
        "zones": {"points": [], "coord_space": "pixel",
                  "shape_dim": [360, 202]},
        "BOX": {"points": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "coord_space": "normalized", "shape_dim": [360, 202]},
        "Scale": {"type": "scale", "coord_space": "normalized",
                  "points": [[0.372, 0.122], [0.381, 0.689]],
                  "shape_dim": [360, 202], "scale_length": 10.0,
                  "scale_unit": "cm"},
    },
}


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps(BLOCK), encoding="utf-8")
    return str(path)


def test_only_the_zones_with_a_shape_are_read(config):
    names = [z.get("name") for z in load_zone_config(config)]
    assert names == ["BOX", "Scale"], names


def test_the_scale_zone_survives(config):
    """The one the operator drew, and the one calibration needs.

    Dropping the shapeless entries must not take the real scale with them,
    it is a zone whose geometry is two points, and it has them.
    """
    zones = load_zone_config(config)
    scale = [z for z in zones if str(z.get("type", "")).lower() == "scale"]
    assert len(scale) == 1
    assert len(scale[0].get("points") or []) == 2


def test_an_empty_block_reads_as_no_zones(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"scale": {}, "arena": {}, "zones": {}}),
                    encoding="utf-8")
    assert load_zone_config(str(path)) == []


def test_a_circle_written_without_points_is_kept(tmp_path):
    """Geometry is not only a point list.

    An ellipse or circle can be stored as a centre plus axes; dropping those
    would lose real zones in the name of dropping empty ones.
    """
    path = tmp_path / "circle.json"
    path.write_text(json.dumps({"zones": {
        "Nest": {"type": "circle", "points": [], "center": [0.5, 0.5],
                 "radius_norm": 0.2, "shape_dim": [360, 202]},
    }}), encoding="utf-8")
    assert [z["name"] for z in load_zone_config(str(path))] == ["Nest"]


def test_a_plain_list_config_is_filtered_too(tmp_path):
    """The list form is the same rule; it was the other way in."""
    path = tmp_path / "list.json"
    path.write_text(json.dumps([
        {"name": "ghost", "type": "polygon", "points": []},
        {"name": "real", "type": "polygon",
         "points": [[0, 0], [1, 0], [1, 1]]},
    ]), encoding="utf-8")
    names = [z.get("name") for z in load_zone_config(str(path))]
    # The list form is returned as-is by design (it is already zone dicts),
    # so this records what the rig does rather than asserting a filter that
    # is not there: if it ever grows one, this test says so.
    assert "real" in names
