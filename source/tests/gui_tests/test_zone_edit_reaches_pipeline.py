"""A zone edit has to reach the pipeline, not just the config file.

Zones live in three places at once: the drawing the operator sees, the
project config on disk, and the pipeline's ``TrackingConfig``. Only the last
is what anything measuring from zones actually reads, the latency probe's LED
window among them. An in-place edit used to update the first two and leave the
third holding the geometry captured when the project was loaded, so the probe
went on describing a window 44 px from the lamp it was visibly drawn around,
with nothing on screen showing a disagreement.
"""
from __future__ import annotations

import pytest

from source.gui.base import MainWindowBase


class FakePipeline:
    def __init__(self):
        self.updates = []

    def update_tracking_config(self, setup_id, **fields):
        self.updates.append((setup_id, fields))


class Box:
    """Only the members the two zone boundaries touch."""

    _mirror_zones_to_pipeline = MainWindowBase._mirror_zones_to_pipeline
    _persist_zone_edit = MainWindowBase._persist_zone_edit
    _set_box_zones = MainWindowBase._set_box_zones

    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.tracking_zones = {1: [{"name": "Green", "points": [[0.1, 0.1]]}]}
        self.active_config_path = "project/experiment_config.json"
        self.changed = []

    # the rest of the two boundaries, stubbed
    def _project_changed(self, reason=""):
        self.changed.append(reason)

    def _ensure_zones_tagged(self, zones):
        return zones

    def _snapshot_zone_baseline(self, setup_id):
        pass

    def _invalidate_zone_layer(self, setup_id):
        pass

    def _check_zone_shape_dim(self, setup_id, zones):
        pass


def test_an_in_place_edit_is_handed_to_the_pipeline():
    pipe = FakePipeline()
    box = Box(pipe)
    box.tracking_zones[1] = [{"name": "Green", "points": [[0.25, 0.17]]}]
    box._persist_zone_edit(1)
    assert pipe.updates, "the pipeline was never told the zones moved"
    setup_id, fields = pipe.updates[-1]
    assert setup_id == 1
    assert fields["zones"] == [{"name": "Green", "points": [[0.25, 0.17]]}]


def test_the_edit_still_reaches_the_config_file():
    pipe = FakePipeline()
    box = Box(pipe)
    box._persist_zone_edit(1)
    assert box.changed == ["zones_adjusted"]


def test_loading_zones_into_a_box_also_reaches_the_pipeline():
    pipe = FakePipeline()
    box = Box(pipe)
    box._set_box_zones(1, [{"name": "Green", "points": [[0.9, 0.9]]}])
    assert pipe.updates[-1] == (1, {"zones": [{"name": "Green",
                                               "points": [[0.9, 0.9]]}]})


def test_the_pipeline_is_given_a_copy_not_the_live_list():
    pipe = FakePipeline()
    box = Box(pipe)
    box._persist_zone_edit(1)
    handed = pipe.updates[-1][1]["zones"]
    box.tracking_zones[1].append({"name": "Red"})
    assert len(handed) == 1, "the pipeline's list followed a later edit"


@pytest.mark.parametrize("pipeline", [None])
def test_no_pipeline_is_not_an_error(pipeline):
    box = Box(pipeline)
    box._persist_zone_edit(1)          # must not raise
    assert box.changed == ["zones_adjusted"]


def test_a_pipeline_that_refuses_does_not_lose_the_edit():
    class Broken(FakePipeline):
        def update_tracking_config(self, setup_id, **fields):
            raise RuntimeError("no such box")

    box = Box(Broken())
    box._persist_zone_edit(1)          # must not raise
    assert box.changed == ["zones_adjusted"]
