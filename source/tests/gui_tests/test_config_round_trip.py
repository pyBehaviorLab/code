"""Everything the operator set has to come back the way they set it.

A setting that saves and does not restore is the worst kind of bug in this
application: it takes effect during the session the operator configured it in,
so they watch it work, and it is gone the next time with nothing said. The
recording is then analysed under settings nobody chose.

So this walks the WHOLE of ``TrackingConfig`` field by field rather than a
hand-picked few, a hand-picked list is a list of the fields somebody
remembered, and the field that gets forgotten is exactly the one that breaks.
"""
import dataclasses
import json

import pytest

from source.video.framebus.types import TrackingConfig

#: Fields whose stored form is deliberately not the in-memory one, with why.
#: Anything not here must round-trip identically.
NOT_PERSISTED = {
    # A user-intent marker, not a setting: it says "the operator pressed Apply
    # in this session". A loaded config has, by definition, been applied.
    "user_applied",
}


def _distinct(field):
    """A value for ``field`` that differs from its default.

    Round-tripping a default proves nothing, the value would survive being
    dropped entirely.
    """
    name = field.name
    if name == "setup_id":
        return 7                       # required, so it has no default at all
    if field.default is not dataclasses.MISSING:
        default = field.default
    elif field.default_factory is not dataclasses.MISSING:
        default = field.default_factory()
    else:
        return None
    if default is None:
        # ``Optional[...]`` fields default to None, so the type annotation is
        # the only thing that says what a real value looks like.
        annotation = str(field.type)
        if "int" in annotation:
            return 5
        if "float" in annotation:
            return 0.42
        return {"blob_blur_mode": "median",
                "blob_bg_mode": "self_norm"}.get(name, f"{name}-value")
    if isinstance(default, bool):
        return not default
    if isinstance(default, int) and not isinstance(default, bool):
        return int(default) + 3
    if isinstance(default, float):
        return round(float(default) + 0.17, 4)
    if isinstance(default, str):
        return {
            "tracker_type": "sleap",
            "sleap_model_type": "centroid",
            "sleap_runtime": "onnx",
            "sleap_device": "cuda:1",
            "dlc_model_type": "pytorch",
            "dlc_precision": "FP16",
            "dlc_device": "cuda:1",
            "pose_colour_mode": "grayscale",
            "pose_input_mode": "crop_track",
            "identity_method": "tracker",
            "blob_blur_mode": "median",
            "blob_bg_mode": "self_norm",
            "zone_change_body_part": "Snout",
        }.get(name, f"{name}-value")
    if isinstance(default, (list, tuple)):
        return {
            "keypoint_names": ["Snout", "Head", "center"],
            "skeleton": [["Snout", "Head"], ["Head", "center"]],
            "identities": ["mouse_a", "mouse_b"],
            "rotation_keypoints": ["Snout", "Tail_Base"],
            "zones": [{"name": "Z1", "type": "rectangle",
                       "points": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4],
                                  [0.1, 0.4]],
                       "coord_space": "normalized"}],
            "triggers": [{"condition": "in_zone", "body_part": "center",
                          "zones": ["Z1"], "event_name": "entered",
                          "threshold": 0.5}],
        }.get(name, [f"{name}-item"])
    return None


@pytest.fixture(scope="module")
def loaded_fields():
    """``TrackingConfig`` with every field set away from its default."""
    values, skipped = {}, []
    for f in dataclasses.fields(TrackingConfig):
        v = _distinct(f)
        if v is None:
            skipped.append(f.name)
            continue
        values[f.name] = v
    return values, skipped


def test_every_field_got_a_distinct_value(loaded_fields):
    """If this trips, the generator below stopped covering a field type and
    the round-trip test quietly stopped testing that field."""
    _values, skipped = loaded_fields
    assert not skipped, f"no distinct value generated for {skipped}"


def test_every_field_survives_a_json_round_trip(loaded_fields):
    values, _ = loaded_fields
    cfg = TrackingConfig(**values)
    back = TrackingConfig.from_json(cfg.to_json())

    stored_before = cfg.to_json()
    stored_after = back.to_json()

    wrong = []
    for name, want in values.items():
        if name in NOT_PERSISTED:
            continue
        got = getattr(back, name)
        # Zones and triggers are rebuilt as objects on load, so the in-memory
        # value legitimately differs in type. Their contract is the STORED
        # form, which is what has to be identical.
        if name in ("zones", "triggers"):
            if stored_before.get(name) != stored_after.get(name):
                wrong.append(f"{name}: {stored_before.get(name)!r} -> "
                             f"{stored_after.get(name)!r}")
        elif isinstance(want, (list, tuple)):
            if [list(x) if isinstance(x, (list, tuple)) else x for x in want] \
                    != [list(x) if isinstance(x, (list, tuple)) else x
                        for x in got]:
                wrong.append(f"{name}: {want!r} -> {got!r}")
        elif isinstance(want, float):
            if abs(float(got) - want) > 1e-6:
                wrong.append(f"{name}: {want!r} -> {got!r}")
        elif got != want:
            wrong.append(f"{name}: {want!r} -> {got!r}")
    assert not wrong, "settings did not survive save and load:\n  " + \
        "\n  ".join(wrong)


def test_the_stored_form_is_actually_json(loaded_fields):
    """``to_json`` has to produce something ``json.dump`` accepts, or the
    project write fails at the last step with the session already over."""
    values, _ = loaded_fields
    text = json.dumps(TrackingConfig(**values).to_json())
    assert json.loads(text)


def test_a_second_round_trip_changes_nothing(loaded_fields):
    """Load-then-save must be a fixed point. If it is not, every open-and-close
    of a project silently rewrites it, and two rigs drift apart."""
    values, _ = loaded_fields
    once = TrackingConfig.from_json(TrackingConfig(**values).to_json())
    twice = TrackingConfig.from_json(once.to_json())
    assert once.to_json() == twice.to_json()


def test_an_empty_config_round_trips(loaded_fields):
    """The other end: a box nobody configured must not gain values on save."""
    bare = TrackingConfig(setup_id=1)
    back = TrackingConfig.from_json(bare.to_json())
    assert back.to_json() == bare.to_json()


def test_unknown_keys_in_a_stored_config_do_not_crash_the_load():
    """A project written by a newer build must still open on an older one,
    the alternative is a lab that cannot read its own recordings."""
    raw = TrackingConfig(setup_id=1).to_json()
    raw["a_setting_from_the_future"] = {"nested": [1, 2, 3]}
    back = TrackingConfig.from_json(raw)
    assert back.setup_id == 1
