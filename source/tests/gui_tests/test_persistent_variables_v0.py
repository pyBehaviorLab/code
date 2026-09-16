"""Host-side persistent-variable round-trip: push at Upload, capture at Stop,
restore at next Upload, keyed by variable NAME, per project + task (no
subject dimension).

These tests use a fake pycboard (no serial / no MCU) so they cover the
host-side ``persistent_variables.json`` round-trip without hardware.

The apply pipeline COLLECTS pushes into ``ApplyResult.pushed`` (one batched
``set_variables`` round-trip is flushed by the caller); capture reads the whole
MCU var set in one ``get_variables()`` call.
"""
from __future__ import annotations

import json

from source.config.experiment import BoxVariableSpec
from source.config.task_variables import (
    PERSISTENT_FILE,
    apply_pre_run_hw,
    capture_persistent,
    read_pv_dict,
    restore_pers_vars,
    write_pers_vars,
)


# -----------------------------------------------------------------------------
# Fake MCU board
# -----------------------------------------------------------------------------


class _FakeSmInfo:
    def __init__(self, variables):
        self.variables = dict(variables)


class _FakePycboard:
    """Just enough of pycboard for the persistent helpers."""

    def __init__(self, variables):
        self.sm_info = _FakeSmInfo(variables)
        self.framework_running = False

    def get_variables(self):
        return dict(self.sm_info.variables)

    def set_variables(self, var_dict, source="s"):
        out = {}
        for k, v in var_dict.items():
            if k in self.sm_info.variables:
                self.sm_info.variables[k] = v
                out[k] = True
        return out


# -----------------------------------------------------------------------------
# apply_pre_run_hw, hw_* collected into .pushed
# -----------------------------------------------------------------------------


def test_apply_pre_run_hw_collects_only_hw_vars():
    pyc = _FakePycboard({"hw_reward_pin": 0, "hw_left_lever": 0, "n_trials": 0})
    out = apply_pre_run_hw(pyc, [],
                           hw_prompt_values={"hw_reward_pin": 5,
                                             "hw_left_lever": 7})
    assert out.pushed == {"hw_reward_pin": 5, "hw_left_lever": 7}
    assert "n_trials" not in out.pushed
    assert not out.hw_missing


def test_apply_pre_run_hw_missing_returns_to_caller():
    pyc = _FakePycboard({"hw_reward_pin": 0})
    out = apply_pre_run_hw(pyc, [], hw_prompt_values=None)
    assert out.pushed == {}
    assert "hw_reward_pin" in out.hw_missing


# -----------------------------------------------------------------------------
# Persistent capture + restore round trip, the name-keyed cycle
# -----------------------------------------------------------------------------


def test_persistent_capture_then_restore_round_trip(tmp_path):
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    task_family = "reversal_learning"

    specs = [
        BoxVariableSpec(name="n_blocks",  persistent=True),
        BoxVariableSpec(name="stage",     persistent=True),
        BoxVariableSpec(name="good_side", persistent=True),
        BoxVariableSpec(name="reward_duration"),                 # NOT persistent
    ]
    pyc_day1 = _FakePycboard({
        "n_blocks":        4,         # final value at Stop on Day 1
        "stage":           3,
        "good_side":       "right",
        "reward_duration": 100,
    })

    # Stop hook: one get_variables() round-trip, filtered to persistent names.
    pers = capture_persistent(pyc_day1, specs)
    assert pers == {"n_blocks": 4, "stage": 3, "good_side": "right"}
    path = write_pers_vars(project_dir, task_family, pers)
    assert path == project_dir / task_family / PERSISTENT_FILE

    raw = json.loads(path.read_text(encoding="utf-8"))
    # Flat, by name, no subject layer.
    assert raw["values"] == {"n_blocks": 4, "stage": 3, "good_side": "right"}
    assert "persistent" not in raw  # obsolete top-level names list dropped

    # Day 2: fresh upload, MCU at module defaults. Restore collects pushes.
    pyc_day2 = _FakePycboard({
        "n_blocks": 0, "stage": 1, "good_side": "left", "reward_duration": 100,
    })
    out = restore_pers_vars(pyc_day2, project_dir, task_family, specs)
    assert out.pushed == {"n_blocks": 4, "stage": 3, "good_side": "right"}
    assert all(src == "(persistent value)" for _, _, src in out.set_lines)


def test_reset_var_is_never_saved_or_restored(tmp_path):
    """A non-persistent (RESET) var is never captured, so nothing restores it,
    it stays at the task-file default the MCU already holds."""
    proj = tmp_path / "p"
    proj.mkdir()
    specs = [
        BoxVariableSpec(name="stage", persistent=True),
        BoxVariableSpec(name="n_trials"),                 # RESET
    ]
    pers = capture_persistent(_FakePycboard({"stage": 3, "n_trials": 99}), specs)
    assert pers == {"stage": 3}
    write_pers_vars(proj, "rl", pers)
    # Day 2 board has n_trials at its default 0; restore leaves it untouched.
    pyc = _FakePycboard({"stage": 1, "n_trials": 0})
    out = restore_pers_vars(pyc, proj, "rl", specs)
    assert out.pushed == {"stage": 3}
    assert "n_trials" not in out.pushed


def test_capture_skips_non_persistent_vars():
    pyc = _FakePycboard({"n_blocks": 7, "reward_duration": 100})
    specs = [
        BoxVariableSpec(name="n_blocks", persistent=True),
        BoxVariableSpec(name="reward_duration"),   # not persistent
    ]
    assert capture_persistent(pyc, specs) == {"n_blocks": 7}


def test_capture_empty_when_board_read_fails():
    """Error-stop guard: a wedged board (get_variables raises) yields {} so the
    caller keeps the previous saved values."""
    class _Wedged(_FakePycboard):
        def get_variables(self):
            raise RuntimeError("serial dead")
    pyc = _Wedged({"n_blocks": 7})
    specs = [BoxVariableSpec(name="n_blocks", persistent=True)]
    assert capture_persistent(pyc, specs) == {}


def test_write_merges_by_name(tmp_path):
    proj = tmp_path / "p"
    proj.mkdir()
    write_pers_vars(proj, "rl", {"n_blocks": 4, "stage": 3})
    write_pers_vars(proj, "rl", {"stage": 5})            # update one, keep other
    data = json.loads((proj / "rl" / PERSISTENT_FILE).read_text(encoding="utf-8"))
    assert data["values"] == {"n_blocks": 4, "stage": 5}


def test_read_pv_dict_empty_without_project():
    assert read_pv_dict(None, "rl") == {}
    assert read_pv_dict("/tmp", "") == {}


def test_restore_no_op_without_project():
    """Persistent restore no-ops when project_dir/task are missing (dry-run /
    no-project path)."""
    pyc = _FakePycboard({"n_blocks": 0})
    specs = [BoxVariableSpec(name="n_blocks", persistent=True)]
    out = restore_pers_vars(pyc, None, "rl", specs)      # no project
    assert out.pushed == {} and out.set_lines == []


def test_legacy_subject_nested_file_is_ignored(tmp_path):
    """A pre-migration file with subject-nested values reads as 'nothing
    remembered yet', and the next write self-heals it to the flat schema."""
    proj = tmp_path / "p"
    (proj / "rl").mkdir(parents=True)
    (proj / "rl" / PERSISTENT_FILE).write_text(json.dumps({
        "persistent": ["stage"],
        "values": {"M01": {"stage": 3}, "M02": {"stage": 2}},
    }), encoding="utf-8")
    assert read_pv_dict(proj, "rl") == {}                # nested → ignored
    write_pers_vars(proj, "rl", {"stage": 7})            # self-heal
    data = json.loads((proj / "rl" / PERSISTENT_FILE).read_text(encoding="utf-8"))
    assert data["values"] == {"stage": 7}                # flat, subjects gone
