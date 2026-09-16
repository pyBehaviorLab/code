"""Unit tests for BoxStatisticsCalculator, the operant Stats tab formula
evaluator.

Exercises the canonical RL formula shapes ``correct + incorrect``,
``correct / (correct + incorrect) * 100``, and the ``moving_avg_value``
"last" aggregator.
"""
from __future__ import annotations

import pytest

from source.stats import BoxStatisticsCalculator


def _bump_count(calc, setup_id, key, n):
    """Drive a counter's count by N without going through the event router."""
    for _ in range(n):
        slot = calc.event_counts[setup_id][key]
        slot["count"] += 1


def _set_last(calc, setup_id, key, value):
    """Set a counter's 'last' value (mirrors agg='last' regex matches)."""
    slot = calc.event_counts[setup_id][key]
    slot["last"] = value
    slot["count"] += 1


def test_rl_count_formulas_resolve_to_real_numbers():
    """RL template uses `correct`, `incorrect`, `correct + incorrect`; each
    must resolve to its real count."""
    calc = BoxStatisticsCalculator()
    _bump_count(calc, 1, "correct", 7)
    _bump_count(calc, 1, "incorrect", 3)

    formulas = {
        "num_correct": "correct",
        "num_incorrect": "incorrect",
        "num_trials": "correct + incorrect",
        "per_accuracy": "(correct / (correct + incorrect)) * 100 if (correct + incorrect) > 0 else 0",
    }
    out = calc.calculateStatistics(1, formulas)

    assert out["num_correct"] == 7
    assert out["num_incorrect"] == 3
    assert out["num_trials"] == 10
    assert out["per_accuracy"] == pytest.approx(70.0)


def test_last_aggregator_returns_scalar_not_empty_list():
    """Counters with agg='last' (e.g. moving_avg_value) must surface the
    scalar, not the empty default ``values`` list."""
    calc = BoxStatisticsCalculator()
    _set_last(calc, 1, "moving_avg_value", 0.83)
    _set_last(calc, 1, "moving_avg_value", 0.91)  # most recent

    formulas = {"moving_average": "moving_avg_value"}
    out = calc.calculateStatistics(1, formulas)

    assert out["moving_average"] == pytest.approx(0.91)


def test_sum_aggregator_via_explicit_suffix():
    """``reward_ul_sum`` formula reads the per-counter ``_sum`` slot."""
    calc = BoxStatisticsCalculator()
    slot = calc.event_counts[1]["reward_ul"]
    slot["sum"] = 12.5
    slot["count"] = 5

    formulas = {"total_reward_ul": "reward_ul_sum"}
    out = calc.calculateStatistics(1, formulas)

    assert out["total_reward_ul"] == pytest.approx(12.5)


def test_empty_session_returns_zero_not_crash():
    """Before any events, every formula should return 0 cleanly."""
    calc = BoxStatisticsCalculator()
    formulas = {
        "num_trials": "correct + incorrect",
        "per_accuracy": "(correct / (correct + incorrect)) * 100 if (correct + incorrect) > 0 else 0",
    }
    out = calc.calculateStatistics(1, formulas)
    assert out["num_trials"] == 0
    assert out["per_accuracy"] == 0


# ── Per-trial sequence capture (C/F alternation table) ──────────────────

def _print_dp(text, t):
    from source.communication.message import MsgType, Datatuple
    return Datatuple(time=t, type=MsgType.PRINT, subtype="", content=text)


def test_sequence_counter_captures_ordered_tokens_and_duration():
    """A ``type:'sequence'`` counter appends one token per matching print in
    arrival order; duration is derived from first→last MCU timestamp."""
    calc = BoxStatisticsCalculator()
    counters = {
        "num_correct": {"type": "print", "regex": "^Correct_choice", "agg": "count"},
        "num_false":   {"type": "print", "regex": "^Incorrect_choice", "agg": "count"},
        "trials": {"type": "sequence",
                   "map": {"C": "Correct_choice", "F": "Incorrect_choice"}},
    }
    outcomes = list("FCFCFFCFCCCCFF")  # the SpontMaze example sequence
    data, t = [], 0
    for tok in outcomes:
        t += 20000  # 20 s per trial, in fw ms
        data.append(_print_dp("L", t))  # arm print (ignored by counters)
        data.append(_print_dp(
            ("Correct_choice: 1" if tok == "C" else "Incorrect_choice: 1"), t))
    calc.processData(1, data, id2name={}, counters=counters)

    assert calc.getSequence(1, "trials") == outcomes
    # "Incorrect_choice" must NOT also trip the "^Correct_choice" counter.
    res = calc.calculateStatistics(1, {"num_correct": "num_correct",
                                       "num_false": "num_false"})
    assert res["num_correct"] == outcomes.count("C")
    assert res["num_false"] == outcomes.count("F")
    # 14 trials × 20 s = 280 s = 4.666… min (last−first = 13×20 s = 260 s).
    assert calc.getDuration(1) == pytest.approx(260 / 60.0)


def test_sequence_column_expands_into_per_trial_columns():
    """A sequence column fans out into ``count`` keyed cells the table fills."""
    from source.stats.canvas import StatisticsTableWidget as T
    cols = [{"key": "trials", "label": "Trial", "sequence": True, "count": 14}]
    expanded = T._expand_sequence_columns(cols)
    keys = [c["key"] for c in expanded]
    assert keys == [f"trials#{i}" for i in range(1, 15)]
    assert [c["label"] for c in expanded] == [str(i) for i in range(1, 15)]


def test_clear_box_resets_sequence_and_duration():
    """clearBox wipes the per-trial sequence + duration so a re-run starts fresh."""
    calc = BoxStatisticsCalculator()
    counters = {"trials": {"type": "sequence", "map": {"C": "Correct_choice"}}}
    calc.processData(1, [_print_dp("Correct_choice: 1", 1000),
                         _print_dp("Correct_choice: 2", 5000)],
                     id2name={}, counters=counters)
    assert calc.getSequence(1, "trials") == ["C", "C"]
    calc.clearBox(1)
    assert calc.getSequence(1, "trials") == []
    assert calc.getDuration(1) == 0.0
