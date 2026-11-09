"""Handoff 3 scoring-contract tests (written and run BEFORE any tuning code existed).

All series are tiny hand-built SYNTHETIC sequences at 5-minute cadence starting 2026-01-01T00:00Z.
"""
import math

import pandas as pd
import pytest

from smart_grid import fixtures
from smart_grid.detectors import FixedLimitDetector, RollingMADDetector
from smart_grid.evaluation import (
    Partition, collect_flags, eligible_slots, evaluate_method, evaluate_partition,
)
from smart_grid.replay import replay
from smart_grid.scoring import detection_latency_min, match_alerts

START = fixtures.START_UTC


def t(minute):
    return START + pd.Timedelta(minutes=minute)


def frame(values):
    return pd.DataFrame({"timestamp_utc": [t(i * 5) for i in range(len(values))],
                         "voltage_v": [math.nan if v is None else float(v) for v in values]})


def faults(*rows):  # (id, start_minute, end_minute_exclusive)
    return pd.DataFrame([{"fault_id": i, "fault_type": "x", "start_utc": t(s), "end_utc_exclusive": t(e)}
                         for i, s, e in rows],
                        columns=["fault_id", "fault_type", "start_utc", "end_utc_exclusive"])


def whole(df):
    return Partition("all", t(-5), t(len(df) * 5))


def fixed():
    return FixedLimitDetector(114, 126)


# ---- eligibility and shared denominator ---------------------------------------------------------

def test_eligible_slots_need_12_prior_valid_and_skip_missing():
    r = frame([120] * 14)
    r.loc[3, "voltage_v"] = math.nan
    e = eligible_slots(r)
    assert list(e[e].index) == [13]  # idx 12 has only 11 valid priors; idx 3 (missing) is never eligible
    assert not e[3]


def test_both_methods_share_eligible_slots_and_denominator():
    df = frame([120] * 20 + [None] + [120] * 5)  # 25 valid -> 13 eligible
    for det in (fixed(), RollingMADDetector()):
        res = evaluate_method(df, faults(), det, whole(df))
        assert res["eligible_slots"] == 13
        assert res["monitored_sensor_hours"] == pytest.approx(13 * 5 / 60)


def test_false_alert_rate_uses_eligible_hours():
    df = frame([120] * 12 + [130] + [120] * 11)  # 12 eligible slots = 1.0 h, one unused alert
    res = evaluate_method(df, faults(), fixed(), whole(df))
    assert res["eligible_slots"] == 12 and res["monitored_sensor_hours"] == pytest.approx(1.0)
    assert res["false_alert_events"] == 1
    assert res["false_alerts_per_24_sensor_hours"] == pytest.approx(24.0)


# ---- warm-up exclusion ---------------------------------------------------------------------------

def test_warmup_flags_do_not_count_for_either_method():
    df = frame([130] * 3 + [120] * 20)
    assert len(collect_flags(df, fixed())) == 3  # the fixed detector does raise raw flags in warm-up
    for det in (fixed(), RollingMADDetector()):
        res = evaluate_method(df, faults(("W", 0, 15)), det, whole(df))
        assert res["alert_events"] == 0 and res["false_alert_events"] == 0
        assert res["faults_detected"] == 0 and res["faults_missed"] == 1
        assert res["per_fault"][0]["eligible_slots_in_fault"] == 0


def test_alert_crossing_warmup_boundary_keeps_only_eligible_flags():
    df = frame([120] * 10 + [130] * 5 + [120] * 10)  # fixed flags idx 10-14; eligible from idx 12
    res = evaluate_method(df, faults(("F", 50, 75)), fixed(), whole(df))
    (alert,) = res["alerts"]
    assert alert.contributing_utc == [t(60), t(65), t(70)]
    pf = res["per_fault"][0]
    assert pf["alert_start_utc"] == t(60) and pf["first_in_fault_utc"] == t(60)
    assert pf["latency_min"] == 10  # measured from fault onset (minute 50), not from the first scored flag


def test_earlier_readings_initialise_rolling_history_for_a_later_partition():
    df = frame([120] * 20 + [130, 120])  # partition starts at idx 15; history comes from idx 0-14
    p = Partition("late", t(75), t(len(df) * 5))
    for det in (fixed(), RollingMADDetector()):
        res = evaluate_method(df, faults(("F", 100, 105)), det, p)
        assert res["eligible_slots"] == 7  # every slot of the partition is already warmed up
        assert res["faults_detected"] == 1 and res["false_alert_events"] == 0


# ---- grouping on eligible slots -----------------------------------------------------------------

def test_missing_slot_breaks_grouping_for_both_methods():
    df = frame([120] * 12 + [130, 130, None, 130, 130, 120])
    for det in (fixed(), RollingMADDetector()):
        res = evaluate_method(df, faults(), det, whole(df))
        assert [len(a.flags) for a in res["alerts"]] == [2, 2]


# ---- matching: in-fault flag required -------------------------------------------------------------

def test_trailing_slot_only_overlap_is_not_a_match():
    df = frame([120] * 13 + [130] + [120] * 5)  # one flag at minute 65 -> alert span [65, 70)
    f = pd.DataFrame([{"fault_id": "F", "fault_type": "x", "start_utc": t(67), "end_utc_exclusive": t(80)}])
    res = evaluate_method(df, f, fixed(), whole(df))
    (alert,) = res["alerts"]
    a_start, a_end = alert.interval(5)
    assert a_start < t(80) and t(67) < a_end  # the old span rule WOULD have called this a match
    assert detection_latency_min(f.iloc[0], alert) is None
    assert (res["faults_detected"], res["faults_missed"], res["false_alert_events"]) == (0, 1, 1)
    assert res["per_fault"][0]["first_in_fault_utc"] is None


def test_flag_at_exact_fault_end_is_outside_half_open_interval():
    df = frame([120] * 13 + [130] + [120] * 5)  # flag at minute 65; fault [60, 65)
    res = evaluate_method(df, faults(("F", 60, 65)), fixed(), whole(df))
    assert res["faults_detected"] == 0 and res["false_alert_events"] == 1


def test_first_in_fault_timestamp_and_latency_reported_separately_from_alert_start():
    df = frame([120] * 12 + [130] * 4 + [120] * 5)  # flags at minutes 60, 65, 70, 75
    res = evaluate_method(df, faults(("F", 65, 80)), fixed(), whole(df))
    pf = res["per_fault"][0]
    assert pf["alert_start_utc"] == t(60)
    assert pf["first_in_fault_utc"] == t(65)
    assert pf["latency_min"] == 0  # not -5
    assert res["mean_latency_min"] == 0


def test_latency_positive_when_first_in_fault_flag_is_after_onset():
    df = frame([120] * 14 + [130] * 2 + [120] * 4)  # flags at minutes 70, 75
    res = evaluate_method(df, faults(("F", 60, 90)), fixed(), whole(df))
    assert res["per_fault"][0]["latency_min"] == 10


# ---- matching: one-to-one and chronological ----------------------------------------------------------

def test_one_alert_cannot_match_two_faults():
    df = frame([120] * 12 + [130] * 9 + [120] * 3)  # one alert, minutes 60-100
    res = evaluate_method(df, faults(("F1", 60, 65), ("F2", 90, 95)), fixed(), whole(df))
    assert [p["detected"] for p in res["per_fault"]] == [True, False]
    assert res["false_alert_events"] == 0


def test_second_alert_for_same_fault_is_a_false_alert():
    df = frame([120] * 12 + [130, 120, 130] + [120] * 5)  # alerts at minutes 60 and 70
    res = evaluate_method(df, faults(("F", 60, 80)), fixed(), whole(df))
    assert res["alert_events"] == 2
    assert res["faults_detected"] == 1 and res["false_alert_events"] == 1
    assert res["per_fault"][0]["alert_start_utc"] == t(60)  # earliest alert wins


def test_faults_are_matched_in_chronological_order_regardless_of_row_order():
    # A (minutes 60-105) covers both faults; B (minute 115) covers only the later one.
    df = frame([120] * 12 + [130] * 10 + [120] + [130] + [120] * 2)
    late_first = faults(("LATE", 100, 120), ("EARLY", 60, 65))
    res = evaluate_method(df, late_first, fixed(), whole(df))
    assert res["faults_detected"] == 2 and res["false_alert_events"] == 0
    by_id = {p["fault_id"]: p for p in res["per_fault"]}
    assert by_id["EARLY"]["alert_start_utc"] == t(60)
    assert by_id["LATE"]["alert_start_utc"] == t(115)


def test_match_alerts_returns_unused_alerts():
    df = frame([120] * 12 + [130, 120, 130] + [120] * 3)
    _, events = replay(df, fixed(), 5)
    matches, unused = match_alerts(events, faults(("F", 60, 65)))
    assert matches[0][1] is events[0] and unused == [events[1]]


# ---- partition handling ---------------------------------------------------------------------------------

def test_faults_outside_partition_are_ignored_and_straddlers_rejected():
    df = frame([120] * 12 + [130] + [120] * 11)
    p = Partition("later", t(60), t(len(df) * 5))
    res = evaluate_method(df, faults(("OLD", 0, 5), ("IN", 60, 65)), fixed(), p)
    assert res["faults_total"] == 1 and res["faults_detected"] == 1
    with pytest.raises(ValueError, match="straddle"):
        evaluate_method(df, faults(("X", 55, 70)), fixed(), p)


def test_evaluate_partition_equals_evaluate_method():
    df = frame([120] * 12 + [130] * 3 + [120] * 5)
    f = faults(("F", 60, 75))
    p = whole(df)
    a = evaluate_method(df, f, fixed(), p)
    b = evaluate_partition(df, f, collect_flags(df, fixed()), p, "fixed_limit")
    assert {k: v for k, v in a.items() if k != "alerts"} == {k: v for k, v in b.items() if k != "alerts"}


# ---- unmatched breakdown and slot counts (Handoff 4) -------------------------------------------------------------

def test_unmatched_breakdown_duplicate_vs_no_in_fault_flag():
    # alerts at minutes 60 (matches F), 70 (second alert inside F -> duplicate), 120 (outside any fault)
    df = frame([120] * 12 + [130, 120, 130] + [120] * 8 + [130] + [120] * 3)
    res = evaluate_method(df, faults(("F", 60, 80)), fixed(), whole(df))
    assert res["alert_events"] == 3 and res["faults_detected"] == 1
    kinds = sorted((u["kind"], u["fault_id"] or "") for u in res["unmatched"])
    assert kinds == [("in_fault_duplicate", "F"), ("no_in_fault_flag", "")]
    assert res["unmatched_in_fault_duplicates"] == 1 and res["unmatched_no_in_fault_flag"] == 1
    assert res["false_alert_events"] == res["unmatched_in_fault_duplicates"] + res["unmatched_no_in_fault_flag"]


def test_slot_counts_scheduled_valid_eligible():
    df = frame([120] * 12 + [None, 120, 120])  # 15 scheduled, 14 valid, 2 eligible (idx 13, 14)
    res = evaluate_method(df, faults(), fixed(), whole(df))
    assert (res["scheduled_slots"], res["valid_slots"], res["eligible_slots"]) == (15, 14, 2)
