"""Scoring clarifications from Handoff 2: latency definition and per-flag threshold/direction."""
import math

import pandas as pd

from smart_grid import fixtures
from smart_grid.detectors import FixedLimitDetector
from smart_grid.replay import replay
from smart_grid.scoring import detection_latency_min, score

START = fixtures.START_UTC


def t(minute):
    return START + pd.Timedelta(minutes=minute)


def frame(values):
    return pd.DataFrame({"timestamp_utc": [t(i * 5) for i in range(len(values))],
                         "voltage_v": [math.nan if v is None else float(v) for v in values]})


def fault(start, end):
    return pd.DataFrame([{"fault_id": "X", "fault_type": "t", "start_utc": t(start), "end_utc_exclusive": t(end)}])


def test_latency_uses_earliest_flag_inside_fault_not_alert_start():
    # Alert flags at minutes 25, 30, 35; fault is [30, 45). Alert start (25) precedes onset.
    df = frame([120] * 5 + [130, 130, 130, 120, 120])
    _, alerts = replay(df, FixedLimitDetector(114, 126), 5)
    f = fault(30, 45)
    assert alerts[0].first_flagged_utc == t(25)
    assert score(alerts, f, df, 5)["latency_min"] == {"X": 0.0}  # not -5


def test_latency_positive_when_detected_after_onset():
    df = frame([120] * 7 + [130, 130, 120])
    _, alerts = replay(df, FixedLimitDetector(114, 126), 5)
    m = score(alerts, fault(30, 45), df, 5)
    assert m["faults_detected"] == 1 and m["latency_min"] == {"X": 5.0}


def test_latency_none_when_no_flag_inside_off_grid_fault():
    df = frame([130, 120])
    _, alerts = replay(df, FixedLimitDetector(114, 126), 5)
    f = fault(3, 10).iloc[0]  # off-grid onset: alert span [0,5) overlaps, but flag at 0 is outside
    assert detection_latency_min(f, alerts[0]) is None


def test_each_contributing_flag_keeps_own_threshold_and_direction():
    df = frame([120, 130, 110, 110, 120])
    _, events = replay(df, FixedLimitDetector(114, 126), 5)
    assert len(events) == 1
    e = events[0]
    assert [f.direction for f in e.flags] == ["high", "low", "low"]
    assert [f.threshold_v for f in e.flags] == [126.0, 114.0, 114.0]
    # first flag is the primary triggering reading for the summary
    assert (e.triggering_voltage_v, e.breached_threshold_v) == (130.0, 126.0)
    assert e.contributing_utc == [t(5), t(10), t(15)]
