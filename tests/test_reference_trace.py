"""Reference-trace tests. Expected values come from the Handoff 1 spec, not from observed output."""
import math
from pathlib import Path

import pandas as pd
import pytest

from smart_grid import fixtures
from smart_grid.detectors import FixedLimitDetector
from smart_grid.replay import replay
from smart_grid.scoring import match_alerts, score

ROOT = Path(__file__).resolve().parents[1]
START = fixtures.START_UTC


def t(minute):
    return START + pd.Timedelta(minutes=minute)


@pytest.fixture(scope="module")
def trace():
    readings = fixtures.load_readings(ROOT / "data" / fixtures.READINGS_FILE)
    faults = fixtures.load_faults(ROOT / "data" / fixtures.FAULTS_FILE)
    flags, alerts = replay(readings, FixedLimitDetector(114, 126), 5)
    return readings, faults, flags, alerts


def test_committed_csvs_match_generator():
    pd.testing.assert_frame_equal(
        fixtures.load_readings(ROOT / "data" / fixtures.READINGS_FILE),
        fixtures.build_readings(), check_dtype=False)
    pd.testing.assert_frame_equal(
        fixtures.load_faults(ROOT / "data" / fixtures.FAULTS_FILE),
        fixtures.build_faults(), check_dtype=False)


def test_data_files_labeled_synthetic():
    for name in (fixtures.READINGS_FILE, fixtures.FAULTS_FILE):
        assert "SYNTHETIC" in (ROOT / "data" / name).read_text().splitlines()[0]
    assert fixtures.SENSOR_ID.startswith("SYNTHETIC")


def test_flagged_readings(trace):
    _, _, flags, _ = trace
    assert [f.timestamp for f in flags] == [t(20), t(40), t(45), t(50)]


def test_alert_grouping(trace):
    _, _, _, alerts = trace
    assert len(alerts) == 2
    spike, shift = alerts
    assert (spike.first_flagged_utc, spike.last_flagged_utc) == (t(20), t(20))
    assert spike.contributing_utc == [t(20)]
    assert (spike.method, spike.triggering_voltage_v, spike.breached_threshold_v) == ("fixed_limit", 130.0, 126.0)
    assert "above upper limit" in spike.reason
    assert (shift.first_flagged_utc, shift.last_flagged_utc) == (t(40), t(50))
    assert shift.contributing_utc == [t(40), t(45), t(50)]
    assert (shift.triggering_voltage_v, shift.breached_threshold_v) == (128.0, 126.0)


def test_matching_and_metrics(trace):
    readings, faults, _, alerts = trace
    m = score(alerts, faults, readings, 5)
    assert m["faults_detected"] == 2 and m["faults_total"] == 2
    assert m["faults_missed"] == 0
    assert m["false_alert_events"] == 0
    assert m["latency_min"] == {"F1": 0, "F2": 0}
    assert (m["valid_samples"], m["total_samples"]) == (11, 12)
    assert m["monitored_sensor_hours"] == pytest.approx(11 * 5 / 60)
    assert m["false_alerts_per_24_sensor_hours"] == 0


def test_benign_load_change_not_flagged(trace):
    _, _, flags, _ = trace
    assert not any(t(10) <= f.timestamp <= t(15) for f in flags)


def test_limits_are_inclusive():
    d = FixedLimitDetector(114, 126)
    assert d.update(t(0), 126.0) is None
    assert d.update(t(0), 114.0) is None
    assert d.update(t(0), 126.1) is not None
    assert d.update(t(0), 113.9) is not None
    assert d.update(t(0), float("nan")) is None


def test_missing_reading_breaks_sequence():
    df = pd.DataFrame({"timestamp_utc": [t(0), t(5), t(10), t(15)],
                       "voltage_v": [130.0, math.nan, 130.0, 130.0]})
    _, alerts = replay(df, FixedLimitDetector(114, 126), 5)
    assert [(a.first_flagged_utc, a.last_flagged_utc) for a in alerts] == [(t(0), t(0)), (t(10), t(15))]


def test_matching_is_one_to_one_and_counts_false_alerts():
    # Two alerts both overlap one fault; only one may match, the other is a false alert.
    df = pd.DataFrame({"timestamp_utc": [t(0), t(5), t(10), t(15), t(20)],
                       "voltage_v": [130.0, math.nan, 130.0, 120.0, 130.0]})
    _, alerts = replay(df, FixedLimitDetector(114, 126), 5)
    faults = pd.DataFrame([{"fault_id": "X", "fault_type": "t", "start_utc": t(0), "end_utc_exclusive": t(15)}])
    matches, unmatched = match_alerts(alerts, faults, 5)
    assert matches[0][1].first_flagged_utc == t(0)  # earliest overlapping alert wins
    assert len(unmatched) == 2  # t(10) alert and t(20) alert
    assert score(alerts, faults, df, 5)["false_alert_events"] == 2


def test_no_lookahead(trace):
    """Flags/alerts for a prefix must equal the full run's results up to that point."""
    readings, _, flags, _ = trace
    for k in range(1, len(readings) + 1):
        prefix_flags, _ = replay(readings.iloc[:k], FixedLimitDetector(114, 126), 5)
        cutoff = readings["timestamp_utc"].iloc[k - 1]
        assert [f.timestamp for f in prefix_flags] == [f.timestamp for f in flags if f.timestamp <= cutoff]
