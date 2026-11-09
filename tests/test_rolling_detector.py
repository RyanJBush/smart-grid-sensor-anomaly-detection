"""Rolling median/MAD tests. Expected values come from the Handoff 2 spec (hand-checkable)."""
import math
import random
from pathlib import Path

import pandas as pd
import pytest

from smart_grid import fixtures
from smart_grid.detectors import FixedLimitDetector, RollingMADDetector
from smart_grid.export import alerts_to_frame, scores_to_frame, write_csv
from smart_grid.replay import replay

ROOT = Path(__file__).resolve().parents[1]
START = fixtures.START_UTC
NAN = math.nan


def t(minute):
    return START + pd.Timedelta(minutes=minute)


def series(values, step=5):
    return pd.DataFrame({"timestamp_utc": [t(i * step) for i in range(len(values))],
                         "voltage_v": [NAN if v is None else float(v) for v in values]})


def run(values, **kw):
    det = RollingMADDetector(**kw)
    flags, events = replay(series(values), det, 5)
    return det, flags, events


@pytest.fixture(scope="module")
def ref():
    readings = fixtures.load_readings(ROOT / "data" / fixtures.ROLLING_READINGS_FILE)
    det = RollingMADDetector(window=12, k=4.0, floor_v=0.5)
    flags, events = replay(readings, det, 5)
    return readings, det, flags, events


# ---- 16-slot reference trace -------------------------------------------------------------

def test_rolling_fixture_matches_generator_and_is_synthetic():
    path = ROOT / "data" / fixtures.ROLLING_READINGS_FILE
    pd.testing.assert_frame_equal(fixtures.load_readings(path), fixtures.build_rolling_readings(),
                                  check_dtype=False)
    assert "SYNTHETIC" in path.read_text().splitlines()[0]
    assert fixtures.ROLLING_SENSOR_ID.startswith("SYNTHETIC")
    v = fixtures.load_readings(path)["voltage_v"].tolist()
    assert len(v) == 16 and v[:12] == [120.0] * 12 and v[12:14] == [121.0, 123.0]
    assert math.isnan(v[14]) and v[15] == 120.0


def test_reference_trace_per_reading(ref):
    _, det, _, _ = ref
    s = det.scores
    assert len(s) == 16
    for i in range(12):  # warm-up
        assert (s[i].status, s[i].history_count, s[i].flagged) == ("warmup", i, False)
        assert s[i].lower_v is None and s[i].upper_v is None
    for i, volts, flagged in [(12, 121.0, False), (13, 123.0, True), (15, 120.0, False)]:
        assert s[i].status == "scored" and s[i].voltage_v == volts
        assert s[i].history_count == 12
        assert (s[i].median_v, s[i].mad_v, s[i].spread_v) == (120.0, 0.0, 0.5)
        assert (s[i].lower_v, s[i].upper_v) == (118.0, 122.0)
        assert s[i].flagged is flagged
    assert (s[13].direction, s[12].direction, s[15].direction) == ("high", None, None)
    assert (s[14].status, s[14].voltage_v, s[14].flagged) == ("missing", None, False)
    assert s[14].lower_v is None and s[14].history_count == 12


def test_reference_trace_one_flag_one_event(ref):
    _, _, flags, events = ref
    assert [f.timestamp for f in flags] == [t(13 * 5)]
    assert len(events) == 1
    e = events[0]
    assert e.method == "rolling_median_mad"
    assert (e.first_flagged_utc, e.last_flagged_utc) == (t(65), t(65))
    assert (e.triggering_voltage_v, e.breached_threshold_v) == (123.0, 122.0)
    f = e.flags[0]
    assert (f.direction, f.lower_v, f.upper_v) == ("high", 118.0, 122.0)


# ---- required behaviours -------------------------------------------------------------------

def test_exactly_12_valid_samples_before_first_evaluation():
    det, _, _ = run([120] * 14)
    scored = [i for i, s in enumerate(det.scores) if s.status == "scored"]
    assert scored[0] == 12
    assert all(det.scores[i].history_count == 12 for i in scored)


def test_missing_does_not_count_toward_warmup():
    det, _, _ = run([120] * 6 + [None] * 3 + [120] * 5 + [120])  # 11 valid, 3 missing, then 12th..
    # valid readings: 6 + 5 = 11 before the final reading's slot -> final is the 12th valid => warmup
    assert det.scores[-1].status == "warmup" and det.scores[-1].history_count == 11
    det2, _, _ = run([120] * 6 + [None] * 3 + [120] * 6 + [120])
    assert det2.scores[-1].status == "scored" and det2.scores[-1].history_count == 12


def test_no_alert_during_warmup_even_for_huge_values():
    det, flags, events = run([120] * 5 + [500] + [120] * 5)  # 11 readings, all warm-up
    assert flags == [] and events == []
    assert all(s.status == "warmup" and not s.flagged for s in det.scores)


def test_flat_history_mad_zero_uses_floor():
    det, _, _ = run([120] * 12 + [120])
    s = det.scores[12]
    assert (s.mad_v, s.spread_v) == (0.0, 0.5)
    assert (s.lower_v, s.upper_v) == (118.0, 122.0)  # 120 -/+ 4.0 * 0.5


def test_mad_scaling_when_above_floor():
    hist = [118, 122] * 6  # median 120, every |dev| = 2 -> MAD 2
    det, _, _ = run(hist + [120])
    s = det.scores[12]
    assert (s.median_v, s.mad_v) == (120.0, 2.0)
    assert s.spread_v == pytest.approx(1.4826 * 2)
    assert s.upper_v == pytest.approx(120 + 4.0 * 1.4826 * 2)
    assert s.lower_v == pytest.approx(120 - 4.0 * 1.4826 * 2)


def test_strict_inequality_at_boundary():
    for volts, flagged in [(122.0, False), (118.0, False), (122.0000001, True), (117.9999999, True)]:
        det, flags, _ = run([120] * 12 + [volts])
        assert det.scores[12].flagged is flagged
        assert bool(flags) is flagged
    det, flags, _ = run([120] * 12 + [118.0])
    assert flags == []
    _, flags, _ = run([120] * 12 + [117.0])
    assert flags[0].direction == "low" and flags[0].threshold_v == 118.0


def test_missing_not_in_history():
    det, _, _ = run([120] * 12 + [None, None, 121])
    assert len(det.history) == 12
    assert det.history == tuple([120.0] * 11 + [121.0])
    assert all(not math.isnan(x) for x in det.history)


def test_history_retained_across_gap():
    # 12 valid readings, then a ~3.5-day wall-clock gap, then a reading: still scored with full history.
    df = series([120] * 12)
    later = pd.DataFrame({"timestamp_utc": [t(5000)], "voltage_v": [123.0]})
    det = RollingMADDetector()
    flags, events = replay(pd.concat([df, later], ignore_index=True), det, 5)
    s = det.scores[12]
    assert s.status == "scored" and s.history_count == 12
    assert (s.lower_v, s.upper_v) == (118.0, 122.0)
    assert [f.timestamp for f in flags] == [t(5000)]


def test_gap_interrupts_alert_grouping():
    _, flags, events = run([120] * 12 + [130, None, 130, 130])
    assert len(flags) == 3 and len(events) == 2
    assert [len(e.flags) for e in events] == [1, 2]


def test_flagged_reading_enters_history_only_after_evaluation():
    det, _, _ = run([120] * 12 + [130])
    s = det.scores[12]
    assert s.flagged and (s.lower_v, s.upper_v) == (118.0, 122.0)  # thresholds ignore the 130
    assert det.history == tuple([120.0] * 11 + [130.0])  # 130 appended afterwards
    # Same thresholds regardless of the evaluated value => it was not part of its own window.
    det2, _, _ = run([120] * 12 + [999])
    assert (det2.scores[12].lower_v, det2.scores[12].upper_v) == (118.0, 122.0)


def test_flagged_value_shifts_later_median():
    # 11 x 120 then 130 (flagged), then 130 x5 more: each flagged value must enter history so
    # the median eventually moves; if flagged values were excluded it would stay at 120.
    det, _, _ = run([120] * 12 + [130] * 8)
    medians = [s.median_v for s in det.scores[12:]]
    assert medians[0] == 120.0 and medians[-1] > 120.0


def test_no_lookahead_future_changes_do_not_alter_earlier_results():
    rng = random.Random(12345)
    base = [round(120 + rng.gauss(0, 1.0), 3) for _ in range(40)]
    base[20], base[27] = None, 131.0
    full_det, full_flags, _ = run(base)
    for j in range(1, len(base) + 1):
        mutated = base[:j] + [rng.choice([None, 0.0, 500.0, 120.0]) for _ in base[j:]]
        det, flags, _ = run(mutated)
        assert det.scores[:j] == full_det.scores[:j]  # scores, thresholds, history counts, flags
        cutoff = t((j - 1) * 5)
        assert [f for f in flags if f.timestamp <= cutoff] == [f for f in full_flags if f.timestamp <= cutoff]


def test_parameters_are_recorded_not_tuned():
    d = RollingMADDetector()
    assert (d.window, d.k, d.floor_v, d.mad_scale) == (12, 4.0, 0.5, 1.4826)


# ---- serialization --------------------------------------------------------------------------

def test_serialized_output_distinguishes_methods(ref, tmp_path):
    _, det, _, rolling_events = ref
    baseline = fixtures.load_readings(ROOT / "data" / fixtures.READINGS_FILE)
    _, fixed_events = replay(baseline, FixedLimitDetector(114, 126), 5)
    fixed_df, roll_df = alerts_to_frame(fixed_events), alerts_to_frame(rolling_events)
    assert len(fixed_df) == 2 and set(fixed_df["method"]) == {"fixed_limit"}
    assert set(roll_df["method"]) == {"rolling_median_mad"}

    combined = pd.concat([fixed_df, roll_df], ignore_index=True)
    path = tmp_path / "alerts.csv"
    write_csv(combined, path)
    assert "SYNTHETIC" in path.read_text().splitlines()[0]
    back = pd.read_csv(path, comment="#", dtype=str)
    assert set(back["data_label"]) == {"SYNTHETIC"}
    assert list(back["method"]) == ["fixed_limit", "fixed_limit", "rolling_median_mad"]
    fixed_row = back[back["method"] == "fixed_limit"].iloc[0]
    assert (fixed_row["lower_threshold_v"], fixed_row["upper_threshold_v"]) == ("114.0", "126.0")
    row = back[back["method"] == "rolling_median_mad"].iloc[0]
    assert (row["lower_threshold_v"], row["upper_threshold_v"], row["breached_threshold_v"]) == ("118.0", "122.0", "122.0")
    assert row["triggering_direction"] == "high"
    assert row["contributing_breached_thresholds_v"] == "122.0"


def test_score_trace_export_is_unrounded_and_labeled(ref, tmp_path):
    _, det, _, _ = ref
    df = scores_to_frame(det.scores, det.method)
    assert len(df) == 16 and set(df["method"]) == {"rolling_median_mad"}
    assert df.loc[13, "upper_threshold_v"] == 122.0 and bool(df.loc[13, "flagged"])
    path = tmp_path / "trace.csv"
    write_csv(df, path)
    assert "SYNTHETIC" in path.read_text().splitlines()[0]
