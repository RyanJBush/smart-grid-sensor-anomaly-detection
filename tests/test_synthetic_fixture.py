"""Seeded SYNTHETIC fixture: determinism, cadence, partitions, event coverage, label separation."""
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from smart_grid import synthetic_series as ss
from smart_grid.evaluation import eligible_slots
from smart_grid.fixtures import load_faults, load_readings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def built():
    return ss.build_synthetic()


def test_seed_is_documented_and_generation_is_deterministic(built):
    assert ss.SEED == 20260101
    again = ss.build_synthetic()
    for a, b in zip(built, again):
        pd.testing.assert_frame_equal(a, b)
    other = ss.build_synthetic(seed=ss.SEED + 1)[0]
    assert not other["voltage_v"].equals(built[0]["voltage_v"])  # the seed really drives the noise
    manifest = json.loads((ROOT / "config" / ss.MANIFEST_FILE).read_text())
    assert manifest["seed"] == ss.SEED and manifest["interval_min"] == 5


def test_five_minute_cadence_and_length(built):
    r = built[0]
    assert len(r) == ss.N_DAYS * 288 == 4320
    assert (r["timestamp_utc"].diff().dropna() == pd.Timedelta(minutes=5)).all()
    assert r["timestamp_utc"].iloc[0] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert str(r["timestamp_utc"].dt.tz) == "UTC"


def test_committed_csvs_match_generator(built):
    readings = load_readings(DATA / ss.READINGS_FILE)
    pd.testing.assert_frame_equal(readings, built[0], check_dtype=False)
    faults = load_faults(DATA / ss.FAULTS_FILE)
    pd.testing.assert_frame_equal(faults, built[1], check_dtype=False)
    for name in (ss.READINGS_FILE, ss.FAULTS_FILE, ss.CONTEXT_FILE):
        assert "SYNTHETIC" in (DATA / name).read_text().splitlines()[0]
    assert ss.SENSOR_ID.startswith("SYNTHETIC")


def test_detector_inputs_carry_no_labels():
    cols = list(load_readings(DATA / ss.READINGS_FILE).columns)
    assert cols == ["timestamp_utc", "sensor_id", "voltage_v"]
    assert ss.FAULTS_FILE != ss.READINGS_FILE and ss.CONTEXT_FILE != ss.READINGS_FILE


def test_partitions_are_contiguous_half_open_utc_five_days():
    p = ss.partitions()
    assert list(p) == ["calibration", "tuning", "heldout"]
    assert p["calibration"].start == ss.SERIES_START
    assert p["calibration"].end == p["tuning"].start and p["tuning"].end == p["heldout"].start
    assert p["heldout"].end == ss.SERIES_START + pd.Timedelta(days=15)
    for part in p.values():
        assert part.end - part.start == pd.Timedelta(days=5)
    assert p["tuning"].start == pd.Timestamp("2026-01-06T00:00:00Z")
    assert p["heldout"].start == pd.Timestamp("2026-01-11T00:00:00Z")


def test_every_partition_has_all_required_behaviours(built):
    _, faults, context = built
    for name in ss.PARTITION_DAYS:
        types = set(faults.loc[faults["partition"] == name, "fault_type"])
        assert types == {"spike", "sustained_shift"}, name
        kinds = set(context.loc[context["partition"] == name, "event_type"])
        assert {"benign_load_change", "outage"} <= kinds, name


def test_faults_and_events_do_not_straddle_or_overlap(built):
    _, faults, context = built
    parts = ss.partitions()
    for df, end_col in ((faults, "end_utc_exclusive"), (context, "end_utc_exclusive")):
        for row in df.itertuples(index=False):
            p = parts[row.partition]
            assert p.start <= row.start_utc and row.end_utc_exclusive <= p.end
    intervals = sorted(
        [(r.start_utc, r.end_utc_exclusive) for r in faults.itertuples(index=False)]
        + [(r.start_utc, r.end_utc_exclusive) for r in context.itertuples(index=False)
           if r.event_type in ("benign_load_change", "outage")])
    for (_, e1), (s2, _) in zip(intervals, intervals[1:]):
        assert e1 <= s2  # no fault / load change / outage overlaps another


def test_missing_data_cases_are_present_and_never_random_inside_faults(built):
    readings, faults, context = built
    miss = readings.set_index("timestamp_utc")["voltage_v"].isna()
    for row in context.itertuples(index=False):
        if row.event_type in ("outage", "dropout_inside_fault", "random_dropout"):
            assert miss[row.start_utc:row.end_utc_exclusive - pd.Timedelta(minutes=5)].all()
    outage = context[context["event_type"] == "outage"].iloc[0]
    assert (outage.end_utc_exclusive - outage.start_utc) == pd.Timedelta(minutes=40)
    for f in faults.itertuples(index=False):
        inside_rand = context[(context["event_type"] == "random_dropout")
                              & (context["start_utc"] >= f.start_utc) & (context["start_utc"] < f.end_utc_exclusive)]
        assert inside_rand.empty
    assert (context["event_type"] == "dropout_inside_fault").sum() == 2  # tuning + held-out


def test_every_fault_has_eligible_warmed_up_slots(built):
    readings, faults, _ = built
    elig = eligible_slots(readings)
    ts = readings["timestamp_utc"]
    for f in faults.itertuples(index=False):
        mask = (ts >= f.start_utc) & (ts < f.end_utc_exclusive)
        assert elig[mask].sum() >= 1, f.fault_id


def test_values_are_plausible_synthetic_and_rounded(built):
    v = built[0]["voltage_v"].dropna()
    assert v.between(100, 140).all()
    assert ((v * 100).round(6) % 1 == 0).all()  # two decimals
    quiet = built[0][(built[0]["timestamp_utc"] < pd.Timestamp("2026-01-02T08:00:00Z"))]["voltage_v"].dropna()
    assert quiet.sub(120).abs().max() < 3  # normal variation stays small before the first scheduled event
    assert not math.isnan(v.mean())
