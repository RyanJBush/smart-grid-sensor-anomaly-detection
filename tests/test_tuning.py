"""Tuning workflow and frozen configuration (SYNTHETIC). Held-out data must never influence selection."""
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from smart_grid import synthetic_series as ss
from smart_grid import tuning
from smart_grid.fixtures import load_faults, load_readings

ROOT = Path(__file__).resolve().parents[1]
DATA, CONFIG = ROOT / "data", ROOT / "config"
CANDIDATES = CONFIG / "tuning_candidates.csv"  # committed artifact; tests never need a prior run_tuning.py


@pytest.fixture(scope="module")
def inputs():
    return load_readings(DATA / ss.READINGS_FILE), load_faults(DATA / ss.FAULTS_FILE), ss.partitions()


@pytest.fixture(scope="module")
def tuned(inputs):
    return tuning.tune(*inputs)


def row(detected, lat, false_alerts, fa24, cons, **extra):
    return {"tuning_detected": detected, "tuning_mean_latency_min": lat, "tuning_false_alerts": false_alerts,
            "tuning_fa_per_24h": fa24, "_conservative": cons, "id": extra.get("id")}


# ---- selection rule (pure unit tests) ---------------------------------------------------------------

def test_selection_prefers_feasible_then_most_detections():
    rows = [row(6, 0.0, 0, 5.0, (-1,), id="infeasible"), row(5, 0.0, 0, 0.5, (-2,), id="fewer"),
            row(4, 0.0, 0, 0.0, (-3,), id="fewest")]
    best, n = tuning.select(rows, budget=1.0)
    assert best["id"] == "fewer" and n == 2


def test_selection_tie_breaks_latency_then_false_alerts_then_conservative():
    rows = [row(3, 5.0, 0, 0.0, (-9,), id="slow"), row(3, 0.0, 2, 0.5, (-9,), id="more_fa"),
            row(3, 0.0, 1, 0.5, (-1,), id="narrow"), row(3, 0.0, 1, 0.5, (-2,), id="wide")]
    best, _ = tuning.select(rows, budget=1.0)
    assert best["id"] == "wide"  # same latency and false alerts as 'narrow'; wider (more conservative) wins
    none_detected = [row(0, None, 0, 0.0, (-1,), id="a"), row(0, None, 0, 0.0, (-2,), id="b")]
    assert tuning.select(none_detected)[0]["id"] == "b"  # missing latency ranks last, conservative tie-break


def test_selection_fallback_when_nothing_feasible():
    rows = [row(6, 0.0, 9, 9.0, (-1,), id="worse"), row(2, 0.0, 3, 3.0, (-2,), id="least_bad")]
    best, n = tuning.select(rows, budget=1.0)
    assert n == 0 and best["id"] == "least_bad"


def test_selection_ignores_calibration_columns():
    a = row(3, 0.0, 1, 0.5, (-1,), id="a")
    b = row(3, 0.0, 1, 0.5, (-2,), id="b")
    for r in (a, b):
        r["calibration_detected"] = 0
    a["calibration_detected"] = 99
    assert tuning.select([a, b])[0]["id"] == "b"


# ---- no held-out influence --------------------------------------------------------------------------------

def test_heldout_data_cannot_change_the_frozen_settings(inputs, tuned):
    readings, faults, parts = inputs
    mutated = readings.copy()
    late = mutated["timestamp_utc"] >= parts["heldout"].start
    mutated.loc[late, "voltage_v"] = 500.0
    mutated.loc[mutated["timestamp_utc"] >= parts["heldout"].start + pd.Timedelta(days=2), "voltage_v"] = math.nan
    mutated_faults = faults[faults["end_utc_exclusive"] <= parts["tuning"].end]  # held-out labels removed
    body, rows = tuning.tune(mutated, mutated_faults, parts)
    assert body == tuned[0]
    assert rows == tuned[1]


def test_tune_only_reads_data_before_heldout_start(inputs):
    readings, faults, parts = inputs
    seen = {}
    real = tuning.collect_flags

    def spy(cut, detector):
        seen["max"] = cut["timestamp_utc"].max()
        return real(cut, detector)

    tuning.collect_flags = spy
    try:
        tuning.tune(readings, faults, parts)
    finally:
        tuning.collect_flags = real
    assert seen["max"] < parts["heldout"].start


# ---- frozen configuration -----------------------------------------------------------------------------------

def test_frozen_config_is_reproducible_from_earlier_data(tuned):
    frozen = json.loads((CONFIG / "frozen_config.json").read_text())
    body = {k: v for k, v in frozen.items() if k != "data_sha256"}
    assert body == json.loads(json.dumps(tuned[0], sort_keys=True))


def test_frozen_config_hashes_match_files():
    frozen = json.loads((CONFIG / "frozen_config.json").read_text())
    paths = {ss.READINGS_FILE: DATA, ss.FAULTS_FILE: DATA, ss.CONTEXT_FILE: DATA, ss.MANIFEST_FILE: CONFIG}
    assert set(frozen["data_sha256"]) == set(paths)
    for name, folder in paths.items():
        assert frozen["data_sha256"][name] == tuning.file_sha256(folder / name)


def test_frozen_config_contents():
    c = json.loads((CONFIG / "frozen_config.json").read_text())
    assert c["heldout_evaluated"] is False and c["status"].startswith("FROZEN")
    assert c["selection_data_end_exclusive_utc"] == "2026-01-11T00:00:00Z"
    assert "SYNTHETIC" in c["label"] and "not operational" in c["label"]
    fx, rl = c["fixed_limit"], c["rolling_median_mad"]
    assert fx["half_width_v"] in tuning.HALF_WIDTHS_V
    assert fx["lower_v"] == fx["center_v"] - fx["half_width_v"] and fx["upper_v"] == fx["center_v"] + fx["half_width_v"]
    assert rl["window"] == 12 and rl["k"] in tuning.ROLLING_K and rl["floor_v"] in tuning.ROLLING_FLOOR_V
    assert rl["mad_scale"] == 1.4826
    assert c["false_alert_budget_per_24_sensor_hours"] == tuning.FALSE_ALERT_BUDGET_PER_24H
    assert c["selection_rule"] == tuning.SELECTION_RULE
    for key in ("fixed_limit", "rolling_median_mad"):
        assert c[key]["selection_partition_results"]["tuning_fa_per_24h"] <= c["false_alert_budget_per_24_sensor_hours"]


def test_candidate_table_marks_exactly_the_frozen_choices():
    c = json.loads((CONFIG / "frozen_config.json").read_text())
    df = pd.read_csv(CANDIDATES, comment="#")
    assert len(df) == len(tuning.HALF_WIDTHS_V) + len(tuning.ROLLING_K) * len(tuning.ROLLING_FLOOR_V)
    sel = df[df["selected"]]
    assert sorted(sel["method"]) == ["fixed_limit", "rolling_median_mad"]
    assert sel[sel["method"] == "fixed_limit"].iloc[0]["half_width_v"] == c["fixed_limit"]["half_width_v"]
    r = sel[sel["method"] == "rolling_median_mad"].iloc[0]
    assert (r["k"], r["floor_v"]) == (c["rolling_median_mad"]["k"], c["rolling_median_mad"]["floor_v"])
    assert "SYNTHETIC" in (CANDIDATES).read_text().splitlines()[0]


def test_both_methods_scored_on_identical_eligible_slots(tuned):
    for r in tuned[1]:
        assert r["calibration_eligible_slots"] == tuned[1][0]["calibration_eligible_slots"]
        assert r["tuning_eligible_slots"] == tuned[1][0]["tuning_eligible_slots"]


# ---- boundary validation (Handoff 4 regression) -----------------------------------------------------------

def _with_fault(faults, fid, start, end):
    extra = pd.DataFrame([{"fault_id": fid, "fault_type": "spike", "start_utc": pd.Timestamp(start),
                           "end_utc_exclusive": pd.Timestamp(end), "magnitude_v": 9.0, "partition": "tuning"}])
    return pd.concat([faults, extra], ignore_index=True)


def test_fault_crossing_tuning_cutoff_is_rejected_not_silently_dropped(inputs):
    readings, faults, parts = inputs
    crossing = _with_fault(faults, "XCUT", "2026-01-10T23:50:00Z", "2026-01-11T00:10:00Z")
    # The old label filter (end <= tuning.end) would have dropped this fault without any error.
    with pytest.raises(ValueError, match="XCUT"):
        tuning.tune(readings, crossing, parts)


def test_fault_crossing_calibration_tuning_boundary_is_rejected(inputs):
    readings, faults, parts = inputs
    crossing = _with_fault(faults, "XCAL", "2026-01-05T23:55:00Z", "2026-01-06T00:05:00Z")
    with pytest.raises(ValueError, match="XCAL"):
        tuning.tune(readings, crossing, parts)


def test_boundary_validation_accepts_exact_boundary_faults(inputs):
    from smart_grid.evaluation import validate_faults_within_partitions
    readings, faults, parts = inputs
    ok = _with_fault(faults, "EDGE", "2026-01-10T23:55:00Z", "2026-01-11T00:00:00Z")  # ends exactly at cutoff
    validate_faults_within_partitions(ok, parts)
    validate_faults_within_partitions(faults, parts)


def test_existing_fixture_still_reproduces_candidate_table_and_config(inputs, tuned):
    body, rows = tuned
    df = pd.read_csv(CANDIDATES, comment="#")
    fresh = pd.DataFrame(rows)
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False)
    frozen = json.loads((CONFIG / "frozen_config.json").read_text())
    assert {k: v for k, v in frozen.items() if k != "data_sha256"} == json.loads(json.dumps(body, sort_keys=True))


def test_candidate_table_regenerates_byte_identically_in_a_temp_location(tuned, tmp_path):
    """The committed candidate table is an explicit dependency; regeneration goes to a temp dir only."""
    from smart_grid.export import write_csv
    regen = tmp_path / "tuning_candidates.csv"
    write_csv(pd.DataFrame(tuned[1]), regen)
    assert regen.read_bytes() == CANDIDATES.read_bytes()
