"""Frozen held-out replay tests (SYNTHETIC). Observed-result pins are regression checks, not expectations."""
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from smart_grid import heldout
from smart_grid import heldout_report as HR
from smart_grid.synthetic_series import partitions
from smart_grid.tuning import file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "frozen_config.json"
HELD = pd.Timestamp("2026-01-11T00:00:00Z"), pd.Timestamp("2026-01-16T00:00:00Z")


@pytest.fixture(scope="module")
def res():
    return heldout.run(ROOT)


@pytest.fixture()
def sandbox(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copytree(ROOT / "config", tmp_path / "config")
    return tmp_path


# ---- freeze verification ------------------------------------------------------------------------------------

def test_run_aborts_if_any_fixture_file_changed(sandbox):
    path = sandbox / "data" / "synthetic_voltage_readings.csv"
    text = path.read_text()
    path.write_text(text.replace(",120.3\n", ",120.4\n", 1))
    assert path.read_text() != text
    with pytest.raises(heldout.FreezeError, match="hash mismatch"):
        heldout.run(sandbox)


def test_run_aborts_if_config_not_a_preevaluation_freeze(sandbox):
    cfg = sandbox / "config" / "frozen_config.json"
    d = json.loads(cfg.read_text())
    d["heldout_evaluated"] = True
    cfg.write_text(json.dumps(d))
    with pytest.raises(heldout.FreezeError, match="pre-evaluation"):
        heldout.run(sandbox)


def test_run_aborts_if_heldout_boundaries_differ(sandbox):
    cfg = sandbox / "config" / "frozen_config.json"
    d = json.loads(cfg.read_text())
    d["partitions_half_open_utc"]["heldout"]["end_utc_exclusive"] = "2026-01-17T00:00:00Z"
    cfg.write_text(json.dumps(d))
    with pytest.raises(heldout.FreezeError, match="boundaries"):
        heldout.run(sandbox)


def test_unchanged_files_pass_verification(sandbox):
    config, _ = heldout.load_frozen_config(sandbox)
    heldout.verify_freeze(config, sandbox)


# ---- detectors and window -----------------------------------------------------------------------------------

def test_detectors_are_built_from_saved_settings(res):
    cfg = res["config"]
    fx, rl = res["methods"]["fixed_limit"]["detector"], res["methods"]["rolling_median_mad"]["detector"]
    assert (fx.lower_v, fx.upper_v) == (cfg["fixed_limit"]["lower_v"], cfg["fixed_limit"]["upper_v"])
    r = cfg["rolling_median_mad"]
    assert (rl.window, rl.k, rl.floor_v, rl.mad_scale) == (r["window"], r["k"], r["floor_v"], r["mad_scale"])


def test_window_is_jan11_to_jan16_exclusive_and_matches_partition(res):
    assert (res["partition"].start, res["partition"].end) == HELD
    assert (partitions()["heldout"].start, partitions()["heldout"].end) == HELD
    s = res["slots"]
    assert s["timestamp_utc"].min() == HELD[0] and s["timestamp_utc"].max() == HELD[1] - pd.Timedelta(minutes=5)
    assert len(s) == 1440


def test_scoring_only_on_shared_eligible_heldout_slots(res):
    evs = [res["methods"][m]["ev"] for m in heldout.METHODS]
    assert evs[0]["eligible_slots"] == evs[1]["eligible_slots"]
    assert evs[0]["monitored_sensor_hours"] == evs[1]["monitored_sensor_hours"]
    assert evs[0]["monitored_sensor_hours"] == pytest.approx(evs[0]["eligible_slots"] * 5 / 60)
    elig_ts = set(res["slots"].loc[res["slots"]["eligible"], "timestamp_utc"])
    for ev in evs:
        for a in ev["alerts"]:
            assert all(f.timestamp in elig_ts for f in a.flags)
            assert HELD[0] <= a.first_flagged_utc and a.last_flagged_utc < HELD[1]


def test_earlier_readings_warm_up_rolling_history(res):
    first = res["slots"].iloc[0]
    assert bool(first["eligible"]) and pd.notna(first["rolling_median_v"])  # scored from the very first held-out slot
    ev = res["methods"]["rolling_median_mad"]["ev"]
    assert ev["eligible_slots"] == ev["valid_slots"]


def test_config_file_is_not_modified_by_replay(tmp_path):
    before = file_sha256(CONFIG)
    r = heldout.run(ROOT)
    HR.write_outputs(r, tmp_path / "out")
    assert file_sha256(CONFIG) == before
    assert json.loads(CONFIG.read_text())["heldout_evaluated"] is False  # freeze record stays as saved


# ---- exports ------------------------------------------------------------------------------------------------

def test_alert_export_has_required_fields_and_links(res):
    a = heldout.alert_records(res)
    for col in ("triggering_voltage_v", "breached_threshold_v", "triggering_direction", "reason",
                "contributing_flagged_utc", "matched_fault_id", "duplicate_of_fault_id", "match_status", "method"):
        assert col in a.columns
    assert set(a["data_label"]) == {"SYNTHETIC"}
    faults = res["faults"].set_index("fault_id")
    for m in heldout.METHODS:
        ev = res["methods"][m]["ev"]
        sub = a[a["method"] == m]
        assert len(sub) == ev["alert_events"]
        assert (sub["match_status"] == "matched").sum() == ev["faults_detected"]
        assert (sub["match_status"] == "unmatched_in_fault_duplicate").sum() == ev["unmatched_in_fault_duplicates"]
        assert (sub["match_status"] == "unmatched_no_in_fault_flag").sum() == ev["unmatched_no_in_fault_flag"]
    for r in a[a["match_status"] == "matched"].itertuples(index=False):
        f = faults.loc[r.matched_fault_id]
        assert f["start_utc"] <= pd.Timestamp(r.first_in_fault_utc) < f["end_utc_exclusive"]
    assert a.loc[a["match_status"] == "unmatched_in_fault_duplicate", "duplicate_of_fault_id"].notna().all()


def test_metrics_report_required_counts(res):
    met = heldout.metrics(res)
    for m in heldout.METHODS:
        v = met["methods"][m]
        assert v["scheduled_slots"] == 1440 and v["eligible_slots"] <= v["valid_slots"] <= 1440
        assert v["coverage_eligible"] == pytest.approx(v["eligible_slots"] / v["scheduled_slots"])
        assert v["unmatched_alert_events"] == v["unmatched_in_fault_duplicates"] + v["unmatched_no_in_fault_flag"]
        assert v["unmatched_per_24_sensor_hours"] == pytest.approx(
            v["unmatched_alert_events"] / v["monitored_sensor_hours"] * 24)
        assert set(v["per_fault_latency_min"]) == {"SF11", "SF12", "SF13", "SF14", "SF15", "SF16"}


def test_observed_heldout_results_regression_pin(res):
    """Pins the observed 2026-01-11..16 results so later refactors cannot change them silently."""
    got = {}
    for m in heldout.METHODS:
        ev = res["methods"][m]["ev"]
        got[m] = (ev["eligible_slots"], ev["faults_detected"], ev["faults_missed"], ev["alert_events"],
                  ev["false_alert_events"], ev["unmatched_in_fault_duplicates"], ev["unmatched_no_in_fault_flag"])
    assert got == {"fixed_limit": (1423, 6, 0, 11, 5, 1, 4), "rolling_median_mad": (1423, 5, 1, 9, 4, 0, 4)}
    rolling = {p["fault_id"]: p["detected"] for p in res["methods"]["rolling_median_mad"]["ev"]["per_fault"]}
    assert rolling["SF16"] is False


def test_chart_report_and_failure_case_outputs(res, tmp_path):
    out = HR.write_outputs(res, tmp_path / "o")
    chart = tmp_path / "o" / "heldout_chart.png"
    assert chart.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and chart.stat().st_size > 50_000
    for name in ("heldout_alerts.csv", "heldout_per_fault.csv", "heldout_failure_case_evidence.csv"):
        assert "SYNTHETIC" in (tmp_path / "o" / name).read_text().splitlines()[0]
    report = (tmp_path / "o" / "heldout_report.md").read_text()
    assert "SYNTHETIC DATA" in report and "Observed failure case" in report and "scoped to this one seeded" in report
    assert out["case"]["kind"] == "missed_fault" and out["case"]["ref"] == "SF16"
    ev = out["case"]["evidence"]
    assert ev["timestamp_utc"].min() < pd.Timestamp("2026-01-15T15:00:00Z") and ev["rolling_flag"].any()


def test_failure_case_selection_rule_is_text_and_deterministic(res):
    assert HR.pick_failure_case(res) == ("missed_fault", "rolling_median_mad", "SF16")
    assert "Fixed before looking at results" in HR.FAILURE_RULE
