"""Handoff 5: zero-alert reporting path and the shipped-freeze guard (SYNTHETIC)."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from smart_grid import heldout
from smart_grid import heldout_report as HR
from smart_grid.export import ALERT_COLUMNS, alerts_to_frame, write_csv
from smart_grid.tuning import file_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def sandbox(tmp_path):
    for name in ("data", "config"):
        shutil.copytree(ROOT / name, tmp_path / name)
    return tmp_path


def _edit_settings(sandbox, **kw):
    """Change frozen DETECTOR SETTINGS only (data hashes are untouched) to force a chosen alert behaviour."""
    cfg = sandbox / "config" / "frozen_config.json"
    d = json.loads(cfg.read_text())
    for key, val in kw.items():
        d[key].update(val)
    cfg.write_text(json.dumps(d))


NEVER_FIXED = {"fixed_limit": {"lower_v": 0.0, "upper_v": 1000.0}}
NEVER_ROLLING = {"rolling_median_mad": {"k": 1e6}}


# ---- zero-alert reporting ---------------------------------------------------------------------------------

def test_alerts_to_frame_keeps_headers_when_empty():
    df = alerts_to_frame([])
    assert list(df.columns) == ALERT_COLUMNS and len(df) == 0


def test_alert_records_keep_full_header_when_no_method_alerts(sandbox):
    _edit_settings(sandbox, **NEVER_FIXED, **NEVER_ROLLING)
    res = heldout.run(sandbox)
    a = heldout.alert_records(res)
    assert len(a) == 0 and list(a.columns) == heldout.ALERT_RECORD_COLUMNS
    assert all(res["methods"][m]["ev"]["alert_events"] == 0 for m in heldout.METHODS)


def test_zero_alerts_for_both_methods_does_not_crash_and_reports_zero_events(sandbox, tmp_path):
    _edit_settings(sandbox, **NEVER_FIXED, **NEVER_ROLLING)
    res = heldout.run(sandbox)
    out = HR.write_outputs(res, tmp_path / "out")
    for m in heldout.METHODS:
        v = out["metrics"]["methods"][m]
        assert v["alert_events"] == 0 and v["unmatched_alert_events"] == 0
        assert v["unmatched_per_24_sensor_hours"] == 0
        assert v["faults_detected"] == 0 and v["faults_missed"] == v["faults_total"] == 6
        assert v["mean_latency_min"] is None
    csv = (tmp_path / "out" / "heldout_alerts.csv").read_text().splitlines()
    assert csv[0].startswith("# SYNTHETIC")
    assert csv[1].split(",") == heldout.ALERT_RECORD_COLUMNS and len(csv) == 2  # header kept, no data rows
    back = pd.read_csv(tmp_path / "out" / "heldout_alerts.csv", comment="#")
    assert list(back.columns) == heldout.ALERT_RECORD_COLUMNS and back.empty
    assert (tmp_path / "out" / "heldout_chart.png").stat().st_size > 10_000
    assert "Observed failure case" in (tmp_path / "out" / "heldout_report.md").read_text()
    assert out["case"]["kind"] == "missed_fault"


def test_one_method_with_zero_alerts_still_reports(sandbox, tmp_path):
    _edit_settings(sandbox, **NEVER_ROLLING)  # fixed limit alerts normally; rolling never alerts
    res = heldout.run(sandbox)
    out = HR.write_outputs(res, tmp_path / "out")
    m = out["metrics"]["methods"]
    assert m["rolling_median_mad"]["alert_events"] == 0 and m["fixed_limit"]["alert_events"] == 11
    assert set(out["alerts"]["method"]) == {"fixed_limit"}


def test_shipped_settings_still_give_nonzero_alert_records():
    a = heldout.alert_records(heldout.run(ROOT))
    assert list(a.columns) == heldout.ALERT_RECORD_COLUMNS and len(a) == 20


def test_write_csv_of_empty_frame_keeps_header(tmp_path):
    write_csv(pd.DataFrame(columns=["a", "b"]), tmp_path / "x.csv")
    assert (tmp_path / "x.csv").read_text().splitlines()[1] == "a,b"


# ---- freeze guard --------------------------------------------------------------------------------------------

def test_shipped_freeze_verifies_and_matches_pinned_handoff4_hash():
    sha = heldout.verify_shipped_freeze(ROOT)
    assert sha == "2735a40ba5b425682b301e3beec4bcbc4f936af6e1f7cf8e818dc8e74accd4f1"
    assert sha == file_sha256(ROOT / "config" / "frozen_config.json")


def test_verify_shipped_freeze_rejects_config_change_data_change_and_missing_pin(sandbox):
    heldout.verify_shipped_freeze(sandbox)
    cfg = sandbox / "config" / "frozen_config.json"
    original = cfg.read_bytes()  # bytes: text mode would rewrite line endings on Windows
    cfg.write_bytes(original.replace(b'"k": 10.0', b'"k": 9.0'))
    with pytest.raises(heldout.FreezeError, match="pinned"):
        heldout.verify_shipped_freeze(sandbox)
    cfg.write_bytes(original)
    data = sandbox / "data" / "synthetic_voltage_faults.csv"
    data.write_text(data.read_text().replace("SF16", "SF99"))
    with pytest.raises(heldout.FreezeError, match="hash mismatch"):
        heldout.verify_shipped_freeze(sandbox)
    (sandbox / "config" / heldout.SIDECAR_FILE).unlink()
    with pytest.raises(heldout.FreezeError, match="missing"):
        heldout.verify_shipped_freeze(sandbox)


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_tuning_refuses_to_overwrite_a_valid_shipped_freeze(sandbox):
    rt = _load_script("run_tuning")
    rt.ROOT = sandbox
    before = {p.name: file_sha256(p) for p in (sandbox / "config").iterdir()}
    assert rt.main([]) == 3
    assert {p.name: file_sha256(p) for p in (sandbox / "config").iterdir()} == before


def test_run_tuning_aborts_and_touches_nothing_if_shipped_freeze_is_invalid(sandbox):
    rt = _load_script("run_tuning")
    rt.ROOT = sandbox
    data = sandbox / "data" / "synthetic_voltage_readings.csv"
    data.write_text(data.read_text().replace(",120.3\n", ",120.4\n", 1))
    snapshot = {p.name: file_sha256(p) for p in (sandbox / "data").iterdir()}
    assert rt.main([]) == 2
    assert {p.name: file_sha256(p) for p in (sandbox / "data").iterdir()} == snapshot


def test_run_tuning_out_dir_regenerates_elsewhere_identically(tmp_path):
    cfg_before = file_sha256(ROOT / "config" / "frozen_config.json")
    proc = subprocess.run([sys.executable, str(ROOT / "run_tuning.py"), "--out-dir", str(tmp_path / "regen")],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DIFFERS" not in proc.stdout and proc.stdout.count("IDENTICAL") == 6
    assert file_sha256(ROOT / "config" / "frozen_config.json") == cfg_before  # shipped tree untouched
    assert file_sha256(tmp_path / "regen" / "config" / "frozen_config.json") == cfg_before


def test_verify_freeze_script_exit_codes():
    ok = subprocess.run([sys.executable, str(ROOT / "verify_freeze.py")], capture_output=True, text=True)
    assert ok.returncode == 0 and "FREEZE OK" in ok.stdout
