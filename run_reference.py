"""Regenerate SYNTHETIC fixtures; run the fixed-limit baseline and the rolling 16-slot unit trace."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from smart_grid.detectors import FixedLimitDetector, RollingMADDetector  # noqa: E402
from smart_grid.export import FMT, alerts_to_frame, scores_to_frame, write_csv  # noqa: E402
from smart_grid.fixtures import (  # noqa: E402
    FAULTS_FILE, INTERVAL_MIN, READINGS_FILE, ROLLING_READINGS_FILE, SYNTHETIC_BANNER,
    load_faults, load_readings, write_fixtures,
)
from smart_grid.replay import replay  # noqa: E402
from smart_grid.scoring import score  # noqa: E402


def run_fixed(data_dir: Path, out_dir: Path) -> dict:
    readings = load_readings(data_dir / READINGS_FILE)
    faults = load_faults(data_dir / FAULTS_FILE)
    flags, alerts = replay(readings, FixedLimitDetector(114.0, 126.0), INTERVAL_MIN)
    metrics = score(alerts, faults, readings, INTERVAL_MIN)
    frame = alerts_to_frame(alerts)
    write_csv(frame, out_dir / "reference_alerts.csv")

    print("=== FIXED-LIMIT BASELINE (reference trace) ===")
    print("Flagged readings:")
    for f in flags:
        print(f"  {f.timestamp.strftime(FMT)}  {f.voltage_v:g} V  {f.direction}")
    print("Alert events:")
    for rec in frame.to_dict("records"):
        print(" ", rec)
    print("Summary metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def run_rolling(data_dir: Path, out_dir: Path):
    readings = load_readings(data_dir / ROLLING_READINGS_FILE)
    det = RollingMADDetector(window=12, k=4.0, floor_v=0.5)
    flags, alerts = replay(readings, det, INTERVAL_MIN)
    trace = scores_to_frame(det.scores, det.method)
    frame = alerts_to_frame(alerts)
    write_csv(trace, out_dir / "rolling_reference_trace.csv")
    write_csv(frame, out_dir / "rolling_reference_alerts.csv")

    print("\n=== ROLLING MEDIAN/MAD (16-slot unit-test trace; provisional k=4.0, floor 0.5 V) ===")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(trace.drop(columns=["data_label", "method"]).to_string(index=False))
    print(f"Flagged readings: {len(flags)}   Grouped alert events: {len(alerts)}")
    for rec in frame.to_dict("records"):
        print(" ", rec)
    return flags, alerts


def main():
    data_dir, out_dir = ROOT / "data", ROOT / "output"
    write_fixtures(data_dir)
    print(f"*** {SYNTHETIC_BANNER} ***\n")
    run_fixed(data_dir, out_dir)
    run_rolling(data_dir, out_dir)


if __name__ == "__main__":
    main()
