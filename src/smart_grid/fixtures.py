"""Synthetic reference fixtures. All values are invented for testing; none are real grid data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SYNTHETIC_BANNER = "SYNTHETIC DATA - invented for software testing; not real grid measurements"
SENSOR_ID = "SYNTHETIC-SENSOR-01"
START_UTC = pd.Timestamp("2026-01-01T00:00:00Z")  # fixed date for reproducibility
INTERVAL_MIN = 5

# Minute offset -> voltage (V). None = missing reading.
_VOLTAGES = [120, 121, 124, 124, 130, 120, 119, None, 128, 129, 128, 120]

# Ground truth, kept separate from readings. Intervals are [start, end) in minutes.
_FAULTS = [
    ("F1", "spike", 20, 25),
    ("F2", "sustained_level_shift", 40, 55),
]

READINGS_FILE = "reference_readings.csv"
FAULTS_FILE = "reference_faults.csv"


def _ts(minute: int) -> pd.Timestamp:
    return START_UTC + pd.Timedelta(minutes=minute)


# Rolling-detector unit-test fixture (16 slots). NOT tuning or held-out evaluation data.
ROLLING_SENSOR_ID = "SYNTHETIC-ROLLING-UNIT-TEST-01"
ROLLING_READINGS_FILE = "rolling_reference_readings.csv"
_ROLLING_VOLTAGES = [120] * 12 + [121, 123, None, 120]


def _build(voltages, sensor_id: str) -> pd.DataFrame:
    rows = [
        {
            "timestamp_utc": _ts(i * INTERVAL_MIN),
            "sensor_id": sensor_id,
            "voltage_v": float("nan") if v is None else float(v),
        }
        for i, v in enumerate(voltages)
    ]
    return pd.DataFrame(rows)


def build_readings() -> pd.DataFrame:
    return _build(_VOLTAGES, SENSOR_ID)


def build_rolling_readings() -> pd.DataFrame:
    return _build(_ROLLING_VOLTAGES, ROLLING_SENSOR_ID)


def build_faults() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fault_id": fid, "fault_type": ftype, "start_utc": _ts(s), "end_utc_exclusive": _ts(e)}
            for fid, ftype, s, e in _FAULTS
        ]
    )


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {SYNTHETIC_BANNER}\n")
        out.to_csv(fh, index=False, lineterminator="\n")


write_labeled_csv = _write_csv  # public alias: SYNTHETIC banner line + UTC-formatted timestamps


def write_fixtures(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(build_readings(), data_dir / READINGS_FILE)
    _write_csv(build_faults(), data_dir / FAULTS_FILE)
    _write_csv(build_rolling_readings(), data_dir / ROLLING_READINGS_FILE)


def load_readings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def load_faults(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    for col in ("start_utc", "end_utc_exclusive"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df
