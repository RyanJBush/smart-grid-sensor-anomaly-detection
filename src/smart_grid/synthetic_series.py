"""Seeded SYNTHETIC voltage series for calibration / tuning / held-out partitions.

Everything here is invented for software testing. The 120 V nominal, noise levels, event sizes and
schedule are arbitrary choices, not measurements and not operational grid thresholds.

Detector inputs (readings) and ground truth (faults, context events) are written to SEPARATE files.
The event schedule below is fixed in code (not drawn from the RNG); only the noise and the
isolated random drop-outs depend on SEED.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd

from .evaluation import Partition
from .fixtures import SYNTHETIC_BANNER, write_labeled_csv

SEED = 20260101
INTERVAL_MIN = 5  # five-minute cadence
SLOTS_PER_DAY = 24 * 60 // INTERVAL_MIN  # 288
N_DAYS = 15
SERIES_START = pd.Timestamp("2026-01-01T00:00:00Z")
SENSOR_ID = "SYNTHETIC-SENSOR-SEEDED-01"
NOMINAL_V = 120.0
DAILY_AMPLITUDE_V = 0.4  # sinusoidal daily component
AR_PHI = 0.6  # AR(1) coefficient of the noise
INNOVATION_SD_V = 0.3
RANDOM_DROPOUT_P = 0.004  # per slot, never inside a fault interval

# Chronological, contiguous, half-open partitions [start, end) in UTC, 5 days each.
PARTITION_DAYS = {"calibration": (0, 5), "tuning": (5, 10), "heldout": (10, 15)}

READINGS_FILE = "synthetic_voltage_readings.csv"
FAULTS_FILE = "synthetic_voltage_faults.csv"
CONTEXT_FILE = "synthetic_voltage_context_events.csv"
MANIFEST_FILE = "fixture_manifest.json"

# (kind, day, "HH:MM", n_slots, magnitude_v). Faults: spike, sustained_shift.
# Non-faults: benign_load_change (ramped step), outage (burst of missing), planned_dropout (1 missing
# slot placed inside a fault). Day index 0 = 2026-01-01.
SCHEDULE = [
    # --- calibration (days 0-4) ---
    ("benign_load_change", 1, "08:00", 18, +4.0),
    ("sustained_shift", 1, "21:00", 36, +7.0),
    ("spike", 2, "03:20", 1, +10.0),
    ("outage", 2, "14:00", 8, None),
    ("sustained_shift", 3, "02:00", 24, -6.5),
    ("benign_load_change", 3, "17:00", 12, -3.5),
    ("spike", 4, "12:05", 1, -9.0),
    # --- tuning (days 5-9) ---
    ("benign_load_change", 5, "09:00", 18, +4.0),
    ("spike", 5, "15:10", 1, +10.0),
    ("spike", 6, "04:45", 2, -9.0),
    ("sustained_shift", 6, "11:00", 36, +7.0),
    ("sustained_shift", 7, "01:00", 24, -6.5),
    ("planned_dropout", 7, "01:50", 1, None),
    ("benign_load_change", 7, "18:00", 12, -3.5),
    ("outage", 8, "10:00", 8, None),
    ("spike", 8, "22:30", 1, +8.0),
    ("benign_load_change", 9, "07:30", 24, +4.5),
    ("sustained_shift", 9, "14:00", 48, +5.5),
    # --- held-out (days 10-14): generated now, NOT used by tuning ---
    ("benign_load_change", 10, "08:30", 18, +4.2),
    ("spike", 10, "20:15", 1, +9.5),
    ("spike", 11, "05:40", 2, -8.5),
    ("sustained_shift", 11, "12:00", 36, +6.8),
    ("sustained_shift", 12, "03:00", 24, -6.2),
    ("planned_dropout", 12, "03:50", 1, None),
    ("benign_load_change", 12, "19:00", 12, -3.8),
    ("outage", 13, "08:00", 8, None),
    ("spike", 13, "13:25", 1, +10.5),
    ("benign_load_change", 14, "06:00", 24, +3.9),
    ("sustained_shift", 14, "15:00", 48, +5.2),
]
FAULT_KINDS = ("spike", "sustained_shift")


def partitions() -> dict[str, Partition]:
    day = pd.Timedelta(days=1)
    return {name: Partition(name, SERIES_START + a * day, SERIES_START + b * day)
            for name, (a, b) in PARTITION_DAYS.items()}


def partition_of(ts: pd.Timestamp) -> str:
    for p in partitions().values():
        if p.start <= ts < p.end:
            return p.name
    raise ValueError(f"{ts} outside all partitions")


def _slot(day: int, hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return day * SLOTS_PER_DAY + (h * 60 + m) // INTERVAL_MIN


def _ts(slot: int) -> pd.Timestamp:
    return SERIES_START + pd.Timedelta(minutes=slot * INTERVAL_MIN)


def build_synthetic(seed: int = SEED):
    """Return (readings, faults, context_events) DataFrames. Deterministic for a given seed."""
    n = N_DAYS * SLOTS_PER_DAY
    offset = [0.0] * n
    forced_missing = [False] * n
    in_fault = [False] * n
    faults, context = [], []

    for kind, day, hhmm, length, mag in SCHEDULE:
        s = _slot(day, hhmm)
        e = s + length
        if kind in FAULT_KINDS:
            for i in range(s, e):
                offset[i] += mag
                in_fault[i] = True
            faults.append((kind, s, e, mag))
        elif kind == "benign_load_change":
            for k, i in enumerate(range(s, e)):
                offset[i] += mag * (0.5 if length >= 3 and k in (0, length - 1) else 1.0)
            context.append((kind, s, e, mag))
        else:  # outage / planned_dropout
            for i in range(s, e):
                forced_missing[i] = True
            context.append((kind if kind == "outage" else "dropout_inside_fault", s, e, None))

    rng = random.Random(seed)
    sd0 = INNOVATION_SD_V / math.sqrt(1 - AR_PHI ** 2)
    err = rng.gauss(0, sd0)
    volts, missing = [], []
    for i in range(n):
        if i > 0:
            err = AR_PHI * err + rng.gauss(0, INNOVATION_SD_V)
        u = rng.random()  # drawn every slot so the stream does not depend on the schedule
        base = (NOMINAL_V + DAILY_AMPLITUDE_V * math.sin(2 * math.pi * (i % SLOTS_PER_DAY) / SLOTS_PER_DAY)
                + err + offset[i])
        random_drop = u < RANDOM_DROPOUT_P and not in_fault[i] and not forced_missing[i]
        volts.append(round(base, 2))
        missing.append(forced_missing[i] or random_drop)
        if random_drop:
            context.append(("random_dropout", i, i + 1, None))

    readings = pd.DataFrame({
        "timestamp_utc": [_ts(i) for i in range(n)],
        "sensor_id": SENSOR_ID,
        "voltage_v": [math.nan if m else v for v, m in zip(volts, missing)],
    })
    faults.sort(key=lambda r: r[1])
    faults_df = pd.DataFrame([
        {"fault_id": f"SF{k + 1:02d}", "fault_type": kind, "start_utc": _ts(s), "end_utc_exclusive": _ts(e),
         "magnitude_v": mag, "partition": partition_of(_ts(s))}
        for k, (kind, s, e, mag) in enumerate(faults)])
    context.sort(key=lambda r: r[1])
    context_df = pd.DataFrame([
        {"event_id": f"SC{k + 1:02d}", "event_type": kind, "start_utc": _ts(s), "end_utc_exclusive": _ts(e),
         "magnitude_v": mag, "partition": partition_of(_ts(s))}
        for k, (kind, s, e, mag) in enumerate(context)])
    return readings, faults_df, context_df


def manifest() -> dict:
    return {
        "label": SYNTHETIC_BANNER,
        "seed": SEED,
        "rng": "python random.Random(seed): one gauss() then one random() per slot, in slot order",
        "interval_min": INTERVAL_MIN,
        "series_start_utc": SERIES_START.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_slots": N_DAYS * SLOTS_PER_DAY,
        "sensor_id": SENSOR_ID,
        "partitions_half_open_utc": {
            p.name: {"start_utc": p.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "end_utc_exclusive": p.end.strftime("%Y-%m-%dT%H:%M:%SZ")}
            for p in partitions().values()},
        "model": {
            "nominal_v": NOMINAL_V, "daily_sine_amplitude_v": DAILY_AMPLITUDE_V, "ar1_phi": AR_PHI,
            "innovation_sd_v": INNOVATION_SD_V, "random_dropout_p": RANDOM_DROPOUT_P, "rounding_decimals": 2,
        },
        "schedule_is_fixed_in_code": True,
        "detector_input_columns": ["timestamp_utc", "sensor_id", "voltage_v"],
        "label_files": [FAULTS_FILE, CONTEXT_FILE],
    }


def write_synthetic_fixture(data_dir: Path, config_dir: Path) -> None:
    readings, faults, context = build_synthetic()
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    write_labeled_csv(readings, data_dir / READINGS_FILE)
    write_labeled_csv(faults, data_dir / FAULTS_FILE)
    write_labeled_csv(context, data_dir / CONTEXT_FILE)
    with open(config_dir / MANIFEST_FILE, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest(), fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_context(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    for col in ("start_utc", "end_utc_exclusive"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df
