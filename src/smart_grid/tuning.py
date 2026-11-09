"""Calibration + tuning workflow (SYNTHETIC data only). Uses ONLY data before the held-out start.

Roles
  calibration partition: sets the fixed-limit centre (median of its valid readings, rounded to 0.1 V);
                         candidate metrics on it are reported for transparency but NOT used to select.
  tuning partition:      the only data used to choose settings, via the selection rule below.
  held-out partition:    never read here. tune() slices its inputs to timestamps < tuning.end first.

Both methods are scored with the Handoff 3 contract in smart_grid.evaluation (same eligible slots).
The rolling window is frozen at the spec value 12 so eligible slots are identical across candidates.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from .detectors import FixedLimitDetector, RollingMADDetector
from .evaluation import WINDOW, Partition, collect_flags, evaluate_partition, validate_faults_within_partitions

HALF_WIDTHS_V = [2.0 + 0.5 * i for i in range(21)]  # fixed band = centre +/- h, h in 2.0 .. 12.0 V
ROLLING_K = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0]
ROLLING_FLOOR_V = [0.25, 0.5, 0.75, 1.0]
MAD_SCALE = 1.4826
FALSE_ALERT_BUDGET_PER_24H = 1.0  # provisional operating budget, an assumption, not a grid requirement

SELECTION_RULE = (
    "Applied identically to each method on the TUNING partition only, using the Handoff 3 scoring contract. "
    "1) Feasible = candidates with false alert events per 24 eligible sensor-hours <= "
    f"{FALSE_ALERT_BUDGET_PER_24H}. "
    "2) Among feasible, maximise faults detected. "
    "3) Ties: minimise mean detection latency (first in-fault flag minus onset) over detected faults. "
    "4) Ties: minimise false alert events. "
    "5) Ties: prefer the most conservative setting (widest fixed band; for rolling, larger k then larger floor). "
    "If no candidate is feasible: rank by false alerts per 24 h ascending, then apply steps 2-5."
)


def _summary(res: dict, prefix: str) -> dict:
    return {
        f"{prefix}_eligible_slots": res["eligible_slots"],
        f"{prefix}_faults_total": res["faults_total"],
        f"{prefix}_detected": res["faults_detected"],
        f"{prefix}_missed": res["faults_missed"],
        f"{prefix}_alert_events": res["alert_events"],
        f"{prefix}_false_alerts": res["false_alert_events"],
        f"{prefix}_fa_per_24h": res["false_alerts_per_24_sensor_hours"],
        f"{prefix}_mean_latency_min": res["mean_latency_min"],
    }


def _rank_key(row: dict):
    """Steps 2-5 of the selection rule (lower tuple wins)."""
    lat = row["tuning_mean_latency_min"]
    return (-row["tuning_detected"], math.inf if lat is None else lat,
            row["tuning_false_alerts"], row["_conservative"])


def select(rows: list[dict], budget: float = FALSE_ALERT_BUDGET_PER_24H) -> tuple[dict, int]:
    """Pick one row by SELECTION_RULE. Returns (row, n_feasible)."""
    def fa(r):
        return math.inf if r["tuning_fa_per_24h"] is None else r["tuning_fa_per_24h"]

    feasible = [r for r in rows if fa(r) <= budget]
    if feasible:
        return min(feasible, key=_rank_key), len(feasible)
    return min(rows, key=lambda r: (fa(r),) + _rank_key(r)), 0


def tune(readings: pd.DataFrame, faults: pd.DataFrame, parts: dict[str, Partition]):
    """Return (config_body, candidate_rows). Reads nothing at or after parts['tuning'].end."""
    cal, tun = parts["calibration"], parts["tuning"]
    # Reject boundary-crossing faults BEFORE the label filter below could silently drop them.
    validate_faults_within_partitions(faults, parts)
    cut =readings[readings["timestamp_utc"] < tun.end].sort_values("timestamp_utc").reset_index(drop=True)
    earlier_faults = faults[faults["end_utc_exclusive"] <= tun.end]

    cal_vals = cut.loc[(cut["timestamp_utc"] >= cal.start) & (cut["timestamp_utc"] < cal.end), "voltage_v"].dropna()
    center = round(float(cal_vals.median()), 1)

    def run(detector, method):
        flags = collect_flags(cut, detector)
        return (evaluate_partition(cut, earlier_faults, flags, cal, method),
                evaluate_partition(cut, earlier_faults, flags, tun, method))

    fixed_rows, rolling_rows = [], []
    for h in HALF_WIDTHS_V:
        lo, hi = center - h, center + h
        c, t = run(FixedLimitDetector(lo, hi), "fixed_limit")
        fixed_rows.append({"method": "fixed_limit", "half_width_v": h, "lower_v": lo, "upper_v": hi,
                           "k": None, "floor_v": None, "_conservative": (-h,),
                           **_summary(c, "calibration"), **_summary(t, "tuning")})
    for k in ROLLING_K:
        for floor in ROLLING_FLOOR_V:
            c, t = run(RollingMADDetector(window=WINDOW, k=k, floor_v=floor, mad_scale=MAD_SCALE),
                       "rolling_median_mad")
            rolling_rows.append({"method": "rolling_median_mad", "half_width_v": None, "lower_v": None,
                                 "upper_v": None, "k": k, "floor_v": floor, "_conservative": (-k, -floor),
                                 **_summary(c, "calibration"), **_summary(t, "tuning")})

    best_fixed, n_fixed = select(fixed_rows)
    best_roll, n_roll = select(rolling_rows)

    def pick(row, keys):
        return {k: row[k] for k in keys}

    result_keys = [k for k in best_fixed if k.startswith(("calibration_", "tuning_"))]
    body = {
        "status": "FROZEN before held-out evaluation",
        "heldout_evaluated": False,
        "label": "SYNTHETIC - arbitrary test settings; not operational grid thresholds",
        "partitions_half_open_utc": {
            p.name: {"start_utc": p.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "end_utc_exclusive": p.end.strftime("%Y-%m-%dT%H:%M:%SZ")} for p in parts.values()},
        "selection_data_end_exclusive_utc": tun.end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scoring_contract": {
            "eligible_slot": f"valid reading with >= {WINDOW} valid readings before it (same for both methods)",
            "grouping": "consecutive flagged eligible slots; missing/ineligible/unflagged slot breaks a group",
            "denominator": "eligible slots in the partition x interval_min / 60 sensor-hours",
            "matching": "chronological one-to-one; requires >=1 contributing flag in [fault start, fault end)",
            "latency": "first in-fault flag timestamp minus fault onset (alert start reported separately)",
            "false_alerts": "alert events not matched to any fault",
            "rolling_history": "earlier readings, including earlier partitions, may initialise history",
        },
        "selection_rule": SELECTION_RULE,
        "false_alert_budget_per_24_sensor_hours": FALSE_ALERT_BUDGET_PER_24H,
        "grids": {"fixed_half_width_v": HALF_WIDTHS_V, "rolling_k": ROLLING_K, "rolling_floor_v": ROLLING_FLOOR_V,
                  "rolling_window_frozen": WINDOW, "mad_scale": MAD_SCALE},
        "fixed_limit": {
            "center_v": center,
            "center_source": "median of valid calibration-partition readings, rounded to 0.1 V",
            "half_width_v": best_fixed["half_width_v"],
            "lower_v": best_fixed["lower_v"],
            "upper_v": best_fixed["upper_v"],
            "n_candidates": len(fixed_rows), "n_feasible": n_fixed,
            "selection_partition_results": pick(best_fixed, result_keys),
        },
        "rolling_median_mad": {
            "window": WINDOW, "k": best_roll["k"], "floor_v": best_roll["floor_v"], "mad_scale": MAD_SCALE,
            "n_candidates": len(rolling_rows), "n_feasible": n_roll,
            "selection_partition_results": pick(best_roll, result_keys),
        },
    }
    selected = {id(best_fixed), id(best_roll)}
    rows = []
    for r in fixed_rows + rolling_rows:
        out = {k: v for k, v in r.items() if k != "_conservative"}
        out["selected"] = id(r) in selected
        rows.append(out)
    return body, rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen_config(body: dict, data_files: dict[str, Path], path: Path) -> dict:
    config = dict(body)
    config["data_sha256"] = {name: file_sha256(p) for name, p in data_files.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return config
