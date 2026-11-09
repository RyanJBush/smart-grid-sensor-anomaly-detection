"""Frozen held-out replay (SYNTHETIC data only).

Loads config/frozen_config.json READ-ONLY, verifies the recorded fixture hashes, builds both detectors
from the saved settings, replays the whole series chronologically (earlier readings initialise
rolling history) and scores ONLY the shared eligible held-out slots with the frozen scoring contract.
Nothing here tunes or re-selects anything, and nothing is written back to the config.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import synthetic_series as ss
from .detectors import FixedLimitDetector, RollingMADDetector
from .evaluation import (
    collect_flags, eligible_slots, evaluate_partition, validate_faults_within_partitions,
)
from .export import ALERT_COLUMNS, alerts_to_frame
from .fixtures import load_faults, load_readings
from .tuning import file_sha256

METHODS = ("fixed_limit", "rolling_median_mad")
PREFIX = {"fixed_limit": "FIX", "rolling_median_mad": "ROL"}
HELDOUT_START = pd.Timestamp("2026-01-11T00:00:00Z")
HELDOUT_END = pd.Timestamp("2026-01-16T00:00:00Z")  # exclusive
RECOVERY_WINDOW = pd.Timedelta(minutes=60)  # informational only; not used in headline metrics
SIDECAR_FILE = "frozen_config.sha256"  # pins the shipped frozen config's SHA-256 (Handoff 4 value)
EXTRA_COLUMNS = ["alert_id", "match_status", "matched_fault_id", "duplicate_of_fault_id", "first_in_fault_utc",
                 "latency_min", "overlaps_benign_load_change_id", "within_60min_after_fault_id"]
ALERT_RECORD_COLUMNS = EXTRA_COLUMNS + ALERT_COLUMNS


class FreezeError(RuntimeError):
    """The saved configuration or fixture does not match the pre-evaluation freeze."""


def load_frozen_config(root: Path):
    path = root / "config" / "frozen_config.json"
    return json.loads(path.read_text(encoding="utf-8")), file_sha256(path)


def _data_paths(root: Path) -> dict[str, Path]:
    return {ss.READINGS_FILE: root / "data" / ss.READINGS_FILE, ss.FAULTS_FILE: root / "data" / ss.FAULTS_FILE,
            ss.CONTEXT_FILE: root / "data" / ss.CONTEXT_FILE, ss.MANIFEST_FILE: root / "config" / ss.MANIFEST_FILE}


def verify_freeze(config: dict, root: Path) -> None:
    if not str(config.get("status", "")).startswith("FROZEN") or config.get("heldout_evaluated") is not False:
        raise FreezeError("config is not a pre-evaluation freeze (status/heldout_evaluated)")
    recorded = config["data_sha256"]
    paths = _data_paths(root)
    if set(recorded) != set(paths):
        raise FreezeError(f"hash set mismatch: {sorted(recorded)} vs {sorted(paths)}")
    bad = [n for n, p in paths.items() if file_sha256(p) != recorded[n]]
    if bad:
        raise FreezeError(f"fixture hash mismatch (file changed since freeze): {bad}")
    held = config["partitions_half_open_utc"]["heldout"]
    if (pd.Timestamp(held["start_utc"]), pd.Timestamp(held["end_utc_exclusive"])) != (HELDOUT_START, HELDOUT_END):
        raise FreezeError(f"held-out boundaries differ from 2026-01-11..2026-01-16: {held}")


def verify_shipped_freeze(root: Path) -> str:
    """Full pre-run check of the SHIPPED freeze: config SHA-256 equals the pinned sidecar value, then the
    recorded fixture hashes, status and held-out boundaries. Returns the config SHA-256."""
    config, sha = load_frozen_config(root)
    sidecar = root / "config" / SIDECAR_FILE
    if not sidecar.exists():
        raise FreezeError(f"missing {SIDECAR_FILE}")
    pinned = sidecar.read_text(encoding="utf-8").split()[0]
    if sha != pinned:
        raise FreezeError(f"frozen_config.json SHA-256 {sha} != pinned {pinned}")
    verify_freeze(config, root)
    return sha


def build_detectors(config: dict) -> dict:
    fx, rl = config["fixed_limit"], config["rolling_median_mad"]
    return {
        "fixed_limit": FixedLimitDetector(fx["lower_v"], fx["upper_v"]),
        "rolling_median_mad": RollingMADDetector(window=rl["window"], k=rl["k"], floor_v=rl["floor_v"],
                                                 mad_scale=rl["mad_scale"]),
    }


def run(root: Path) -> dict:
    """Run the frozen held-out replay. Raises FreezeError if the freeze cannot be verified."""
    config, config_sha = load_frozen_config(root)
    verify_freeze(config, root)
    paths = _data_paths(root)
    parts = ss.partitions()
    held = parts["heldout"]
    readings = load_readings(paths[ss.READINGS_FILE])
    faults = load_faults(paths[ss.FAULTS_FILE])
    context = ss.load_context(paths[ss.CONTEXT_FILE])
    validate_faults_within_partitions(faults, parts)

    cut = readings[readings["timestamp_utc"] < held.end].sort_values("timestamp_utc").reset_index(drop=True)
    detectors = build_detectors(config)
    out = {"config": config, "config_sha256": config_sha, "partition": held, "readings": cut, "faults": faults,
           "context": context, "methods": {}}
    for m in METHODS:
        det = detectors[m]
        flags = collect_flags(cut, det)
        out["methods"][m] = {"detector": det, "flags": flags,
                             "ev": evaluate_partition(cut, faults, flags, held, m)}
    out["slots"] = slot_table(out)
    return out


# ---- tables ------------------------------------------------------------------------------------------------------

def _fault_at(ts, df):
    hit = df[(df["start_utc"] <= ts) & (ts < df["end_utc_exclusive"])]
    return hit.iloc[0] if len(hit) else None


def slot_table(res: dict) -> pd.DataFrame:
    """One row per held-out scheduled slot with voltage, eligibility, thresholds and flags for both methods."""
    held, cut = res["partition"], res["readings"]
    elig = eligible_slots(cut)
    rol = res["methods"]["rolling_median_mad"]
    fix = res["methods"]["fixed_limit"]["detector"]
    scores = rol["detector"].scores
    rows = []
    for i, r in cut.iterrows():
        ts = r["timestamp_utc"]
        if not (held.start <= ts < held.end):
            continue
        s = scores[i]
        f = _fault_at(ts, res["faults"])
        b = _fault_at(ts, res["context"][res["context"]["event_type"] == "benign_load_change"])
        rows.append({
            "timestamp_utc": ts, "voltage_v": r["voltage_v"], "eligible": bool(elig[i]),
            "fault_id": None if f is None else f["fault_id"],
            "benign_event_id": None if b is None else b["event_id"],
            "fixed_lower_v": fix.lower_v, "fixed_upper_v": fix.upper_v,
            "fixed_flag": ts in res["methods"]["fixed_limit"]["flags"] and bool(elig[i]),
            "rolling_median_v": s.median_v, "rolling_lower_v": s.lower_v, "rolling_upper_v": s.upper_v,
            "rolling_flag": ts in rol["flags"] and bool(elig[i]),
        })
    return pd.DataFrame(rows)


def alert_records(res: dict) -> pd.DataFrame:
    """All alert events for both methods, with match status and links to matched faults."""
    ctx = res["context"]
    benign = ctx[ctx["event_type"] == "benign_load_change"]
    faults = res["faults"][(res["faults"]["start_utc"] >= res["partition"].start)
                           & (res["faults"]["end_utc_exclusive"] <= res["partition"].end)]
    frames = []
    for m in METHODS:
        ev = res["methods"][m]["ev"]
        matched = {id(a): fid for fid, a in ev["matches"] if a is not None}
        unmatched = {id(u["alert"]): (u["kind"], u["fault_id"]) for u in ev["unmatched"]}
        base = alerts_to_frame(ev["alerts"])
        extra = []
        for n, a in enumerate(ev["alerts"], 1):
            fid, dup_of, first_in, lat = None, None, None, None
            if id(a) in matched:
                status, fid = "matched", matched[id(a)]
                f = faults[faults["fault_id"] == fid].iloc[0]
                first_in = min(x.timestamp for x in a.flags if f.start_utc <= x.timestamp < f.end_utc_exclusive)
                lat = (first_in - f.start_utc).total_seconds() / 60
            else:
                kind, dup_of = unmatched[id(a)]
                status = "unmatched_" + kind
            ben = next((b.event_id for b in benign.itertuples(index=False)
                        if any(b.start_utc <= x.timestamp < b.end_utc_exclusive for x in a.flags)), None)
            rec = next((f.fault_id for f in faults.itertuples(index=False)
                        if id(a) not in matched and f.end_utc_exclusive <= a.first_flagged_utc
                        <= f.end_utc_exclusive + RECOVERY_WINDOW and dup_of is None), None)
            extra.append({"alert_id": f"{PREFIX[m]}-{n:03d}", "match_status": status, "matched_fault_id": fid,
                          "duplicate_of_fault_id": dup_of,
                          "first_in_fault_utc": None if first_in is None else first_in.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "latency_min": lat, "overlaps_benign_load_change_id": ben,
                          "within_60min_after_fault_id": rec})
        frames.append(pd.concat([pd.DataFrame(extra, columns=EXTRA_COLUMNS), base], axis=1))
    frames = [f for f in frames if len(f)]
    if not frames:  # neither method alerted: keep the full header so downstream code and CSVs stay valid
        return pd.DataFrame(columns=ALERT_RECORD_COLUMNS)
    return pd.concat(frames, ignore_index=True).reindex(columns=ALERT_RECORD_COLUMNS)


def per_fault_records(res: dict) -> pd.DataFrame:
    fl = res["faults"].set_index("fault_id")
    rows = []
    for m in METHODS:
        for pf in res["methods"][m]["ev"]["per_fault"]:
            f = fl.loc[pf["fault_id"]]
            rows.append({
                "method": m, "fault_id": pf["fault_id"], "fault_type": f["fault_type"],
                "magnitude_v": f["magnitude_v"], "fault_start_utc": f["start_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fault_end_utc_exclusive": f["end_utc_exclusive"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "detected": pf["detected"], "eligible_slots_in_fault": pf["eligible_slots_in_fault"],
                "alert_start_utc": None if pf["alert_start_utc"] is None else pf["alert_start_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "first_in_fault_utc": None if pf["first_in_fault_utc"] is None else pf["first_in_fault_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "latency_min": pf["latency_min"],
            })
    return pd.DataFrame(rows)


def metrics(res: dict) -> dict:
    alerts = alert_records(res)
    out = {"label": "SYNTHETIC replay - scoped to this seeded fixture; not real-equipment performance",
           "heldout_window_utc": {"start": res["partition"].start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                  "end_exclusive": res["partition"].end.strftime("%Y-%m-%dT%H:%M:%SZ")},
           "seed": ss.SEED,
           "frozen_config_sha256": res["config_sha256"],
           "fixture_hashes_verified": res["config"]["data_sha256"],
           "frozen_settings": {k: res["config"][k] for k in ("fixed_limit", "rolling_median_mad")},
           "methods": {}}
    for m in METHODS:
        ev = res["methods"][m]["ev"]
        a = alerts[alerts["method"] == m]
        sched, valid, elig = ev["scheduled_slots"], ev["valid_slots"], ev["eligible_slots"]
        out["methods"][m] = {
            "scheduled_slots": sched, "valid_slots": valid, "eligible_slots": elig,
            "coverage_valid": valid / sched, "coverage_eligible": elig / sched,
            "monitored_sensor_hours": ev["monitored_sensor_hours"],
            "faults_total": ev["faults_total"], "faults_detected": ev["faults_detected"],
            "faults_missed": ev["faults_missed"], "alert_events": ev["alert_events"],
            "unmatched_alert_events": ev["false_alert_events"],
            "unmatched_per_24_sensor_hours": ev["false_alerts_per_24_sensor_hours"],
            "unmatched_in_fault_duplicates": ev["unmatched_in_fault_duplicates"],
            "unmatched_no_in_fault_flag": ev["unmatched_no_in_fault_flag"],
            "unmatched_overlapping_benign_load_change": int(
                ((a["match_status"] != "matched") & a["overlaps_benign_load_change_id"].notna()).sum()),
            "unmatched_within_60min_after_a_fault_end": int(
                ((a["match_status"] != "matched") & a["within_60min_after_fault_id"].notna()).sum()),
            "mean_latency_min": ev["mean_latency_min"],
            "per_fault_latency_min": {p["fault_id"]: p["latency_min"] for p in ev["per_fault"]},
        }
    return out
