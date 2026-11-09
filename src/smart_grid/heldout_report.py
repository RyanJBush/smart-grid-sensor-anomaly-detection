"""Chart, failure-case selection and report for the frozen held-out replay (SYNTHETIC data only)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import heldout as H
from .export import write_csv
from .synthetic_series import INTERVAL_MIN

BANNER = "SYNTHETIC DATA - seeded test fixture; results do not describe real equipment"
STEP = pd.Timedelta(minutes=INTERVAL_MIN)
OKABE = {"matched": "#009E73", "unmatched_in_fault_duplicate": "#E69F00", "unmatched_no_in_fault_flag": "#D55E00"}
MARK = {"matched": "o", "unmatched_in_fault_duplicate": "^", "unmatched_no_in_fault_flag": "X"}
LABEL = {"matched": "alert matched to a fault", "unmatched_in_fault_duplicate": "unmatched: in-fault duplicate",
         "unmatched_no_in_fault_flag": "unmatched: no in-fault flag"}


def _ts(t):
    return None if t is None or pd.isna(t) else pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def _v(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{nd}f}"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---- chart ------------------------------------------------------------------------------------------------------

def render_chart(res: dict, alerts: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    s = res["slots"]
    held = res["partition"]
    faults = res["faults"][(res["faults"]["start_utc"] >= held.start) & (res["faults"]["end_utc_exclusive"] <= held.end)]
    benign = res["context"][(res["context"]["event_type"] == "benign_load_change")
                            & (res["context"]["start_utc"] >= held.start) & (res["context"]["end_utc_exclusive"] <= held.end)]
    cfg = res["config"]
    fig, axes = plt.subplots(2, 1, figsize=(18, 9.5), sharex=True, sharey=True)
    ymin, ymax = s["voltage_v"].min() - 1.5, s["voltage_v"].max() + 2.5
    missing = s[s["voltage_v"].isna()]["timestamp_utc"]

    for ax, m, title in (
            (axes[0], "fixed_limit",
             f"Fixed limit  [{cfg['fixed_limit']['lower_v']:g}, {cfg['fixed_limit']['upper_v']:g}] V"),
            (axes[1], "rolling_median_mad",
             f"Rolling median/MAD  window {cfg['rolling_median_mad']['window']}, k={cfg['rolling_median_mad']['k']:g}, "
             f"floor {cfg['rolling_median_mad']['floor_v']:g} V")):
        for b in benign.itertuples(index=False):
            ax.axvspan(b.start_utc, b.end_utc_exclusive, color="#56B4E9", alpha=0.22, lw=0)
        for f in faults.itertuples(index=False):
            ax.axvspan(f.start_utc, f.end_utc_exclusive + STEP * 0, color="#F0E442", alpha=0.55, lw=0)
            ax.axvspan(f.start_utc, f.end_utc_exclusive, ymin=0.93, ymax=1.0, color="#B8A800", alpha=0.9, lw=0)
            ax.text(f.start_utc, ymax - 0.15, f.fault_id, fontsize=8, va="top", ha="left", color="#4d4400")
        for t in missing:
            ax.axvspan(t, t + STEP, color="#7f7f7f", alpha=0.55, lw=0)
        if m == "fixed_limit":
            ax.axhline(cfg["fixed_limit"]["lower_v"], color="#444", ls="--", lw=0.9)
            ax.axhline(cfg["fixed_limit"]["upper_v"], color="#444", ls="--", lw=0.9)
        else:
            e = s[s["rolling_lower_v"].notna()]
            ax.fill_between(e["timestamp_utc"], e["rolling_lower_v"], e["rolling_upper_v"], step="post",
                            color="#999999", alpha=0.28, lw=0)
        ax.plot(s["timestamp_utc"], s["voltage_v"], color="#1f2d3d", lw=0.8)
        a = alerts[alerts["method"] == m]
        for row in a.itertuples(index=False):
            ev = next(x for x in res["methods"][m]["ev"]["alerts"] if x.first_flagged_utc.strftime("%Y-%m-%dT%H:%M:%SZ") == row.first_flagged_utc)
            pts = pd.DataFrame({"t": [x.timestamp for x in ev.flags], "v": [x.voltage_v for x in ev.flags]})
            ax.scatter(pts["t"], pts["v"], marker=MARK[row.match_status], s=30 if row.match_status == "matched" else 44,
                       color=OKABE[row.match_status], edgecolor="white", linewidth=0.4, zorder=5)
        ev_m = res["methods"][m]["ev"]
        ax.set_title(f"{title}   |   detected {ev_m['faults_detected']}/{ev_m['faults_total']} faults, "
                     f"{ev_m['alert_events']} alert events, {ev_m['false_alert_events']} unmatched "
                     f"({ev_m['unmatched_in_fault_duplicates']} in-fault duplicate, "
                     f"{ev_m['unmatched_no_in_fault_flag']} no in-fault flag)", fontsize=11, loc="left")
        ax.set_ylabel("voltage (V)")
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.2)

    handles = [Line2D([0], [0], color="#1f2d3d", lw=1, label="voltage reading"),
               Patch(color="#F0E442", alpha=0.55, label="fault interval [start, end)"),
               Patch(color="#56B4E9", alpha=0.22, label="benign load change (context, not a fault)"),
               Patch(color="#7f7f7f", alpha=0.55, label="missing reading (gap)"),
               Patch(color="#999999", alpha=0.28, label="rolling threshold band"),
               Line2D([0], [0], color="#444", ls="--", lw=0.9, label="fixed limits")]
    handles += [Line2D([0], [0], marker=MARK[k], color="w", markerfacecolor=OKABE[k], markeredgecolor="white",
                       markersize=9, label=LABEL[k]) for k in OKABE]
    axes[0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8.5,
                   frameon=False)
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].set_xlim(held.start, held.end)
    axes[1].set_xlabel("UTC (held-out window 2026-01-11 to 2026-01-16, end excluded)")
    fig.suptitle("SYNTHETIC DATA - frozen held-out replay (seeded test fixture, not real equipment)",
                 fontsize=13, fontweight="bold", color="#8b0000")
    fig.text(0.995, 0.005, "SYNTHETIC", ha="right", va="bottom", fontsize=9, color="#8b0000")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---- failure case ---------------------------------------------------------------------------------------------------

FAILURE_RULE = ("Fixed before looking at results. 1) The earliest-onset fault missed by either method (fixed limit "
                "first on ties). 2) Else the earliest sustained shift whose in-fault slots the rolling detector "
                "stopped flagging (adaptation). 3) Else the earliest unmatched alert overlapping a benign load change.")


def pick_failure_case(res: dict):
    faults = res["faults"][(res["faults"]["start_utc"] >= res["partition"].start)
                           & (res["faults"]["end_utc_exclusive"] <= res["partition"].end)].sort_values("start_utc")
    s = res["slots"]
    for f in faults.itertuples(index=False):  # rule 1
        for m in H.METHODS:
            pf = next(p for p in res["methods"][m]["ev"]["per_fault"] if p["fault_id"] == f.fault_id)
            if not pf["detected"]:
                return "missed_fault", m, f.fault_id
    for f in faults[faults["fault_type"] == "sustained_shift"].itertuples(index=False):  # rule 2
        sub = s[(s["fault_id"] == f.fault_id) & s["eligible"]]
        if 0 < sub["rolling_flag"].sum() < len(sub):
            return "rolling_adaptation", "rolling_median_mad", f.fault_id
    al = H.alert_records(res)  # rule 3
    ub = al[(al["match_status"] != "matched") & al["overlaps_benign_load_change_id"].notna()]
    if len(ub):
        r = ub.sort_values(["first_flagged_utc", "method"]).iloc[0]
        return "benign_change_alert", r["method"], r["alert_id"]
    return None, None, None


def _cols(m):
    p = "fixed" if m == "fixed_limit" else "rolling"
    return f"{p}_lower_v", f"{p}_upper_v", f"{p}_flag"


def failure_case(res: dict):
    kind, m, ref = pick_failure_case(res)
    s = res["slots"]
    if kind is None:
        return {"kind": None, "paragraphs": ["No missed fault, rolling adaptation or benign-change alert occurred."],
                "evidence": pd.DataFrame()}
    f = res["faults"].set_index("fault_id")
    paras, windows = [], []

    def window(label, lo, hi):
        w = s[(s["timestamp_utc"] >= lo) & (s["timestamp_utc"] < hi)].copy()
        w.insert(0, "window", label)
        windows.append(w)

    if kind in ("missed_fault", "rolling_adaptation"):
        fr = f.loc[ref]
        lo_c, up_c, fl_c = _cols(m)
        sub = s[(s["fault_id"] == ref) & s["eligible"]].copy()
        sub["margin"] = pd.concat([sub["voltage_v"] - sub[up_c], sub[lo_c] - sub["voltage_v"]], axis=1).max(axis=1)
        best = sub.loc[sub["margin"].idxmax()]
        other = next(x for x in H.METHODS if x != m)
        o_pf = next(p for p in res["methods"][other]["ev"]["per_fault"] if p["fault_id"] == ref)
        before = s[(s["timestamp_utc"] < fr["start_utc"]) & s["eligible"]].iloc[-1]
        last = sub.iloc[-1]
        name = "fixed-limit" if m == "fixed_limit" else "rolling median/MAD"
        paras.append(
            f"**Case type: {kind.replace('_', ' ')} - {name} detector, fault {ref}** "
            f"({fr['fault_type'].replace('_', ' ')}, {fr['magnitude_v']:+g} V injected, "
            f"{_ts(fr['start_utc'])} to {_ts(fr['end_utc_exclusive'])}, {len(sub)} eligible slots).")
        if kind == "missed_fault":
            nflag = int(sub[fl_c].sum())
            paras.append(
                f"No alert had a flag inside the fault interval: {nflag} of {len(sub)} in-fault slots were flagged. "
                f"In-fault voltage ranged {_v(sub['voltage_v'].min())} to {_v(sub['voltage_v'].max())} V against "
                f"{name} thresholds spanning {_v(sub[lo_c].min())} to {_v(sub[up_c].max())} V. "
                f"The closest approach was {_v(-best['margin'])} V inside the nearest threshold at "
                f"{_ts(best['timestamp_utc'])} (voltage {_v(best['voltage_v'])} V, thresholds "
                f"[{_v(best[lo_c])}, {_v(best[up_c])}] V).")
            if m == "rolling_median_mad":
                paras.append(
                    f"The rolling median was {_v(before['rolling_median_v'])} V in the last eligible slot before onset "
                    f"({_ts(before['timestamp_utc'])}) and {_v(last['rolling_median_v'])} V at the last in-fault slot "
                    f"({_ts(last['timestamp_utc'])}): the window moved with the shifted level, so the threshold band "
                    f"([{_v(last[lo_c])}, {_v(last[up_c])}] V at that slot) followed the voltage. "
                    f"The fixed-limit detector on the same slots {'detected it with latency ' + _v(o_pf['latency_min'], 0) + ' min' if o_pf['detected'] else 'also missed it'}.")
                wide = sub.loc[(sub[up_c] - sub[lo_c]).idxmax()]
                paras.append(
                    f"Just before onset the band half-width was {_v((before[up_c] - before[lo_c]) / 2)} V (k x spread). "
                    f"The first in-fault voltages ({' and '.join(_v(x) for x in sub['voltage_v'].iloc[:2])} V) stayed under the "
                    f"upper threshold ({' and '.join(_v(x) for x in sub[up_c].iloc[:2])} V). Once the 12-slot window held a mix of "
                    f"pre- and post-shift values the band widened to {_v(wide[up_c] - wide[lo_c])} V total at "
                    f"{_ts(wide['timestamp_utc'])}, which further reduced the chance of a flag.")
            else:
                paras.append(f"The rolling detector on the same slots "
                             f"{'detected it with latency ' + _v(o_pf['latency_min'], 0) + ' min' if o_pf['detected'] else 'also missed it'}.")
            window("onset", fr["start_utc"] - 3 * STEP, fr["start_utc"] + 12 * STEP)
            window("end", fr["end_utc_exclusive"] - 4 * STEP, fr["end_utc_exclusive"] + 6 * STEP)
        else:
            first = sub[sub[fl_c]].iloc[0]
            lastflag = sub[sub[fl_c]].iloc[-1]
            paras.append(
                f"The rolling detector flagged {int(sub[fl_c].sum())} of {len(sub)} in-fault slots, first at "
                f"{_ts(first['timestamp_utc'])} and last at {_ts(lastflag['timestamp_utc'])}, then stopped while the "
                f"shift continued to {_ts(fr['end_utc_exclusive'])}. Its median moved from {_v(before['rolling_median_v'])} V "
                f"before onset to {_v(last['rolling_median_v'])} V at the last in-fault slot.")
            window("onset", fr["start_utc"] - 3 * STEP, fr["start_utc"] + 12 * STEP)
            window("end", fr["end_utc_exclusive"] - 4 * STEP, fr["end_utc_exclusive"] + 6 * STEP)
    else:  # benign_change_alert
        al = H.alert_records(res)
        a = al[al["alert_id"] == ref].iloc[0]
        ben = res["context"].set_index("event_id").loc[a["overlaps_benign_load_change_id"]]
        paras.append(
            f"**Case type: benign-change alert - alert {ref}** ({a['method']}): flags {a['contributing_flagged_utc']} "
            f"overlap benign load change {a['overlaps_benign_load_change_id']} ({ben['magnitude_v']:+g} V, "
            f"{_ts(ben['start_utc'])} to {_ts(ben['end_utc_exclusive'])}). Triggering voltage {_v(a['triggering_voltage_v'])} V "
            f"crossed {a['triggering_direction']} threshold {_v(a['breached_threshold_v'])} V. "
            "The labels mark this interval as a benign load change, not a fault, so the alert is unmatched.")
        window("alert", pd.Timestamp(a["first_flagged_utc"]) - 3 * STEP, pd.Timestamp(a["first_flagged_utc"]) + 12 * STEP)

    ev = pd.concat(windows, ignore_index=True)
    ev["rolling_margin_v"] = pd.concat([ev["voltage_v"] - ev["rolling_upper_v"], ev["rolling_lower_v"] - ev["voltage_v"]],
                                       axis=1).max(axis=1)
    keep = ["window", "timestamp_utc", "voltage_v", "fault_id", "benign_event_id", "eligible", "fixed_lower_v",
            "fixed_upper_v", "fixed_flag", "rolling_median_v", "rolling_lower_v", "rolling_upper_v", "rolling_margin_v",
            "rolling_flag"]
    return {"kind": kind, "method": m, "ref": ref, "paragraphs": paras, "evidence": ev[keep]}


# ---- report -------------------------------------------------------------------------------------------------------------

def build_report(res: dict, met: dict, alerts: pd.DataFrame, per_fault: pd.DataFrame, case: dict) -> str:
    cfg = res["config"]
    fx, rl = cfg["fixed_limit"], cfg["rolling_median_mad"]
    M = met["methods"]
    fm, rm = M["fixed_limit"], M["rolling_median_mad"]
    L = [f"# Held-out replay report\n\n**{BANNER}.** Conclusions below are scoped to this one seeded synthetic "
         "replay. They are not evidence about real equipment and settings were not changed after seeing results.\n"]
    L.append("## Freeze record\n")
    L.append(f"- Window: 2026-01-11T00:00:00Z to 2026-01-16T00:00:00Z (end excluded), 5-minute cadence, seed {met['seed']}.")
    L.append(f"- Frozen settings read from `config/frozen_config.json` (SHA-256 `{met['frozen_config_sha256']}`; file not modified):")
    L.append(f"  fixed limit [{fx['lower_v']:g}, {fx['upper_v']:g}] V; rolling median/MAD window {rl['window']}, "
             f"k {rl['k']:g}, floor {rl['floor_v']:g} V, MAD scale {rl['mad_scale']}.")
    L.append("- All four recorded fixture SHA-256 hashes matched before the run. Earlier readings initialised rolling history; "
             "only shared eligible held-out slots were grouped and scored.\n")
    L.append("## Headline metrics (frozen scoring rules)\n")
    rows = [
        ("Scheduled slots", fm["scheduled_slots"], rm["scheduled_slots"]),
        ("Valid slots", fm["valid_slots"], rm["valid_slots"]),
        ("Eligible slots (shared)", fm["eligible_slots"], rm["eligible_slots"]),
        ("Coverage: valid / scheduled", f"{fm['coverage_valid']:.4f}", f"{rm['coverage_valid']:.4f}"),
        ("Coverage: eligible / scheduled", f"{fm['coverage_eligible']:.4f}", f"{rm['coverage_eligible']:.4f}"),
        ("Monitored sensor-hours", _v(fm["monitored_sensor_hours"], 3), _v(rm["monitored_sensor_hours"], 3)),
        ("Faults detected / total", f"{fm['faults_detected']} / {fm['faults_total']}", f"{rm['faults_detected']} / {rm['faults_total']}"),
        ("Faults missed", fm["faults_missed"], rm["faults_missed"]),
        ("Alert events", fm["alert_events"], rm["alert_events"]),
        ("Unmatched alert events", fm["unmatched_alert_events"], rm["unmatched_alert_events"]),
        ("Unmatched per 24 monitored sensor-hours", _v(fm["unmatched_per_24_sensor_hours"], 3), _v(rm["unmatched_per_24_sensor_hours"], 3)),
        ("  of which in-fault duplicates", fm["unmatched_in_fault_duplicates"], rm["unmatched_in_fault_duplicates"]),
        ("  of which no in-fault flag", fm["unmatched_no_in_fault_flag"], rm["unmatched_no_in_fault_flag"]),
        ("Mean latency over detected faults (min)", _v(fm["mean_latency_min"], 1), _v(rm["mean_latency_min"], 1)),
    ]
    L.append(md_table(["Metric", "Fixed limit", "Rolling median/MAD"], rows))
    L.append("\nUnmatched = alert events not matched one-to-one to a fault. *In-fault duplicate*: has a flag inside a fault "
             "interval whose match went to another alert. *No in-fault flag*: no flag inside any fault interval.\n")
    L.append("Informational counts (not part of the headline metrics): unmatched events overlapping a labelled benign load "
             f"change - fixed {fm['unmatched_overlapping_benign_load_change']}, rolling {rm['unmatched_overlapping_benign_load_change']}; "
             "unmatched events starting within 60 min after a fault ends - "
             f"fixed {fm['unmatched_within_60min_after_a_fault_end']}, rolling {rm['unmatched_within_60min_after_a_fault_end']}.\n")
    L.append("## Per-fault latency\n")
    pf = per_fault.pivot(index="fault_id", columns="method", values="latency_min")
    det = per_fault.pivot(index="fault_id", columns="method", values="detected")
    meta = per_fault[per_fault["method"] == "fixed_limit"].set_index("fault_id")
    rows = []
    for fid in pf.index:
        def cell(m):
            return f"{pf.loc[fid, m]:.0f}" if bool(det.loc[fid, m]) else "missed"
        rows.append((fid, meta.loc[fid, "fault_type"], f"{meta.loc[fid, 'magnitude_v']:+g}", meta.loc[fid, "fault_start_utc"],
                     cell("fixed_limit"), cell("rolling_median_mad")))
    L.append(md_table(["Fault", "Type", "Injected (V)", "Onset (UTC)", "Fixed latency (min)", "Rolling latency (min)"], rows))
    L.append("\nLatency = first in-fault flag minus fault onset. Alert start is reported separately in `heldout_per_fault.csv`.\n")
    L.append("## Observed failure case\n")
    L.append(f"Selection rule: {FAILURE_RULE}\n")
    L.extend(p + "\n" for p in case["paragraphs"])
    ev = case["evidence"]
    if len(ev):
        t = ev.copy()
        t["timestamp_utc"] = t["timestamp_utc"].map(_ts)
        rows = [(r.window, r.timestamp_utc[5:16].replace("T", " "), _v(r.voltage_v), "" if pd.isna(r.fault_id) else r.fault_id, _v(r.fixed_upper_v, 1),
                 "Y" if r.fixed_flag else "", _v(r.rolling_median_v), f"[{_v(r.rolling_lower_v)}, {_v(r.rolling_upper_v)}]",
                 "Y" if r.rolling_flag else "") for r in t.itertuples(index=False)]
        L.append(md_table(["window", "UTC (MM-DD HH:MM)", "V", "fault", "fixed upper", "fixed flag", "rolling median",
                           "rolling band", "rolling flag"], rows))
        L.append("\nFull slot-level evidence: `heldout_failure_case_evidence.csv`.\n")
    L.append("## Chart\n\n![held-out chart](heldout_chart.png)\n")
    L.append("## Files\n\n`heldout_metrics.json`, `heldout_alerts.csv` (per-event triggering voltage, threshold, direction, reason, "
             "contributing flags, matched fault), `heldout_per_fault.csv`, `heldout_failure_case_evidence.csv`, `heldout_chart.png`.\n")
    L.append("## Scope and open assumptions\n")
    L.append("- One seeded synthetic series; six faults per method. No confidence intervals, no repeated seeds. "
             "Differences between methods here may not carry over to other seeds or to any real data.\n"
             "- Frozen settings came from a narrow grid and a provisional 1.0 per 24 h false-alert budget.\n"
             "- Under the contract a sustained shift split into several alerts adds unmatched events, and a detector "
             "that flags the return to normal after a shift ends is charged an unmatched event.\n"
             "- The 60-minute after-fault count is informational only.\n")
    return "\n".join(L)


def write_outputs(res: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    alerts, per_fault, met = H.alert_records(res), H.per_fault_records(res), H.metrics(res)
    case = failure_case(res)
    write_csv(alerts, out_dir / "heldout_alerts.csv")
    write_csv(per_fault, out_dir / "heldout_per_fault.csv")
    ev = case["evidence"].copy()
    if len(ev):
        ev["timestamp_utc"] = ev["timestamp_utc"].map(_ts)
    write_csv(ev, out_dir / "heldout_failure_case_evidence.csv")
    met["failure_case"] = {"kind": case["kind"], "method": case.get("method"), "ref": case.get("ref"), "rule": FAILURE_RULE}
    met["heldout_evaluated"] = True
    # Explicit LF so output bytes do not depend on the OS that generates them.
    text = json.dumps(met, indent=2, sort_keys=True, default=str) + "\n"
    (out_dir / "heldout_metrics.json").write_text(text, encoding="utf-8", newline="\n")
    render_chart(res, alerts, out_dir / "heldout_chart.png")
    report = build_report(res, met, alerts, per_fault, case)
    (out_dir / "heldout_report.md").write_text(report, encoding="utf-8", newline="\n")
    return {"metrics": met, "alerts": alerts, "per_fault": per_fault, "case": case}
