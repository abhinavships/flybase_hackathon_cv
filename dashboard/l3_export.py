"""
l3_export.py — compute every Level 3 aggregate and write CSVs + a findings.md.

    python l3_export.py --data flytbase_out --out flytbase_out/l3_export

The markdown is a first draft of the writeup with the real numbers substituted in,
including the caveats that must travel with them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import l3_core as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--count-y", type=float, default=None)
    ap.add_argument("--limit", type=float, default=C.SPEED_LIMIT_KMH)
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--legs", type=int, default=4)
    ap.add_argument("--edie-dy", type=float, default=20.0)
    ap.add_argument("--edie-dt", type=float, default=10.0)
    a = ap.parse_args()

    data = C.find_data_dir(a.data)
    out = Path(a.out) if a.out else data / "l3_export"
    out.mkdir(parents=True, exist_ok=True)

    kin = C.load_kinematics(data)
    summary = C.track_summary(kin)
    summary, legs = C.discover_legs(summary, k=a.legs)
    mov = kin[kin["is_moving"]]

    count_y = a.count_y if a.count_y is not None else float(np.median(mov["Y_s"]))

    tables = {}
    tables["per_track_summary"] = summary
    tables["legs"] = legs
    tables["class_table"] = C.class_table(summary)
    tables["od_matrix"] = C.od_matrix(summary).reset_index()
    tables["turning_movements"] = C.movement_table(summary)

    cross = C.line_crossings(mov, count_y)
    tables["line_crossings"] = cross
    tables["counts_by_interval"] = C.counts_by_interval(cross, a.interval)
    tables["headways"] = C.headways(cross)

    tables["speed_profile"] = C.speed_profile(mov, 5.0, a.limit)
    tables["speed_profile_by_class"] = C.speed_profile(mov, 5.0, a.limit, by_class=True,
                                                       by_direction=False)
    tables["speed_field"] = C.speed_field(mov)
    tables["speeding_hotspots"] = C.speeding_hotspots(mov, a.limit)

    lanes_by_dir = {d: C.discover_lanes(cross, d) for d in cross["dir_label"].unique()} \
        if not cross.empty else {}
    crl = C.assign_lanes(cross, lanes_by_dir) if not cross.empty else pd.DataFrame()
    tables["lane_centres"] = pd.concat(
        [l.assign(direction=d) for d, l in lanes_by_dir.items() if l is not None and not l.empty],
        ignore_index=True) if lanes_by_dir else pd.DataFrame()
    tables["lane_modal_split"] = C.lane_modal_split(crl) if not crl.empty else pd.DataFrame()
    y10, y90 = np.nanpercentile(mov["Y_s"], [20, 80])
    tables["lateral_dispersion"] = C.lateral_dispersion(mov, y10, y90)

    q = C.queues(kin)
    tables["queues"] = q
    tables["queue_timeseries"] = C.queue_timeseries(q)
    tables["stop_bar"] = C.stop_bar_estimate(q)

    e = C.edie(kin, a.edie_dy, a.edie_dt)
    tables["fundamental_diagram"] = e
    tables["occupancy"] = C.occupancy(kin, count_y)
    tables["interaction_density"] = C.interaction_density(kin)

    for name, df in tables.items():
        if df is not None and len(df):
            df.to_csv(out / f"{name}.csv", index=False)
            print(f"  {name}.csv  ({len(df)} rows)")

    ov = C.overview(kin, summary)
    json.dump(ov, open(out / "overview.json", "w"), indent=2, default=float)

    write_findings(out, ov, tables, count_y, a.limit, e, q)
    print(f"\nwrote {out}")


def write_findings(out: Path, ov, T, count_y, limit, e, q):
    ct = T["class_table"]
    sp = T["speed_profile"]
    ld = T["lateral_dispersion"]
    sb = T["stop_bar"]
    hw = T["headways"]

    L = []
    L.append("# Level 3 — Aggregate Insight\n")
    L.append(f"Window {ov['window_s']:.0f} s · {ov['n_moving']} moving road users "
             f"({ov['n_static']} static tracks excluded from every statistic below).\n")

    L.append("## Headline\n")
    L.append(f"- Throughput **{ov['throughput_vph']:,.0f} veh/h**, "
             f"**{ov['pcu_ph']:,.0f} PCU/h**.")
    L.append(f"- Two-wheelers are **{ov['two_wheeler_share']*100:.0f}%** of motorised "
             "road users. Capacity and emissions models built on a four-wheeler fleet "
             "do not describe this junction.")
    L.append(f"- Median speed **{ov['median_speed_kmh']:.1f} km/h**, "
             f"median stopped delay **{ov['median_delay_s']:.1f} s** per road user.\n")

    if not ct.empty:
        L.append("## Fleet composition\n")
        L.append(ct.to_markdown(index=False, floatfmt=".2f"))
        small = ct[ct["reliable"] != "yes"]["class_final"].tolist()
        if small:
            L.append(f"\n> Sample-size caveat: {', '.join(small)} have fewer than 20 tracks. "
                     "Those rows are observations, not distributions.\n")

    if not T["turning_movements"].empty:
        L.append("\n## Turning movements and OD\n")
        tm = T["turning_movements"]
        tot = tm.groupby("movement")["count"].sum().sort_values(ascending=False)
        L.append("Movement split: " + ", ".join(f"**{k}** {v}" for k, v in tot.items()) + ".")
        L.append("\nOD matrix (rows = entry leg, columns = exit leg):\n")
        L.append(T["od_matrix"].to_markdown(index=False))
        L.append("\n> Legs are k-means clusters over track entry/exit points; compass "
                 "labels are in the corridor frame (+Y = 'N'), not true north.\n")

    if not sp.empty:
        worst = sp.loc[sp["v_med"].idxmin()]
        fastest = sp.loc[sp["v_med"].idxmax()]
        over = sp.loc[sp["frac_over"].idxmax()]
        L.append("\n## Speed along the corridor\n")
        L.append(f"- Slowest point: Y = **{worst['y_bin']:.0f} m** at "
                 f"{worst['v_med']:.1f} km/h median ({worst['dir_label']}).")
        L.append(f"- Fastest point: Y = **{fastest['y_bin']:.0f} m** at "
                 f"{fastest['v_med']:.1f} km/h median.")
        L.append(f"- Speeding concentrates at Y = **{over['y_bin']:.0f} m**, where "
                 f"{over['frac_over']*100:.0f}% of samples exceed {limit:.0f} km/h.")
        L.append("- Speeds are along-track and GCP-calibrated, so these are the reliable "
                 "numbers in this report.\n")

    if not T["lane_modal_split"].empty:
        L.append("\n## Lanes and modal split\n")
        L.append(T["lane_modal_split"].to_markdown(index=False, floatfmt=".2f"))
    if not ld.empty:
        L.append("\nLateral dispersion by class:\n")
        L.append(ld.to_markdown(index=False, floatfmt=".2f"))
        L.append("\n> Lateral (X) distances come from the weak axis of a thin calibration "
                 "quad. Lane counts and modal mix are sound; absolute lane widths are "
                 "indicative only.\n")

    if not q.empty:
        L.append("\n## Queues\n")
        L.append(f"- Longest queue observed **{q['length_m'].max():.1f} m** "
                 f"({int(q['n_veh'].max())} vehicles).")
        L.append(f"- Mean queue length **{q['length_m'].mean():.1f} m**.")
        if not sb.empty:
            for _, r in sb.iterrows():
                L.append(f"- Inferred stop line, direction {r['direction']}: "
                         f"Y = **{r['stop_bar_y_m']:.1f} m** "
                         f"(IQR {r['y_head_iqr']:.1f} m).")
        L.append("- Stop-line position is inferred purely from where vehicles repeatedly "
                 "come to rest — no signal-phase data was used.\n")

    if not e.empty:
        cap = e.loc[e["flow_vph"].idxmax()]
        L.append("\n## Flow–density (Edie's definitions)\n")
        L.append(f"- Observed capacity **{cap['flow_vph']:,.0f} veh/h** at "
                 f"**{cap['density_vpkm']:.0f} veh/km**, space-mean speed "
                 f"{cap['speed_kmh']:.1f} km/h.")
        L.append(f"- Density range observed: {e['density_vpkm'].min():.0f}–"
                 f"{e['density_vpkm'].max():.0f} veh/km over "
                 f"{len(e)} space–time cells.")
        L.append("- Flow and density are computed as total distance and total time per "
                 "space–time cell, not from a detector line, so the estimate does not "
                 "depend on where a counter happens to be placed.\n")

    if not hw.empty:
        L.append(f"\n## Headways\n\nMedian time headway **{hw['headway_s'].median():.2f} s** "
                 f"at the counting line (Y = {count_y:.0f} m); "
                 f"15th percentile {hw['headway_s'].quantile(0.15):.2f} s.\n")

    L.append("\n## Method notes\n")
    L.append("- Trajectories at 10 Hz in a metric ground frame; positions differentiated "
             "with Savitzky–Golay (window 15, order 2).")
    L.append(f"- Static tracks (net displacement < {C.MOVING_MIN_DISP_M:.0f} m) excluded "
             "from every flow and speed statistic.")
    L.append(f"- Stopped threshold {C.STOP_SPEED_MPS} m/s; queue clustering gap "
             f"{C.QUEUE_GAP_M:.0f} m.")
    L.append("- PCU factors IRC-style: motorcycle 0.5, car 1.0, three-wheeler 1.2, "
             "LGV 1.4, HGV 2.2.")
    L.append("- Along-track (Y) geometry is GCP-calibrated and reliable; across-track (X) "
             "geometry is approximate because the calibration quad is a thin strip.")

    (out / "findings.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
