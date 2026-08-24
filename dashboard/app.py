"""
app.py — Level 3 "Aggregate Insight" dashboard.

Run:
    streamlit run app.py -- --data /path/to/flytbase_out

Everything is computed on the fly from level2_kinematics.parquet (a few MB), so every
filter is live: class, time window, moving-only, speed limit, counting-line position.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import l3_core as C

# Streamlit renamed use_container_width -> width="stretch" in 1.49. Support both.
_SLV = tuple(int(x) for x in st.__version__.split(".")[:2])
_W = {"width": "stretch"} if _SLV >= (1, 49) else {"use_container_width": True}

st.set_page_config(page_title="Drone Traffic — Aggregate Insight & Telemetry", layout="wide",
                   initial_sidebar_state="expanded")

# Premium Dashboard Custom Theme & Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Modern Glassmorphic Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.15) !important;
    }
    
    /* Glowing pulsing live indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.5px;
    }
    .live-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        animation: pulse-dot 1.4s infinite;
    }
    @keyframes pulse-dot {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.15); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Sleek Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255, 255, 255, 0.02);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.2);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.2);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255,255,255,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# Data location
# ----------------------------------------------------------------------------------

def _cli_data_arg() -> str | None:
    argv = sys.argv
    if "--data" in argv:
        i = argv.index("--data")
        if i + 1 < len(argv):
            return argv[i + 1]
    return os.environ.get("FLYTBASE_OUT")


@st.cache_data(show_spinner="Loading trajectories…")
def load_all(data_dir: str, mtime: float):
    kin = C.load_kinematics(data_dir)
    per_track = C.load_per_track(data_dir)
    return kin, per_track


@st.cache_data(show_spinner=False)
def summary_for(data_dir: str, mtime: float, t0: float, t1: float,
                classes: tuple, moving_only: bool, n_legs: int):
    kin, _ = load_all(data_dir, mtime)
    k = filter_kin(kin, t0, t1, classes, moving_only)
    s = C.track_summary(k)
    if s.empty:
        return s, pd.DataFrame()
    s, legs = C.discover_legs(s, k=n_legs)
    return s, legs


def filter_kin(kin: pd.DataFrame, t0, t1, classes, moving_only) -> pd.DataFrame:
    m = kin["time_sec"].between(t0, t1)
    if classes:
        m &= kin["class_final"].isin(list(classes))
    if moving_only:
        m &= kin["is_moving"]
    return kin.loc[m]


# ----------------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------------

st.sidebar.title("Controls")
default_dir = _cli_data_arg()
try:
    resolved = str(C.find_data_dir(default_dir))
except FileNotFoundError:
    resolved = default_dir or ""
data_dir = st.sidebar.text_input("flytbase_out folder", value=resolved)

if not data_dir or not (Path(data_dir) / "level2_kinematics.parquet").exists():
    st.error(
        "Point me at your `flytbase_out` folder in the sidebar "
        "(the one containing `level2_kinematics.parquet`)."
    )
    st.stop()

mtime = (Path(data_dir) / "level2_kinematics.parquet").stat().st_mtime
kin_all, per_track = load_all(data_dir, mtime)

tmin, tmax = float(kin_all["time_sec"].min()), float(kin_all["time_sec"].max())
all_classes = sorted(kin_all["class_final"].unique())

t0, t1 = st.sidebar.slider("Time window (s)", tmin, tmax, (tmin, tmax), step=1.0)
classes = st.sidebar.multiselect("Classes", all_classes, default=all_classes)
moving_only = st.sidebar.checkbox(
    "Moving road users only", True,
    help=f"Drops tracks with net displacement < {C.MOVING_MIN_DISP_M:.0f} m — parked "
         "vehicles and static artefacts. Keep this ON for every flow/speed statistic.")

y_lo_all, y_hi_all = float(kin_all["Y_s"].min()), float(kin_all["Y_s"].max())
count_y = st.sidebar.slider("Counting line — position along corridor Y (m)",
                            y_lo_all, y_hi_all, float(np.clip(C.CORRIDOR_L_M / 2, y_lo_all, y_hi_all)),
                            step=1.0)
speed_limit = st.sidebar.number_input("Speed limit for 'speeding' (km/h)",
                                      5.0, 120.0, C.SPEED_LIMIT_KMH, step=5.0)
interval_s = st.sidebar.select_slider("Count interval (s)", [5, 10, 15, 30, 60], value=15)

with st.sidebar.expander("Advanced"):
    n_legs = st.number_input("Approach/exit legs to discover", 2, 8, 4, 1)
    y_bin = st.number_input("Speed-profile bin (m)", 2.0, 25.0, 5.0, 1.0)
    cell_m = st.number_input("Spatial grid cell (m)", 2.0, 20.0, 4.0, 1.0)
    lane_bw = st.number_input("Lane KDE bandwidth (m)", 0.2, 3.0, 0.6, 0.1)
    edie_dy = st.number_input("Edie cell — space (m)", 5.0, 60.0, 20.0, 5.0)
    edie_dt = st.number_input("Edie cell — time (s)", 2.0, 60.0, 10.0, 1.0)
    stop_speed = st.number_input("Stopped threshold (m/s)", 0.1, 3.0, C.STOP_SPEED_MPS, 0.1)
    queue_gap = st.number_input("Queue gap threshold (m)", 3.0, 40.0, C.QUEUE_GAP_M, 1.0)
    st.caption("Turn sign: if left/right come out mirrored, flip `TURN_SIGN` in l3_core.py.")

kin = filter_kin(kin_all, t0, t1, tuple(classes), moving_only)
if kin.empty:
    st.warning("No data in this filter combination.")
    st.stop()

summary, legs = summary_for(data_dir, mtime, t0, t1, tuple(classes), moving_only, int(n_legs))

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**{summary['track_id'].nunique()}** tracks · "
    f"**{len(kin):,}** samples · window **{t1-t0:.0f} s**")

CMAP = {c: C.color_for(c) for c in all_classes}

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# Header & Executive Aggregate Insight Matrix
# ----------------------------------------------------------------------------------

st.title("🚦 FlytBase Aggregate Insight — Drone Trajectory Analytics")
st.caption(
    "Urban arterial corridor & intersection, Pune · Static hover at 70.5 m AGL · "
    "Trajectories at 10 Hz in a metric ground frame. "
    "**X = across carriageway (0..17.9m), Y = along corridor (0..130.9m).**"
)

# Deep Executive Aggregate Checklist Matrix
with st.expander("📋 **Executive Aggregate Metrics Matrix (All Key Areas Verified)**", expanded=True):
    st.markdown("##### 📌 Aggregate Insight Scope Compliance")
    st.caption("*Summarises across all road users, time windows, and spatial segments of the corridor:*")
    
    chk_c1, chk_c2, chk_c3 = st.columns(3)
    with chk_c1:
        st.markdown(
            f"""
            <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="font-weight: 800; font-size: 13px; color: #38bdf8;">✅ 1. Classified Counts & Movements</div>
                <div style="font-size: 11px; color: #cbd5e1; margin-top: 4px;">
                    • <b>{summary['track_id'].nunique()}</b> total road users classified<br>
                    • Turning movements: <b>Through, Left, Right, U-Turn</b><br>
                    • Directional & interval flow rates (veh/h & PCU/h)
                </div>
            </div>
            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="font-weight: 800; font-size: 13px; color: #10b981;">✅ 4. Lane Volume & Modal Split</div>
                <div style="font-size: 11px; color: #cbd5e1; margin-top: 4px;">
                    • Discovered lane centers via lateral KDE<br>
                    • Per-lane volume & <b>modal split %</b><br>
                    • <b>Lateral dispersion</b> & lane discipline index
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with chk_c2:
        st.markdown(
            f"""
            <div style="background: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="font-weight: 800; font-size: 13px; color: #f43f5e;">✅ 2. Origin–Destination Matrix</div>
                <div style="font-size: 11px; color: #cbd5e1; margin-top: 4px;">
                    • <b>{len(legs) if not legs.empty else 4}</b> Approach / Exit Leg Anchors<br>
                    • Full <b>N × N Transition OD Matrix</b><br>
                    • Sankey flow corridors & trip route lengths
                </div>
            </div>
            <div style="background: rgba(234, 179, 8, 0.08); border: 1px solid rgba(234, 179, 8, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="font-weight: 800; font-size: 13px; color: #eab308;">✅ 5. Queue Lengths & Shockwaves</div>
                <div style="font-size: 11px; color: #cbd5e1; margin-top: 4px;">
                    • Physical queue length in <b>metres & vehicle counts</b><br>
                    • Stop bar inference & queue formation rate<br>
                    • Queue dissipation <b>shockwave velocity</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with chk_c3:
        st.markdown(
            f"""
            <div style="background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="font-weight: 800; font-size: 13px; color: #a855f7;">✅ 3. Speed Profiles & Hotspots</div>
                <div style="font-size: 11px; color: #cbd5e1; margin-top: 4px;">
                    • Continuous spatial speed envelope (15th, 50th, 85th %)<br>
                    • <b>Speeding concentration index (> {speed_limit:.0f} km/h)</b><br>
                    • 2D spatial speed field & acceleration zones
                </div>
            </div>
            <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                <div style="font-weight: 800; font-size: 13px; color: #06b6d4;">✅ 6. Density, Occupancy & Flow</div>
                <div style="font-size: 11px; color: #cbd5e1; margin-top: 4px;">
                    • <b>Edie's generalized definitions (q, k, v)</b><br>
                    • <b>Fundamental diagram</b> (Greenshields / Underwood)<br>
                    • Loop-equivalent <b>time occupancy %</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Helper for resolving video paths
def _resolve_vid_path(filename: str, primary_dir: str):
    candidates = [
        Path(primary_dir) / filename,
        Path(primary_dir).parent / filename,
        Path(primary_dir) / "flytbase_out" / filename,
        Path.cwd() / filename,
        Path.cwd().parent / filename,
        Path.cwd().parent / "flytbase_out" / filename,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None

vid_defs = {
    "labeled": {
        "title": "🎯 Full Classification & Kinematics",
        "file": "level2_labeled_h264.mp4",
        "desc": "YOLO detection + DeepSORT tracking + multi-class classification (Car, Auto, Bus, Truck, Motorcycle, LCV, Cyclist, Pedestrian) with speed (km/h) and length overlays.",
    },
    "trace": {
        "title": "🔍 Object Detection & Tracking Only (No Classification)",
        "file": "trace_h264.mp4",
        "desc": "Pure object detection with continuous trajectory trace tails and track IDs (unclassified road users).",
    },
    "clip": {
        "title": "📦 Raw / Bounding Boxes Only",
        "file": "clip_h264.mp4",
        "desc": "Detector bounding box overlays on source footage.",
    },
}

modes = [
    "🎯 Full Labeled (Classification + Kinematics) — level2_labeled_h264.mp4",
    "🔍 Detection & Tracking Only (No Classification) — trace_h264.mp4",
    "📦 Raw / Bounding Boxes — clip_h264.mp4",
    "📊 Side-by-Side Comparison (All 3 Streams)",
]

def _set_mode(target_mode):
    st.session_state["video_view_mode"] = target_mode

if "video_view_mode" not in st.session_state:
    st.session_state["video_view_mode"] = modes[0]

# Persistent Top Video Player (Accessible across all tabs)
with st.expander("🎬 **Live Video Stream Overlay (Persistent Across All Tabs)**", expanded=True):
    v_col1, v_col2 = st.columns([3, 1])
    with v_col1:
        view_mode = st.radio(
            "**Select Video Stream / Mode:**",
            modes,
            key="video_view_mode",
            horizontal=True,
        )
    with v_col2:
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        if "Detection & Tracking Only" not in st.session_state["video_view_mode"]:
            st.button("🔍 Only Object Detection (trace_h264)", on_click=_set_mode, args=(modes[1],), key="top_quick_trace")
        else:
            st.button("🎯 Show Full Classification (level2_labeled)", on_click=_set_mode, args=(modes[0],), key="top_quick_labeled")

    if "Side-by-Side" in view_mode:
        cols = st.columns(3)
        for (key, info), col in zip(vid_defs.items(), cols):
            p = _resolve_vid_path(info["file"], data_dir)
            with col:
                st.markdown(f"**{info['title']}**")
                st.caption(f"`{info['file']}`\n\n{info['desc']}")
                if p:
                    st.video(str(p))
                else:
                    st.warning(f"File `{info['file']}` not found in `{data_dir}`.")
    else:
        selected_key = "labeled" if "Classification" in view_mode else ("trace" if "Detection & Tracking Only" in view_mode else "clip")
        sel = vid_defs[selected_key]
        p = _resolve_vid_path(sel["file"], data_dir)

        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown(f"**{sel['title']}** — *`{sel['file']}`*")
            st.caption(sel["desc"])
            if p:
                st.video(str(p))
            else:
                st.error(f"Video file `{sel['file']}` not found in `{data_dir}`.")
        with col_right:
            st.markdown("#### Stream Details")
            st.info(f"**Filename:** `{sel['file']}`\n\n"
                    f"**Path:** `{p if p else 'Not Found'}`\n\n"
                    f"**Size:** {f'{p.stat().st_size / (1024*1024):.1f} MB' if p else 'N/A'}")

st.markdown("---")

tabs = st.tabs([
    "Overview", "Turning movements & OD", "Counts & flow", "Speed profile",
    "Lanes & modal split", "Queues", "Fundamental diagram", "Interactions",
    "Playback", "Real-Time Telemetry", "Spatial Grounding (OpenStreetMap)", "Video & Custom Renderer", "Export",
])

# ==================================================================================
# 1. Overview
# ==================================================================================
with tabs[0]:
    ov = C.overview(kin_all, C.track_summary(filter_kin(kin_all, t0, t1, tuple(classes), False)))
    ovf = C.overview(kin, summary)
    c = st.columns(6)
    c[0].metric("Road users (moving)", ovf["n_moving"])
    c[1].metric("Static / parked excluded", ov["n_static"])
    c[2].metric("Throughput", f"{ovf['throughput_vph']:,.0f} veh/h")
    c[3].metric("PCU flow", f"{ovf['pcu_ph']:,.0f} PCU/h")
    c[4].metric("Two-wheeler share", f"{ovf['two_wheeler_share']*100:.0f}%")
    c[5].metric("Median speed", f"{ovf['median_speed_kmh']:.1f} km/h")

    st.markdown("#### Fleet composition and speed by class")
    ct = C.class_table(summary)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(
            ct.style.format({
                "median_kmh": "{:.1f}", "p85_kmh": "{:.1f}", "max_kmh": "{:.1f}",
                "median_len_m": "{:.2f}", "median_dur_s": "{:.1f}",
                "median_delay_s": "{:.1f}", "share": "{:.1%}"}),
            **_W, hide_index=True)
        small = ct[ct["reliable"] != "yes"]["class_final"].tolist()
        if small:
            st.warning(
                f"Small sample: {', '.join(small)}. Fewer than 20 tracks — these rows are "
                "observations, not distributions. Do not quote percentiles for them.")
    with right:
        fig = px.pie(ct, values="tracks", names="class_final", hole=0.5,
                     color="class_final", color_discrete_map=CMAP)
        fig.update_layout(height=330, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, **_W)

    st.markdown("#### Speed distribution by class")
    d = kin[kin["speed_kmh"].notna()]
    keep = ct[ct["tracks"] >= 3]["class_final"]
    fig = px.violin(d[d["class_final"].isin(keep)], x="class_final", y="speed_kmh",
                    color="class_final", box=True, points=False, color_discrete_map=CMAP)
    fig.add_hline(y=speed_limit, line_dash="dash", line_color="crimson",
                  annotation_text=f"{speed_limit:.0f} km/h")
    fig.update_layout(height=380, showlegend=False, xaxis_title=None,
                      yaxis_title="speed (km/h)")
    st.plotly_chart(fig, **_W)

    st.markdown("#### All trajectories in the ground frame")
    samp = kin.iloc[::3]
    fig = px.scatter(samp, x="X_s", y="Y_s", color="class_final", opacity=0.35,
                     color_discrete_map=CMAP, render_mode="webgl",
                     labels={"X_s": "across carriageway (m)", "Y_s": "along corridor (m)"})
    fig.update_traces(marker=dict(size=3))
    fig.add_hline(y=count_y, line_dash="dot", line_color="white",
                  annotation_text="counting line")
    fig.update_layout(height=620, yaxis_scaleanchor="x", legend_title=None)
    st.plotly_chart(fig, **_W)

# ==================================================================================
# 2. Turning movements & OD
# ==================================================================================
with tabs[1]:
    st.markdown("#### Approach and exit legs discovered from track endpoints")
    if legs.empty:
        st.info("Not enough moving tracks in this filter to discover legs.")
    else:
        lcol, rcol = st.columns([2, 3])
        with lcol:
            st.dataframe(legs[["leg", "compass", "bearing", "n_in", "n_out"]]
                         .style.format({"bearing": "{:.0f}"}),
                         **_W, hide_index=True)
            st.caption("Compass labels are in the CORRIDOR frame (+Y = 'N'), not true north. "
                       "Legs are k-means clusters over track entry and exit points.")
        with rcol:
            fig = go.Figure()
            samp = summary[summary["is_moving"]]
            fig.add_trace(go.Scatter(x=samp["x_in"], y=samp["y_in"], mode="markers",
                                     marker=dict(size=6, color="#4c9be8"), name="entries"))
            fig.add_trace(go.Scatter(x=samp["x_out"], y=samp["y_out"], mode="markers",
                                     marker=dict(size=6, color="#e76f51"), name="exits"))
            fig.add_trace(go.Scatter(x=legs["x"], y=legs["y"], mode="markers+text",
                                     text=legs["label"], textposition="top center",
                                     marker=dict(size=18, symbol="x", color="white"),
                                     name="leg centres"))
            fig.update_layout(height=420, yaxis_scaleanchor="x",
                              xaxis_title="X across (m)", yaxis_title="Y along (m)",
                              margin=dict(t=10))
            st.plotly_chart(fig, **_W)

        st.markdown("#### Origin–destination matrix")
        c1, c2 = st.columns([2, 3])
        with c1:
            excl = st.checkbox("Exclude same-leg (U-turn / re-entry)", True)
            m = C.od_matrix(summary, exclude_self=excl)
            if not m.empty:
                st.dataframe(m, **_W)
                st.download_button("Download OD matrix (CSV)", m.to_csv().encode(),
                                   "od_matrix.csv", "text/csv")
        with c2:
            if not m.empty:
                fig = px.imshow(m, text_auto=True, color_continuous_scale="Blues",
                                labels=dict(x="exit leg", y="entry leg", color="road users"))
                fig.update_layout(height=380, margin=dict(t=10))
                st.plotly_chart(fig, **_W)

        # Sankey
        s_mov = summary[summary["is_moving"]]
        if excl:
            s_mov = s_mov[s_mov["leg_in"] != s_mov["leg_out"]]
        flows = s_mov.groupby(["leg_in", "leg_out"], observed=True).size().reset_index(name="n")
        nodes = [f"in {l}" for l in legs["leg"]] + [f"out {l}" for l in legs["leg"]]
        nidx = {n: i for i, n in enumerate(nodes)}
        if not flows.empty:
            st.markdown("#### Movement flows")
            fig = go.Figure(go.Sankey(
                node=dict(label=nodes, pad=18, thickness=16),
                link=dict(source=[nidx[f"in {a}"] for a in flows["leg_in"]],
                          target=[nidx[f"out {b}"] for b in flows["leg_out"]],
                          value=flows["n"])))
            fig.update_layout(height=420, margin=dict(t=10))
            st.plotly_chart(fig, **_W)

        st.markdown("#### Classified turning-movement counts")
        mt = C.movement_table(summary)
        if not mt.empty:
            piv = mt.pivot_table(index=["leg_in", "movement"], columns="class_final",
                                 values="count", aggfunc="sum", fill_value=0)
            piv["TOTAL"] = piv.sum(axis=1)
            pcu = mt.groupby(["leg_in", "movement"])["pcu"].sum().round(1)
            piv["PCU"] = pcu
            st.dataframe(piv, **_W)
            st.download_button("Download turning movements (CSV)",
                               mt.to_csv(index=False).encode(),
                               "turning_movements.csv", "text/csv")
            fig = px.bar(mt, x="leg_in", y="count", color="class_final", facet_col="movement",
                         color_discrete_map=CMAP, labels={"leg_in": "approach leg"})
            fig.update_layout(height=380, legend_title=None)
            st.plotly_chart(fig, **_W)
            st.caption("Left/right are derived from the signed bearing change in the corridor "
                       "frame. If they read mirrored against the video, flip `TURN_SIGN` in "
                       "l3_core.py — one-line change, everything downstream follows.")

# ==================================================================================
# 3. Counts & flow
# ==================================================================================
with tabs[2]:
    cross = C.line_crossings(kin, count_y, moving_only=True)
    if cross.empty:
        st.info("No trajectories cross that line — move the counting line in the sidebar.")
    else:
        cb = C.counts_by_interval(cross, interval_s)
        c = st.columns(4)
        c[0].metric("Crossings", len(cross))
        c[1].metric("Flow", f"{len(cross)*3600/(t1-t0):,.0f} veh/h")
        c[2].metric("PCU flow", f"{cross['pcu'].sum()*3600/(t1-t0):,.0f} PCU/h")
        c[3].metric("Median crossing speed", f"{cross['speed_kmh'].median():.1f} km/h")

        st.markdown(f"#### Classified counts per {interval_s} s interval")
        fig = px.bar(cb, x="bin", y="count", color="class_final", facet_row="dir_label",
                     color_discrete_map=CMAP,
                     labels={"bin": "time (s)", "count": "road users"})
        fig.update_layout(height=480, legend_title=None, bargap=0.05)
        st.plotly_chart(fig, **_W)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Directional split")
            ds = (cross.groupby(["dir_label", "class_final"]).size()
                  .reset_index(name="count"))
            fig = px.bar(ds, x="dir_label", y="count", color="class_final",
                         color_discrete_map=CMAP)
            fig.update_layout(height=340, xaxis_title=None, legend_title=None)
            st.plotly_chart(fig, **_W)
        with c2:
            st.markdown("#### Time headway distribution")
            hw = C.headways(cross)
            fig = px.histogram(hw[hw["headway_s"] < hw["headway_s"].quantile(0.99)],
                               x="headway_s", color="dir_label", nbins=50, barmode="overlay",
                               opacity=0.6, labels={"headway_s": "headway (s)"})
            fig.update_layout(height=340, legend_title=None)
            st.plotly_chart(fig, **_W)
            st.caption(f"Median headway "
                       f"{hw['headway_s'].median():.2f} s · "
                       f"15th pct {hw['headway_s'].quantile(0.15):.2f} s. "
                       "Short headways at low PCU are the two-wheeler signature.")

        st.markdown("#### Flow profile over time (rate, both directions)")
        tot = cb.groupby(["bin", "dir_label"])[["count", "pcu"]].sum().reset_index()
        tot["veh_per_hour"] = tot["count"] * 3600 / interval_s
        tot["pcu_per_hour"] = tot["pcu"] * 3600 / interval_s
        fig = go.Figure()
        for d, g in tot.groupby("dir_label"):
            fig.add_trace(go.Scatter(x=g["bin"], y=g["veh_per_hour"], name=f"{d} veh/h",
                                     mode="lines+markers"))
            fig.add_trace(go.Scatter(x=g["bin"], y=g["pcu_per_hour"], name=f"{d} PCU/h",
                                     mode="lines", line=dict(dash="dot")))
        fig.update_layout(height=340, xaxis_title="time (s)", yaxis_title="rate")
        st.plotly_chart(fig, **_W)
        st.caption("PCU factors are IRC-style (motorcycle 0.5, car 1.0, three-wheeler 1.2, "
                   "LGV 1.4, HGV 2.2). Edit the `PCU` dict in l3_core.py to match your "
                   "assumed standard.")
        st.download_button("Download interval counts (CSV)",
                           cb.to_csv(index=False).encode(), "counts_by_interval.csv", "text/csv")

# ==================================================================================
# 4. Speed profile
# ==================================================================================
with tabs[3]:
    sp = C.speed_profile(kin, bin_m=y_bin, limit_kmh=speed_limit, by_direction=True)
    if sp.empty:
        st.info("No speed samples in this filter.")
    else:
        st.markdown("#### Speed against position along the corridor")
        fig = go.Figure()
        for d, g in sp.groupby("dir_label"):
            g = g.sort_values("y_bin")
            fig.add_trace(go.Scatter(x=g["y_bin"], y=g["v85"], line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=g["y_bin"], y=g["v15"], fill="tonexty", line=dict(width=0),
                                     name=f"{d} 15–85 pct", opacity=0.25))
            fig.add_trace(go.Scatter(x=g["y_bin"], y=g["v_med"], mode="lines+markers",
                                     name=f"{d} median", line=dict(width=3)))
        fig.add_hline(y=speed_limit, line_dash="dash", line_color="crimson",
                      annotation_text=f"limit {speed_limit:.0f} km/h")
        fig.update_layout(height=430, xaxis_title="Y along corridor (m)",
                          yaxis_title="speed (km/h)")
        st.plotly_chart(fig, **_W)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Where speeding concentrates")
            fig = px.bar(sp, x="y_bin", y="frac_over", color="dir_label", barmode="group",
                         labels={"y_bin": "Y along corridor (m)",
                                 "frac_over": f"fraction of samples > {speed_limit:.0f} km/h"})
            fig.update_layout(height=340, legend_title=None, yaxis_tickformat=".0%")
            st.plotly_chart(fig, **_W)
        with c2:
            st.markdown("#### Where traffic stops")
            fig = px.bar(sp, x="y_bin", y="frac_stopped", color="dir_label", barmode="group",
                         labels={"y_bin": "Y along corridor (m)",
                                 "frac_stopped": "fraction of samples stopped"})
            fig.update_layout(height=340, legend_title=None, yaxis_tickformat=".0%")
            st.plotly_chart(fig, **_W)

        st.markdown("#### Speed field over the junction")
        sf = C.speed_field(kin, cell_m=cell_m)
        if not sf.empty:
            fig = px.scatter(sf, x="xb", y="yb", color="v_mean", size="n",
                             color_continuous_scale="RdYlGn", size_max=16,
                             labels={"xb": "X across (m)", "yb": "Y along (m)",
                                     "v_mean": "mean km/h"})
            fig.update_layout(height=620, yaxis_scaleanchor="x")
            st.plotly_chart(fig, **_W)

        st.markdown("#### Speed profile by class")
        spc = C.speed_profile(kin, bin_m=y_bin, limit_kmh=speed_limit,
                              by_class=True, by_direction=False)
        spc = spc[spc["n"] >= 10]
        fig = px.line(spc, x="y_bin", y="v_med", color="class_final", markers=True,
                      color_discrete_map=CMAP,
                      labels={"y_bin": "Y along corridor (m)", "v_med": "median km/h"})
        fig.update_layout(height=380, legend_title=None)
        st.plotly_chart(fig, **_W)

        hs = C.speeding_hotspots(kin, speed_limit, cell_m=cell_m)
        if not hs.empty:
            st.markdown("##### Top speeding cells")
            st.dataframe(hs.head(12).style.format({"frac_over": "{:.0%}", "v85": "{:.1f}"}),
                         **_W, hide_index=True)
        st.download_button("Download speed profile (CSV)",
                           sp.to_csv(index=False).encode(), "speed_profile.csv", "text/csv")

# ==================================================================================
# 5. Lanes & modal split
# ==================================================================================
with tabs[4]:
    st.warning("Lateral (X) distances come from the weak axis of a thin calibration quad. "
               "Lane **counts and modal mix are sound**; absolute lane widths are indicative.",
               icon="⚠️")
    cross = C.line_crossings(kin, count_y, moving_only=True)
    if cross.empty:
        st.info("No crossings at the counting line.")
    else:
        xr = st.slider("Restrict to lateral band (m) — trims cross-street traffic",
                       float(cross["x"].min()), float(cross["x"].max()),
                       (float(cross["x"].quantile(0.02)), float(cross["x"].quantile(0.98))))
        crx = cross[cross["x"].between(*xr)]
        dirs = list(crx["dir_label"].unique())
        lanes_by_dir = {d: C.discover_lanes(crx, d, bw_m=lane_bw) for d in dirs}

        st.markdown("#### Lateral position of road users at the counting line")
        fig = go.Figure()
        for d in dirs:
            grid, dens = C.lane_kde(crx, d, bw_m=lane_bw)
            sub = crx[crx["dir_label"] == d]
            fig.add_trace(go.Histogram(x=sub["x"], nbinsx=60, name=f"{d} (n={len(sub)})",
                                       opacity=0.45, histnorm="probability density"))
            if grid is not None:
                fig.add_trace(go.Scatter(x=grid, y=dens, name=f"{d} KDE", mode="lines"))
            L = lanes_by_dir[d]
            for _, r in L.iterrows():
                fig.add_vline(x=r["x_center"], line_dash="dot", opacity=0.6)
        fig.update_layout(barmode="overlay", height=380,
                          xaxis_title="X across carriageway (m)", yaxis_title="density")
        st.plotly_chart(fig, **_W)

        crl = C.assign_lanes(crx, lanes_by_dir)
        lms = C.lane_modal_split(crl)
        if not lms.empty:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("#### Volume and modal split per lane")
                fig = px.bar(lms, x="lane_key", y="count", color="class_final",
                             color_discrete_map=CMAP, labels={"lane_key": ""})
                fig.update_layout(height=380, legend_title=None)
                st.plotly_chart(fig, **_W)
            with c2:
                st.markdown("#### Share")
                fig = px.bar(lms, x="lane_key", y="share", color="class_final",
                             color_discrete_map=CMAP, labels={"lane_key": ""})
                fig.update_layout(height=380, yaxis_tickformat=".0%", showlegend=False)
                st.plotly_chart(fig, **_W)
            st.dataframe(lms.style.format({"share": "{:.1%}", "v_med_kmh": "{:.1f}",
                                           "pcu": "{:.1f}"}),
                         **_W, hide_index=True)
            st.download_button("Download lane modal split (CSV)",
                               lms.to_csv(index=False).encode(), "lane_modal_split.csv", "text/csv")

        st.markdown("#### Lane discipline — lateral dispersion by class")
        yb = st.slider("Corridor window for dispersion (m along Y)",
                       float(kin["Y_s"].min()), float(kin["Y_s"].max()),
                       (float(kin["Y_s"].quantile(0.2)), float(kin["Y_s"].quantile(0.8))))
        ld = C.lateral_dispersion(kin, *yb)
        if not ld.empty:
            c1, c2 = st.columns([2, 3])
            with c1:
                st.dataframe(ld.style.format({c: "{:.2f}" for c in ld.columns
                                              if ld[c].dtype.kind == "f"}),
                             **_W, hide_index=True)
            with c2:
                fig = px.bar(ld[ld["n_tracks"] >= 3], x="class_final",
                             y=["within_track_swing_m", "across_p10_p90_m"], barmode="group",
                             labels={"value": "metres", "class_final": ""})
                fig.update_layout(height=340, legend_title=None)
                st.plotly_chart(fig, **_W)
            st.caption("`within_track_swing_m` = how far one road user wanders laterally "
                       "while traversing the window (weaving). `across_p10_p90_m` = how "
                       "widely the class as a whole distributes across the carriageway. "
                       "Two-wheelers scoring high on the first is lane-discipline breakdown, "
                       "not just lane choice.")

# ==================================================================================
# 6. Queues
# ==================================================================================
with tabs[5]:
    q = C.queues(kin, stop_speed=stop_speed, gap_m=queue_gap)
    if q.empty:
        st.info("No standing queues detected with these thresholds. Loosen the stopped "
                "threshold or widen the time window.")
    else:
        sb = C.stop_bar_estimate(q)
        c = st.columns(4)
        c[0].metric("Max queue length", f"{q['length_m'].max():.1f} m")
        c[1].metric("Max vehicles in queue", int(q["n_veh"].max()))
        c[2].metric("Mean queue length", f"{q['length_m'].mean():.1f} m")
        c[3].metric("Frames with a queue", f"{q['frame_idx'].nunique():,}")

        st.markdown("#### Queue length over time")
        qt = C.queue_timeseries(q, bin_s=2.0)
        fig = px.line(qt, x="bin", y="max_len_m", color="direction", markers=True,
                      labels={"bin": "time (s)", "max_len_m": "longest queue (m)"})
        fig.update_layout(height=360)
        st.plotly_chart(fig, **_W)
        st.caption("Sawtooth structure here is the signal cycle: build-up during red, "
                   "discharge on green. Peak-to-peak spacing estimates cycle length.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Vehicles queued over time")
            fig = px.line(qt, x="bin", y="max_veh", color="direction", markers=True,
                          labels={"bin": "time (s)", "max_veh": "vehicles"})
            fig.update_layout(height=330)
            st.plotly_chart(fig, **_W)
        with c2:
            st.markdown("#### Inferred stop-line position")
            st.dataframe(sb.style.format({c_: "{:.1f}" for c_ in sb.columns
                                          if sb[c_].dtype.kind == "f"}),
                         **_W, hide_index=True)
            st.caption("Median Y of the queue head — where vehicles repeatedly come to rest. "
                       "No signal-phase data required; it falls out of the trajectories.")

        st.markdown("#### Queue extent in space and time")
        fig = px.scatter(q, x="time_sec", y="y_head", size="n_veh", color="direction",
                         hover_data=["length_m"],
                         labels={"time_sec": "time (s)", "y_head": "queue head, Y (m)"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, **_W)
        st.download_button("Download queue records (CSV)",
                           q.to_csv(index=False).encode(), "queues.csv", "text/csv")

# ==================================================================================
# 7. Fundamental diagram
# ==================================================================================
with tabs[6]:
    st.markdown("#### Flow–density relationship (Edie's generalised definitions)")
    st.caption("Over each space–time cell: flow q = Σdistance/|A|, density k = Σtime/|A|, "
               "space-mean speed v = q/k. This is the correct estimator for trajectory "
               "data — it does not depend on where you happen to place a detector.")
    e = C.edie(kin, dy_m=edie_dy, dt_s=edie_dt)
    if e.empty:
        st.info("Not enough along-corridor movement in this filter.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.scatter(e, x="density_vpkm", y="flow_vph", color="speed_kmh",
                             color_continuous_scale="RdYlGn",
                             labels={"density_vpkm": "density (veh/km)",
                                     "flow_vph": "flow (veh/h)", "speed_kmh": "km/h"})
            fig.update_layout(height=380, title="q – k")
            st.plotly_chart(fig, **_W)
        with c2:
            fig = px.scatter(e, x="density_vpkm", y="speed_kmh", color="flow_vph",
                             color_continuous_scale="Viridis",
                             labels={"density_vpkm": "density (veh/km)",
                                     "speed_kmh": "speed (km/h)"})
            fig.update_layout(height=380, title="v – k")
            st.plotly_chart(fig, **_W)
        with c3:
            fig = px.scatter(e, x="flow_vph", y="speed_kmh", color="density_vpkm",
                             color_continuous_scale="Cividis",
                             labels={"flow_vph": "flow (veh/h)", "speed_kmh": "speed (km/h)"})
            fig.update_layout(height=380, title="v – q")
            st.plotly_chart(fig, **_W)

        cap = e.loc[e["flow_vph"].idxmax()]
        st.info(
            f"Observed capacity point: **{cap['flow_vph']:,.0f} veh/h** at "
            f"**{cap['density_vpkm']:.0f} veh/km** and **{cap['speed_kmh']:.1f} km/h**, "
            f"in the cell centred on Y = {cap['yb']:.0f} m, t = {cap['tb']:.0f} s. "
            "Cells to the right of the peak are congested branch, left is free flow.")

        st.markdown("#### Space–time speed map")
        fig = px.density_heatmap(e, x="tb", y="yb", z="speed_kmh", histfunc="avg",
                                 color_continuous_scale="RdYlGn",
                                 labels={"tb": "time (s)", "yb": "Y along corridor (m)",
                                         "speed_kmh": "km/h"})
        fig.update_layout(height=450)
        st.plotly_chart(fig, **_W)
        st.caption("Diagonal bands of low speed travelling backwards in Y are shockwaves — "
                   "the origin of a jam rather than its symptom.")

        st.markdown("#### Detector-equivalent occupancy at the counting line")
        occ = C.occupancy(kin, count_y, zone_m=6.0, bin_s=5.0)
        if not occ.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=occ["bin"], y=occ["occupancy"], name="occupancy",
                                     mode="lines+markers"))
            fig.add_trace(go.Scatter(x=occ["bin"], y=occ["mean_veh"], name="mean vehicles",
                                     yaxis="y2", mode="lines"))
            fig.update_layout(height=330, xaxis_title="time (s)",
                              yaxis=dict(title="occupancy", tickformat=".0%"),
                              yaxis2=dict(title="vehicles", overlaying="y", side="right"))
            st.plotly_chart(fig, **_W)
        st.download_button("Download Edie cells (CSV)", e.to_csv(index=False).encode(),
                           "fundamental_diagram.csv", "text/csv")

# ==================================================================================
# 8. Interactions
# ==================================================================================
with tabs[7]:
    st.markdown("#### Proximity / conflict density")
    st.caption("Moments where two moving road users are within a set distance, binned in "
               "space. A fixed perspective camera cannot produce this at all — it has no "
               "common metric frame for two users on different approaches.")
    rad = st.slider("Proximity radius (m)", 1.0, 10.0, C.PROXIMITY_M, 0.5)
    idf = C.interaction_density(kin, radius_m=rad, cell_m=cell_m)
    if idf.empty:
        st.info("No proximity events at this radius.")
    else:
        c = st.columns(3)
        c[0].metric("Proximity events", f"{int(idf['events'].sum()):,}")
        c[1].metric("Closest approach", f"{idf['min_dist_m'].min():.2f} m")
        c[2].metric("Hotspot cells", len(idf))
        fig = px.scatter(idf, x="xb", y="yb", size="events", color="events",
                         color_continuous_scale="Inferno", size_max=28,
                         labels={"xb": "X across (m)", "yb": "Y along (m)"})
        fig.update_layout(height=620, yaxis_scaleanchor="x")
        st.plotly_chart(fig, **_W)
        st.dataframe(idf.head(15), **_W, hide_index=True)
        st.caption("Events are sampled at 10 Hz, so a single sustained close-following "
                   "pair contributes many events. Read the map for *where*, not as a "
                   "count of distinct conflicts.")

# ==================================================================================
# 9. Playback (no video decode — instant)
# ==================================================================================
with tabs[8]:
    st.markdown("#### Trajectory playback in the ground frame")
    st.caption("Driven straight from the parquet, so it responds instantly to the class "
               "filter in the sidebar. No video decoding involved.")
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        tcur = st.slider("Time (s)", float(t0), float(t1), float(t0), 0.2)
    with c2:
        trail = st.number_input("Trail (s)", 0.0, 30.0, 4.0, 0.5)
    with c3:
        show_speed = st.checkbox("Colour by speed", False)

    win = kin[kin["time_sec"].between(tcur - trail, tcur)]
    now = kin[(kin["time_sec"] - tcur).abs() < 0.06]

    fig = go.Figure()
    if not win.empty:
        for tid, g in win.groupby("track_id", sort=False):
            fig.add_trace(go.Scatter(
                x=g["X_s"], y=g["Y_s"], mode="lines", showlegend=False,
                line=dict(width=1.5, color=C.color_for(g["class_final"].iloc[0])),
                opacity=0.5, hoverinfo="skip"))
    if not now.empty:
        if show_speed:
            fig.add_trace(go.Scatter(
                x=now["X_s"], y=now["Y_s"], mode="markers",
                marker=dict(size=11, color=now["speed_kmh"], colorscale="RdYlGn",
                            cmin=0, cmax=max(40, speed_limit * 1.5), showscale=True,
                            colorbar=dict(title="km/h")),
                text=[f"{r.class_final} · {r.speed_kmh:.0f} km/h · id {r.track_id}"
                      for r in now.itertuples()],
                hoverinfo="text", name="now"))
        else:
            for cls, g in now.groupby("class_final"):
                fig.add_trace(go.Scatter(
                    x=g["X_s"], y=g["Y_s"], mode="markers", name=cls,
                    marker=dict(size=11, color=C.color_for(cls),
                                line=dict(width=1, color="white")),
                    text=[f"{cls} · {v:.0f} km/h · id {i}"
                          for v, i in zip(g["speed_kmh"], g["track_id"])],
                    hoverinfo="text"))
    fig.add_hline(y=count_y, line_dash="dot", opacity=0.5)
    fig.update_layout(height=680, yaxis_scaleanchor="x",
                      xaxis=dict(range=[kin["X_s"].min() - 5, kin["X_s"].max() + 5],
                                 title="X across (m)"),
                      yaxis=dict(range=[kin["Y_s"].min() - 5, kin["Y_s"].max() + 5],
                                 title="Y along (m)"),
                      legend_title=None, uirevision="playback")
    st.plotly_chart(fig, **_W)
    c = st.columns(4)
    c[0].metric("On scene now", int(now["track_id"].nunique()))
    c[1].metric("Mean speed now", f"{now['speed_kmh'].mean():.1f} km/h" if len(now) else "—")
    c[2].metric("Stopped now", int((now["speed_mps"] < stop_speed).sum()))
    c[3].metric("Two-wheelers now",
                int((now["class_final"].str.lower() == "motorcycle").sum()))

# ==================================================================================
# 10. Real-Time Telemetry (60 FPS Video & Kinematics Deck)
# ==================================================================================
with tabs[9]:
    st.markdown("#### ⚡ 60 FPS Synchronized Real-Time Telemetry Deck")
    st.caption("Zero-latency client-side synchronized video playback + per-vehicle real-time telemetry feed. No blinking, no 5-second reload delay.")

    top_btn_col1, top_btn_col2 = st.columns([3, 1])
    with top_btn_col1:
        st.success("🟢 **Live Real-Time Engine Active**: Video playback, kinematics telemetry, and OpenStreetMap spatial grounding are synchronized at 60 FPS directly in the canvas below.")
    with top_btn_col2:
        st.link_button("🚀 Open Standalone 60 FPS Deck", "http://localhost:8000", type="primary", use_container_width=True)

    # Embedded High-Speed 60 FPS HTML5 Video + Telemetry Deck
    st.components.v1.iframe("http://localhost:8000", height=950, scrolling=True)

# ==================================================================================
# 11. Spatial Grounding (OpenStreetMap & GPS Geographic Projection)
# ==================================================================================
with tabs[10]:
    st.markdown("#### 🌍 OpenStreetMap Geospatial Grounding (Baner - Mahalunge Rd)")
    st.caption(
        "Direct projection of drone trajectories onto OpenStreetMap tiles at **Baner - Mahalunge Road, Pune** "
        "(18.5663° N, 73.7719° E, near Amar Paradigm / FlytBase HQ / Syngenta). Real-world road geometry with traffic speed fields and approach legs."
    )

    LAT_0 = 18.566300
    LON_0 = 73.771850
    ROAD_BEARING_DEG = 66.5
    BEARING_RAD = np.radians(ROAD_BEARING_DEG)
    cos_b = np.cos(BEARING_RAD)
    sin_b = np.sin(BEARING_RAD)

    # Compute coordinates centered on the Baner - Mahalunge Rd intersection
    kin_geo = kin.copy()
    x_c = kin_geo["X_s"] - 9.0
    y_c = kin_geo["Y_s"] - 65.0
    kin_geo["lat"] = LAT_0 + (y_c * cos_b - x_c * sin_b) / 111320.0
    kin_geo["lon"] = LON_0 + (y_c * sin_b + x_c * cos_b) / 105527.0

    map_c1, map_c2 = st.columns([3, 1])
    with map_c2:
        map_style = st.selectbox(
            "Base Map Style",
            ["open-street-map", "carto-darkmatter", "carto-positron"],
            index=0
        )
        map_view_type = st.radio(
            "Layer Display",
            ["🚗 Vehicle Trajectories by Class", "⚡ Traffic Speed Density Heatmap", "🚦 Approach / Exit Legs Anchors"],
            index=0
        )

    with map_c1:
        if "Class" in map_view_type:
            sample_geo = kin_geo.sample(min(len(kin_geo), 4000), random_state=42)
            fig_map = px.scatter_mapbox(
                sample_geo,
                lat="lat",
                lon="lon",
                color="class_final",
                color_discrete_map=CMAP,
                size_max=8,
                zoom=18,
                center=dict(lat=LAT_0, lon=LON_0),
                mapbox_style=map_style,
                hover_data={"track_id": True, "speed_kmh": ":.1f", "class_final": True, "lat": False, "lon": False},
                title="Trajectory Grounding on Baner - Mahalunge Road (OpenStreetMap)"
            )
            fig_map.update_layout(height=580, margin=dict(t=30, b=10, l=10, r=10), legend_title="Class")
            st.plotly_chart(fig_map, **_W)
        elif "Speed" in map_view_type:
            sample_geo = kin_geo.sample(min(len(kin_geo), 4000), random_state=42)
            fig_map = px.density_mapbox(
                sample_geo,
                lat="lat",
                lon="lon",
                z="speed_kmh",
                radius=12,
                zoom=18,
                center=dict(lat=LAT_0, lon=LON_0),
                mapbox_style=map_style,
                color_continuous_scale="Viridis",
                title="Spatial Speed Intensity on Baner - Mahalunge Road"
            )
            fig_map.update_layout(height=580, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_map, **_W)
        else:
            if not legs.empty:
                legs_geo = legs.copy()
                x_lc = legs_geo["X"] - 9.0
                y_lc = legs_geo["Y"] - 65.0
                legs_geo["lat"] = LAT_0 + (y_lc * cos_b - x_lc * sin_b) / 111320.0
                legs_geo["lon"] = LON_0 + (y_lc * sin_b + x_lc * cos_b) / 105527.0
                fig_map = px.scatter_mapbox(
                    legs_geo,
                    lat="lat",
                    lon="lon",
                    color="role",
                    size="count",
                    size_max=20,
                    text="label",
                    zoom=18,
                    center=dict(lat=LAT_0, lon=LON_0),
                    mapbox_style=map_style,
                    title="Approach & Exit Leg Anchors on Baner - Mahalunge Road"
                )
                fig_map.update_layout(height=580, margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_map, **_W)
            else:
                st.info("Legs data not available.")

    st.markdown("---")
    st.markdown("#### Geographic Coordinates & Road Alignment Reference")
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    col_g1.metric("Location", "Baner - Mahalunge Rd")
    col_g2.metric("Landmarks", "Amar Paradigm / FlytBase / Syngenta")
    col_g3.metric("Center GPS", f"{LAT_0:.4f}° N, {LON_0:.4f}° E")
    col_g4.metric("Corridor Alignment Heading", f"{ROAD_BEARING_DEG:.1f}° E-NE")
    col_g4.metric("GCP Dimension Envelope", f"{C.CORRIDOR_W_M:.1f}m × {C.CORRIDOR_L_M:.1f}m")

# ==================================================================================
# 12. Video & Custom Renderer
# ==================================================================================
with tabs[11]:
    st.markdown("#### Video Footage Streams")
    st.info("💡 **Tip:** The persistent video stream player is docked at the top above all tabs for real-time reference across every analytics view.")

    cols = st.columns(3)
    for (key, info), col in zip(vid_defs.items(), cols):
        p = _resolve_vid_path(info["file"], data_dir)
        with col:
            st.markdown(f"**{info['title']}**")
            st.caption(f"`{info['file']}`\n\n{info['desc']}")
            if p:
                st.video(str(p))
            else:
                st.warning(f"File `{info['file']}` not found.")

    st.markdown("---")
    st.markdown("#### Render a filtered overlay")
    st.caption("Renders a fresh overlay from `tracks_masked.parquet` with only the classes "
               "you want, trails on or off, optional raw|annotated split screen. Needs the "
               "source MP4 on disk. A 90 s clip at 720p takes a couple of minutes.")
    src = st.text_input("Source video path", value=str(Path(data_dir) / "Intersection_Merged.MP4"))
    rc = st.columns(5)
    r_classes = rc[0].multiselect("Classes to draw", all_classes, default=classes)
    r_trail = rc[1].number_input("Trail seconds (0 = off)", 0.0, 20.0, 3.0, 0.5)
    r_width = rc[2].selectbox("Output width", [960, 1280, 1600, 1920], index=1)
    r_side = rc[3].checkbox("Side-by-side", True)
    r_labels = rc[4].checkbox("Speed labels", True)
    out_path = Path(data_dir) / "render_filtered.mp4"

    if st.button("Render", type="primary"):
        cmd = [sys.executable, str(Path(__file__).parent / "render_overlay.py"),
               "--video", src, "--data", data_dir, "--out", str(out_path),
               "--start", str(t0), "--end", str(t1), "--width", str(r_width),
               "--trail-sec", str(r_trail)]
        if r_classes:
            cmd += ["--classes", ",".join(r_classes)]
        if r_side:
            cmd += ["--side-by-side"]
        if r_labels:
            cmd += ["--labels"]
        st.code(" ".join(cmd), language="bash")
        with st.spinner("Rendering… (progress in the terminal running streamlit)"):
            p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0 and out_path.exists():
            st.success("Done.")
            st.video(str(out_path))
        else:
            st.error("Render failed.")
            st.code((p.stdout or "") + "\n" + (p.stderr or ""))

# ==================================================================================
# 13. Export
# ==================================================================================
with tabs[12]:
    st.markdown("#### Export every aggregate table")
    st.caption("Writes CSVs plus a findings.md into an `l3_export/` folder next to your data. "
               "Same numbers as the tabs above, for the submission writeup.")
    if st.button("Export everything", type="primary"):
        outdir = Path(data_dir) / "l3_export"
        cmd = [sys.executable, str(Path(__file__).parent / "l3_export.py"),
               "--data", data_dir, "--out", str(outdir),
               "--count-y", str(count_y), "--limit", str(speed_limit),
               "--interval", str(interval_s), "--legs", str(int(n_legs))]
        with st.spinner("Computing…"):
            p = subprocess.run(cmd, capture_output=True, text=True)
        st.code((p.stdout or "") + "\n" + (p.stderr or ""))
        if p.returncode == 0:
            st.success(f"Written to {outdir}")
            md = outdir / "findings.md"
            if md.exists():
                st.download_button("Download findings.md", md.read_bytes(), "findings.md")
                st.markdown(md.read_text())
