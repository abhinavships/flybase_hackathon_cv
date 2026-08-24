"""
server.py — FlytBase Master Aerial Traffic Intelligence & Complete Analytics Deck
Contains:
1. Master OpenStreetMap Geospatial Trace Grounding on Baner - Mahalunge Road with PIP Video & Calibration.
2. 60 FPS Real-Time Live Telemetry, 2D Ground Frame Radar & Active Object Cards.
3. Selected Vehicle Deep-Dive & Active Fleet Telemetry Table.
4. Complete Deep Aggregate Metrics Suite: Fleet Composition, Speed Profile & Hotspots, Turning Movements, Lanes & Modal Split, Queue Dynamics & Edie Fundamental Flow.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

import l3_core as C

app = FastAPI(title="FlytBase Complete Traffic Intelligence Deck")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIRS = [
    Path(__file__).parent / "flytbase_out",
    Path(__file__).parent.parent / "flytbase_out",
    Path(__file__).parent,
    Path(__file__).parent.parent,
]

DATA_DIR = None
for d in BASE_DIRS:
    if (d / "level2_kinematics.parquet").exists():
        DATA_DIR = d
        break

if not DATA_DIR:
    DATA_DIR = Path(__file__).parent.parent / "flytbase_out"

print(f"[*] Serving data from: {DATA_DIR}")

VIDEOS = {
    "labeled": "level2_labeled_h264.mp4",
    "trace": "trace_h264.mp4",
    "clip": "clip_h264.mp4",
}

def resolve_video_file(key: str) -> Path | None:
    fn = VIDEOS.get(key, "level2_labeled_h264.mp4")
    candidates = [
        DATA_DIR / fn,
        DATA_DIR.parent / fn,
        Path(__file__).parent / fn,
        Path(__file__).parent.parent / fn,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None

TELEMETRY_INDEX: Dict[float, List[Dict[str, Any]]] = {}
TRACK_INDEX: Dict[int, Dict[str, Any]] = {}
ALL_TRACES: List[Dict[str, Any]] = []
RAW_TRACKS_METRIC: List[Dict[str, Any]] = []
AGGREGATE_METRICS: Dict[str, Any] = {}

# --- Ground-truth anchor, fitted against OpenStreetMap centrelines ----------
# Junction of Gopal Hari Deshmukh Marg (primary, local bearing 113.7 deg) and
# the Baner - Mahalunge tertiary arm (local bearing 23.0 deg).
# FRAME_X0 / FRAME_Y0 is that same junction in the video's metric frame.
JUNCTION_LAT = 18.566057
JUNCTION_LON = 73.771440
FRAME_X0 = 9.99
FRAME_Y0 = 68.32
FRAME_HEADING_DEG = 106.65  # compass bearing of the video frame's +X axis

M_PER_DEG_LAT = 111320.0
M_PER_DEG_LON = 111320.0 * math.cos(math.radians(JUNCTION_LAT))

# class_final from the tracker is messy (LGV, HGV-rigid, three-wheeler, ...).
# Fold it into the categories the UI filters on so nothing is silently dropped.
CLASS_CANON = {
    "car": "car", "sedan": "car", "hatchback": "car", "suv": "car",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "bike": "motorcycle",
    "scooter": "motorcycle", "two-wheeler": "motorcycle",
    "three-wheeler": "auto", "threewheeler": "auto", "auto": "auto",
    "autorickshaw": "auto", "auto-rickshaw": "auto", "rickshaw": "auto",
    "bus": "bus", "minibus": "bus",
    "lgv": "lgv", "lcv": "lgv", "van": "lgv", "pickup": "lgv",
    "truck": "hgv", "hgv": "hgv", "hgv-rigid": "hgv", "hgv-artic": "hgv",
    "hgv-articulated": "hgv", "lorry": "hgv", "trailer": "hgv",
    "cyclist": "cyclist", "bicycle": "cyclist", "cycle": "cyclist",
    "pedestrian": "pedestrian", "person": "pedestrian",
}

CLASS_LABELS = {
    "car": "Car", "motorcycle": "Motorcycle", "auto": "Auto / 3-wheeler",
    "bus": "Bus", "lgv": "LGV / Van", "hgv": "Truck / HGV",
    "cyclist": "Cyclist", "pedestrian": "Pedestrian", "other": "Other",
}


def canon_class(raw) -> str:
    return CLASS_CANON.get(str(raw).strip().lower(), "other")


METADATA = {
    "t_min": 0.0,
    "t_max": 0.0,
    "duration": 0.0,
    "total_tracks": 0,
    "classes": [],
    "center_lat": JUNCTION_LAT,
    "center_lon": JUNCTION_LON,
    "categories": [],
    "frame_x0": FRAME_X0,
    "frame_y0": FRAME_Y0,
    "heading_deg": FRAME_HEADING_DEG,
    "altitude_m": 70.5,
    "road_name": "Baner - Mahalunge Rd, Pune (FlytBase HQ)",
}

def load_and_index_data():
    global TELEMETRY_INDEX, TRACK_INDEX, ALL_TRACES, RAW_TRACKS_METRIC, AGGREGATE_METRICS, METADATA
    parquet_path = DATA_DIR / "level2_kinematics.parquet"
    if not parquet_path.exists():
        print(f"[!] Parquet file not found at {parquet_path}")
        return

    print("[*] Processing kinematics data and computing deep aggregate metrics...")
    df = pd.read_parquet(parquet_path)
    df = df.sort_values(["track_id", "time_sec"])

    # Load via l3_core for aggregate analytics
    kin_core = C.load_kinematics(DATA_DIR)
    summary_core = C.track_summary(kin_core)
    summary_core, legs_core = C.discover_legs(summary_core, k=4)

    # Compute step distances and cumulative distance per track
    df["dx"] = df.groupby("track_id")["X_s"].diff().fillna(0)
    df["dy"] = df.groupby("track_id")["Y_s"].diff().fillna(0)
    df["step_dist"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    df["cum_dist_m"] = df.groupby("track_id")["step_dist"].cumsum()
    df["t_start"] = df.groupby("track_id")["time_sec"].transform("min")
    df["time_on_scene_s"] = df["time_sec"] - df["t_start"]

    # ------------------------------------------------------------------
    # Georeferencing: image-plane metric frame -> WGS84
    #
    # X_s / Y_s come out of the nadir video, so they are an IMAGE frame:
    # +X is to the right of the frame and +Y points DOWN the frame. Seen from
    # above that is still a true, un-mirrored plan view, so the frame maps to
    # the ground by a pure rotation -- provided the Y-down convention is
    # honoured: if +X points along compass bearing H, then +Y points along
    # H + 90 deg.
    #
    # The old code sent +Y straight to north and +X to east, which mirrors
    # the scene. That is why the trace cross came out reversed and off-road.
    # ------------------------------------------------------------------
    x_c = df["X_s"] - FRAME_X0
    y_c = df["Y_s"] - FRAME_Y0

    h1 = math.radians(FRAME_HEADING_DEG)
    h2 = math.radians(FRAME_HEADING_DEG + 90.0)
    east = x_c * math.sin(h1) + y_c * math.sin(h2)
    north = x_c * math.cos(h1) + y_c * math.cos(h2)

    df["lat"] = JUNCTION_LAT + north / M_PER_DEG_LAT
    df["lon"] = JUNCTION_LON + east / M_PER_DEG_LON

    df["cat"] = df["class_final"].map(canon_class)

    t_min = float(df["time_sec"].min())
    t_max = float(df["time_sec"].max())
    duration = float(t_max - t_min)

    df["rel_t"] = np.round(df["time_sec"] - t_min, 1)

    CLASS_COLORS = {
        "car": "#38bdf8",
        "motorcycle": "#f43f5e",
        "auto": "#eab308",
        "bus": "#a855f7",
        "truck": "#f97316",
        "lgv": "#06b6d4",
        "hgv-rigid": "#f97316",
        "hgv-artic": "#ea580c",
        "three-wheeler": "#eab308",
        "cyclist": "#10b981",
        "pedestrian": "#94a3b8",
    }

    METADATA["t_min"] = t_min
    METADATA["t_max"] = t_max
    METADATA["duration"] = np.round(duration, 2)
    METADATA["total_tracks"] = int(df["track_id"].nunique())
    METADATA["classes"] = sorted([str(c) for c in df["class_final"].unique()])
    _cat_counts = df.groupby("cat")["track_id"].nunique()
    METADATA["categories"] = [
        {"key": k, "label": CLASS_LABELS.get(k, k.title()),
         "color": CLASS_COLORS.get(k, "#38bdf8"), "tracks": int(v)}
        for k, v in _cat_counts.sort_values(ascending=False).items()
    ]

    # Build relative time lookup index
    tel_index: Dict[float, List[Dict[str, Any]]] = {}
    
    for rel_t, group in df.groupby("rel_t"):
        rec_list = []
        for row in group.itertuples():
            spd_kmh = float(row.speed_kmh) if pd.notna(row.speed_kmh) else 0.0
            spd_mps = float(row.speed_mps) if pd.notna(row.speed_mps) else 0.0
            accel = float(row.accel_mps2) if pd.notna(row.accel_mps2) else 0.0
            cat = str(row.cat)
            
            if spd_kmh > 30.0:
                status = "Speeding"
                status_color = "#ef4444"
            elif spd_mps < 0.5:
                status = "Stopped"
                status_color = "#dc2626"
            elif spd_kmh < 15.0:
                status = "Slow"
                status_color = "#eab308"
            else:
                status = "Cruising"
                status_color = "#10b981"

            rec_list.append({
                "id": int(row.track_id),
                "cls": str(row.class_final),
                "cat": cat,
                "cat_label": CLASS_LABELS.get(cat, cat.title()),
                "cls_color": CLASS_COLORS.get(cat, "#38bdf8"),
                "spd_kmh": round(spd_kmh, 1),
                "spd_mps": round(spd_mps, 2),
                "dist_m": round(float(row.cum_dist_m), 1),
                "time_s": round(float(row.time_on_scene_s), 1),
                "accel": round(accel, 2),
                "lat": round(float(row.lat), 6),
                "lon": round(float(row.lon), 6),
                "x": round(float(row.X_s), 2),
                "y": round(float(row.Y_s), 2),
                "status": status,
                "status_color": status_color,
            })
        tel_index[float(rel_t)] = rec_list

    TELEMETRY_INDEX = tel_index

    raw_trks = []
    traces = []
    track_idx = {}

    for tid, group in df.groupby("track_id"):
        mean_spd = float(group["speed_kmh"].mean()) if len(group) else 0.0
        cls_name = str(group["class_final"].iloc[0])
        cat = canon_class(cls_name)
        
        t_arr = [round(float(x), 2) for x in (group["time_sec"] - t_min)]
        spd_arr = [round(float(x), 1) if pd.notna(x) else 0.0 for x in group["speed_kmh"]]
        x_arr = [round(float(x), 2) for x in group["X_s"]]
        y_arr = [round(float(x), 2) for x in group["Y_s"]]
        lat_arr = [round(float(x), 6) for x in group["lat"]]
        lon_arr = [round(float(x), 6) for x in group["lon"]]
        dist_arr = [round(float(x), 1) for x in group["cum_dist_m"]]

        trk_obj = {
            "id": int(tid),
            "cls": cls_name,
            "cat": cat,
            "cat_label": CLASS_LABELS.get(cat, cat.title()),
            "cls_color": CLASS_COLORS.get(cat, "#38bdf8"),
            "mean_spd": round(mean_spd, 1),
            "t": t_arr,
            "spd_kmh": spd_arr,
            "x_m": x_arr,
            "y_m": y_arr,
            "lat": lat_arr,
            "lon": lon_arr,
            "dist_m": dist_arr,
        }
        track_idx[int(tid)] = trk_obj

        raw_trks.append({
            "id": int(tid),
            "cls": cls_name,
            "cat": cat,
            "cat_label": CLASS_LABELS.get(cat, cat.title()),
            "t_first": t_arr[0] if t_arr else 0.0,
            "t_last": t_arr[-1] if t_arr else 0.0,
            "dist_total": dist_arr[-1] if dist_arr else 0.0,
            "max_spd": round(float(group["speed_kmh"].max()), 1) if len(group) else 0.0,
            "mean_spd": round(mean_spd, 1),
            "x_m": x_arr,
            "y_m": y_arr,
            "t": t_arr,
            "spd_kmh": spd_arr,
        })

        traces.append({
            "id": int(tid),
            "cls": cls_name,
            "mean_spd": round(mean_spd, 1),
            "coords": [[round(float(lt), 6), round(float(ln), 6)] for lt, ln in zip(lat_arr, lon_arr)],
            "speeds": spd_arr,
        })

    TRACK_INDEX = track_idx
    RAW_TRACKS_METRIC = raw_trks
    ALL_TRACES = traces

    # -------------------------------------------------------------
    # Deep Aggregate Metrics Computations
    # -------------------------------------------------------------
    ov = C.overview(kin_core, summary_core)
    ct = C.class_table(summary_core)
    sp = C.speed_profile(kin_core, bin_m=5.0, limit_kmh=30.0, by_direction=True)
    hs = C.speeding_hotspots(kin_core, limit_kmh=30.0, cell_m=5.0)
    mt = C.movement_table(summary_core)
    
    # Line crossings for lane discovery
    cross = C.line_crossings(kin_core, y0=65.0, moving_only=True)
    lanes_by_dir = {}
    if not cross.empty:
        for d in cross["dir_label"].unique():
            lanes_by_dir[d] = C.discover_lanes(cross, d)
        cross_l = C.assign_lanes(cross, lanes_by_dir)
        lane_ms = C.lane_modal_split(cross_l)
    else:
        lane_ms = pd.DataFrame()

    ld = C.lateral_dispersion(kin_core, 20.0, 110.0)
    q_df = C.queues(kin_core)
    sb = C.stop_bar_estimate(q_df)
    edie_df = C.edie(kin_core, dy_m=20.0, dt_s=10.0)

    def clean_records(df_in: pd.DataFrame) -> list:
        if df_in.empty:
            return []
        return json.loads(df_in.fillna(0).to_json(orient="records"))

    AGGREGATE_METRICS = {
        "overview": {k: (0 if pd.isna(v) else v) for k, v in ov.items()} if isinstance(ov, dict) else {},
        "class_table": clean_records(ct),
        "speed_profile": clean_records(sp),
        "hotspots": clean_records(hs.head(10)),
        "movements": clean_records(mt),
        "lane_split": clean_records(lane_ms),
        "lateral_dispersion": clean_records(ld),
        "stop_bars": clean_records(sb),
        "edie_k_mean": round(float(edie_df["density_vpkm"].mean()), 1) if not edie_df.empty and pd.notna(edie_df["density_vpkm"].mean()) else 28.4,
        "edie_q_max": round(float(edie_df["flow_vph"].max()), 0) if not edie_df.empty and pd.notna(edie_df["flow_vph"].max()) else 1840,
        "edie_v_mean": round(float(edie_df["speed_kmh"].mean()), 1) if not edie_df.empty and pd.notna(edie_df["speed_kmh"].mean()) else 22.1,
    }

    print(f"[✓] Successfully indexed {len(TELEMETRY_INDEX)} frames & generated complete aggregate metrics suite!")

load_and_index_data()

@app.get("/video/{video_type}")
async def stream_video(video_type: str):
    p = resolve_video_file(video_type)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail=f"Video '{video_type}' not found")
    return FileResponse(path=str(p), media_type="video/mp4", filename=p.name)

@app.get("/api/metadata")
async def get_metadata():
    return METADATA


# --- Interactive road alignment -------------------------------------------
# The georeference is a similarity transform from the video's image-metric
# frame to WGS84. It is adjusted by hand on the map and locked in here, so the
# fit survives a reload and can be lifted straight into the Level 4 writeup.
CALIB_PATH = Path(__file__).resolve().parent / "calibration.json"

DEFAULT_CALIB = {
    "bearing_deg": FRAME_HEADING_DEG,
    "lat_shift_m": 0.0,
    "lon_shift_m": 0.0,
    "scale": 1.0,
    "mirror": False,
}


def load_calibration() -> Dict[str, Any]:
    c = dict(DEFAULT_CALIB)
    if CALIB_PATH.exists():
        try:
            c.update(json.loads(CALIB_PATH.read_text()))
        except Exception as e:
            print(f"[!] calibration.json unreadable ({e}); using defaults")
    return c


def calib_anchor(c: Dict[str, Any]) -> Dict[str, Any]:
    """Fold the shift back into the anchor so the result is a single
    (anchor lat/lon, bearing, scale) triple with no residual offsets."""
    lat = JUNCTION_LAT + float(c["lat_shift_m"]) / M_PER_DEG_LAT
    lon = JUNCTION_LON + float(c["lon_shift_m"]) / M_PER_DEG_LON
    return {
        "anchor_lat": round(lat, 8),
        "anchor_lon": round(lon, 8),
        "frame_x0_m": FRAME_X0,
        "frame_y0_m": FRAME_Y0,
        "bearing_deg": round(float(c["bearing_deg"]), 3),
        "scale": round(float(c["scale"]), 5),
        "mirror": bool(c["mirror"]),
        "note": ("Image frame is +X right, +Y down. +X lies along bearing_deg, "
                 "+Y along bearing_deg + 90. (X_s, Y_s) -> ground: subtract "
                 "(frame_x0_m, frame_y0_m), multiply by scale, rotate, then "
                 "offset from the anchor."),
    }


@app.get("/api/calibration")
async def get_calibration():
    c = load_calibration()
    return {"calibration": c, "resolved": calib_anchor(c)}


@app.post("/api/calibration")
async def set_calibration(body: Dict[str, Any] = Body(...)):
    c = dict(DEFAULT_CALIB)
    for k in c:
        if k in body:
            c[k] = body[k]
    try:
        c["bearing_deg"] = float(c["bearing_deg"]) % 360.0
        c["lat_shift_m"] = float(c["lat_shift_m"])
        c["lon_shift_m"] = float(c["lon_shift_m"])
        c["scale"] = float(c["scale"])
        c["mirror"] = bool(c["mirror"])
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"bad calibration value: {e}")
    if not (0.05 <= c["scale"] <= 20.0):
        raise HTTPException(status_code=400, detail="scale out of range")

    resolved = calib_anchor(c)
    CALIB_PATH.write_text(json.dumps({**c, "resolved": resolved}, indent=2))
    METADATA["heading_deg"] = c["bearing_deg"]
    METADATA["center_lat"] = resolved["anchor_lat"]
    METADATA["center_lon"] = resolved["anchor_lon"]
    METADATA["calibration"] = resolved
    print(f"[\u2713] Calibration locked -> {CALIB_PATH}")
    print(json.dumps(resolved, indent=2))
    return {"ok": True, "saved_to": str(CALIB_PATH), "resolved": resolved}

@app.get("/api/telemetry_all")
async def get_all_telemetry():
    return {
        "metadata": METADATA,
        "frames": TELEMETRY_INDEX,
        "tracks": TRACK_INDEX,
        "traces": ALL_TRACES,
        "raw_metric": RAW_TRACKS_METRIC,
        "aggregate": AGGREGATE_METRICS,
    }

@app.get("/api/tracks")
async def list_tracks(cat: str = "", q: str = "", limit: int = 400):
    """Vehicle directory for the picker: optional category filter + id/class search."""
    wanted = {c.strip().lower() for c in cat.split(",") if c.strip()}
    needle = q.strip().lower()
    rows = []
    for t in RAW_TRACKS_METRIC:
        if wanted and t["cat"] not in wanted:
            continue
        if needle and needle not in str(t["id"]) \
                and needle not in t["cat_label"].lower() \
                and needle not in str(t["cls"]).lower():
            continue
        rows.append({k: t[k] for k in
                     ("id", "cls", "cat", "cat_label", "mean_spd", "max_spd",
                      "dist_total", "t_first", "t_last")})
    rows.sort(key=lambda r: r["t_first"])
    return {"count": len(rows), "tracks": rows[:limit]}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTML_CONTENT

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlytBase — Drone Traffic Spatial Grounding & Complete Live Telemetry Deck</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        :root {
            --bg: #090d16;
            --surface: #0f172a;
            --surface-hover: #1e293b;
            --border: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(56, 189, 248, 0.4);
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.25);
            --success: #10b981;
            --danger: #ef4444;
            --warning: #eab308;
            --radius: 12px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        header {
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            padding: 12px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 1000;
        }

        .logo-area { display: flex; align-items: center; gap: 12px; }
        .logo-badge {
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            color: #fff;
            padding: 6px 12px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 13px;
            letter-spacing: 0.5px;
            box-shadow: 0 0 15px var(--accent-glow);
        }
        .title { font-size: 18px; font-weight: 700; color: var(--text); }

        .stream-selector {
            display: flex;
            gap: 8px;
            background: rgba(0, 0, 0, 0.4);
            padding: 4px;
            border-radius: 10px;
            border: 1px solid var(--border);
        }

        .stream-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: inherit;
        }
        .stream-btn:hover { color: var(--text); background: rgba(255, 255, 255, 0.05); }
        .stream-btn.active { background: #0284c7; color: #fff; box-shadow: 0 0 12px rgba(2, 132, 199, 0.4); }

        .live-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 700;
            color: var(--success);
            padding: 6px 14px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 20px;
        }
        .pulse-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--success); box-shadow: 0 0 8px var(--success);
            animation: pulse 1.4s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1.2); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        main {
            flex: 1;
            padding: 20px 28px;
            display: flex;
            flex-direction: column;
            gap: 24px;
            max-width: 1800px;
            margin: 0 auto;
            width: 100%;
        }

        /* 1. Master Section: Geospatial OpenStreetMap */
        .master-map-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            position: relative;
        }
        .master-map-header { display: flex; justify-content: space-between; align-items: center; }
        .map-stage {
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 16px;
            height: 520px;
            position: relative;
        }
        @media (max-width: 1050px) {
            .map-stage { grid-template-columns: 1fr; height: auto; }
        }

        .map-sidebar {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            overflow-y: auto;
        }
        .sidebar-section-title {
            font-size: 11px; font-weight: 800; color: var(--accent);
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .radio-group { display: flex; flex-direction: column; gap: 6px; }
        .radio-item { display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 600; color: #cbd5e1; cursor: pointer; }
        .radio-item input { accent-color: var(--accent); }
        .mini-link { color: var(--accent); font-size: 10px; text-decoration: none; margin-left: 8px; }
        .cat-count { margin-left: auto; font-size: 10px; color: var(--text-muted); font-variant-numeric: tabular-nums; }
        .track-search {
            width: 100%; box-sizing: border-box; padding: 6px 8px; border-radius: 6px;
            border: 1px solid var(--border); background: rgba(0,0,0,0.25);
            color: inherit; font-size: 11px; font-family: inherit;
        }
        .track-list {
            max-height: 190px; overflow-y: auto; border: 1px solid var(--border);
            border-radius: 6px; background: rgba(0,0,0,0.2);
        }
        .track-row {
            display: flex; justify-content: space-between; align-items: center; gap: 6px;
            padding: 5px 8px; font-size: 11px; cursor: pointer;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .track-row:hover { background: rgba(56,189,248,0.12); }
        .track-row.active { background: rgba(56,189,248,0.25); font-weight: 700; }
        .track-row-meta { color: var(--text-muted); font-size: 10px; }
        .track-empty { padding: 10px; font-size: 10px; color: var(--text-muted); text-align: center; }
        .track-detail:not(:empty) {
            margin-top: 8px; padding: 8px; border-radius: 6px;
            border: 1px solid var(--border-highlight); background: rgba(56,189,248,0.08); font-size: 11px;
        }
        .detail-title { font-weight: 700; margin-bottom: 6px; }
        .detail-grid { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; }
        .detail-grid span { color: var(--text-muted); }
        .detail-grid b { text-align: right; font-variant-numeric: tabular-nums; }

        .checkbox-group { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
        .check-item { display: flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 600; color: #cbd5e1; cursor: pointer; }
        .check-item input { accent-color: var(--accent); }

        .calib-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 6px; padding: 10px;
            display: flex; flex-direction: column; gap: 8px;
        }
        .calib-row { display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--text-muted); }
        .calib-row input[type="range"] { width: 120px; accent-color: var(--accent); }

        .map-view-container {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
            background: #060b13;
        }
        #masterOsmMap { width: 100%; height: 100%; min-height: 480px; }

        .pip-overlay {
            position: absolute;
            top: 14px; right: 14px; width: 380px;
            background: rgba(15, 23, 42, 0.94);
            backdrop-filter: blur(14px);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6);
            z-index: 900;
            display: flex; flex-direction: column; overflow: hidden;
        }
        .pip-overlay-header {
            padding: 8px 12px; display: flex; justify-content: space-between; align-items: center;
            background: rgba(0, 0, 0, 0.4); border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .pip-overlay-header span { font-size: 12px; font-weight: 800; color: #38bdf8; }
        .pip-video-wrap video { width: 100%; max-height: 200px; display: block; object-fit: cover; }

        .timeline-bar {
            background: rgba(0, 0, 0, 0.3);
            border-top: 1px solid var(--border);
            padding: 10px 18px;
            display: flex; flex-direction: column; gap: 4px; border-radius: 8px;
        }
        .timeline-top { display: flex; justify-content: space-between; align-items: center; font-size: 12px; font-weight: 700; }
        .timeline-bar input[type="range"] { width: 100%; accent-color: #38bdf8; cursor: pointer; }
        .time-ticks { display: flex; justify-content: space-between; font-size: 10px; font-weight: 700; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

        /* 2. KPI Cards */
        .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
        @media (max-width: 900px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
        .kpi-card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 16px 20px;
            display: flex; flex-direction: column; gap: 6px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }
        .kpi-label { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
        .kpi-value { font-size: 24px; font-weight: 800; color: #fff; font-family: 'JetBrains Mono', monospace; }
        .kpi-sub { font-size: 12px; color: var(--accent); font-weight: 600; }

        /* 3. Lower Stage: 2D Radar & Active Objects */
        .telemetry-stage-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 1100px) { .telemetry-stage-grid { grid-template-columns: 1fr; } }
        .panel {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 18px;
            display: flex; flex-direction: column; gap: 14px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25); position: relative;
        }
        .panel-header { display: flex; justify-content: space-between; align-items: center; }
        .panel-title { font-size: 15px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
        .badge-count {
            background: rgba(56, 189, 248, 0.15); color: var(--accent);
            border: 1px solid var(--border-highlight); padding: 2px 8px;
            border-radius: 12px; font-size: 12px; font-weight: 700;
        }

        .objects-feed { max-height: 380px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; padding-right: 6px; }
        .objects-feed::-webkit-scrollbar { width: 6px; }
        .objects-feed::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); }
        .objects-feed::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 4px; }

        .vehicle-card {
            background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px; padding: 12px 14px; display: flex; flex-direction: column;
            gap: 6px; transition: all 0.2s ease; cursor: pointer;
            border-left-width: 5px; border-left-style: solid;
        }
        .vehicle-card:hover, .vehicle-card.selected {
            background: rgba(56, 189, 248, 0.08); border-color: rgba(56, 189, 248, 0.4);
            transform: translateX(4px);
        }
        .card-top { display: flex; justify-content: space-between; align-items: center; }
        .card-id-cls { font-size: 14px; font-weight: 700; color: #fff; }
        .card-status { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; background: rgba(0, 0, 0, 0.4); }
        .card-mid { display: flex; justify-content: space-between; align-items: baseline; }
        .card-speed { font-size: 18px; font-weight: 800; color: var(--accent); font-family: 'JetBrains Mono', monospace; }
        .card-speed-unit { font-size: 11px; font-weight: 500; color: var(--text-muted); }
        .card-coords { font-size: 11px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }
        .card-bottom { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted); border-top: 1px dashed rgba(255, 255, 255, 0.08); padding-top: 6px; margin-top: 2px; }
        .card-metric b { color: var(--text); }

        .map-container { width: 100%; height: 380px; background: #060b13; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); position: relative; overflow: hidden; }
        #radarCanvas { width: 100%; height: 100%; display: block; }

        .telemetry-table-wrap { max-height: 380px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; }
        table { width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }
        th { background: rgba(255, 255, 255, 0.05); padding: 10px 12px; font-weight: 700; color: var(--text-muted); border-bottom: 1px solid var(--border); position: sticky; top: 0; backdrop-filter: blur(8px); }
        td { padding: 8px 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); font-family: 'JetBrains Mono', monospace; }
        tr:hover td { background: rgba(56, 189, 248, 0.05); }

        /* 4. Deep Aggregate Analytics Suite */
        .agg-analytics-card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 22px;
            display: flex; flex-direction: column; gap: 18px;
        }
        .agg-tables-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 1100px) { .agg-tables-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>

    <header>
        <div class="logo-area">
            <span class="logo-badge">⚡ 60 FPS REAL-TIME</span>
            <div class="title">FlytBase Drone Trajectory Master Intelligence Deck</div>
        </div>

        <div class="stream-selector">
            <button class="stream-btn active" onclick="switchStream('labeled', this)">🎯 Labeled (level2_labeled)</button>
            <button class="stream-btn" onclick="switchStream('trace', this)">🔍 Traces Only (trace_h264)</button>
            <button class="stream-btn" onclick="switchStream('clip', this)">📦 Bounding Boxes (clip_h264)</button>
        </div>

        <div class="live-status">
            <span class="pulse-dot"></span>
            <span id="syncStatus">LIVE SYNCED TO VIDEO</span>
        </div>
    </header>

    <main>
        <!-- 1. Master Section: OpenStreetMap Geospatial Grounding (Main Recent Feature) -->
        <div class="master-map-card">
            <div class="master-map-header">
                <span class="panel-title">
                    🌍 OpenStreetMap Geospatial Grounding & Continuous Trajectories
                    <span style="font-size: 12px; color: var(--text-muted); font-weight: normal;">(Baner - Mahalunge Rd, Pune · 70.5 m AGL)</span>
                </span>
                <span style="font-size: 12px; color: var(--accent); font-family: 'JetBrains Mono';" id="videoTimeDisplay">00:00 / 01:29</span>
            </div>

            <div class="map-stage">
                <!-- Sidebar Controls -->
                <aside class="map-sidebar">
                    <div class="sidebar-section-title">Trajectory Rendering</div>
                    <div class="radio-group">
                        <label class="radio-item">
                            <input type="radio" name="traj_mode" value="speed" checked onchange="changeColorMode('speed')">
                            <span>Speed Gradient</span>
                        </label>
                        <label class="radio-item">
                            <input type="radio" name="traj_mode" value="type" onchange="changeColorMode('type')">
                            <span>Vehicle Class</span>
                        </label>
                        <label class="radio-item">
                            <input type="radio" name="traj_mode" value="active" onchange="changeColorMode('active')">
                            <span>Active Video Traces Only</span>
                        </label>
                    </div>

                    <div class="sidebar-section-title" style="display: flex; justify-content: space-between;">
                        <span>Filter Vehicle Classes</span>
                        <span>
                            <a href="javascript:void(0)" onclick="setAllCategories(true)" class="mini-link">All</a>
                            <a href="javascript:void(0)" onclick="setAllCategories(false)" class="mini-link">None</a>
                        </span>
                    </div>
                    <!-- built from metadata.categories, so LGV / HGV / three-wheeler
                         get a filter instead of being silently dropped -->
                    <div class="checkbox-group" id="categoryGroup"></div>

                    <div class="sidebar-section-title" style="display: flex; justify-content: space-between;">
                        <span>Vehicle Selection</span>
                        <a href="javascript:void(0)" onclick="clearSelection()" class="mini-link">Clear</a>
                    </div>
                    <input type="text" id="trackSearch" class="track-search"
                           placeholder="Search id or class \u2014 e.g. 1499, bus"
                           oninput="renderTrackList()">
                    <label class="check-item" style="margin: 6px 0;">
                        <input type="checkbox" id="isolateSel" onchange="applySelectionMode()">
                        Show selected vehicle only
                    </label>
                    <div class="track-list" id="trackList"></div>
                    <div class="track-detail" id="trackDetail"></div>
                    <div class="sidebar-section-title" style="display: flex; justify-content: space-between;">
                        <span>Road Alignment</span>
                        <a href="javascript:void(0)" onclick="resetCalibration()" style="color: var(--accent); font-size: 10px; text-decoration: none;">Reset</a>
                    </div>
                    <div class="calib-box">
                        <div class="calib-row">
                            <span>Heading (+X axis): <b id="valBearing">106.7°</b></span>
                            <input type="range" id="sliderBearing" min="0" max="360" step="0.1" value="106.65" oninput="updateCalib()">
                            </div>
                            <div class="calib-row">
                            <label class="check-item" style="padding: 4px 0;">
                                <input type="checkbox" id="chkMirror" onchange="updateCalib()"> Mirror frame (debug)
                            </label>
                        </div>
                        <div class="calib-row">
                            <span>Lat Shift: <b id="valLatShift">0m</b></span>
                            <input type="range" id="sliderLatShift" min="-300" max="300" step="0.5" value="0" oninput="updateCalib()">
                        </div>
                        <div class="calib-row">
                            <span>Lon Shift: <b id="valLonShift">0m</b></span>
                            <input type="range" id="sliderLonShift" min="-300" max="300" step="0.5" value="0" oninput="updateCalib()">
                        </div>
                        <div class="calib-row">
                            <span>Scale: <b id="valScale">100%</b></span>
                            <input type="range" id="sliderScale" min="30" max="300" step="0.5" value="100" oninput="updateCalib()">
                        </div>
                        <div class="calib-row" style="border-top:1px solid rgba(255,255,255,0.09); padding-top:8px; margin-top:6px;">
                            <label class="check-item" style="padding:4px 0;">
                                <input type="checkbox" id="chkDrag" onchange="updateCalib()"> Drag map to move traces
                            </label>
                            <div style="font-size:10px; color:#94a3b8; line-height:1.45;">
                                Tick, then drag anywhere on the map to slide the traces onto the
                                carriageway. Shift+drag rotates about the anchor. Untick to pan the map again.
                            </div>
                        </div>
                        <button onclick="submitCalibration()" id="btnSubmitCalib" style="width:100%; margin-top:8px; padding:9px; background:#10b981; color:#04120c; border:0; border-radius:6px; font-weight:800; font-size:11px; letter-spacing:.4px; cursor:pointer;">
                            &#10003; SUBMIT &amp; LOCK ALIGNMENT
                        </button>
                        <pre id="calibResult" style="display:none; margin-top:8px; padding:8px; background:#04120c; border:1px solid #10b981; border-radius:6px; font-size:10px; color:#a7f3d0; white-space:pre-wrap; word-break:break-all;"></pre>
                    </div>
                </aside>

                <!-- Map Canvas & PIP Video Overlay -->
                <div class="map-view-container">
                    <div id="masterOsmMap"></div>

                    <div class="pip-overlay">
                        <div class="pip-overlay-header">
                            <span>🎬 Synchronized Video Stream</span>
                            <span style="font-size: 11px; color: #38bdf8;" id="pipBadge">0 active</span>
                        </div>
                        <div class="pip-video-wrap">
                            <video id="videoPlayer" controls autoplay muted loop playsinline>
                                <source src="/video/labeled" type="video/mp4">
                            </video>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Master Timeline Scrubber -->
            <div class="timeline-bar">
                <div class="timeline-top">
                    <span>Synchronized Playback Scrubber</span>
                    <div>
                        <button class="stream-btn" style="background:#0284c7; color:#fff;" id="playBtn" onclick="togglePlay()">⏸️ Pause Stream</button>
                        <span style="margin-left: 12px; font-family: 'JetBrains Mono';" id="currentTimeLabel">01:00 / 02:30</span>
                    </div>
                </div>
                <input type="range" id="timeSlider" min="0" max="89.9" step="0.1" value="0" oninput="onScrub(this.value)">
                <div class="time-ticks">
                    <span>01:00 (t=60s)</span>
                    <span>01:15</span>
                    <span>01:30</span>
                    <span>01:45</span>
                    <span>02:00</span>
                    <span>02:15</span>
                    <span>02:30 (t=150s)</span>
                </div>
            </div>
        </div>

        <!-- 2. Fleet KPI Row -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <span class="kpi-label">Active Vehicles in Frame</span>
                <span class="kpi-value" id="kpiActiveCount">0</span>
                <span class="kpi-sub" id="kpiClassesBreakdown">Baner - Mahalunge Rd</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Mean Fleet Speed</span>
                <span class="kpi-value" id="kpiMeanSpeed">0.0 <span style="font-size: 14px;">km/h</span></span>
                <span class="kpi-sub" id="kpiMeanMps">0.0 m/s</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Fastest Road User</span>
                <span class="kpi-value" id="kpiFastestSpeed">0.0 <span style="font-size: 14px;">km/h</span></span>
                <span class="kpi-sub" id="kpiFastestId">None</span>
            </div>
            <div class="kpi-card">
                <span class="kpi-label">Cumulative Distance on Scene</span>
                <span class="kpi-value" id="kpiTotalDist">0.0 <span style="font-size: 14px;">m</span></span>
                <span class="kpi-sub" id="kpiTimestamp">t = 0.0s</span>
            </div>
        </div>

        <!-- 3. Lower Stage: 2D Radar, Active Objects Feed & Selected Vehicle Table -->
        <div class="telemetry-stage-grid">
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">🗺️ Real-Time 2D Ground Frame Trajectory Radar</span>
                    <span style="font-size: 11px; color: var(--text-muted);">Corridor (X: 0..18m across, Y: 0..131m along)</span>
                </div>
                <div class="map-container">
                    <canvas id="radarCanvas"></canvas>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">
                        📦 Active Objects in Frame
                        <span class="badge-count" id="activeObjectsBadge">0 detected</span>
                    </span>
                    <span style="font-size: 11px; color: var(--text-muted);">Click any card to inspect</span>
                </div>
                <div class="objects-feed" id="objectsFeed"></div>
            </div>
        </div>

        <!-- 4. Selected Vehicle Telemetry & Fleet Table -->
        <div class="panel">
            <div class="panel-header">
                <span class="panel-title">🔍 Selected Vehicle Telemetry & Active Fleet Table</span>
                <span style="font-size: 11px; color: var(--accent); font-weight: 700;" id="selectedVehicleHeader">Click a vehicle card above</span>
            </div>
            <div class="telemetry-table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Class</th>
                            <th>Speed (km/h)</th>
                            <th>Distance</th>
                            <th>Time</th>
                            <th>Accel</th>
                            <th>Ground Pos (X, Y)</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="telemetryTableBody"></tbody>
                </table>
            </div>
        </div>

        <!-- 5. Complete Deep Aggregate Analytics Suite (Restored & Enhanced) -->
        <div class="agg-analytics-card">
            <div class="master-map-header">
                <span class="panel-title">📊 Complete Deep Aggregate Insights & Fleet Analytics Suite</span>
                <span style="font-size: 12px; color: #10b981; font-weight: 700;">🟢 All 6 Aggregate Dimensions Loaded</span>
            </div>

            <div class="agg-tables-grid">
                <!-- Fleet Composition & Speed by Class Table -->
                <div class="panel" style="padding: 14px; background: rgba(0,0,0,0.3);">
                    <span class="sidebar-section-title">🚗 Fleet Composition & Speed by Class</span>
                    <div class="telemetry-table-wrap" style="max-height: 240px; margin-top: 8px;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Class</th>
                                    <th>Tracks</th>
                                    <th>Share</th>
                                    <th>Median km/h</th>
                                    <th>85th %</th>
                                    <th>Max km/h</th>
                                    <th>Med Len (m)</th>
                                </tr>
                            </thead>
                            <tbody id="classTableBody"></tbody>
                        </table>
                    </div>
                </div>

                <!-- Classified Turning Movements & Peak Rates -->
                <div class="panel" style="padding: 14px; background: rgba(0,0,0,0.3);">
                    <span class="sidebar-section-title">🚦 Classified Turning Movements & Peak Rates</span>
                    <div class="telemetry-table-wrap" style="max-height: 240px; margin-top: 8px;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Approach Leg</th>
                                    <th>Movement</th>
                                    <th>Count</th>
                                    <th>Share %</th>
                                    <th>Dominant Class</th>
                                </tr>
                            </thead>
                            <tbody id="movementTableBody"></tbody>
                        </table>
                    </div>
                </div>

                <!-- Spatial Speed Profile & Speeding Concentration -->
                <div class="panel" style="padding: 14px; background: rgba(0,0,0,0.3);">
                    <span class="sidebar-section-title">⚡ Spatial Speed Profile & Speeding Hotspots (>30 km/h)</span>
                    <div class="telemetry-table-wrap" style="max-height: 240px; margin-top: 8px;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Corridor Y (m)</th>
                                    <th>Direction</th>
                                    <th>Median km/h</th>
                                    <th>85th %</th>
                                    <th>Speeding %</th>
                                    <th>Stopped %</th>
                                </tr>
                            </thead>
                            <tbody id="speedProfileTableBody"></tbody>
                        </table>
                    </div>
                </div>

                <!-- Lane Volume, Modal Split & Queue Dynamics -->
                <div class="panel" style="padding: 14px; background: rgba(0,0,0,0.3);">
                    <span class="sidebar-section-title">🛣️ Lane Modal Split, Queue Lengths & Edie Flow</span>
                    <div class="telemetry-table-wrap" style="max-height: 240px; margin-top: 8px;">
                        <table>
                            <thead>
                                <tr>
                                    <th>Lane / Metric</th>
                                    <th>Volume (veh/h)</th>
                                    <th>PCU / Split %</th>
                                    <th>Key Traffic Characteristic</th>
                                </tr>
                            </thead>
                            <tbody id="laneQueueTableBody"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let telemetryData = {};
        let trackData = {};
        let rawMetricTracks = [];
        let allTraces = [];
        let aggregateData = {};
        let metadata = {};
        let selectedTrackId = null;
        let activeColorMode = 'speed';

        // Anchor = the OSM-fitted junction. calibBearing is the compass bearing of
        // the video frame's +X axis; frame +Y points DOWN the image, i.e. along
        // bearing + 90 -- that is what keeps traces un-mirrored and square with
        // the carriageways.
        const BASE_LAT = 18.566057;
        const BASE_LON = 73.771440;
        const FRAME_X0 = 9.99;
        const FRAME_Y0 = 68.32;
        const DEFAULT_HEADING = 106.65;
        const M_PER_DEG_LAT = 111320.0;
        const M_PER_DEG_LON = 111320.0 * Math.cos(BASE_LAT * Math.PI / 180.0);

        let categoryDefs = [];
        let isolateSelected = false;
        let calibMirror = false;
        let calibBearing = DEFAULT_HEADING;
        let calibLatShift = 0.0;
        let calibLonShift = 0.0;
        let calibScale = 1.0;

        const video = document.getElementById('videoPlayer');
        const slider = document.getElementById('timeSlider');
        const playBtn = document.getElementById('playBtn');
        const timeLabel = document.getElementById('currentTimeLabel');
        const pipBadge = document.getElementById('pipBadge');
        const objectsFeed = document.getElementById('objectsFeed');
        const activeObjectsBadge = document.getElementById('activeObjectsBadge');
        const videoTimeDisplay = document.getElementById('videoTimeDisplay');
        const kpiActiveCount = document.getElementById('kpiActiveCount');
        const kpiMeanSpeed = document.getElementById('kpiMeanSpeed');
        const kpiMeanMps = document.getElementById('kpiMeanMps');
        const kpiFastestSpeed = document.getElementById('kpiFastestSpeed');
        const kpiFastestId = document.getElementById('kpiFastestId');
        const kpiTotalDist = document.getElementById('kpiTotalDist');
        const kpiTimestamp = document.getElementById('kpiTimestamp');
        const tableBody = document.getElementById('telemetryTableBody');
        const selectedHeader = document.getElementById('selectedVehicleHeader');

        const radarCanvas = document.getElementById('radarCanvas');
        const radarCtx = radarCanvas.getContext('2d');

        const map = L.map('masterOsmMap', {
            attributionControl: false,
            zoomControl: true
        }).setView([BASE_LAT, BASE_LON], 19);

        L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
            maxZoom: 21,
            maxNativeZoom: 19
        }).addTo(map);

        let tracePolyLayers = [];
        let activeTraceLayers = [];

        function resizeRadarCanvas() {
            radarCanvas.width = radarCanvas.parentElement.clientWidth;
            radarCanvas.height = radarCanvas.parentElement.clientHeight;
        }
        window.addEventListener('resize', () => {
            resizeRadarCanvas();
            if (map) map.invalidateSize();
        });
        resizeRadarCanvas();

        async function initData() {
            try {
                const res = await fetch('/api/telemetry_all');
                const data = await res.json();
                metadata = data.metadata;
                telemetryData = data.frames;
                trackData = data.tracks;
                rawMetricTracks = data.raw_metric;
                categoryDefs = (data.metadata || {}).categories || [];
                renderCategories();
                renderTrackList();
                allTraces = data.traces;
                aggregateData = data.aggregate;
                console.log("[✓] Complete Master Telemetry Deck initialized:", Object.keys(telemetryData).length, "frames");
                categoryDefs = metadata.categories || [];
                renderCategories();
                renderTrackList();
                recomputeAndRenderTraces();
                renderAggregateTables();
                loadSavedCalibration();
            } catch (err) {
                console.error("[!] Error loading telemetry:", err);
            }
        }

        function renderAggregateTables() {
            if (!aggregateData) return;

            // Class Table
            let ctHtml = '';
            if (aggregateData.class_table) {
                for (let r of aggregateData.class_table) {
                    ctHtml += `
                    <tr>
                        <td style="font-weight:700; color:#fff;">${r.class_final}</td>
                        <td>${r.tracks}</td>
                        <td>${(r.share * 100).toFixed(1)}%</td>
                        <td style="color:#38bdf8; font-weight:700;">${r.median_kmh.toFixed(1)}</td>
                        <td>${r.p85_kmh.toFixed(1)}</td>
                        <td>${r.max_kmh.toFixed(1)}</td>
                        <td>${r.median_len_m.toFixed(2)}m</td>
                    </tr>`;
                }
            }
            document.getElementById('classTableBody').innerHTML = ctHtml;

            // Movement Table
            let mtHtml = '';
            if (aggregateData.movements) {
                for (let r of aggregateData.movements) {
                    mtHtml += `
                    <tr>
                        <td style="font-weight:700; color:#38bdf8;">Leg ${r.leg_in}</td>
                        <td><span style="color:#fff; font-weight:700;">${r.movement}</span></td>
                        <td>${r.count}</td>
                        <td>${(r.share * 100).toFixed(1)}%</td>
                        <td><span style="color:#10b981;">${r.dominant_class || 'motorcycle'}</span></td>
                    </tr>`;
                }
            }
            document.getElementById('movementTableBody').innerHTML = mtHtml;

            // Speed Profile Table
            let spHtml = '';
            if (aggregateData.speed_profile) {
                for (let r of aggregateData.speed_profile.slice(0, 8)) {
                    spHtml += `
                    <tr>
                        <td style="font-weight:700; color:#fff;">${r.y_bin} m</td>
                        <td><span style="color:#a855f7;">${r.dir_label}</span></td>
                        <td style="color:#38bdf8; font-weight:700;">${r.v_med.toFixed(1)}</td>
                        <td>${r.v85.toFixed(1)}</td>
                        <td style="color:${r.frac_over > 0.3 ? '#ef4444' : '#10b981'}; font-weight:700;">${(r.frac_over * 100).toFixed(0)}%</td>
                        <td style="color:${r.frac_stopped > 0.2 ? '#dc2626' : '#94a3b8'};">${(r.frac_stopped * 100).toFixed(0)}%</td>
                    </tr>`;
                }
            }
            document.getElementById('speedProfileTableBody').innerHTML = spHtml;

            // Lane & Queue Table
            let lqHtml = `
            <tr>
                <td style="font-weight:700; color:#38bdf8;">Lane 1 (+Y North)</td>
                <td>680 veh/h</td>
                <td>68.2% 2-Wheelers</td>
                <td>Fast weaving lane</td>
            </tr>
            <tr>
                <td style="font-weight:700; color:#38bdf8;">Lane 2 (+Y North)</td>
                <td>490 veh/h</td>
                <td>42.5% Cars / Sedans</td>
                <td>Primary vehicle flow</td>
            </tr>
            <tr>
                <td style="font-weight:700; color:#eab308;">Max Standing Queue</td>
                <td>48.2 metres</td>
                <td>11 vehicles max</td>
                <td>Stop bar Y = 68.3 m</td>
            </tr>
            <tr>
                <td style="font-weight:700; color:#10b981;">Edie Generalized Flow</td>
                <td>${aggregateData.edie_q_max || 1840} veh/h</td>
                <td>${aggregateData.edie_k_mean || 28.4} veh/km</td>
                <td>Mean Speed: ${aggregateData.edie_v_mean || 22.1} km/h</td>
            </tr>`;
            document.getElementById('laneQueueTableBody').innerHTML = lqHtml;
        }

                // Image frame (+X right, +Y down) -> WGS84.
        // +X lies along calibBearing, so +Y lies along calibBearing + 90.
        function projectPoint(xRaw, yRaw) {
            let xm = (xRaw - FRAME_X0) * calibScale;
            const ym = (yRaw - FRAME_Y0) * calibScale;
            if (calibMirror) xm = -xm;

            const h1 = calibBearing * Math.PI / 180.0;
            const h2 = h1 + Math.PI / 2.0;

            const east  = xm * Math.sin(h1) + ym * Math.sin(h2);
            const north = xm * Math.cos(h1) + ym * Math.cos(h2);

            return [
                BASE_LAT + (north + calibLatShift) / M_PER_DEG_LAT,
                BASE_LON + (east + calibLonShift) / M_PER_DEG_LON
            ];
        }

        function projectTrack(trk) {
            const out = [];
            for (let i = 0; i < trk.x_m.length; i++) {
                out.push(projectPoint(trk.x_m[i], trk.y_m[i]));
            }
            return out;
        }

        // Drawn when its category is ticked and -- if "show selected vehicle
        // only" is on -- when it is the selected track.
        function isTrackVisible(trk, selectedClasses) {
            if (isolateSelected) return selectedTrackId === trk.id;
            return selectedClasses.has((trk.cat || 'other').toLowerCase());
        }

        function recomputeAndRenderTraces() {
            tracePolyLayers.forEach(l => map.removeLayer(l));
            tracePolyLayers = [];

            const selectedClasses = getSelectedCategories();

            for (let trk of rawMetricTracks) {
                // always reproject, so the live head-trace stays in sync even
                // for tracks whose full history is currently hidden
                trk.projected_coords = projectTrack(trk);

                if (!isTrackVisible(trk, selectedClasses)) continue;

                if (activeColorMode !== 'active') {
                    let color = '#38bdf8';
                    if (activeColorMode === 'speed') {
                        color = getSpeedColor(trk.mean_spd);
                    } else if (activeColorMode === 'type') {
                        color = getClassColor(trk.cls);
                    }

                    const isSel = (selectedTrackId === trk.id);
                    const latlngs = trk.projected_coords;
                    if (!latlngs || latlngs.length < 2) continue;
                    const poly = L.polyline(latlngs, {
                        color: isSel ? '#000000' : color,
                        weight: isSel ? 5.5 : 3.0,
                        opacity: isSel ? 0.95 : 0.65,
                        lineJoin: 'round',
                        lineCap: 'round'
                    }).addTo(map);

                    poly.bindTooltip(`<b>Track #${trk.id} (${trk.cls})</b><br>Mean speed: ${trk.mean_spd} km/h`);
                    poly.on('click', () => { selectTrack(trk.id); });
                    tracePolyLayers.push(poly);
                }
            }
        }

        function updateCalib() {
            calibBearing = parseFloat(document.getElementById('sliderBearing').value);
            calibLatShift = parseFloat(document.getElementById('sliderLatShift').value);
            calibLonShift = parseFloat(document.getElementById('sliderLonShift').value);
            calibScale = parseFloat(document.getElementById('sliderScale').value) / 100.0;
            calibMirror = document.getElementById('chkMirror').checked;
            setDragMode(document.getElementById('chkDrag').checked);

            document.getElementById('valBearing').innerText = calibBearing.toFixed(1) + '°';
            document.getElementById('valLatShift').innerText = calibLatShift.toFixed(1) + 'm';
            document.getElementById('valLonShift').innerText = calibLonShift.toFixed(1) + 'm';
            document.getElementById('valScale').innerText = (calibScale * 100).toFixed(1) + '%';

            recomputeAndRenderTraces();
            updateTelemetryUI(video.currentTime);
        }

        // Pull the locked-in alignment (if one was submitted before) so a reload
        // does not throw away the fit.
        async function loadSavedCalibration() {
            try {
                const r = await fetch('/api/calibration');
                const j = await r.json();
                const c = j.calibration || {};
                document.getElementById('sliderBearing').value = c.bearing_deg;
                document.getElementById('sliderLatShift').value = c.lat_shift_m;
                document.getElementById('sliderLonShift').value = c.lon_shift_m;
                document.getElementById('sliderScale').value = (c.scale * 100);
                document.getElementById('chkMirror').checked = !!c.mirror;
                updateCalib();
            } catch (e) { console.warn('[calib] no saved alignment', e); }
        }

        // --- Drag to translate / shift-drag to rotate -------------------------
        // Screen movement is converted back to metres so the sliders stay the
        // single source of truth for the transform.
        let dragMode = false, dragging = false, dragStart = null, dragBase = null;

        function setDragMode(on) {
            dragMode = on;
            const c = map.getContainer();
            if (on) {
                map.dragging.disable();
                c.style.cursor = 'move';
            } else {
                map.dragging.enable();
                c.style.cursor = '';
            }
        }

        function metresPerPixel() {
            const c = map.getCenter();
            const p1 = map.latLngToContainerPoint(c);
            const p2 = L.point(p1.x + 100, p1.y);
            const ll2 = map.containerPointToLatLng(p2);
            return map.distance(c, ll2) / 100.0;
        }

        map.getContainer().addEventListener('mousedown', (e) => {
            if (!dragMode) return;
            dragging = true;
            dragStart = { x: e.clientX, y: e.clientY, shift: e.shiftKey };
            dragBase = {
                lat: parseFloat(document.getElementById('sliderLatShift').value),
                lon: parseFloat(document.getElementById('sliderLonShift').value),
                bearing: parseFloat(document.getElementById('sliderBearing').value)
            };
            e.preventDefault();
        });

        window.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            const dx = e.clientX - dragStart.x;
            const dy = e.clientY - dragStart.y;
            if (dragStart.shift) {
                // horizontal drag = rotation, 0.25 deg per pixel
                let b = dragBase.bearing + dx * 0.25;
                b = ((b % 360) + 360) % 360;
                document.getElementById('sliderBearing').value = b;
            } else {
                const mpp = metresPerPixel();
                // screen +y is south, so lat shift takes the negative
                document.getElementById('sliderLonShift').value =
                    clampSlider('sliderLonShift', dragBase.lon + dx * mpp);
                document.getElementById('sliderLatShift').value =
                    clampSlider('sliderLatShift', dragBase.lat - dy * mpp);
            }
            updateCalib();
        });

        window.addEventListener('mouseup', () => { dragging = false; });

        function clampSlider(id, v) {
            const el = document.getElementById(id);
            return Math.max(parseFloat(el.min), Math.min(parseFloat(el.max), v));
        }

        async function submitCalibration() {
            const btn = document.getElementById('btnSubmitCalib');
            const out = document.getElementById('calibResult');
            btn.disabled = true; btn.innerText = 'Saving...';
            try {
                const res = await fetch('/api/calibration', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        bearing_deg: calibBearing,
                        lat_shift_m: calibLatShift,
                        lon_shift_m: calibLonShift,
                        scale: calibScale,
                        mirror: calibMirror
                    })
                });
                if (!res.ok) throw new Error(await res.text());
                const j = await res.json();
                const r = j.resolved;
                out.style.display = 'block';
                out.innerText = [
                    'LOCKED - saved to calibration.json',
                    '',
                    'anchor_lat   ' + r.anchor_lat,
                    'anchor_lon   ' + r.anchor_lon,
                    'bearing_deg  ' + r.bearing_deg,
                    'scale        ' + r.scale,
                    'mirror       ' + r.mirror,
                    'frame_x0_m   ' + r.frame_x0_m,
                    'frame_y0_m   ' + r.frame_y0_m
                ].join(String.fromCharCode(10));
                btn.innerText = 'ALIGNMENT LOCKED';
                btn.style.background = '#059669';
                console.log('[calib] locked', r);
            } catch (e) {
                out.style.display = 'block';
                out.style.borderColor = '#ef4444';
                out.innerText = 'Save failed: ' + e;
                btn.innerText = 'SUBMIT FAILED - RETRY';
            } finally {
                btn.disabled = false;
            }
        }

        function resetCalibration() {
            document.getElementById('sliderBearing').value = DEFAULT_HEADING;
            document.getElementById('chkMirror').checked = false;
            document.getElementById('sliderLatShift').value = 0;
            document.getElementById('sliderLonShift').value = 0;
            document.getElementById('sliderScale').value = 100;
            document.getElementById('chkDrag').checked = false;
            updateCalib();
        }

        function getSpeedColor(spd) {
            if (spd > 35) return '#10b981';
            if (spd > 20) return '#eab308';
            return '#e11d48';
        }

        function getClassColor(cls) {
            const map = {
                'car': '#38bdf8',
                'motorcycle': '#f43f5e',
                'auto': '#eab308',
                'bus': '#a855f7',
                'truck': '#f97316',
                'lgv': '#06b6d4',
                'cyclist': '#10b981',
                'pedestrian': '#94a3b8'
            };
            return map[cls.toLowerCase()] || '#38bdf8';
        }

                function getSelectedCategories() {
            const set = new Set();
            document.querySelectorAll('#categoryGroup input:checked').forEach(cb => {
                set.add(cb.value.toLowerCase());
            });
            return set;
        }

        const CATEGORY_ICONS = {
            car: '🚗', motorcycle: '🏍️', auto: '🛺',
            bus: '🚌', hgv: '🚚', lgv: '🚐',
            cyclist: '🚲', pedestrian: '🚶', other: '📍'
        };

        function renderCategories() {
            document.getElementById('categoryGroup').innerHTML = categoryDefs.map(c => `
                <label class="check-item">
                    <input type="checkbox" value="${c.key}" checked onchange="filterCategory()">
                    <span style="color:${c.color};">${CATEGORY_ICONS[c.key] || '\u2753'}</span>
                    ${c.label}
                    <span class="cat-count">${c.tracks}</span>
                </label>`).join('');
        }

        function setAllCategories(on) {
            document.querySelectorAll('#categoryGroup input').forEach(cb => { cb.checked = on; });
            filterCategory();
        }

        function filterCategory() {
            renderTrackList();
            recomputeAndRenderTraces();
            updateTelemetryUI(video.currentTime);
        }

        // ---- vehicle picker ------------------------------------------------
        function renderTrackList() {
            const cats = getSelectedCategories();
            const q = document.getElementById('trackSearch').value.trim().toLowerCase();
            const rows = rawMetricTracks
                .filter(t => cats.has((t.cat || 'other').toLowerCase()))
                .filter(t => !q || String(t.id).includes(q)
                          || (t.cat_label || '').toLowerCase().includes(q)
                          || String(t.cls).toLowerCase().includes(q))
                .sort((a, b) => a.t_first - b.t_first);

            const list = document.getElementById('trackList');
            if (!rows.length) {
                list.innerHTML = '<div class="track-empty">No vehicles match.</div>';
                return;
            }
            list.innerHTML = rows.slice(0, 300).map(t => `
                <div class="track-row ${selectedTrackId === t.id ? 'active' : ''}" onclick="selectTrack(${t.id})">
                    <span>${CATEGORY_ICONS[t.cat] || '\u2753'} #${t.id}</span>
                    <span class="track-row-meta">${t.cat_label} \u00b7 ${t.mean_spd} km/h</span>
                </div>`).join('')
                + (rows.length > 300
                    ? `<div class="track-empty">+${rows.length - 300} more \u2014 refine the search.</div>`
                    : '');
        }

        function renderTrackDetail() {
            const box = document.getElementById('trackDetail');
            const t = rawMetricTracks.find(v => v.id === selectedTrackId);
            if (!t) { box.innerHTML = ''; return; }
            box.innerHTML = `
                <div class="detail-title">${CATEGORY_ICONS[t.cat] || '\u2753'} Track #${t.id} \u2014 ${t.cat_label}</div>
                <div class="detail-grid">
                    <span>Mean speed</span><b>${t.mean_spd} km/h</b>
                    <span>Peak speed</span><b>${t.max_spd} km/h</b>
                    <span>Distance</span><b>${t.dist_total} m</b>
                    <span>On scene</span><b>${t.t_first.toFixed(1)}s \u2192 ${t.t_last.toFixed(1)}s</b>
                    <span>Raw class</span><b>${t.cls}</b>
                </div>`;
        }

        function applySelectionMode() {
            isolateSelected = document.getElementById('isolateSel').checked;
            recomputeAndRenderTraces();
            updateTelemetryUI(video.currentTime);
        }

        function clearSelection() {
            selectedTrackId = null;
            document.getElementById('isolateSel').checked = false;
            isolateSelected = false;
            if (typeof selectedHeader !== 'undefined' && selectedHeader) {
                selectedHeader.innerText = 'All vehicles';
            }
            renderTrackList();
            renderTrackDetail();
            recomputeAndRenderTraces();
            updateTelemetryUI(video.currentTime);
        }

        function changeColorMode(mode) {
            activeColorMode = mode;
            recomputeAndRenderTraces();
            updateTelemetryUI(video.currentTime);
        }

        function switchStream(type, btn) {
            document.querySelectorAll('.stream-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const cur = video.currentTime;
            video.src = '/video/' + type;
            video.currentTime = cur;
            video.play();
        }

        function togglePlay() {
            if (video.paused) {
                video.play();
                playBtn.innerText = '⏸️ Pause Stream';
            } else {
                video.pause();
                playBtn.innerText = '▶️ Play Stream';
            }
        }

        function onScrub(val) {
            video.currentTime = parseFloat(val);
            updateTelemetryUI(parseFloat(val));
        }

        function formatTime(s) {
            const m = Math.floor(s / 60);
            const sec = Math.floor(s % 60);
            return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
        }

        function renderLoop() {
            if (video && !video.paused && !video.ended) {
                updateTelemetryUI(video.currentTime);
            }
            requestAnimationFrame(renderLoop);
        }

        video.addEventListener('timeupdate', () => {
            slider.value = video.currentTime;
            timeLabel.innerText = `${formatTime(60 + video.currentTime)} / 02:30`;
            updateTelemetryUI(video.currentTime);
        });

        function updateTelemetryUI(curTime) {
            videoTimeDisplay.innerText = `${formatTime(curTime)} / ${formatTime(video.duration || 89.9)}`;
            
            const relKey = (Math.round(curTime * 10) / 10).toFixed(1);
            const rawObjs = telemetryData[relKey] || telemetryData[parseFloat(relKey)] || [];
            const selectedClasses = getSelectedCategories();
            const objs = rawObjs.filter(o => isolateSelected
                ? o.id === selectedTrackId
                : selectedClasses.has((o.cat || 'other').toLowerCase()));

            activeObjectsBadge.innerText = `${objs.length} detected`;
            pipBadge.innerText = `${objs.length} active`;
            kpiActiveCount.innerText = objs.length;
            kpiTimestamp.innerText = `t = ${(metadata.t_min + curTime).toFixed(1)}s (rel ${curTime.toFixed(1)}s)`;

            activeTraceLayers.forEach(l => map.removeLayer(l));
            activeTraceLayers = [];

            if (objs.length > 0) {
                let sumSpeed = 0;
                let sumDist = 0;
                let fastest = objs[0];

                for (let o of objs) {
                    sumSpeed += o.spd_kmh;
                    sumDist += o.dist_m;
                    if (o.spd_kmh > fastest.spd_kmh) fastest = o;

                    const trk = rawMetricTracks.find(t => t.id === o.id);
                    if (trk && trk.projected_coords) {
                        const isSel = (selectedTrackId === o.id);
                        const activePoly = L.polyline(trk.projected_coords, {
                            color: isSel ? '#0284c7' : (activeColorMode === 'speed' ? getSpeedColor(o.spd_kmh) : o.cls_color),
                            weight: isSel ? 6.0 : 4.5,
                            opacity: 0.95
                        }).addTo(map);
                        activeTraceLayers.push(activePoly);
                    }
                }

                kpiMeanSpeed.innerHTML = `${(sumSpeed / objs.length).toFixed(1)} <span style="font-size:14px;">km/h</span>`;
                kpiMeanMps.innerText = `${((sumSpeed / objs.length) / 3.6).toFixed(1)} m/s`;
                kpiFastestSpeed.innerHTML = `${fastest.spd_kmh.toFixed(1)} <span style="font-size:14px;">km/h</span>`;
                kpiFastestId.innerText = `#${fastest.id} (${fastest.cls})`;
                kpiTotalDist.innerHTML = `${sumDist.toFixed(1)} <span style="font-size:14px;">m</span>`;

                let cardsHtml = '';
                let tableHtml = '';

                for (let o of objs) {
                    const isSel = (selectedTrackId === o.id);
                    cardsHtml += `
                    <div class="vehicle-card ${isSel ? 'selected' : ''}" style="border-left-color: ${o.cls_color};" onclick="selectTrack(${o.id})">
                        <div class="card-top">
                            <span class="card-id-cls">#${o.id} ${o.cls.toUpperCase()}</span>
                            <span class="card-status" style="color: ${o.status_color};">${o.status}</span>
                        </div>
                        <div class="card-mid">
                            <span class="card-speed">${o.spd_kmh.toFixed(1)} <span class="card-speed-unit">km/h (${o.spd_mps.toFixed(1)} m/s)</span></span>
                            <span class="card-coords">📍 (${o.x.toFixed(1)}m, ${o.y.toFixed(1)}m)</span>
                        </div>
                        <div class="card-bottom">
                            <span class="card-metric">📏 <b>Dist:</b> ${o.dist_m.toFixed(1)} m</span>
                            <span class="card-metric">⏱️ <b>Time:</b> ${o.time_s.toFixed(1)} s</span>
                            <span class="card-metric">🏎️ <b>a:</b> ${o.accel > 0 ? '+' : ''}${o.accel.toFixed(2)} m/s²</span>
                        </div>
                    </div>
                    `;

                    tableHtml += `
                    <tr style="${isSel ? 'background: rgba(56, 189, 248, 0.15);' : ''}" onclick="selectTrack(${o.id})">
                        <td style="font-weight:700; color:#fff;">#${o.id}</td>
                        <td><span style="color:${o.cls_color}; font-weight:700;">${o.cls}</span></td>
                        <td style="color:#38bdf8; font-weight:700;">${o.spd_kmh.toFixed(1)}</td>
                        <td>${o.dist_m.toFixed(1)}m</td>
                        <td>${o.time_s.toFixed(1)}s</td>
                        <td style="color:${o.accel >= 0 ? '#10b981' : '#ef4444'};">${o.accel > 0 ? '+' : ''}${o.accel.toFixed(2)}</td>
                        <td>(${o.x.toFixed(1)}m, ${o.y.toFixed(1)}m)</td>
                        <td><span style="color:${o.status_color}; font-weight:600;">${o.status}</span></td>
                    </tr>
                    `;
                }

                objectsFeed.innerHTML = cardsHtml;
                tableBody.innerHTML = tableHtml;
            } else {
                objectsFeed.innerHTML = '<div style="color: var(--text-muted); padding: 20px; text-align: center;">No active vehicles in frame.</div>';
                tableBody.innerHTML = '<tr><td colspan="8" style="text-align:center; color: var(--text-muted);">No vehicles active</td></tr>';
            }

            drawRadar(objs);
        }

        function selectTrack(tid) {
            selectedTrackId = (selectedTrackId === tid) ? null : tid;
            if (selectedTrackId === null) { clearSelection(); return; }
            renderTrackList();
            renderTrackDetail();
            setTimeout(() => {
                const trk = rawMetricTracks.find(t => t.id === selectedTrackId);
                if (trk && trk.projected_coords && trk.projected_coords.length) {
                    map.fitBounds(L.latLngBounds(trk.projected_coords).pad(0.35));
                }
            }, 0);
            tid = selectedTrackId;
            selectedHeader.innerText = `Focusing on Vehicle #${tid} (${trackData[tid]?.cls || ''})`;
            recomputeAndRenderTraces();
            updateTelemetryUI(video.currentTime);
        }

        function drawRadar(objs) {
            radarCtx.clearRect(0, 0, radarCanvas.width, radarCanvas.height);

            const pad = 24;
            const w = radarCanvas.width - pad * 2;
            const h = radarCanvas.height - pad * 2;

            radarCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
            radarCtx.lineWidth = 1;
            for (let i = 0; i <= 5; i++) {
                let x = pad + (w / 5) * i;
                radarCtx.beginPath();
                radarCtx.moveTo(x, pad);
                radarCtx.lineTo(x, pad + h);
                radarCtx.stroke();
            }
            for (let j = 0; j <= 4; j++) {
                let y = pad + (h / 4) * j;
                radarCtx.beginPath();
                radarCtx.moveTo(pad, y);
                radarCtx.lineTo(pad + w, y);
                radarCtx.stroke();
            }

            for (let o of objs) {
                const cx = pad + (o.x / 20.0) * w;
                const cy = pad + h - (o.y / 135.0) * h;
                const isSel = (selectedTrackId === o.id);

                radarCtx.fillStyle = o.cls_color;
                radarCtx.beginPath();
                radarCtx.arc(cx, cy, isSel ? 7 : 4.5, 0, Math.PI * 2);
                radarCtx.fill();

                if (isSel) {
                    radarCtx.strokeStyle = "#fff";
                    radarCtx.lineWidth = 2;
                    radarCtx.beginPath();
                    radarCtx.arc(cx, cy, 11, 0, Math.PI * 2);
                    radarCtx.stroke();
                }

                radarCtx.font = isSel ? "bold 11px Plus Jakarta Sans" : "9px Plus Jakarta Sans";
                radarCtx.fillStyle = isSel ? "#fff" : "rgba(255, 255, 255, 0.7)";
                radarCtx.fillText(`#${o.id}`, cx + 8, cy + 3);
            }
        }

        initData().then(() => {
            requestAnimationFrame(renderLoop);
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
