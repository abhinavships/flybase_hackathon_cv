"""
l3_core.py — Level 3 "Aggregate Insight" analytics over drone trajectory data.

Pure functions. No Streamlit, no plotting. Everything returns DataFrames so it can be
driven from the dashboard (app.py), from a notebook, or from the exporter (l3_export.py).

Input (from flytbase_out/):
    level2_kinematics.parquet   per-frame per-track ground coords + kinematics  [REQUIRED]
    level2_per_track.csv        one row per road user                            [optional]
    tracks_masked.parquet       image-space boxes (only needed for video render) [optional]

Coordinate frame (per CONTEXT.md §7):
    X_s = metres ACROSS the carriageway (GCP quad spans 0..17.9 m)
    Y_s = metres ALONG the carriageway  (GCP quad spans 0..130.91 m)
    Bearing convention used here: 0 deg = +Y, 90 deg = +X  (i.e. atan2(dX, dY)).
    Tracks outside the GCP quad extrapolate beyond those ranges; that is expected.

Caveat baked in everywhere (CONTEXT.md §4): the GCP quad is a thin strip, so ALONG-track
(Y) distances and speeds are reliable, ACROSS-track (X) distances are approximate.
Lane-level numbers therefore carry a warning flag.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Configuration — edit freely, all of it is surfaced in the dashboard sidebar too
# ----------------------------------------------------------------------------------

CORRIDOR_W_M = 17.9      # GCP quad width  (across carriageway)
CORRIDOR_L_M = 130.91    # GCP quad length (along corridor)

MOVING_MIN_DISP_M = 20.0  # net displacement above which a track is a real road user
STOP_SPEED_MPS = 0.5      # below this a vehicle counts as stopped (queue detection)
QUEUE_GAP_M = 12.0        # gap along Y that separates two distinct queues
QUEUE_MIN_VEH = 3         # minimum stopped vehicles to call it a queue
SPEED_LIMIT_KMH = 30.0    # urban arterial working assumption; adjust in UI
PROXIMITY_M = 3.0         # pair distance that counts as an "interaction"

# Turn sign: with bearing = atan2(dX, dY), a POSITIVE bearing change means the vehicle
# swung toward +X. Whether that is a real-world left or right depends on how the GCP
# quad was digitised. Flip this after eyeballing one turning vehicle in the video.
TURN_SIGN = +1            # +1 => positive bearing change is labelled "right"

THROUGH_DEG = 35.0        # |turn| below this = through movement
UTURN_DEG = 150.0         # |turn| above this = U-turn

# PCU factors, IRC:106-style for Indian urban roads. Approximate — edit to taste.
PCU = {
    "pedestrian": 0.0,
    "bicycle": 0.4,
    "cyclist": 0.4,
    "motorcycle": 0.5,
    "motor": 0.5,
    "three-wheeler": 1.2,
    "tricycle": 1.2,
    "awning-tricycle": 1.2,
    "car": 1.0,
    "van": 1.4,
    "lgv": 1.4,
    "hgv-rigid": 2.2,
    "truck": 2.2,
    "bus": 2.2,
    "hgv-artic": 3.7,
}

VRU = {"pedestrian", "bicycle", "cyclist", "motorcycle"}

CLASS_COLORS = {
    "motorcycle": "#f4a261",
    "car": "#4c9be8",
    "pedestrian": "#e76f51",
    "LGV": "#2a9d8f",
    "HGV-rigid": "#8e7dbe",
    "HGV-artic": "#6a4c93",
    "three-wheeler": "#e9c46a",
    "bicycle": "#90be6d",
    "cyclist": "#90be6d",
    "bus": "#577590",
    "van": "#2a9d8f",
    "truck": "#8e7dbe",
}


def color_for(cls: str) -> str:
    return CLASS_COLORS.get(str(cls), "#9aa0a6")


def pcu_for(cls: str) -> float:
    return PCU.get(str(cls).strip().lower(), 1.0)


# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------

KIN_REQUIRED = ["frame_idx", "time_sec", "track_id", "class_final", "X_s", "Y_s"]


def find_data_dir(hint: str | Path | None = None) -> Path:
    """Locate flytbase_out/ without making the user think about it."""
    cands = []
    if hint:
        cands.append(Path(hint))
    cands += [
        Path("flytbase_out"),
        Path("../flytbase_out"),
        Path("./data"),
        Path.cwd(),
        Path.home() / "Downloads" / "flytbase_out",
    ]
    for c in cands:
        c = Path(c).expanduser()
        if (c / "level2_kinematics.parquet").exists():
            return c.resolve()
    raise FileNotFoundError(
        "Could not find level2_kinematics.parquet. Pass --data /path/to/flytbase_out "
        "or set the FLYTBASE_OUT environment variable."
    )


def load_kinematics(data_dir: str | Path) -> pd.DataFrame:
    """Load the main per-frame dataset and derive the columns everything else needs."""
    data_dir = Path(data_dir)
    kin = pd.read_parquet(data_dir / "level2_kinematics.parquet")

    missing = [c for c in KIN_REQUIRED if c not in kin.columns]
    if missing:
        raise ValueError(f"level2_kinematics.parquet is missing columns: {missing}")

    kin = kin.sort_values(["track_id", "frame_idx"]).reset_index(drop=True)
    kin["class_final"] = kin["class_final"].astype(str)

    if "speed_mps" not in kin.columns and "speed_kmh" in kin.columns:
        kin["speed_mps"] = kin["speed_kmh"] / 3.6
    if "speed_kmh" not in kin.columns:
        kin["speed_kmh"] = kin["speed_mps"] * 3.6

    g = kin.groupby("track_id", sort=False)

    # per-sample step distance and step time (Edie's definitions need both)
    kin["dX"] = g["X_s"].diff()
    kin["dY"] = g["Y_s"].diff()
    kin["dt"] = g["time_sec"].diff()
    kin["ds"] = np.hypot(kin["dX"], kin["dY"])

    # net displacement per track -> moving flag (149/316 of your tracks are parked)
    first = g[["X_s", "Y_s"]].transform("first")
    last = g[["X_s", "Y_s"]].transform("last")
    kin["net_disp_calc"] = np.hypot(last["X_s"] - first["X_s"], last["Y_s"] - first["Y_s"])
    kin["is_moving"] = kin["net_disp_calc"] >= MOVING_MIN_DISP_M
    kin["is_vru"] = kin["class_final"].str.lower().isin(VRU)
    kin["pcu"] = kin["class_final"].map(pcu_for).astype(float)
    return kin


def load_per_track(data_dir: str | Path) -> pd.DataFrame | None:
    p = Path(data_dir) / "level2_per_track.csv"
    return pd.read_csv(p) if p.exists() else None


def nominal_dt(kin: pd.DataFrame) -> float:
    d = kin["dt"].dropna()
    return float(d.median()) if len(d) else 0.1001


# ----------------------------------------------------------------------------------
# Per-track summary: entry/exit geometry, bearings, turn angle
# ----------------------------------------------------------------------------------

def _stable_bearing(x: np.ndarray, y: np.ndarray, span_m: float, from_end: bool) -> float:
    """Bearing over the first (or last) `span_m` of travel — robust to endpoint jitter."""
    if len(x) < 2:
        return np.nan
    if from_end:
        x, y = x[::-1], y[::-1]
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    j = int(np.searchsorted(d, span_m))
    j = min(max(j, 1), len(x) - 1)
    dx, dy = x[j] - x[0], y[j] - y[0]
    if from_end:                       # we walked backwards; flip to direction of travel
        dx, dy = -dx, -dy
    if dx == 0 and dy == 0:
        return np.nan
    return math.degrees(math.atan2(dx, dy)) % 360.0


def wrap180(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def track_summary(kin: pd.DataFrame, bearing_span_m: float = 8.0) -> pd.DataFrame:
    """One row per track: endpoints, bearings, turn angle, duration, speeds."""
    rows = []
    dt0 = nominal_dt(kin)  # constant across tracks; don't recompute the median per row
    for tid, t in kin.groupby("track_id", sort=False):
        x = t["X_s"].to_numpy(float)
        y = t["Y_s"].to_numpy(float)
        if len(t) < 2:
            continue
        b_in = _stable_bearing(x, y, bearing_span_m, from_end=False)
        b_out = _stable_bearing(x, y, bearing_span_m, from_end=True)
        turn = float(wrap180(b_out - b_in)) if np.isfinite(b_in) and np.isfinite(b_out) else np.nan
        spd = t["speed_mps"].to_numpy(float)
        rows.append(
            dict(
                track_id=tid,
                class_final=t["class_final"].iloc[0],
                n=len(t),
                t_in=float(t["time_sec"].iloc[0]),
                t_out=float(t["time_sec"].iloc[-1]),
                dur_s=float(t["time_sec"].iloc[-1] - t["time_sec"].iloc[0]),
                x_in=x[0], y_in=y[0], x_out=x[-1], y_out=y[-1],
                bearing_in=b_in, bearing_out=b_out, turn_deg=turn,
                path_len_m=float(np.nansum(np.hypot(np.diff(x), np.diff(y)))),
                net_disp=float(np.hypot(x[-1] - x[0], y[-1] - y[0])),
                v_med_kmh=float(np.nanmedian(spd) * 3.6),
                v85_kmh=float(np.nanpercentile(spd, 85) * 3.6) if np.isfinite(spd).any() else np.nan,
                v_max_kmh=float(np.nanmax(spd) * 3.6) if np.isfinite(spd).any() else np.nan,
                stopped_s=float((spd < STOP_SPEED_MPS).sum() * dt0),
                L=float(t["L"].median()) if "L" in t.columns else np.nan,
                is_moving=bool(t["is_moving"].iloc[0]),
                pcu=pcu_for(t["class_final"].iloc[0]),
            )
        )
    s = pd.DataFrame(rows)
    if s.empty:
        return s
    s["movement"] = classify_movement(s["turn_deg"])
    s["v_mean_kmh"] = np.where(s["dur_s"] > 0, s["path_len_m"] / s["dur_s"] * 3.6, np.nan)
    s["delay_s"] = s["stopped_s"]
    return s


def classify_movement(turn_deg) -> pd.Series:
    t = pd.Series(np.asarray(turn_deg, dtype=float))
    out = pd.Series(["unknown"] * len(t), index=t.index, dtype=object)
    a = t.abs()
    out[a <= THROUGH_DEG] = "through"
    right = "right" if TURN_SIGN > 0 else "left"
    left = "left" if TURN_SIGN > 0 else "right"
    out[(a > THROUGH_DEG) & (a <= UTURN_DEG) & (t > 0)] = right
    out[(a > THROUGH_DEG) & (a <= UTURN_DEG) & (t < 0)] = left
    out[a > UTURN_DEG] = "u-turn"
    out[~np.isfinite(t)] = "unknown"
    return out.values


# ----------------------------------------------------------------------------------
# Approach/exit legs — discovered from the data, not hand-drawn
# ----------------------------------------------------------------------------------

def discover_legs(summary: pd.DataFrame, k: int = 4, seed: int = 0):
    """
    K-means over track ENTRY points and EXIT points jointly, so a leg used as both an
    approach and an exit gets one identity. Legs are then named by their bearing from
    the scene centroid (corridor frame, 0 deg = +Y).

    Returns (summary_with_leg_cols, legs_dataframe).
    """
    from sklearn.cluster import KMeans

    s = summary[summary["is_moving"]].copy()
    if len(s) < k * 2:
        summary = summary.copy()
        summary["leg_in"] = "?"
        summary["leg_out"] = "?"
        return summary, pd.DataFrame(columns=["leg", "x", "y", "bearing", "n"])

    pts = np.vstack([s[["x_in", "y_in"]].to_numpy(), s[["x_out", "y_out"]].to_numpy()])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(pts)

    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    cen = km.cluster_centers_
    bear = (np.degrees(np.arctan2(cen[:, 0] - cx, cen[:, 1] - cy))) % 360.0
    order = np.argsort(bear)                       # name legs clockwise from +Y
    remap = {old: i for i, old in enumerate(order)}
    names = [f"{chr(65 + i)}" for i in range(k)]   # A, B, C, D...

    lab = np.array([remap[l] for l in km.labels_])
    n = len(s)
    s["leg_in"] = [names[i] for i in lab[:n]]
    s["leg_out"] = [names[i] for i in lab[n:]]

    legs = pd.DataFrame(
        dict(
            leg=names,
            x=cen[order, 0],
            y=cen[order, 1],
            bearing=bear[order],
            compass=[_compass(b) for b in bear[order]],
        )
    )
    legs["n_in"] = [int((s["leg_in"] == nm).sum()) for nm in names]
    legs["n_out"] = [int((s["leg_out"] == nm).sum()) for nm in names]
    legs["label"] = legs["leg"] + " (" + legs["compass"] + ")"

    out = summary.copy()
    out["leg_in"] = "?"
    out["leg_out"] = "?"
    out.loc[s.index, "leg_in"] = s["leg_in"]
    out.loc[s.index, "leg_out"] = s["leg_out"]
    out["od"] = out["leg_in"] + "\u2192" + out["leg_out"]
    return out, legs


def _compass(b: float) -> str:
    """Name a corridor-frame bearing. NOT true north — corridor +Y is 'N' by convention."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[int(((b + 22.5) % 360) // 45)]


def od_matrix(summary: pd.DataFrame, classes=None, exclude_self=True) -> pd.DataFrame:
    s = summary[summary["is_moving"]].copy()
    if classes:
        s = s[s["class_final"].isin(classes)]
    if exclude_self:
        s = s[s["leg_in"] != s["leg_out"]]
    if s.empty:
        return pd.DataFrame()
    m = pd.crosstab(s["leg_in"], s["leg_out"])
    return m


def movement_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Turning movement counts by (origin leg, movement type, class) — the classic
    traffic-engineering turning-movement count, plus PCU-weighted totals."""
    s = summary[summary["is_moving"]].copy()
    if s.empty:
        return pd.DataFrame()
    t = (
        s.groupby(["leg_in", "movement", "class_final"], observed=True)
        .agg(count=("track_id", "size"), pcu=("pcu", "sum"), v_med_kmh=("v_med_kmh", "median"))
        .reset_index()
    )
    return t.sort_values(["leg_in", "movement", "count"], ascending=[True, True, False])


# ----------------------------------------------------------------------------------
# Virtual counting line -> classified counts by interval, headways, lane position
# ----------------------------------------------------------------------------------

def line_crossings(kin: pd.DataFrame, y0: float, moving_only: bool = True) -> pd.DataFrame:
    """
    Every time a trajectory crosses the line Y = y0, linearly interpolated.
    Returns time of crossing, class, direction (+1 = travelling toward +Y), and the
    X position at the crossing (used for lane assignment).
    """
    df = kin[kin["is_moving"]] if moving_only else kin
    out = []
    for tid, t in df.groupby("track_id", sort=False):
        y = t["Y_s"].to_numpy(float)
        x = t["X_s"].to_numpy(float)
        ts = t["time_sec"].to_numpy(float)
        sp = t["speed_kmh"].to_numpy(float)
        a = y[:-1] - y0
        b = y[1:] - y0
        idx = np.where((a <= 0) & (b > 0) | (a >= 0) & (b < 0))[0]
        for i in idx:
            denom = (y[i + 1] - y[i])
            f = 0.0 if denom == 0 else (y0 - y[i]) / denom
            out.append(
                dict(
                    track_id=tid,
                    class_final=t["class_final"].iloc[0],
                    t=ts[i] + f * (ts[i + 1] - ts[i]),
                    x=x[i] + f * (x[i + 1] - x[i]),
                    speed_kmh=sp[i] + f * (sp[i + 1] - sp[i]),
                    direction=1 if b[i] > 0 else -1,
                    pcu=pcu_for(t["class_final"].iloc[0]),
                )
            )
    c = pd.DataFrame(out)
    if not c.empty:
        c = c.sort_values("t").reset_index(drop=True)
        c["dir_label"] = np.where(c["direction"] > 0, "+Y (up-corridor)", "-Y (down-corridor)")
    return c


def counts_by_interval(cross: pd.DataFrame, bin_s: float = 15.0) -> pd.DataFrame:
    """Classified counts per interval, scaled to an hourly flow rate."""
    if cross.empty:
        return pd.DataFrame()
    c = cross.copy()
    c["bin"] = (c["t"] // bin_s) * bin_s
    g = (
        c.groupby(["bin", "dir_label", "class_final"], observed=True)
        .agg(count=("track_id", "size"), pcu=("pcu", "sum"), v_med_kmh=("speed_kmh", "median"))
        .reset_index()
    )
    g["flow_vph"] = g["count"] * 3600.0 / bin_s
    g["flow_pcuph"] = g["pcu"] * 3600.0 / bin_s
    return g


def headways(cross: pd.DataFrame) -> pd.DataFrame:
    """Time headway between successive road users crossing the line, per direction."""
    if cross.empty:
        return pd.DataFrame()
    out = []
    for d, c in cross.groupby("dir_label"):
        c = c.sort_values("t")
        h = c["t"].diff()
        out.append(pd.DataFrame(dict(dir_label=d, headway_s=h.values,
                                     class_final=c["class_final"].values, t=c["t"].values)))
    h = pd.concat(out, ignore_index=True).dropna(subset=["headway_s"])
    return h


# ----------------------------------------------------------------------------------
# Speed profile along the corridor + 2D speed field + speeding hotspots
# ----------------------------------------------------------------------------------

def speed_profile(kin: pd.DataFrame, bin_m: float = 5.0, limit_kmh: float = SPEED_LIMIT_KMH,
                  by_class: bool = False, by_direction: bool = True) -> pd.DataFrame:
    """Speed statistics vs position ALONG the corridor (Y). This is the reliable axis."""
    d = kin[kin["is_moving"] & kin["speed_kmh"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["y_bin"] = (d["Y_s"] // bin_m) * bin_m + bin_m / 2
    d["dir_label"] = np.where(d["dY"].fillna(0) >= 0, "+Y (up-corridor)", "-Y (down-corridor)")
    keys = ["y_bin"]
    if by_direction:
        keys.append("dir_label")
    if by_class:
        keys.append("class_final")
    g = (
        d.groupby(keys, observed=True)
        .agg(
            n=("speed_kmh", "size"),
            v_mean=("speed_kmh", "mean"),
            v_med=("speed_kmh", "median"),
            v15=("speed_kmh", lambda s: np.nanpercentile(s, 15)),
            v85=("speed_kmh", lambda s: np.nanpercentile(s, 85)),
            frac_over=("speed_kmh", lambda s: float((s > limit_kmh).mean())),
            frac_stopped=("speed_kmh", lambda s: float((s < STOP_SPEED_MPS * 3.6).mean())),
            n_tracks=("track_id", "nunique"),
        )
        .reset_index()
    )
    return g.sort_values("y_bin")


def speed_field(kin: pd.DataFrame, cell_m: float = 4.0, min_samples: int = 3) -> pd.DataFrame:
    """2D grid of mean speed — shows WHERE in the junction speed collapses."""
    d = kin[kin["is_moving"] & kin["speed_kmh"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["xb"] = (d["X_s"] // cell_m) * cell_m + cell_m / 2
    d["yb"] = (d["Y_s"] // cell_m) * cell_m + cell_m / 2
    g = (
        d.groupby(["xb", "yb"], observed=True)
        .agg(v_mean=("speed_kmh", "mean"), n=("speed_kmh", "size"),
             n_tracks=("track_id", "nunique"))
        .reset_index()
    )
    return g[g["n"] >= min_samples]


def speeding_hotspots(kin: pd.DataFrame, limit_kmh: float, cell_m: float = 5.0,
                      min_samples: int = 5) -> pd.DataFrame:
    d = kin[kin["is_moving"] & kin["speed_kmh"].notna()].copy()
    if d.empty:
        return pd.DataFrame()
    d["xb"] = (d["X_s"] // cell_m) * cell_m + cell_m / 2
    d["yb"] = (d["Y_s"] // cell_m) * cell_m + cell_m / 2
    d["over"] = d["speed_kmh"] > limit_kmh
    g = (
        d.groupby(["xb", "yb"], observed=True)
        .agg(frac_over=("over", "mean"), n=("over", "size"),
             v85=("speed_kmh", lambda s: np.nanpercentile(s, 85)),
             n_tracks=("track_id", "nunique"))
        .reset_index()
    )
    return g[g["n"] >= min_samples].sort_values("frac_over", ascending=False)


# ----------------------------------------------------------------------------------
# Lanes: discover lateral bands, volume + modal split per lane, lane discipline
# ----------------------------------------------------------------------------------

def lane_kde(cross: pd.DataFrame, direction: str, bw_m: float = 0.6, n_grid: int = 600):
    """Return (grid_x, density) for the lateral position histogram of one direction."""
    from scipy.stats import gaussian_kde

    c = cross[cross["dir_label"] == direction]
    if len(c) < 8:
        return None, None
    x = c["x"].to_numpy(float)
    sd = float(np.std(x))
    if sd <= 0:
        return None, None
    grid = np.linspace(x.min() - 2, x.max() + 2, n_grid)
    try:
        kde = gaussian_kde(x, bw_method=bw_m / sd)   # bandwidth expressed in METRES
        return grid, kde(grid)
    except Exception:
        return None, None


def discover_lanes(cross: pd.DataFrame, direction: str, max_lanes: int = 5,
                   bw_m: float = 0.6) -> pd.DataFrame:
    """
    1D KDE over the lateral (X) position of line crossings for one direction; peaks are
    lane centres. Data-driven, so it also works when lane markings are ignored, which is
    the interesting case here. `bw_m` is the smoothing bandwidth in metres — lower it if
    lanes merge into one blob, raise it if noise splits a lane in two.
    """
    from scipy.signal import find_peaks

    grid, dens = lane_kde(cross, direction, bw_m=bw_m)
    if grid is None:
        return pd.DataFrame(columns=["lane", "x_center"])
    step = grid[1] - grid[0]
    min_sep = max(int(2.0 / step), 3)               # lanes at least 2 m apart
    peaks, props = find_peaks(dens, prominence=dens.max() * 0.05, distance=min_sep)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(dens))])
        props = {"prominences": np.array([1.0])}
    if len(peaks) > max_lanes:                      # keep the most prominent
        keep = np.argsort(props["prominences"])[-max_lanes:]
        peaks = np.sort(peaks[keep])
    centers = grid[peaks]
    lanes = pd.DataFrame(dict(lane=[f"L{i+1}" for i in range(len(centers))],
                              x_center=centers))
    return lanes


def assign_lanes(cross: pd.DataFrame, lanes_by_dir: dict) -> pd.DataFrame:
    c = cross.copy()
    c["lane"] = "?"
    for d, lanes in lanes_by_dir.items():
        if lanes is None or lanes.empty:
            continue
        m = c["dir_label"] == d
        if not m.any():
            continue
        cx = lanes["x_center"].to_numpy(float)
        idx = np.abs(c.loc[m, "x"].to_numpy(float)[:, None] - cx[None, :]).argmin(axis=1)
        c.loc[m, "lane"] = lanes["lane"].to_numpy()[idx]
    c["lane_key"] = c["dir_label"] + " " + c["lane"]
    return c


def lane_modal_split(cross_l: pd.DataFrame) -> pd.DataFrame:
    if cross_l.empty:
        return pd.DataFrame()
    g = (
        cross_l.groupby(["lane_key", "class_final"], observed=True)
        .agg(count=("track_id", "size"), pcu=("pcu", "sum"),
             v_med_kmh=("speed_kmh", "median"))
        .reset_index()
    )
    tot = g.groupby("lane_key")["count"].transform("sum")
    g["share"] = g["count"] / tot
    return g.sort_values(["lane_key", "count"], ascending=[True, False])


def lateral_dispersion(kin: pd.DataFrame, y_lo: float, y_hi: float) -> pd.DataFrame:
    """
    Per-class spread of lateral position inside a corridor window. Motorcycles wandering
    laterally far more than cars is the quantitative statement of "no lane discipline".
    NOTE: X is the weakly-constrained axis (thin GCP quad) — treat as indicative.
    """
    d = kin[(kin["is_moving"]) & (kin["Y_s"].between(y_lo, y_hi))]
    if d.empty:
        return pd.DataFrame()
    per_track = (
        d.groupby(["track_id", "class_final"], observed=True)["X_s"]
        .agg(["std", "min", "max", "size"])
        .reset_index()
    )
    per_track["swing_m"] = per_track["max"] - per_track["min"]
    per_track = per_track[per_track["size"] >= 5]
    # across-track spread: how widely does this class distribute itself over the width?
    mean_x = (
        d.groupby(["track_id", "class_final"], observed=True)["X_s"].mean()
        .reset_index().groupby("class_final")["X_s"]
    )
    across = mean_x.agg(["std", lambda s: float(np.subtract(*np.nanpercentile(s, [90, 10])))])
    across.columns = ["across_sd_m", "across_p10_p90_m"]

    g = (
        per_track.groupby("class_final", observed=True)
        .agg(n_tracks=("track_id", "nunique"),
             within_track_sd_m=("std", "median"),
             within_track_swing_m=("swing_m", "median"))
        .reset_index()
        .merge(across.reset_index(), on="class_final", how="left")
        .sort_values("n_tracks", ascending=False)
    )
    return g


# ----------------------------------------------------------------------------------
# Queues
# ----------------------------------------------------------------------------------

def queues(kin: pd.DataFrame, stop_speed: float = STOP_SPEED_MPS,
           gap_m: float = QUEUE_GAP_M, min_veh: int = QUEUE_MIN_VEH,
           exclude_ped: bool = True) -> pd.DataFrame:
    """
    Per frame: cluster stopped road users along Y (separately per travel direction),
    and measure the physical extent of each cluster in metres.

    Queue length = span along Y + one mean vehicle length (so a 3-vehicle standing
    queue is not reported as the distance between first and last bumper only).
    """
    d = kin[kin["is_moving"] & (kin["speed_mps"] < stop_speed)].copy()
    if exclude_ped:
        d = d[~d["class_final"].str.lower().isin({"pedestrian"})]
    if d.empty:
        return pd.DataFrame()

    # travel direction of the parent track, so opposing standing queues don't merge
    trk_dir = (
        kin[kin["is_moving"]].groupby("track_id")["Y_s"]
        .agg(lambda s: 1 if (s.iloc[-1] - s.iloc[0]) >= 0 else -1)
    )
    d["tdir"] = d["track_id"].map(trk_dir).fillna(1)

    out = []
    for (fi, tdir), grp in d.groupby(["frame_idx", "tdir"], sort=False):
        grp = grp.sort_values("Y_s")
        y = grp["Y_s"].to_numpy(float)
        brk = np.where(np.diff(y) > gap_m)[0] + 1
        for chunk in np.split(np.arange(len(y)), brk):
            if len(chunk) < min_veh:
                continue
            sub = grp.iloc[chunk]
            Lm = float(sub["L"].median()) if "L" in sub.columns and sub["L"].notna().any() else 4.0
            out.append(
                dict(
                    frame_idx=fi,
                    time_sec=float(sub["time_sec"].iloc[0]),
                    direction="+Y" if tdir > 0 else "-Y",
                    n_veh=len(sub),
                    y_head=float(sub["Y_s"].max() if tdir > 0 else sub["Y_s"].min()),
                    y_tail=float(sub["Y_s"].min() if tdir > 0 else sub["Y_s"].max()),
                    length_m=float(y.max() - y.min()) if len(chunk) > 1 else 0.0,
                    x_mean=float(sub["X_s"].mean()),
                )
            )
            out[-1]["length_m"] = float(sub["Y_s"].max() - sub["Y_s"].min()) + Lm
    q = pd.DataFrame(out)
    if not q.empty:
        q = q.sort_values("time_sec").reset_index(drop=True)
    return q


def queue_timeseries(q: pd.DataFrame, bin_s: float = 2.0) -> pd.DataFrame:
    if q.empty:
        return pd.DataFrame()
    qq = q.copy()
    qq["bin"] = (qq["time_sec"] // bin_s) * bin_s
    return (
        qq.groupby(["bin", "direction"], observed=True)
        .agg(max_len_m=("length_m", "max"), max_veh=("n_veh", "max"),
             n_queues=("length_m", "size"))
        .reset_index()
    )


def stop_bar_estimate(q: pd.DataFrame) -> pd.DataFrame:
    """Where queues repeatedly terminate = inferred stop line position, per direction."""
    if q.empty:
        return pd.DataFrame()
    return (
        q.groupby("direction")
        .agg(stop_bar_y_m=("y_head", "median"),
             y_head_iqr=("y_head", lambda s: float(np.subtract(*np.nanpercentile(s, [75, 25])))),
             max_queue_m=("length_m", "max"),
             mean_queue_m=("length_m", "mean"),
             max_veh=("n_veh", "max"))
        .reset_index()
    )


# ----------------------------------------------------------------------------------
# Fundamental diagram — Edie's generalised definitions
# ----------------------------------------------------------------------------------

def edie(kin: pd.DataFrame, dy_m: float = 20.0, dt_s: float = 10.0,
         y_lo: float | None = None, y_hi: float | None = None,
         along_only: bool = True) -> pd.DataFrame:
    """
    Edie (1963): over a space-time region A of size dy * dt,
        flow    q = (total distance travelled in A) / |A|
        density k = (total time spent in A)         / |A|
        speed   v = q / k
    This is the correct estimator for trajectory data — far more stable than counting
    vehicles at a line and far more honest than instantaneous snapshots.
    """
    d = kin[kin["is_moving"]].copy()
    if along_only:
        # keep road users whose motion is predominantly along the corridor
        tot = d.groupby("track_id")[["dX", "dY"]].sum().abs()
        keep = tot.index[(tot["dY"] >= tot["dX"])]
        d = d[d["track_id"].isin(keep)]
    if y_lo is not None:
        d = d[d["Y_s"].between(y_lo, y_hi)]
    d = d.dropna(subset=["ds", "dt"])
    if d.empty:
        return pd.DataFrame()

    d["ymid"] = (d["Y_s"] + d["Y_s"].shift(1).where(d["track_id"].eq(d["track_id"].shift(1)), d["Y_s"])) / 2
    d["tmid"] = d["time_sec"] - d["dt"] / 2
    d["yb"] = (d["ymid"] // dy_m) * dy_m + dy_m / 2
    d["tb"] = (d["tmid"] // dt_s) * dt_s + dt_s / 2
    d["dY_abs"] = d["ds"]

    g = (
        d.groupby(["yb", "tb"], observed=True)
        .agg(dist_m=("dY_abs", "sum"), time_s=("dt", "sum"), n_tracks=("track_id", "nunique"))
        .reset_index()
    )
    area = dy_m * dt_s
    g["flow_vph"] = g["dist_m"] / area * 3600.0
    g["density_vpkm"] = g["time_s"] / area * 1000.0
    g["speed_kmh"] = np.where(g["density_vpkm"] > 0,
                              g["flow_vph"] / g["density_vpkm"], np.nan)
    return g[g["n_tracks"] > 0]


def occupancy(kin: pd.DataFrame, y0: float, zone_m: float = 6.0,
              bin_s: float = 5.0) -> pd.DataFrame:
    """Loop-detector-equivalent occupancy: fraction of time the zone holds >=1 vehicle."""
    d = kin[kin["is_moving"] & kin["Y_s"].between(y0 - zone_m / 2, y0 + zone_m / 2)]
    if d.empty:
        return pd.DataFrame()
    dt = nominal_dt(kin)
    frames = kin["frame_idx"].unique()
    occ_frames = d.groupby("frame_idx")["track_id"].nunique()
    s = pd.Series(0, index=np.sort(frames), dtype=float)
    s.loc[occ_frames.index] = occ_frames.values
    df = pd.DataFrame(dict(frame_idx=s.index, n=s.values))
    tmap = kin.groupby("frame_idx")["time_sec"].first()
    df["time_sec"] = df["frame_idx"].map(tmap)
    df["bin"] = (df["time_sec"] // bin_s) * bin_s
    return (
        df.groupby("bin")
        .agg(occupancy=("n", lambda v: float((v > 0).mean())),
             mean_veh=("n", "mean"), max_veh=("n", "max"))
        .reset_index()
    )


# ----------------------------------------------------------------------------------
# Interaction / proximity density (bonus: what fixed cameras structurally cannot do)
# ----------------------------------------------------------------------------------

def interaction_density(kin: pd.DataFrame, radius_m: float = PROXIMITY_M,
                        cell_m: float = 5.0, min_speed_mps: float = 1.0) -> pd.DataFrame:
    """
    Count moments where two moving road users are within `radius_m`, binned spatially.
    Hotspots are conflict areas. Cheap: one KD-tree per frame.
    """
    from scipy.spatial import cKDTree

    d = kin[kin["is_moving"] & (kin["speed_mps"] > min_speed_mps)]
    if d.empty:
        return pd.DataFrame()
    recs = []
    for fi, grp in d.groupby("frame_idx", sort=False):
        if len(grp) < 2:
            continue
        pts = grp[["X_s", "Y_s"]].to_numpy(float)
        tree = cKDTree(pts)
        for i, j in tree.query_pairs(radius_m):
            recs.append(((pts[i, 0] + pts[j, 0]) / 2, (pts[i, 1] + pts[j, 1]) / 2,
                         float(np.hypot(*(pts[i] - pts[j])))))
    if not recs:
        return pd.DataFrame()
    r = pd.DataFrame(recs, columns=["x", "y", "dist_m"])
    r["xb"] = (r["x"] // cell_m) * cell_m + cell_m / 2
    r["yb"] = (r["y"] // cell_m) * cell_m + cell_m / 2
    return (
        r.groupby(["xb", "yb"])
        .agg(events=("dist_m", "size"), min_dist_m=("dist_m", "min"))
        .reset_index()
        .sort_values("events", ascending=False)
    )


# ----------------------------------------------------------------------------------
# Headline numbers for the overview panel
# ----------------------------------------------------------------------------------

def overview(kin: pd.DataFrame, summary: pd.DataFrame) -> dict:
    mov = summary[summary["is_moving"]]
    dur = float(kin["time_sec"].max() - kin["time_sec"].min())
    motor = mov[~mov["class_final"].str.lower().isin({"pedestrian", "bicycle", "cyclist"})]
    tw = mov[mov["class_final"].str.lower().eq("motorcycle")]
    return dict(
        window_s=dur,
        n_tracks=int(summary["track_id"].nunique()),
        n_moving=int(mov["track_id"].nunique()),
        n_static=int(summary["track_id"].nunique() - mov["track_id"].nunique()),
        throughput_vph=float(len(mov) * 3600.0 / dur) if dur > 0 else np.nan,
        pcu_ph=float(mov["pcu"].sum() * 3600.0 / dur) if dur > 0 else np.nan,
        two_wheeler_share=float(len(tw) / len(motor)) if len(motor) else np.nan,
        median_speed_kmh=float(mov["v_med_kmh"].median()),
        median_delay_s=float(mov["delay_s"].median()),
        classes=int(mov["class_final"].nunique()),
    )


def class_table(summary: pd.DataFrame) -> pd.DataFrame:
    mov = summary[summary["is_moving"]]
    if mov.empty:
        return pd.DataFrame()
    t = (
        mov.groupby("class_final", observed=True)
        .agg(tracks=("track_id", "nunique"),
             median_kmh=("v_med_kmh", "median"),
             p85_kmh=("v85_kmh", "median"),
             max_kmh=("v_max_kmh", "max"),
             median_len_m=("L", "median"),
             median_dur_s=("dur_s", "median"),
             median_delay_s=("delay_s", "median"))
        .reset_index()
        .sort_values("tracks", ascending=False)
    )
    t["share"] = t["tracks"] / t["tracks"].sum()
    # honesty flag straight from CONTEXT.md §6
    t["reliable"] = np.where(t["tracks"] >= 20, "yes", "SINGLE-OBS / small-n")
    return t
