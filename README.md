# FlytBase Computer Vision Hackathon — Drone Traffic Analytics & Dashboard

A comprehensive computer vision pipeline and Level 3 Aggregate Insight dashboard for analyzing aerial drone video traffic trajectories.

---

## 📁 Repository Structure

```
├── dashboard/              # Level 3 Streamlit Dashboard & Analytics Engine
│   ├── app.py              # Main interactive Streamlit dashboard
│   ├── server.py           # Backend server interface
│   ├── l3_core.py          # Analytics engine (pure analytical functions & metrics)
│   ├── l3_export.py        # Batch data & report exporter
│   ├── render_overlay.py   # Visual overlay renderer (trails, filters, split-screen)
│   ├── fix_geo_and_selection.py  # Georeferencing and ROI selection helper
│   ├── calibration.json    # Camera calibration parameters
│   └── requirements.txt    # Dashboard Python dependencies
│
├── notebooks/              # Tracking Pipelines & Jupyter Notebooks
│   ├── 01_track.py         # Primary vehicle detection & tracking script
│   ├── submission1.ipynb   # Initial pipeline submission notebook
│   ├── level_2_final.ipynb # Level 2 trajectory kinematics notebook
│   ├── final_one_sub_1.ipynb # Full pipeline integration notebook
│   ├── labels_my-project-name_2026-08-22-02-44-31.json  # Annotation dataset
│   └── Intersection_1080p.srt # Drone telemetry subtitles
│
├── flytbase_out/           # Analytical Datasets & Outputs
│   ├── level2_kinematics.parquet  # Per-frame trajectory & ground coordinates
│   ├── tracks_final.parquet       # Filtered & smoothed trajectory tracks
│   ├── level2_per_track.csv       # Summary per road user
│   ├── l3_export/                 # Exported analytics CSV tables & findings report
│   └── *.png                      # Generated analytics visualization plots
│
├── CONTEXT.md              # Technical specifications & coordinate definitions
└── requirements.txt        # Top-level dependencies
```

---

## 🚀 Quick Start

### 1. Installation

Set up a virtual environment and install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard

Run the Level 3 Aggregate Insight dashboard:

```bash
streamlit run dashboard/app.py
```

The application will launch at `http://localhost:8501`.

### 3. Real-Time Tracking (live)

`01_track.py` is the accurate **offline** pipeline (tiled inference for maximum
small-object recall). For **live** tracking as traffic moves — webcam, RTSP
drone/CCTV stream, or a video file — use the whole-frame real-time script,
which leans on Ultralytics' built-in ByteTrack/BoT-SORT tracker:

```bash
python notebooks/realtime_track.py --source 0                 # webcam
python notebooks/realtime_track.py --source intersection.mp4 --imgsz 960
python notebooks/realtime_track.py --source rtsp://host/stream # live drone/CCTV
python notebooks/realtime_track.py --benchmark 120 --imgsz 736 # throughput check

# Track a chosen 2-min segment (skip the first 90s) and save a shareable clip:
python notebooks/realtime_track.py --source Intersection_Merged.MP4 \
    --start-sec 90 --max-seconds 120 --imgsz 960 \
    --no-show --out clip.mp4 --out-width 640 --out-fps 10

# Keep up with live playback on slow hardware by processing every Nth frame:
python notebooks/realtime_track.py --source Intersection_Merged.MP4 \
    --start-sec 90 --max-seconds 30 --frame-stride 6
```

A live window shows boxes, track IDs, and a running FPS counter (`q` quits).
Key flags:

- `--imgsz` — inference size; lower = faster, less small-object recall.
- `--start-sec` / `--max-seconds` — track only a time window of a file.
- `--frame-stride N` — process every Nth frame (frame-skip). The `[result]`
  line reports real-time against the rate needed to keep up, `source_fps / N`.
- `--out` / `--out-width` / `--out-fps` — write an H.264 (avc1) annotated clip,
  optionally downscaled / fps-subsampled, at true playback speed.
- `--benchmark N` — process N frames headless and print detect+track FPS
  (synthetic frames when no `--source` is given).

**Measured throughput** (VisDrone YOLO11s, Apple M-series **MPS** GPU):

| Input                         | FPS   | Real-time?                              |
|-------------------------------|-------|-----------------------------------------|
| synthetic 1280p, `--imgsz 640`| ~31   | ✅ yes                                   |
| synthetic 1280p, `--imgsz 736`| ~29   | ✅ yes                                   |
| synthetic 1280p, `--imgsz 960`| ~19   | ✅ yes                                   |
| real **4K** footage, `--imgsz 960` | ~5 | ❌ batch (4K decode + ~100 objects/frame) |

At native 4K on a laptop GPU the run is **offline batch** (~5 FPS, ~6× slower
than 30 fps playback) — the limiter is 4K decode and object count, not the
tracker. To hit real time: feed 1080p instead of 4K, run on a CUDA GPU, or use
`--frame-stride` (e.g. `6` so ~5 FPS × 6 keeps up with 30 fps playback).
`01_track.py`'s tiled inference stays offline-only, for maximum recall on tiny
objects in 4K frames.

### 4. Run Batch Data Export

To generate summary tables (CSVs) and automated markdown findings (`findings.md`):

```bash
python dashboard/l3_export.py --data flytbase_out --out flytbase_out/l3_export
```

---

## 📊 Analytics Features

- **Classified Volume & Flow**: Modal split (cars, motorcycles, trucks, buses, VRUs) with PCU/h calculations.
- **Origin-Destination Distributions**: Dynamic approach leg clustering, OD matrices, and Sankey diagrams.
- **Speed Profiles & Speeding Hotspots**: Position-wise 15th/85th percentile speed bands and spatial speed field mapping.
- **Queue Lengths & Stop Line Estimation**: Vehicle stoppage clustering and signal cycle time-series analysis.
- **Fundamental Diagrams & Interaction Density**: Edie's generalized traffic state definitions ($q$, $k$, $v$) and conflict proximity density.
