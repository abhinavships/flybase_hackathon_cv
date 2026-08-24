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

### 3. Run Batch Data Export

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
