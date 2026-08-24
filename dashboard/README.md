# Level 3 — Aggregate Insight dashboard

Drop these four files anywhere. They read `flytbase_out/` and nothing else.

```
l3_core.py         all the analytics (pure functions, no UI)
app.py             the Streamlit dashboard
render_overlay.py  filtered overlay video renderer (class filter, trails, split-screen)
l3_export.py       batch export: every table as CSV + an auto-written findings.md
```

## Run it (2 minutes)

```bash
cd <folder with these files>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py -- --data /path/to/flytbase_out
```

Opens on `http://localhost:8501`. If you skip `--data` it looks for `flytbase_out/`
in the current folder, the parent, and `~/Downloads/` — or just paste the path into
the sidebar box once it's open.

Only **`level2_kinematics.parquet`** is required. `level2_per_track.csv` is used if
present; `tracks_masked.parquet` is only needed for the video renderer.

Batch export without the UI:

```bash
python l3_export.py --data flytbase_out --out flytbase_out/l3_export
```

That writes 21 CSVs plus `findings.md` — a first draft of the writeup with your real
numbers already substituted in, caveats included.

## What each tab answers, against the brief

| Brief item | Tab | What you get |
|---|---|---|
| 1. Classified counts by movement and interval | **Turning movements & OD**, **Counts & flow** | Approach legs discovered by k-means over track endpoints; turning-movement counts cross-tabbed by class; movable virtual counting line with per-interval classified counts, directional split, veh/h and PCU/h |
| 2. Origin–destination distributions | **Turning movements & OD** | OD matrix, heatmap, Sankey |
| 3. Speed profiles across the segment, and where speeding concentrates | **Speed profile** | Median + 15/85 percentile band vs position along corridor, per direction and per class; fraction-over-limit by position; 2D speed field; ranked speeding cells |
| 4. Lane volume and modal split by lane | **Lanes & modal split** | Lane centres from a KDE of lateral position at the counting line (no lane markings needed), volume and modal mix per lane, plus lateral dispersion as a lane-discipline metric |
| 5. Queue lengths | **Queues** | Per-frame clustering of stopped vehicles along Y, queue length in metres, sawtooth time series (= signal cycle), inferred stop-line position |
| 6. Density, occupancy, flow–density | **Fundamental diagram** | Edie's generalised definitions (q = Σdistance/\|A\|, k = Σtime/\|A\|), q–k, v–k, v–q, space–time speed map with shockwaves, detector-equivalent occupancy |
| bonus | **Interactions** | Proximity/conflict density map — the thing a fixed perspective camera structurally cannot produce |

**Playback** tab animates the trajectories straight from the parquet with a time slider,
trail length, and the sidebar class filter — instant, no video decoding. **Video** tab
plays your existing renders side by side and can render a new filtered one.

## Filters

Sidebar filters apply to every tab at once: time window, class multiselect, moving-only,
counting-line position, speed limit, count interval. Under **Advanced**: number of legs,
bin sizes, lane KDE bandwidth, Edie cell size, stopped threshold, queue gap.

## Filtered overlay video

```bash
# motorcycles only, 4 s trails, raw|tracked split screen
python render_overlay.py --video Intersection_Merged.MP4 --data flytbase_out \
  --out moto.mp4 --start 60 --end 150 --classes motorcycle --trail-sec 4 \
  --side-by-side --labels

# everything, boxes only
python render_overlay.py --video Intersection_Merged.MP4 --data flytbase_out \
  --out all.mp4 --start 60 --end 150 --trail-sec 0
```

Install `ffmpeg` if you want the output to play in the browser (it re-encodes to H.264
automatically when ffmpeg is on PATH).

## Two things to check on first run

1. **Turn direction.** Left/right come from the signed bearing change in the corridor
   frame. Watch one turning vehicle in the video; if left and right are swapped, flip
   `TURN_SIGN = +1` to `-1` in `l3_core.py`. Everything downstream follows.
2. **Counting line position.** Default is mid-corridor. Slide it until it sits on a clean
   section of carriageway — the lane analysis uses lateral position at that line, so a
   line placed inside the junction box will mix approaches together.

## Caveats the code already carries for you

- Static tracks (net displacement < 20 m) are excluded from every flow and speed
  statistic. The Overview tab shows how many were dropped.
- Classes with fewer than 20 tracks are flagged `SINGLE-OBS / small-n` in the class table
  and in `findings.md`. Only car and motorcycle support distributional claims.
- Along-track (Y) geometry is GCP-calibrated and reliable. Across-track (X) geometry is
  approximate because the calibration quad is a thin strip — the Lanes tab shows this
  warning inline. Lane *counts and modal mix* are sound; absolute lane *widths* are not.
- PCU factors are IRC-style and editable in the `PCU` dict at the top of `l3_core.py`.

## Should you hand this to Claude Code?

Not for building the dashboard — it's built and tested. Hand Claude Code the two things
that need repo access:

1. **Re-run the tracker over the full 6:39 video** so the aggregate statistics rest on
   400 s instead of 90 s. Ninety seconds is thin for flow statistics and a grader will
   notice. Calibration transfers unchanged since the drone never moved.
2. **Any scene-specific tuning** — turn sign, leg naming, counting-line placement — once
   you've watched the video with the dashboard open next to it.

A prompt that works:

> This repo has `flytbase_out/` with Level 2 trajectory outputs and a Level 3 dashboard
> (`app.py`, `l3_core.py`). Re-run detection and tracking over the full
> `Intersection_Merged.MP4` using the documented pipeline in `CONTEXT.md` §3 (tiled 1024 px
> inference, class-agnostic NMS at 0.55, conf 0.10, stride 3), apply the same stitching,
> ROI mask and GCP homography, and write `level2_kinematics.parquet` in the identical
> schema so the dashboard picks it up with no changes. Then run
> `python l3_export.py --data flytbase_out` and show me the diff in headline numbers
> between the 90 s window and the full video.

If you also want the deeper Level 3 differentiators later, the highest-value additions
on top of this are: signal cycle and phase recovery from the queue sawtooth, saturation
flow rate at the stop line, and per-class control delay. All three come from tables this
code already computes.

## Deploying beyond localhost

You don't need to, but if you want a URL to put in the submission:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501   # LAN
```

For a public link, `pip install streamlit` on any small VM and reverse-proxy it, or push
the folder plus `flytbase_out/` (it's only a few MB of parquet) to a GitHub repo and
deploy free on Streamlit Community Cloud — set the main file to `app.py` and put
`flytbase_out/` in the repo root so the auto-discovery finds it. Do not commit the 6 GB
MP4s; the dashboard degrades gracefully without them (the Video tab just shows a note).
