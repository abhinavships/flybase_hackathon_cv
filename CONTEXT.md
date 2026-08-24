# FlytBase Visual Intelligence Hackathon — Project Context

**Read this first.** It is the complete state of the project as of end of Level 2.
Levels 1 and 2 are submitted. The current task is **Level 3 — Aggregate Insight (max 200 pts)**.

---

## 1. The problem statement (verbatim intent)

Build a **Traffic Analysis Agent** from drone footage.

Current fixed-camera traffic systems are alarm generators: they detect a closed
vocabulary of events (stopped vehicle, wrong-way driver, pedestrian on carriageway,
debris) from a single perspective view of one approach. Three gaps follow:

1. **Interactions are unmeasured.** A perspective view of one approach cannot place
   two road users in a common coordinate frame, so near-misses, conflicts and failed
   merges are invisible.
2. **The event vocabulary is closed.** Anything not pre-configured is never detected.
3. **Coverage is fixed to infrastructure.** Data exists only where a camera was installed.

A drone removes all three constraints: the full scene sits in one frame, every road
user's complete path is captured natively, no fixed installation required.

**Trajectories are the enabling primitive.** From them you derive the existing event
vocabulary *plus* what fixed systems cannot produce: interaction metrics, full speed
profiles, turning demand by movement and class, queues measured in metres, and the
origin of a jam rather than its symptoms.

> "You are given hours of drone footage with telemetry, and no annotations. Build a
> system that extracts trajectories and derives insight from them. **Submissions are
> assessed on the depth of insight produced — detection and tracking are the
> foundation, not the objective.**"

### Level structure (sequential unlock)

| Level | Name | Max pts | Status |
|---|---|---|---|
| 1 | Detection & Tracking | 100 | **submitted** |
| 2 | Object-Level Insight | 150 | **submitted** |
| 3 | Aggregate Insight | 200 | **CURRENT TASK** |
| 4 | Spatial Grounding | 100 | locked |
| 5 | Network Reasoning | 100 | locked |

### Level 3 brief (current task)

> Anything that summarises across many road users, a time window, or a region of the
> road counts here. Some examples:
> 1. Classified counts by movement and interval — turning movements at intersections,
>    directional and lane volumes on open road
> 2. Origin–destination distributions
> 3. Speed profiles across the segment, and where within it speeding concentrates
> 4. Lane volume and modal split by lane
> 5. Queue lengths
> 6. Density, occupancy and flow–density relationships

**Deliverable decided:** a **local dashboard** built from the exported data.
`flytbase_out/` has been downloaded to the laptop. Claude Code will be used for build.

---

## 2. Dataset

Two files, provided via Google Drive:

- `Intersection_Merged.MP4` — 6.0 GB, 06:39, 3840×2160 @ 29.97 fps ← **all work so far uses this**
- `Multi_Road_Merged.MP4` — 4.6 GB, 05:05 ← untouched, reserved for Level 5

Per-frame telemetry supplied as SRT (`int.srt`, also `Intersection_1080p.srt`).

### Scene

Urban arterial intersection in **Pune, India** at **18.566227 N, 73.771846 E**,
recorded **2026-08-21 17:39** (weekday evening). Landmarks visible in frame:
Croma, Samsung, Bosch, Wooden Street, a highrise residential tower, and active
construction on the east side.

Geometry: a **main diagonal arterial** running lower-left → upper-right, crossed by a
secondary road descending past Croma/Samsung. Signalised, with a pedestrian crossing
at bottom-right.

### Telemetry (measured over frames 1798–4496, the 90 s working window)

| field | value |
|---|---|
| `rel_alt` | 70.469 m, σ = 0.0067 |
| `abs_alt` | 607.269 m |
| `gb_pitch` | −63.1°, **exactly constant** |
| `gb_roll` | 0.0, **exactly constant** |
| `gb_yaw` | −126.10° median, σ = 0.26° (drifts −125.5 → −126.7 over full video) |
| `focal_len` | 24.00 (35 mm equivalent) |
| `dzoom_ratio` | 1.00 (no digital zoom) |
| `latitude/longitude` | static to 5 decimal places |

**Implication: static hover.** No ego-motion stabilisation was needed, and a single
fixed homography is valid for the entire clip.

**SRT parsing gotcha:** `FrameCnt` has skips and does NOT run 1:1 with frame index.
There are 11,971 SRT blocks against ~11,968 video frames, so **block order IS the
frame index**. Index the SRT by block position, not by `FrameCnt`.

---

## 3. What was built

### Compute environment
Google Colab, single **T4 GPU**. Video mounted from Google Drive (shortcut, not copy).
Throughput measured at **~1.8 sampled-frames/sec**. Full 6:39 video ≈ 36 min;
the 90 s working clip ≈ 8 min.

### Working window
**t = 60–150 s** (frames 1798–4496). Chosen for speed of iteration. Everything below
describes this window unless stated.

### Pipeline

**Detection** — `dronefreak/visdrone-yolov11s` (best.pt, 19 MB, 10 VisDrone classes)
from HuggingFace Hub.

- Tiled inference: **1024 px tiles, 20% overlap**, batched **12 tiles per forward pass**.
  Batching was the single biggest speedup — `sv.InferenceSlicer` calls the model once
  per tile sequentially and never fills the GPU.
- **Class-agnostic NMS across tile seams** (`with_nms(threshold=0.55, class_agnostic=True)`).
  Critical: one vehicle appears in up to 4 overlapping tiles and would otherwise be
  counted several times. Verified fixed — objects/frame settled at ~172–178, which is
  correct for this junction.
- `conf=0.10` deliberately low so ByteTrack's second-stage low-score association has
  detections to work with. Filtering at 0.25 upstream defeats the tracker's own
  recovery mechanism.
- Frame sampling: `cap.grab()` for skipped frames (demux without BGR conversion),
  `retrieve()` only on kept frames. **stride 3 → 10 Hz**, the standard rate for
  trajectory datasets.

**Tracking** — `sv.ByteTrack` (deprecated in supervision 0.28, removal planned 0.31).
NOTE: the tuned-kwargs path silently fell back to defaults (`[tracker kwargs] {}` was
printed), so ByteTrack ran **untuned**. This did not end up mattering because
stitching did the work instead.

**Stitching (the important post-process)** — raw ByteTrack fragmented badly:
1,320 tracks with median lifetime 2.5 s at a junction where a crossing takes 10–30 s.
Fragments are rejoined by:
- extrapolating each track's terminal velocity across gaps up to 2 s
- matching within a spatial gate scaled to object size (`3× sqrt(w*h)`, min 60 px,
  loosened proportionally with gap length)
- constrained to same class
- union-find to merge chains

Result: **1,320 → 838 tracks, median lifetime 2.5 s → 7.0 s, p90 48 s.**
Pedestrian count fell 580 → 314; the excess was one person counted as several.

**Class assignment** — confidence-weighted majority vote over the whole track, not
per-frame. A vehicle flickering car/van across 200 frames resolves to one label.

**ROI masking** — the **complement** of the drivable area was digitised in
makesense.ai as four polygons (buildings, glass frontage, forecourt parking, tree
canopy), then inverted. This removes two systematic error sources: parked vehicles in
forecourts producing permanent static tracks, and specular reflections in the Croma
glazing producing phantom detections. Tracks with <30% of points inside the ROI are dropped.

**Trace completeness metric** — each track's endpoints are tested against the ROI
boundary (within 140 px = "at the edge"). A correctly tracked road user enters AND
exits at the boundary; a fragment begins or ends mid-road. This is a **fragmentation
rate computed with no annotation**, reported per class. It was the differentiator in
the Level 1 submission.

---

## 4. Metric calibration — the hard part, and how it was resolved

This took several failed attempts. Documented so it is not repeated.

**Attempt 1 — analytic homography from gimbal telemetry.** Camera pose → ground plane
via pinhole geometry. First implementation had a wrong translation column and produced
a nearly-flat perspective gradient. Corrected version gave GSD 2.54 (bottom) → 3.93
(top) cm/px, ratio 1.55. Cars measured 3.17 m.

**Attempt 2 — fit tilt by optimising pedestrian speed uniformity. THIS WAS WRONG.**
Searched for the pitch making walking speed depth-invariant. Converged on −36.25°
with near 1.307 / far 1.314 m/s — looked excellent. **It was a false positive.**
Far-field pedestrians are tiny and jittery; tracking noise makes a standing person
read as ~1.3 m/s. The optimiser found where near-field *walking* equals far-field
*noise*. Detected by the frame extent: 421 m × 296 m, absurd for this junction.
Car length inflated to 6.32 m, HGV p75 to 29 m. **Do not use speed-uniformity fits
without a magnitude constraint.**

**Attempt 3 — ground control points. THIS IS WHAT SHIPPED.**
Four points on the main diagonal corridor, distances measured on Google Maps satellite:

```python
IMG = np.float32([[3400,0],[3780,310],[530,2060],[0,1780]])   # image px, clockwise
Wm, Dm = 17.9, 130.91                                          # metres, from Maps
GND = np.float32([[0,0],[Wm,0],[Wm,Dm],[0,Dm]])
H_i2g = cv2.getPerspectiveTransform(IMG, GND)

def to_ground(u, v):
    p = H_i2g @ np.array([u, v, 1.0])
    return float(p[0]/p[2]), float(p[1]/p[2])
```

Resulting GSD: **near 2.475, mid 2.806, far 3.224 cm per px** (across-track),
ratio 1.30. Agrees with the telemetry homography within ~10% despite sharing no inputs.

### Validation — three checks, none of them fitted

| check | result | expected |
|---|---|---|
| pedestrian walking speed | **1.232 m/s** | 1.2–1.5 |
| vehicle lengths, ordered | HGV 8.14 / LGV 5.52 / car 3.37 / three-wheeler 2.96 / cyclist 2.17 / motorcycle 1.51 m | correct order, realistic magnitudes |
| 6 full-corridor traversals | 149.3–151.3 m in 18.6–27.5 s | matches surveyed 130.91 m corridor + approaches |
| acceleration p1/p99 | −2.88 / +2.85 m/s² | within ±3 |

**Calibration note:** car median 3.37 m is correct for this fleet, not an error.
India's best-sellers are sub-3.7 m hatchbacks (Alto 3.40, S-Presso 3.57, WagonR 3.66).
An earlier assumption of 4.2 m (European average) caused a false alarm.

### Known calibration limitation
The GCP quad is a **thin strip** (17.9 × 130.91 m). Thin quads constrain across-track
scale well but the depth direction weakly — along-track GSD comes out nearly constant
(4.67–4.76) where it should vary more. **Speeds along the corridor are reliable;
cross-corridor distances are approximate.** State this in any Level 3 writeup that
uses lateral measurements.

---

## 5. Kinematics method

Positions differentiated with **Savitzky-Golay** (window 15, polynomial order 2),
not finite differences. SG fits a local polynomial and differentiates that, so
tracking jitter is not amplified — this matters far more for the second derivative
than the first. Window 11 gave accel p1/p99 = ±4; window 15 gave ±2.88.

```python
vx = savgol_filter(X, win, 2, deriv=1, delta=dt)
ax = savgol_filter(X, win, 2, deriv=2, delta=dt)
speed = np.hypot(vx, vy)
accel_tangential = (ax*vx + ay*vy) / speed     # component along heading
heading = np.degrees(np.arctan2(vx, vy)) % 360
```
`dt = stride/fps = 3/29.97 = 0.1001 s`

**Fine-grained classification by measured physical length** (VisDrone's van/truck
head is unreliable at this scale):
- car < 5.0 m · LGV 5.0–7.5 m · HGV-rigid 7.5–10.5 m · HGV-artic > 10.5 m
- vulnerable road users (pedestrian, cyclist, motorcycle, three-wheeler) keep their
  detection class — length does not disambiguate them
- length = median over frames where `speed_mps > 2` (axis-aligned boxes on stationary
  vehicles are unreliable)

---

## 6. Results as submitted (90 s window)

- **316 total tracks, 167 moving** (net displacement > 20 m). The other 149 are parked
  vehicles and static artefacts, excluded from all speed statistics.

| class | tracks | median km/h | IQR | max | median length |
|---|---|---|---|---|---|
| motorcycle | 97 | 16.8 | 11.4–20.2 | 55.9 | 1.51 m |
| car | 55 | 18.6 | 13.8–24.3 | 77.8 | 3.37 m |
| pedestrian | 10 | 4.2 | 3.2–4.9 | 12.0 | 1.18 m |
| LGV | 3 | 16.3 | 12.7–19.2 | 31.9 | 5.52 m |
| HGV-rigid | 1 | 20.1 | 16.2–27.8 | 49.5 | 8.14 m |
| three-wheeler | 1 | 25.2 | 23.0–33.4 | 41.7 | 2.96 m |

**Findings that carried the Level 2 submission:**
- **Motorcycles outnumber cars 97:55** — 58% of motorised road users are two-wheelers.
  Any capacity or emissions model assuming a four-wheeler fleet is wrong here.
- Heavy vehicles nearly absent (5 tracks total across LGV + HGV + three-wheeler).
- Motorcycles run **10% slower than cars** at the median despite greater agility,
  with a wider low-speed tail — they absorb the corridor's friction.

**Sample-size caveat that must be repeated in Level 3:** HGV-rigid and three-wheeler
are **one track each**; LGV is three. Those rows are single observations, not
distributions. Only car (55) and motorcycle (97) support statistical claims.

---

## 7. Files in `flytbase_out/` (downloaded to laptop)

### Level 2 primary — start here
| file | contents |
|---|---|
| `level2_kinematics.parquet` | **the main dataset.** Per-frame, per-track: `frame_idx, time_sec, track_id, class_final, X_s, Y_s, speed_mps, speed_kmh, accel_mps2, heading_deg, L, Wd, net_disp, interpolated` |
| `level2_per_track.csv` | one row per road user: `track_id, L, Wd, det_cls, net_disp, dur_s, v_med, v85, class_final` |
| `level2_summary.csv` | the speed table by class |
| `level2_validation.json` | GCP points, corridor dims, GSD, pedestrian check, accel percentiles |

### Level 1 artefacts
| file | contents |
|---|---|
| `tracks_masked.parquet` | ROI-filtered tracks, **image-space boxes**: `frame_idx, time_sec, track_id, class_id, class_name, class_group, conf, x1, y1, x2, y2, cx, cy_foot, w, h, interpolated, in_road` |
| `tracks_final.parquet` | pre-mask stitched tracks |
| `track_quality.csv` | per-track entry/exit completeness: `track_id, cls, n, dur, d_in, d_out, disp, enters, exits, complete, stationary` |
| `class_refined.csv` | per-track length + refined class |

### Media
`trace_h264.mp4` (trace tails), `clip_h264.mp4` (boxes only),
`level2_labeled_h264.mp4` (boxes + class + live speed + length + legend),
`all_traces.png` (every trajectory on one frame), `mask_check.png` (ROI overlay),
`l2_kinematics.png`, `level2_table.png`

### Telemetry
`int.srt` — per-frame drone telemetry, full video

**Key join:** `tracks_masked.parquet` has image-space boxes; `level2_kinematics.parquet`
has ground coordinates and speeds. Merge on `['frame_idx','track_id']` when you need both.

**Coordinate frames:** `X_s, Y_s` in `level2_kinematics.parquet` are **metres in the
GCP corridor frame** — X across the carriageway (0–17.9 m), Y along it (0–130.91 m).
An earlier version used an East/North rotation via gimbal yaw; the shipped version does
NOT — it is the raw corridor frame. Verify axis semantics before writing directional logic.

---

## 8. Reproducing / extending

Reference implementations that exist as scripts in Colab (not in `flytbase_out`):
`track.py` (detect + track), `stitch.py` (fragment rejoin), `render.py` (box video),
`trace.py` (trace-tail video), `label.py` (labelled video). Recreate as needed; the
methods are documented above.

To extend to the **full 6:39 video** (~36 min on one T4):
```
python track.py --video "$VID" --out-dir full --stride 3 --device cuda
python stitch.py full/tracks.parquet full/tracks_final.parquet 3
```
Then re-apply the ROI mask and the GCP homography unchanged — the drone never moved,
so calibration holds for the entire video.

To shard across Kaggle's 2×T4: run non-overlapping `--start-sec`/`--max-seconds`
windows with `CUDA_VISIBLE_DEVICES=0/1`, offset each shard's `track_id` by
`shard_idx * 1_000_000`, concat.

---

## 9. Recommendations for Level 3

The brief rewards **aggregate** insight. Highest-value items given what already exists:

1. **Turning movements by class.** Define approach/exit zones at the junction, classify
   each moving track by (entry zone → exit zone), cross-tabulate against `class_final`.
   This is the single most requested traffic-engineering output and the data supports it
   directly. Origin–destination distribution falls out of the same computation.
2. **Speed profile along the corridor.** `Y_s` is distance along the carriageway —
   bin it and plot speed vs position to show *where* speeding or slowdown concentrates,
   not just that it exists. Directly answers brief item 3.
3. **Flow–density relationship.** Count road users per 10 s interval (flow) and mean
   occupancy per corridor segment (density). A fundamental diagram from drone data is
   a strong, quantitative deliverable.
4. **Modal split by lane.** `X_s` is lateral position across the 17.9 m carriageway.
   Cluster it to recover lanes, then report volume and modal mix per lane. Expect
   motorcycle lateral position to be far more dispersed than cars — that dispersion
   is itself a finding about lane discipline.
5. **Queue length in metres.** Identify stopped clusters (`speed_mps < 0.5`) and measure
   their extent along `Y_s`. Brief item 5, and directly enables signal-timing insight.

**Caveats to carry forward into any Level 3 claim:**
- Only car and motorcycle have adequate sample size for per-class statistics.
- Cross-corridor (lateral, `X_s`) distances are approximate — the GCP quad is thin.
  Lane-level work should acknowledge this or re-fit the homography with a wider quad.
- 90 s is a short window for flow statistics. Consider running the full 6:39 for
  Level 3 aggregate work; the calibration transfers unchanged.
- 149 of 316 tracks are static and must stay excluded from flow/speed aggregates.

**Dashboard build:** data is small (a few MB of parquet/CSV) — everything can be
precomputed and served client-side. No backend required.
