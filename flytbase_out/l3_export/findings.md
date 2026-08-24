# Level 3 — Aggregate Insight

Window 90 s · 167 moving road users (149 static tracks excluded from every statistic below).

## Headline

- Throughput **6,688 veh/h**, **4,449 PCU/h**.
- Two-wheelers are **62%** of motorised road users. Capacity and emissions models built on a four-wheeler fleet do not describe this junction.
- Median speed **17.6 km/h**, median stopped delay **0.0 s** per road user.

## Fleet composition

| class_final   |   tracks |   median_kmh |   p85_kmh |   max_kmh |   median_len_m |   median_dur_s |   median_delay_s |   share | reliable             |
|:--------------|---------:|-------------:|----------:|----------:|---------------:|---------------:|-----------------:|--------:|:---------------------|
| motorcycle    |       97 |        17.27 |     20.97 |     55.93 |           1.51 |          17.92 |             0.00 |    0.58 | yes                  |
| car           |       55 |        19.54 |     25.20 |     77.80 |           3.60 |          22.12 |             0.00 |    0.33 | yes                  |
| pedestrian    |       10 |         4.50 |      5.02 |     11.97 |           1.20 |          37.74 |             0.10 |    0.06 | SINGLE-OBS / small-n |
| LGV           |        3 |        16.81 |     19.91 |     31.92 |           5.14 |          13.11 |             0.00 |    0.02 | SINGLE-OBS / small-n |
| HGV-rigid     |        1 |        20.13 |     29.17 |     49.50 |           8.14 |          13.31 |             0.00 |    0.01 | SINGLE-OBS / small-n |
| three-wheeler |        1 |        25.23 |     35.80 |     41.73 |           2.67 |          18.32 |             0.00 |    0.01 | SINGLE-OBS / small-n |

> Sample-size caveat: pedestrian, LGV, HGV-rigid, three-wheeler have fewer than 20 tracks. Those rows are observations, not distributions.


## Turning movements and OD

Movement split: **through** 108, **left** 31, **right** 21, **u-turn** 7.

OD matrix (rows = entry leg, columns = exit leg):

| leg_in   |   A |   B |   C |   D |
|:---------|----:|----:|----:|----:|
| A        |   0 |   7 |   5 |   0 |
| B        |   4 |   0 |   2 |  13 |
| C        |   9 |  44 |   0 |  18 |
| D        |   5 |  26 |   5 |   0 |

> Legs are k-means clusters over track entry/exit points; compass labels are in the corridor frame (+Y = 'N'), not true north.


## Speed along the corridor

- Slowest point: Y = **62 m** at 4.6 km/h median (+Y (up-corridor)).
- Fastest point: Y = **138 m** at 34.2 km/h median.
- Speeding concentrates at Y = **138 m**, where 69% of samples exceed 30 km/h.
- Speeds are along-track and GCP-calibrated, so these are the reliable numbers in this report.


## Lanes and modal split

| lane_key              | class_final   |   count |   pcu |   v_med_kmh |   share |
|:----------------------|:--------------|--------:|------:|------------:|--------:|
| +Y (up-corridor) L1   | motorcycle    |       2 |  1.00 |        4.34 |    1.00 |
| +Y (up-corridor) L2   | motorcycle    |      11 |  5.50 |       15.84 |    0.65 |
| +Y (up-corridor) L2   | car           |       5 |  5.00 |       17.84 |    0.29 |
| +Y (up-corridor) L2   | three-wheeler |       1 |  1.20 |       26.40 |    0.06 |
| +Y (up-corridor) L3   | car           |       3 |  3.00 |       12.82 |    0.50 |
| +Y (up-corridor) L3   | motorcycle    |       3 |  1.50 |       14.19 |    0.50 |
| +Y (up-corridor) L4   | pedestrian    |       5 |  0.00 |        4.80 |    1.00 |
| +Y (up-corridor) L5   | pedestrian    |       1 |  0.00 |       10.95 |    1.00 |
| -Y (down-corridor) L1 | motorcycle    |      27 | 13.50 |       17.79 |    0.66 |
| -Y (down-corridor) L1 | car           |      14 | 14.00 |       18.75 |    0.34 |
| -Y (down-corridor) L2 | motorcycle    |      16 |  8.00 |       11.14 |    0.52 |
| -Y (down-corridor) L2 | car           |      13 | 13.00 |       13.27 |    0.42 |
| -Y (down-corridor) L2 | HGV-rigid     |       1 |  2.20 |       15.58 |    0.03 |
| -Y (down-corridor) L2 | pedestrian    |       1 |  0.00 |        5.03 |    0.03 |

Lateral dispersion by class:

| class_final   |   n_tracks |   within_track_sd_m |   within_track_swing_m |   across_sd_m |   across_p10_p90_m |
|:--------------|-----------:|--------------------:|-----------------------:|--------------:|-------------------:|
| motorcycle    |         89 |                0.82 |                   2.70 |          9.94 |              19.24 |
| car           |         50 |                1.08 |                   3.33 |         12.45 |              22.71 |
| pedestrian    |         10 |                9.93 |                  31.77 |         16.96 |              39.12 |
| LGV           |          3 |                0.42 |                   1.40 |         10.59 |              15.65 |
| HGV-rigid     |          1 |                0.54 |                   1.87 |        nan    |               0.00 |
| three-wheeler |          1 |                0.78 |                   2.65 |        nan    |               0.00 |

> Lateral (X) distances come from the weak axis of a thin calibration quad. Lane counts and modal mix are sound; absolute lane widths are indicative only.


## Queues

- Longest queue observed **21.2 m** (3 vehicles).
- Mean queue length **18.5 m**.
- Inferred stop line, direction -Y: Y = **54.8 m** (IQR 0.1 m).
- Stop-line position is inferred purely from where vehicles repeatedly come to rest — no signal-phase data was used.


## Flow–density (Edie's definitions)

- Observed capacity **7,316 veh/h** at **404 veh/km**, space-mean speed 18.1 km/h.
- Density range observed: 1–581 veh/km over 81 space–time cells.
- Flow and density are computed as total distance and total time per space–time cell, not from a detector line, so the estimate does not depend on where a counter happens to be placed.


## Headways

Median time headway **1.00 s** at the counting line (Y = 64 m); 15th percentile 0.34 s.


## Method notes

- Trajectories at 10 Hz in a metric ground frame; positions differentiated with Savitzky–Golay (window 15, order 2).
- Static tracks (net displacement < 20 m) excluded from every flow and speed statistic.
- Stopped threshold 0.5 m/s; queue clustering gap 12 m.
- PCU factors IRC-style: motorcycle 0.5, car 1.0, three-wheeler 1.2, LGV 1.4, HGV 2.2.
- Along-track (Y) geometry is GCP-calibrated and reliable; across-track (X) geometry is approximate because the calibration quad is a thin strip.