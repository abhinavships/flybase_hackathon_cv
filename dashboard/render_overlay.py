"""
render_overlay.py — render a filtered tracking overlay from the source MP4.

Draws only the classes you ask for, with optional trajectory trails and live speed
labels, and can put raw footage next to the annotated version in one frame so the
"before / after tracking" comparison is a single file.

Examples
--------
# motorcycles only, 4 s trails, split screen
python render_overlay.py --video Intersection_Merged.MP4 --data flytbase_out \
    --out moto.mp4 --start 60 --end 150 --classes motorcycle --trail-sec 4 --side-by-side

# everything, boxes only, no trails
python render_overlay.py --video Intersection_Merged.MP4 --data flytbase_out \
    --out all.mp4 --start 60 --end 150 --trail-sec 0
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import l3_core as C


def bgr(cls: str):
    h = C.color_for(cls).lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--data", required=True, help="flytbase_out folder")
    ap.add_argument("--out", default="render_filtered.mp4")
    ap.add_argument("--start", type=float, default=None, help="seconds (track time base)")
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--classes", default="", help="comma-separated class_final values")
    ap.add_argument("--trail-sec", type=float, default=3.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--labels", action="store_true", help="draw class + live speed")
    ap.add_argument("--boxes", dest="boxes", action="store_true", default=True)
    ap.add_argument("--no-boxes", dest="boxes", action="store_false")
    a = ap.parse_args()

    data = Path(a.data)
    tk_path = data / "tracks_masked.parquet"
    if not tk_path.exists():
        tk_path = data / "tracks_final.parquet"
    if not tk_path.exists():
        sys.exit("Need tracks_masked.parquet (or tracks_final.parquet) in --data")

    tk = pd.read_parquet(tk_path)
    need = {"frame_idx", "track_id", "x1", "y1", "x2", "y2"}
    if not need.issubset(tk.columns):
        sys.exit(f"{tk_path.name} missing columns: {need - set(tk.columns)}")

    # attach the refined class + speed from the Level 2 outputs
    kin = pd.read_parquet(data / "level2_kinematics.parquet")
    cls_map = kin.groupby("track_id")["class_final"].first()
    tk["class_final"] = tk["track_id"].map(cls_map)
    tk["class_final"] = tk["class_final"].fillna(
        tk["class_name"] if "class_name" in tk.columns else "unknown")

    spd = kin.set_index(["frame_idx", "track_id"])["speed_kmh"]
    tk = tk.join(spd, on=["frame_idx", "track_id"])

    if a.classes:
        wanted = [c.strip() for c in a.classes.split(",") if c.strip()]
        tk = tk[tk["class_final"].isin(wanted)]
    if a.start is not None and "time_sec" in tk.columns:
        tk = tk[tk["time_sec"] >= a.start]
    if a.end is not None and "time_sec" in tk.columns:
        tk = tk[tk["time_sec"] <= a.end]
    if tk.empty:
        sys.exit("No detections left after filtering.")

    frames_wanted = np.sort(tk["frame_idx"].unique())
    by_frame = {f: g for f, g in tk.groupby("frame_idx", sort=False)}
    stride = int(np.median(np.diff(frames_wanted))) if len(frames_wanted) > 1 else 1

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {a.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = src_fps / max(stride, 1)

    scale = a.width / W
    ow, oh = a.width, int(round(H * scale / 2) * 2)
    fw = ow * 2 if a.side_by_side else ow

    tmp = str(Path(a.out).with_suffix(".raw.mp4"))
    writer = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*fourcc), out_fps, (fw, oh))
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        sys.exit("Could not open a VideoWriter.")

    trail_frames = int(a.trail_sec * out_fps)
    trails = defaultdict(lambda: deque(maxlen=max(trail_frames, 1)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames_wanted[0]))
    pos = int(frames_wanted[0])
    written = 0
    total = len(frames_wanted)

    for target in frames_wanted:
        while pos < target:                 # demux without decoding to BGR
            cap.grab()
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if not ok:
            break

        raw = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        ann = raw.copy()
        g = by_frame.get(target)
        if g is not None:
            for r in g.itertuples():
                col = bgr(r.class_final)
                x1, y1 = int(r.x1 * scale), int(r.y1 * scale)
                x2, y2 = int(r.x2 * scale), int(r.y2 * scale)
                cxp, cyp = (x1 + x2) // 2, (y1 + y2) // 2
                if trail_frames > 0:
                    trails[r.track_id].append((cxp, cyp, col))
                if a.boxes:
                    cv2.rectangle(ann, (x1, y1), (x2, y2), col, 2)
                if a.labels:
                    v = getattr(r, "speed_kmh", np.nan)
                    txt = f"{r.class_final}"
                    if v == v:
                        txt += f" {v:.0f}"
                    cv2.putText(ann, txt, (x1, max(y1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        if trail_frames > 0:
            live = set(g["track_id"]) if g is not None else set()
            for tid in list(trails.keys()):
                pts = trails[tid]
                if tid not in live:
                    pts.popleft() if pts else trails.pop(tid, None)
                    continue
                for i in range(1, len(pts)):
                    cv2.line(ann, pts[i - 1][:2], pts[i][:2], pts[i][2],
                             1 + int(2 * i / len(pts)), cv2.LINE_AA)

        n = 0 if g is None else len(g)
        hud = f"t={target/src_fps:6.1f}s  frame {target}  objects {n}"
        cv2.rectangle(ann, (0, 0), (ow, 26), (0, 0, 0), -1)
        cv2.putText(ann, hud, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        if a.side_by_side:
            cv2.putText(raw, "RAW", (8, oh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ann, "TRACKED", (8, oh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            frame_out = np.hstack([raw, ann])
        else:
            frame_out = ann

        writer.write(frame_out)
        written += 1
        if written % 50 == 0:
            print(f"  {written}/{total} frames", flush=True)

    writer.release()
    cap.release()

    # re-encode to H.264 so browsers (and st.video) will play it
    ff = shutil.which("ffmpeg")
    if ff:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", tmp,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", a.out],
                       check=False)
        if Path(a.out).exists():
            Path(tmp).unlink(missing_ok=True)
        else:
            shutil.move(tmp, a.out)
    else:
        shutil.move(tmp, a.out)
        print("ffmpeg not found — output may not play in the browser. "
              "brew install ffmpeg / apt install ffmpeg to fix.")

    print(f"wrote {a.out}  ({written} frames @ {out_fps:.2f} fps)")


if __name__ == "__main__":
    main()
