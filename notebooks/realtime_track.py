#!/usr/bin/env python3
"""
realtime_track.py

Real-time vehicle detection + tracking, meant to run *live* as traffic moves
(webcam / RTSP / video file), as opposed to the offline tiled pipeline in
01_track.py.

Why a separate script: 01_track.py runs YOLO through supervision's
InferenceSlicer (many tiled inferences per frame) for maximum small-object
recall on 4K drone frames. That is accurate but far too slow for live use. For
real time we run the SAME VisDrone-tuned YOLO11s whole-frame at a tunable
--imgsz, and lean on Ultralytics' built-in tracker (ByteTrack / BoT-SORT) via
model.track(..., persist=True) — no extra tracking dependency.

Sources:
  --source 0                 default webcam
  --source path/to/vid.mp4   a video file
  --source rtsp://host/...    a live RTSP stream (drone / CCTV)

Modes:
  (default)      annotated live window with a running FPS overlay
  --no-show      headless; writes an annotated mp4 to --out (no GUI needed)
  --benchmark N  process N frames, no display, print detect+track FPS. If no
                 --source is given it feeds synthetic frames so you can measure
                 the loop's raw throughput on this machine with no camera/video.

Usage:
  python realtime_track.py --source 0
  python realtime_track.py --source intersection.mp4 --imgsz 960
  python realtime_track.py --benchmark 120 --imgsz 736      # throughput check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# VisDrone-tuned YOLO11s — the exact checkpoint 01_track.py already uses.
HF_REPO_ID = "dronefreak/visdrone-yolov11s"
HF_FILENAME = "best.pt"

# VisDrone class ids that are motor vehicles / road users worth tracking live.
# (VisDrone order: 0 pedestrian,1 people,2 bicycle,3 car,4 van,5 truck,
#  6 tricycle,7 awning-tricycle,8 bus,9 motor)
DEFAULT_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]


def pick_device(forced: str = "auto") -> str:
    import torch
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if forced != "auto":
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"


def load_model(weights: str | None):
    from ultralytics import YOLO
    if weights:
        return YOLO(weights)
    from huggingface_hub import hf_hub_download
    print(f"[info] downloading {HF_REPO_ID}/{HF_FILENAME} ...", file=sys.stderr)
    return YOLO(hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME))


def synthetic_frames(n: int, wh=(1280, 720)):
    """Yield n frames with a few moving bright blobs. YOLO will mostly find
    nothing in these, which is fine: --benchmark measures loop throughput
    (decode-free detect+track FPS), not detection quality."""
    w, h = wh
    rng = np.random.default_rng(0)
    blobs = rng.uniform([0, 0], [w, h], size=(6, 2))
    vel = rng.uniform(-6, 6, size=(6, 2))
    for _ in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        blobs = (blobs + vel) % [w, h]
        for (bx, by) in blobs.astype(int):
            frame[max(by - 15, 0):by + 15, max(bx - 25, 0):bx + 25] = (60, 200, 255)
        yield frame


def run(args) -> None:
    device = pick_device(args.device)
    model = load_model(args.weights)
    print(f"[info] device={device} imgsz={args.imgsz} tracker={args.tracker} conf={args.conf}",
          file=sys.stderr)

    track_kwargs = dict(
        persist=True, stream=True, tracker=f"{args.tracker}.yaml",
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=device,
        classes=args.classes, verbose=False,
    )

    # Benchmark with no real source: feed synthetic frames one at a time.
    if args.benchmark and not args.source:
        _loop(model, synthetic_frames(args.benchmark), track_kwargs,
              show=False, out=None, limit=args.benchmark, label="synthetic")
        return

    # Real source: let Ultralytics own the decode loop (stream=True is a generator).
    src = int(args.source) if args.source.isdigit() else args.source
    results = model.track(source=src, **track_kwargs)
    _loop(model, results, None, show=not args.no_show, out=args.out,
          limit=args.benchmark or 0, label=str(src), is_results=True)


def _loop(model, iterable, track_kwargs, show, out, limit, label, is_results=False):
    """Drive frames through detect+track, overlay FPS, optionally show/write.
    Two entry shapes: raw frames (call model.track per frame) or an Ultralytics
    results generator (already tracked)."""
    import cv2

    writer = None
    n = 0
    t0 = time.time()
    fps = 0.0
    try:
        for item in iterable:
            if is_results:
                result = item
                frame = result.plot()  # BGR annotated frame
            else:
                # raw-frame path (synthetic benchmark): track this single frame.
                # stream=True is dropped here since we pass frames one at a time.
                kw = {k: v for k, v in track_kwargs.items() if k != "stream"}
                result = model.track(item, **kw)[0]
                frame = result.plot()

            n += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                fps = n / elapsed
            ntrk = 0 if result.boxes is None or result.boxes.id is None else len(result.boxes.id)

            if show or out:
                cv2.putText(frame, f"{fps:5.1f} FPS  |  {ntrk} tracked",
                            (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if out:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(fps, 10), (w, h))
                writer.write(frame)
            if show:
                cv2.imshow("realtime_track — q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if limit and n >= limit:
                break
    finally:
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    elapsed = time.time() - t0
    fps = n / elapsed if elapsed > 0 else 0.0
    print(f"[result] source={label} frames={n} elapsed={elapsed:.2f}s "
          f"=> {fps:.2f} FPS  ({'REAL-TIME' if fps >= 15 else 'below 15 fps'})")
    if out:
        print(f"[info] wrote annotated video -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="", help="0 for webcam, a video path, or an rtsp:// url")
    p.add_argument("--weights", default="", help="path to a .pt (default: download VisDrone YOLO11s)")
    p.add_argument("--imgsz", type=int, default=960, help="inference size; lower = faster")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--classes", type=int, nargs="*", default=DEFAULT_CLASSES,
                   help="VisDrone class ids to keep (default: all road users)")
    p.add_argument("--no-show", action="store_true", help="headless; write --out instead of a window")
    p.add_argument("--out", default="realtime_annotated.mp4", help="output path when --no-show")
    p.add_argument("--benchmark", type=int, default=0, metavar="N",
                   help="process N frames headless and print FPS (synthetic if no --source)")
    args = p.parse_args()

    if args.no_show is False and args.benchmark:
        args.no_show = True
    if not args.no_show:
        args.out = ""  # live window only, don't also write a file
    args.weights = args.weights or None
    run(args)


if __name__ == "__main__":
    main()
