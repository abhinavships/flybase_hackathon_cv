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


def read_window(path, start_sec, max_seconds, frame_stride=1):
    """Yield frames of `path` in [start_sec, start_sec+max_seconds), keeping only
    every `frame_stride`-th frame (skipped frames are grabbed, not decoded to BGR,
    which is cheaper). Returns (generator, effective_fps, n_frames_yielded)."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = int(round(start_sec * fps))
    end_f = int(round((start_sec + max_seconds) * fps)) if max_seconds > 0 else total
    end_f = min(end_f, total) if total > 0 else end_f
    if start_f > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    s = max(1, frame_stride)
    n_yield = (end_f - start_f + s - 1) // s

    def gen():
        i = start_f
        try:
            while i < end_f:
                if (i - start_f) % s != 0:
                    cap.grab()          # advance without the BGR decode/copy
                    i += 1
                    continue
                ok, fr = cap.read()
                if not ok:
                    break
                yield fr
                i += 1
        finally:
            cap.release()

    return gen(), fps / s, max(n_yield, 0)


def run(args) -> None:
    device = pick_device(args.device)
    model = load_model(args.weights)
    print(f"[info] device={device} imgsz={args.imgsz} tracker={args.tracker} conf={args.conf}",
          file=sys.stderr)

    kw = dict(
        persist=True, tracker=f"{args.tracker}.yaml", conf=args.conf, iou=args.iou,
        imgsz=args.imgsz, device=device, classes=args.classes, verbose=False,
    )

    # Benchmark with no real source: feed synthetic frames one at a time.
    if args.benchmark and not args.source:
        _loop(model, synthetic_frames(args.benchmark), kw, show=False, out=None,
              limit=args.benchmark, label="synthetic")
        return

    src = int(args.source) if args.source.isdigit() else args.source

    # Windowed file source: decode with cv2, seek to the window, track per frame.
    if isinstance(src, str) and (args.start_sec > 0 or args.max_seconds > 0):
        frames, eff_fps, nwin = read_window(src, args.start_sec, args.max_seconds,
                                            frame_stride=args.frame_stride)
        end = args.start_sec + args.max_seconds if args.max_seconds > 0 else -1
        # eff_fps = source_fps / frame_stride is the rate we must process at to keep
        # up with live playback. Subsample the WRITTEN file further to --out-fps.
        stride = max(1, round(eff_fps / args.out_fps)) if args.out_fps > 0 else 1
        print(f"[info] window [{args.start_sec:.0f}s, {end:.0f}s) frame_stride={args.frame_stride} "
              f"-> {nwin} frames @ {eff_fps:.2f} fps effective; need >= {eff_fps:.1f} FPS for real-time",
              file=sys.stderr)
        _loop(model, frames, kw, show=not args.no_show, out=(args.out or None), limit=0,
              label=f"{Path(str(src)).name}[{args.start_sec:.0f}-{end:.0f}s]/stride{args.frame_stride}",
              total=nwin, out_fps=eff_fps / stride, out_width=args.out_width,
              write_stride=stride, realtime_target=eff_fps)
        return

    # Whole source: let Ultralytics own the decode loop (stream=True generator).
    results = model.track(source=src, stream=True, **kw)
    _loop(model, results, None, show=not args.no_show, out=(args.out or None),
          limit=args.benchmark or 0, label=str(src), is_results=True,
          out_width=args.out_width)


def _loop(model, iterable, kw, show, out, limit, label, is_results=False,
          total=0, out_fps=0.0, out_width=0, write_stride=1, realtime_target=15.0):
    """Drive frames through detect+track, overlay FPS, optionally show/write.
    Two entry shapes: raw frames (call model.track per frame) or an Ultralytics
    results generator (already tracked). `out_fps` sets the output file's playback
    rate (use the source fps so a saved clip plays at real speed); `out_width`
    downscales the written/shown frame."""
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
                result = model.track(item, **kw)[0]  # one frame at a time
                frame = result.plot()

            n += 1
            elapsed = time.time() - t0
            if elapsed > 0:
                fps = n / elapsed
            ntrk = 0 if result.boxes is None or result.boxes.id is None else len(result.boxes.id)

            if out_width and frame.shape[1] > out_width:
                s = out_width / frame.shape[1]
                frame = cv2.resize(frame, (out_width, int(frame.shape[0] * s)))

            if show or out:
                cv2.putText(frame, f"{fps:5.1f} FPS  |  {ntrk} tracked",
                            (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if out and (n - 1) % write_stride == 0:
                if writer is None:
                    h, w = frame.shape[:2]
                    fps_out = out_fps or max(fps, 10)
                    # avc1 (H.264) compresses ~5-10x smaller than mp4v; fall back if
                    # the platform's OpenCV lacks the encoder.
                    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), fps_out, (w, h))
                    if not writer.isOpened():
                        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))
                writer.write(frame)
            if show:
                cv2.imshow("realtime_track — q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if n % 100 == 0:
                pct = f" ({100*n/total:.0f}%)" if total else ""
                print(f"[info] {n}{('/'+str(total)) if total else ''} frames{pct} "
                      f"| {fps:.1f} FPS live | {ntrk} tracked", file=sys.stderr)
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
    verdict = (f"REAL-TIME (>= {realtime_target:.1f} FPS needed)" if fps >= realtime_target
               else f"below real-time ({realtime_target:.1f} FPS needed)")
    print(f"[result] source={label} frames={n} elapsed={elapsed:.2f}s => {fps:.2f} FPS  ({verdict})")
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
    p.add_argument("--start-sec", type=float, default=0.0, help="window start in source seconds")
    p.add_argument("--max-seconds", type=float, default=0.0, help="window length in seconds (0 = to end)")
    p.add_argument("--frame-stride", type=int, default=1, help="process every Nth frame (N>1 = real-time via frame-skip on slow hardware)")
    p.add_argument("--out-width", type=int, default=0, help="downscale written/shown frame to this width (0 = full)")
    p.add_argument("--out-fps", type=float, default=0.0, help="subsample the written file to this fps (0 = every frame); playback stays real-time")
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
