#!/usr/bin/env python3
"""
01_track.py

Detect and track objects in Intersection_Merged.MP4 with a YOLOv11s model
fine-tuned on VisDrone, run through supervision.InferenceSlicer for tiled
inference on large frames, and sv.ByteTrack for multi-object tracking.

Outputs (in --out-dir, default: current directory):
  - tracks.parquet        one row per tracked detection, for every --stride-th
                           frame within [--start-sec, --start-sec + --max-seconds)
                           of the source video.
  - tracks_annotated.mp4  boxes + class label + track ID, covering the first
                           --annotate-seconds of that same window.

By default only a --max-seconds window (90s) of the source video is processed,
not the whole file -- pass a larger/0 --max-seconds to cover more (0 = to end).

Usage:
  python 01_track.py                                   # first 90s, auto device
  python 01_track.py --start-sec 120 --max-seconds 30   # a 30s slice starting at t=120s
  python 01_track.py --max-seconds 0 --device cuda      # full video on a CUDA GPU
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

# NOTE: the actual Hugging Face repo id is "dronefreak/visdrone-yolov11s"
# (words swapped relative to the commonly-quoted "yolov11s-visdrone"); the
# repo hosts a `best.pt` Ultralytics YOLO11s checkpoint fine-tuned on VisDrone.
HF_REPO_ID = "dronefreak/visdrone-yolov11s"
HF_FILENAME = "best.pt"

DEFAULT_VIDEO = "Intersection_Merged.MP4"
DEFAULT_OUT_DIR = "."

STRIDE_DEFAULT = 5          # process every Nth frame of the source video
START_SEC_DEFAULT = 0.0     # start of the processing window, source-video seconds
MAX_SECONDS_DEFAULT = 90.0  # length of the processing window; <=0 means "to end"
SLICE_WH = (1024, 1024)     # InferenceSlicer tile size (w, h)
OVERLAP_RATIO = 0.2         # 20% tile overlap
ANNOTATE_SECONDS = 30.0     # length, within the processing window, of the annotated clip
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5


def pick_device(forced: str = "auto") -> tuple[str, bool]:
    """Resolve the torch device to run on. fp16 is only used on CUDA; MPS/CPU
    fall back to fp32 since half precision is unsupported or unstable there
    for this model. `forced` may be "auto", "cuda", "mps", or "cpu"."""
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if forced == "cuda":
        if not torch.cuda.is_available():
            sys.exit("--device cuda requested but no CUDA GPU is available")
        return "cuda", True
    if forced == "mps":
        if not has_mps:
            sys.exit("--device mps requested but MPS is not available")
        return "mps", False
    if forced == "cpu":
        return "cpu", False
    if forced != "auto":
        sys.exit(f"unknown --device value: {forced!r}")

    # auto: cuda -> mps -> cpu
    if torch.cuda.is_available():
        return "cuda", True
    if has_mps:
        print(
            "[warn] no CUDA GPU found; using Apple MPS. fp16 is unreliable on "
            "MPS for this model, running inference in fp32 instead.",
            file=sys.stderr,
        )
        return "mps", False
    print(
        "[warn] no GPU found; running on CPU in fp32. This will be slow.",
        file=sys.stderr,
    )
    return "cpu", False


def build_slicer(model: YOLO, device: str, use_half: bool) -> sv.InferenceSlicer:
    def callback(image_slice: np.ndarray) -> sv.Detections:
        predict_kwargs = dict(device=device, conf=CONF_THRESHOLD, verbose=False)
        if use_half:
            # recent ultralytics replaced the bare `half=True` flag with a
            # unified `quantize` scheme; 16 selects fp16.
            predict_kwargs["quantize"] = 16
        result = model.predict(image_slice, **predict_kwargs)[0]
        return sv.Detections.from_ultralytics(result)

    base_kwargs = dict(
        callback=callback,
        slice_wh=SLICE_WH,
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
        iou_threshold=IOU_THRESHOLD,
        thread_workers=1,
    )
    # supervision >=0.26 removed overlap_ratio_wh in favor of pixel-based overlap_wh;
    # its deprecation shims make inspect.signature unreliable, so just try both.
    try:
        return sv.InferenceSlicer(
            overlap_wh=(int(SLICE_WH[0] * OVERLAP_RATIO), int(SLICE_WH[1] * OVERLAP_RATIO)),
            **base_kwargs,
        )
    except TypeError:
        return sv.InferenceSlicer(overlap_ratio_wh=(OVERLAP_RATIO, OVERLAP_RATIO), **base_kwargs)


def build_tracker(frame_rate: float) -> sv.ByteTrack:
    # supervision >=0.24 renamed ByteTrack's constructor args; its deprecation
    # shims make inspect.signature unreliable, so just try both naming schemes.
    try:
        return sv.ByteTrack(
            track_activation_threshold=CONF_THRESHOLD,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=max(frame_rate, 1.0),
        )
    except TypeError:
        return sv.ByteTrack(
            track_thresh=CONF_THRESHOLD,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=max(frame_rate, 1.0),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="path to the source video")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="directory for outputs")
    parser.add_argument("--stride", type=int, default=STRIDE_DEFAULT, help="process every Nth frame")
    parser.add_argument("--start-sec", type=float, default=START_SEC_DEFAULT,
                         help="start of the processing window, in source-video seconds")
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS_DEFAULT,
                         help="length of the processing window in seconds; <=0 means to end of video")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                         help="inference device; auto tries cuda, then mps, then cpu")
    parser.add_argument("--annotate-seconds", type=float, default=ANNOTATE_SECONDS,
                         help="length, within the processing window, of the annotated clip")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "tracks.parquet"
    annotated_path = out_dir / "tracks_annotated.mp4"

    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    device, use_half = pick_device(args.device)
    print(f"[info] device={device} fp16={use_half}")

    print(f"[info] downloading weights {HF_REPO_ID}/{HF_FILENAME} from the Hugging Face Hub ...")
    weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    model = YOLO(weights_path)
    class_names = model.names  # {class_id: class_name}

    slicer = build_slicer(model, device, use_half)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_fps = source_fps / args.stride
    video_duration_s = total_frames / source_fps

    end_sec = args.start_sec + args.max_seconds if args.max_seconds > 0 else video_duration_s
    end_sec = min(end_sec, video_duration_s)
    window_s = max(end_sec - args.start_sec, 0.0)
    expected_processed = max(int(window_s * source_fps / args.stride), 1)

    print(
        f"[info] source: {width}x{height} @ {source_fps:.2f} fps, "
        f"{total_frames} frames (~{video_duration_s:.1f}s)"
    )
    print(
        f"[info] window: [{args.start_sec:.1f}s, {end_sec:.1f}s) every {args.stride} frame(s) -> "
        f"~{expected_processed} sampled frames at {sampled_fps:.2f} fps; "
        f"annotating first {args.annotate_seconds:.0f}s of that window"
    )

    # Keyframe-based seeking on mp4/h264 is approximate; good enough to bound a test window.
    start_frame = int(round(args.start_sec * source_fps))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tracker = build_tracker(sampled_fps)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        sampled_fps,
        (width, height),
    )

    records: list[dict] = []
    frame_idx = start_frame
    processed = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_s = frame_idx / source_fps
            if timestamp_s >= end_sec:
                break

            if frame_idx % args.stride == 0:
                detections = slicer(frame)
                detections = tracker.update_with_detections(detections)

                labels = []
                for i in range(len(detections)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                    cls_id = int(detections.class_id[i]) if detections.class_id is not None else -1
                    conf = float(detections.confidence[i]) if detections.confidence is not None else float("nan")
                    cname = class_names.get(cls_id, str(cls_id))

                    records.append(
                        {
                            "frame_idx": frame_idx,
                            "timestamp_s": timestamp_s,
                            "track_id": tid,
                            "class_id": cls_id,
                            "class_name": cname,
                            "conf": conf,
                            "x1": float(x1),
                            "y1": float(y1),
                            "x2": float(x2),
                            "y2": float(y2),
                            "cx": float((x1 + x2) / 2.0),
                            "cy_foot": float(y2),  # bottom-centre of the box
                        }
                    )
                    labels.append(f"#{tid} {cname} {conf:.2f}")

                if (timestamp_s - args.start_sec) <= args.annotate_seconds:
                    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
                    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                    writer.write(annotated)

                processed += 1
                if processed >= 20 and processed % 20 == 0:
                    elapsed = time.time() - t0
                    measured_fps = processed / elapsed if elapsed > 0 else 0.0
                    remaining = max(expected_processed - processed, 0)
                    eta_s = remaining / measured_fps if measured_fps > 0 else float("nan")
                    print(
                        f"[info] {processed}/{expected_processed} sampled frames | "
                        f"t={timestamp_s:.1f}s | {measured_fps:.2f} sampled-frames/s | "
                        f"ETA {eta_s:.1f}s | elapsed {elapsed:.1f}s",
                        end="\r",
                    )

            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    print()  # newline after the carriage-return progress line

    df = pd.DataFrame.from_records(
        records,
        columns=[
            "frame_idx", "timestamp_s", "track_id", "class_id", "class_name",
            "conf", "x1", "y1", "x2", "y2", "cx", "cy_foot",
        ],
    )
    df.to_parquet(parquet_path, index=False)
    print(f"[info] wrote {len(df)} detection rows -> {parquet_path}")
    print(f"[info] wrote annotated clip (first {args.annotate_seconds:.0f}s) -> {annotated_path}")

    valid = df[df["track_id"] >= 0]
    track_lengths = valid.groupby("track_id").size()
    n_tracks = int(track_lengths.shape[0])
    mean_len_frames = float(track_lengths.mean()) if n_tracks else 0.0
    mean_len_seconds = mean_len_frames * args.stride / source_fps if n_tracks else 0.0

    print(f"Total unique tracks: {n_tracks}")
    print(
        f"Mean track length: {mean_len_frames:.2f} sampled frames "
        f"(~{mean_len_seconds:.2f}s)"
    )


if __name__ == "__main__":
    main()
