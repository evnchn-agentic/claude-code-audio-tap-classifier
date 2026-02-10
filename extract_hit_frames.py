#!/usr/bin/env python3
"""
Extract specific frames from mysterious.webm to study how instrument blocks
flash when hit by balls in the Algodoo/DoodleChaos physics simulation.

Instruments and their colors:
  Snare  = green
  Kick   = black/gray
  Cymbal = brown
  Hat    = olive/yellow-green
  Crash  = dark
  Ride   = brown

We extract frames at three time ranges:
  1. t=7.0s-7.1s   every frame   - "Hat" hit region
  2. t=12.2s-12.3s  every frame   - "Snare" hit region
  3. t=35.0s-35.5s  every 2 frames - active drum section
"""

import os
import cv2

VIDEO_PATH = "/Users/evnchn/ClaudeCodeAudioTask/mysterious.webm"
OUTPUT_DIR = "/Users/evnchn/ClaudeCodeAudioTask/frames/hit_study"

# Each entry: (label, start_sec, end_sec, frame_step)
RANGES = [
    ("hat_hit",    7.0,  7.1,  1),
    ("snare_hit", 12.2, 12.3,  1),
    ("active",    35.0, 35.5,  2),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Could not open video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
    print(f"Duration:   {total_frames / fps:.2f}s")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    saved_total = 0

    for label, t_start, t_end, step in RANGES:
        frame_start = int(t_start * fps)
        frame_end = int(t_end * fps)

        print(f"--- Range: {label} ---")
        print(f"    Time: {t_start:.3f}s - {t_end:.3f}s")
        print(f"    Frames: {frame_start} - {frame_end}  (step={step})")

        saved_count = 0
        frame_idx = frame_start

        while frame_idx <= frame_end:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"    WARNING: Could not read frame {frame_idx}")
                frame_idx += step
                continue

            timestamp_s = frame_idx / fps

            filename = f"{label}_f{frame_idx:06d}_t{timestamp_s:.4f}s.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            frame_idx += step

        print(f"    Saved {saved_count} frames")
        saved_total += saved_count

    cap.release()

    print()
    print(f"Total frames saved: {saved_total}")
    print(f"All files in: {OUTPUT_DIR}")

    # List saved files
    print()
    files = sorted(os.listdir(OUTPUT_DIR))
    for f in files:
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  {f}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
