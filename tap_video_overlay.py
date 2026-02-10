"""
Overlay tap-test predictions onto the source video.

Reads classification results from tap_classify.json, then composites
a prediction banner onto each video frame:
  - Green bar for SOLID, red bar for HOLLOW
  - Persists for DISPLAY_DURATION seconds (not just 1 frame)
  - Shows tap number, classification, and a mini-timeline

Output: output/tile1-prediction.webm, output/tile2-prediction.webm
"""

import json
import os
import sys
import subprocess
import numpy as np
import cv2

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = 'output'
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'tap_classify.json')

# How long each prediction stays visible (seconds)
DISPLAY_DURATION = 1.0
# Fade-out: prediction is full opacity for HOLD seconds, then fades for the rest
HOLD_DURATION = 0.6

# Overlay geometry (pixels, at original resolution)
BANNER_HEIGHT = 220       # prediction banner at bottom
TIMELINE_HEIGHT = 40      # mini-timeline strip within banner

# Colors (BGR for OpenCV)
COLOR_SOLID = (80, 200, 80)       # green
COLOR_HOLLOW = (70, 70, 230)      # red
COLOR_IDLE = (50, 50, 50)         # dark gray
COLOR_WHITE = (255, 255, 255)
COLOR_TIMELINE_BG = (30, 30, 30)

VIDEO_MAP = {
    'tile1.opus': 'tile1-video.webm',
    'tile2.opus': 'tile2-video.webm',
}


def lerp_color(c1, c2, t):
    """Linearly interpolate between two BGR colors. t=0 → c1, t=1 → c2."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1):
    """Draw a filled rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    # For simplicity, draw a regular filled rect (rounded corners are slow)
    cv2.rectangle(img, pt1, pt2, color, thickness)


def render_overlay_video(audio_file, video_file, taps, output_path):
    """Render one prediction overlay video."""
    if not os.path.isfile(video_file):
        print(f"  WARNING: {video_file} not found, skipping")
        return

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"  Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"  Taps to overlay: {len(taps)}")

    # Output video (raw, no audio — we'll mux audio later)
    out_height = height + BANNER_HEIGHT
    temp_video = output_path + '.temp.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, out_height))

    if not writer.isOpened():
        print(f"  ERROR: Cannot open video writer for {temp_video}")
        cap.release()
        return

    # Pre-sort taps by time
    taps_sorted = sorted(taps, key=lambda t: t['time'])
    tap_times = [t['time'] for t in taps_sorted]
    tap_classes = [t['class'] for t in taps_sorted]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        # --- Determine active prediction ---
        active_tap_idx = None
        active_age = None  # seconds since tap

        for i in range(len(taps_sorted) - 1, -1, -1):
            age = current_time - tap_times[i]
            if 0 <= age <= DISPLAY_DURATION:
                active_tap_idx = i
                active_age = age
                break

        # --- Build banner ---
        banner = np.zeros((BANNER_HEIGHT, width, 3), dtype=np.uint8)
        banner[:] = COLOR_IDLE

        if active_tap_idx is not None:
            tap_class = tap_classes[active_tap_idx]
            base_color = COLOR_SOLID if tap_class == 'SOLID' else COLOR_HOLLOW

            # Opacity: full for HOLD_DURATION, then fade to idle
            if active_age <= HOLD_DURATION:
                alpha = 1.0
            else:
                fade_progress = (active_age - HOLD_DURATION) / (DISPLAY_DURATION - HOLD_DURATION)
                alpha = 1.0 - fade_progress

            bg_color = lerp_color(COLOR_IDLE, base_color, alpha)
            text_alpha = alpha

            # Fill banner background
            banner[:BANNER_HEIGHT - TIMELINE_HEIGHT] = bg_color

            # Large classification text
            label = tap_class
            tap_num = active_tap_idx + 1
            main_text = f"#{tap_num}  {label}"
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Calculate text size for centering
            text_size = cv2.getTextSize(main_text, font, 2.0, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (BANNER_HEIGHT - TIMELINE_HEIGHT) // 2 + text_size[1] // 2

            # Text color with alpha
            text_color = tuple(int(c * text_alpha + 50 * (1 - text_alpha))
                               for c in COLOR_WHITE)
            cv2.putText(banner, main_text, (text_x, text_y),
                        font, 2.0, text_color, 3, cv2.LINE_AA)

            # Sub-text: time and bandwidth info
            tap_data = taps_sorted[active_tap_idx]
            sub_text = f"t={tap_data['time']:.2f}s  centroid={tap_data.get('centroid_hz', 0):.0f}Hz"
            sub_size = cv2.getTextSize(sub_text, font, 0.7, 1)[0]
            sub_x = (width - sub_size[0]) // 2
            sub_y = text_y + 40
            sub_color = tuple(int(c * text_alpha * 0.7) for c in COLOR_WHITE)
            cv2.putText(banner, sub_text, (sub_x, sub_y),
                        font, 0.7, sub_color, 1, cv2.LINE_AA)
        else:
            # Idle state
            font = cv2.FONT_HERSHEY_SIMPLEX
            idle_text = "Listening..."
            text_size = cv2.getTextSize(idle_text, font, 1.2, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (BANNER_HEIGHT - TIMELINE_HEIGHT) // 2 + text_size[1] // 2
            cv2.putText(banner, idle_text, (text_x, text_y),
                        font, 1.2, (100, 100, 100), 2, cv2.LINE_AA)

        # --- Mini-timeline strip ---
        tl_y = BANNER_HEIGHT - TIMELINE_HEIGHT
        banner[tl_y:] = COLOR_TIMELINE_BG

        if duration > 0:
            # Draw all tap markers
            for i, (t, cls) in enumerate(zip(tap_times, tap_classes)):
                x = int(t / duration * width)
                color = COLOR_SOLID if cls == 'SOLID' else COLOR_HOLLOW
                # Small colored rectangle for each tap
                x1 = max(0, x - 2)
                x2 = min(width - 1, x + 2)
                banner[tl_y + 4:BANNER_HEIGHT - 4, x1:x2] = color

            # Playhead
            px = int(current_time / duration * width)
            px = max(0, min(width - 1, px))
            cv2.line(banner, (px, tl_y), (px, BANNER_HEIGHT), COLOR_WHITE, 2)

            # Time label
            time_text = f"{current_time:.1f}s / {duration:.1f}s"
            cv2.putText(banner, time_text, (10, BANNER_HEIGHT - 10),
                        font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # --- Composite: original frame + banner ---
        canvas = np.zeros((out_height, width, 3), dtype=np.uint8)
        canvas[:height] = frame
        canvas[height:] = banner

        writer.write(canvas)
        frame_idx += 1

        if frame_idx % 300 == 0:
            print(f"    Frame {frame_idx}/{total_frames} ({current_time:.1f}s)")

    cap.release()
    writer.release()
    print(f"  Rendered {frame_idx} frames")

    # Mux with original audio
    print(f"  Muxing audio from {video_file}...")
    final_path = output_path.replace('.webm', '.mp4')
    cmd = [
        'ffmpeg', '-y',
        '-i', temp_video,
        '-i', video_file,
        '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        '-shortest',
        final_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output_path = final_path

    if os.path.isfile(temp_video):
        os.remove(temp_video)

    if result.returncode == 0:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved {output_path} ({size_mb:.1f} MB)")
    else:
        print(f"  ERROR muxing: {result.stderr[-200:]}")


# ============================================================
# Main
# ============================================================
print("Loading classification results...")
with open(RESULTS_FILE) as f:
    results = json.load(f)

taps = results['taps']
print(f"  {len(taps)} classified taps")

# Group taps by source file
taps_by_file = {}
for tap in taps:
    f = tap['file']
    if f not in taps_by_file:
        taps_by_file[f] = []
    taps_by_file[f].append(tap)

# Render each video
for audio_file, file_taps in taps_by_file.items():
    video_file = VIDEO_MAP.get(audio_file)
    if not video_file:
        print(f"\nNo video mapping for {audio_file}, skipping")
        continue

    basename = os.path.splitext(video_file)[0]
    output_path = os.path.join(OUTPUT_DIR, f'{basename}-prediction.webm')

    print(f"\n{'='*60}")
    print(f"Processing {audio_file} → {video_file}")
    print(f"  {len(file_taps)} taps ({sum(1 for t in file_taps if t['class']=='SOLID')} SOLID, "
          f"{sum(1 for t in file_taps if t['class']=='HOLLOW')} HOLLOW)")

    render_overlay_video(audio_file, video_file, file_taps, output_path)

print(f"\nDone!")
