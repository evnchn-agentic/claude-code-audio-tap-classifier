"""
Extract ground truth drum hit events from the video by detecting
when labeled blocks change color (flash) upon being hit by a ball.

Improved approach (v2):
  1. Read video frame-by-frame
  2. REJECT scene transitions (high mean diff across whole frame)
  3. Compute pixel-wise difference between consecutive frames
  4. Find changed regions of block-appropriate size
  5. Filter out purple ramps, white background, and ball-only regions
  6. Require that at least one side classifies as a known instrument
  7. Require that prev/curr colors differ (actual flash, not just movement)
  8. Aggressively merge multi-frame events
  9. Output timestamped hit events with instrument labels

The labeled blocks have these base colors (observed from video frames):
  - Snare  = bright green  (label text on green rectangle)
  - Kick   = black/dark gray (label text on black/gray rectangle)
  - Cymbal = brown          (label text on brown rectangle)
  - Hat    = olive/yellow-green (label text on olive rectangle)
  - Crash  = dark olive/brown (label text on dark rectangle)
  - Ride   = brown          (label text on brown rectangle)

When hit, blocks briefly flash to a brighter/different color for 1-3 frames.
The purple ramps/platforms move constantly and must be filtered out.
"""

import cv2
import numpy as np
import os
import json
import sys

VIDEO_FILE = 'mysterious.webm'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Configuration
# ============================================================
# Scene transition detection: if mean diff > this, skip the frame
SCENE_TRANSITION_THRESHOLD = 2.5  # mean pixel diff (0-255 scale on gray)
TRIM_END_SECONDS = 10.0           # exclude last N seconds (parody/credits)

# Changed region size filters (at SCALE resolution)
MIN_BLOCK_AREA = 400        # pixels^2 at scaled res — blocks are ~40-75px wide
MAX_BLOCK_AREA = 40000      # filter out huge regions
DIFF_THRESHOLD = 35         # per-channel abs diff threshold
MIN_CHANGED_PIXELS = 100    # minimum changed pixels in a region at scaled res

# We work at reduced resolution for speed
SCALE = 0.5  # process at 960x540

# Aspect ratio: blocks are roughly 2:1 to 4:1, not super elongated
MAX_ASPECT_RATIO = 6.0

# Merging
MERGE_TIME_WINDOW = 0.10    # 100ms — generous to catch multi-frame flashes
MERGE_DIST = 150            # pixels (at original resolution)

# Colors for visualization (BGR for OpenCV, will convert for matplotlib)
VIZ_COLORS = {
    'Snare': (0, 200, 0),
    'Kick': (40, 40, 40),
    'Hat': (0, 180, 180),
    'Cymbal': (0, 140, 200),
    'Crash': (30, 80, 100),
    'Ride': (50, 120, 180),
    'Unknown': (128, 128, 128),
}


def classify_block_color(roi_bgr):
    """Classify a region's dominant color to an instrument name.

    Returns (instrument_name, confidence) where confidence is how
    strongly the color matches the instrument template.
    """
    if roi_bgr.size == 0:
        return 'Unknown'

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    b_mean = np.mean(roi_bgr[:, :, 0])
    g_mean = np.mean(roi_bgr[:, :, 1])
    r_mean = np.mean(roi_bgr[:, :, 2])

    # === Kick: black/dark gray, very low saturation and value ===
    if s_mean < 60 and v_mean < 160:
        # Distinguish from white background
        if v_mean < 120:
            return 'Kick'

    # === Snare: bright green — high G channel, moderate R/B ===
    if 35 <= h_mean <= 80 and s_mean > 80 and v_mean > 80:
        return 'Snare'

    # === Hat: olive/yellow-green — roughly equal R and G, low B ===
    if 20 <= h_mean <= 38 and s_mean > 50 and v_mean > 100:
        if g_mean > 100 and r_mean > 100 and b_mean < 100:
            return 'Hat'

    # === Cymbal: medium brown — moderate R, lower G, low B ===
    if 8 <= h_mean <= 25 and s_mean > 80 and v_mean > 80:
        return 'Cymbal'

    # === Crash: dark brown/olive — similar to cymbal but darker ===
    if 8 <= h_mean <= 35 and s_mean > 50 and 40 <= v_mean <= 100:
        return 'Crash'

    # === Ride: similar brown range ===
    if 10 <= h_mean <= 25 and 60 <= s_mean <= 150 and 60 <= v_mean <= 130:
        return 'Ride'

    return 'Unknown'


def is_purple_ramp(roi_bgr):
    """Check if a region is a purple ramp/platform (ignore these).

    Purple ramps in the video are lavender/light purple bars.
    They have moderate saturation, blue-ish hue, and moderate value.
    """
    if roi_bgr.size == 0:
        return True
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    b_mean = np.mean(roi_bgr[:, :, 0])
    g_mean = np.mean(roi_bgr[:, :, 1])
    r_mean = np.mean(roi_bgr[:, :, 2])

    # Purple ramps: hue in blue-purple range, moderate saturation
    # They appear as light lavender (R~170, G~160, B~200)
    if 95 <= h_mean <= 150 and 15 <= s_mean <= 130 and v_mean > 100:
        return True
    # Also catch by RGB: B > R > G pattern with all above 130
    if b_mean > 150 and r_mean > 130 and g_mean > 120 and b_mean > r_mean:
        return True
    return False


def is_background(roi_bgr):
    """Check if a region is mostly white/light background."""
    if roi_bgr.size == 0:
        return True
    mean_val = np.mean(roi_bgr)
    return mean_val > 220


def is_ball_region(w, h, area):
    """Check if a region is ball-sized (too small to be a block)."""
    # Balls are small circles, roughly 10-25px diameter at full res
    # At SCALE=0.5, that's 5-12px. Area would be ~20-110px^2
    # Blocks are bigger: 40-80px wide, 20-50px tall at SCALE=0.5
    min_dim = min(w, h)
    max_dim = max(w, h)
    if max_dim < 15:  # too small — ball or noise
        return True
    return False


# ============================================================
# Phase 1: Scan video for block flash events
# ============================================================
print(f"Opening {VIDEO_FILE}...")
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"ERROR: Cannot open {VIDEO_FILE}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps if fps > 0 else 0

print(f"  {width}x{height} @ {fps:.1f}fps, {total_frames} frames, {duration:.1f}s")
print(f"  Processing at {SCALE}x = {int(width*SCALE)}x{int(height*SCALE)}")

hit_events = []
prev_frame = None
frame_idx = 0

diff_magnitudes = []
frame_times = []
scene_transition_frames = []

print("\nPhase 1: Scanning video for block color changes...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
    timestamp = frame_idx / fps

    # Skip last N seconds (parody/credits)
    if timestamp > duration - TRIM_END_SECONDS:
        break

    if prev_frame is not None:
        diff = cv2.absdiff(small, prev_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        mean_diff = np.mean(diff_gray)
        diff_magnitudes.append(mean_diff)
        frame_times.append(timestamp)

        # === SCENE TRANSITION REJECTION ===
        if mean_diff > SCENE_TRANSITION_THRESHOLD:
            scene_transition_frames.append(frame_idx)
            prev_frame = small.copy()
            frame_idx += 1
            continue

        # Threshold to find changed pixels
        _, thresh = cv2.threshold(diff_gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Morphological close to connect nearby changed pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BLOCK_AREA or area > MAX_BLOCK_AREA:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Aspect ratio filter
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if aspect > MAX_ASPECT_RATIO:
                continue

            # Skip if too small (ball-sized)
            if is_ball_region(w, h, area):
                continue

            # Get ROIs from both frames
            roi_prev = prev_frame[y:y+h, x:x+w]
            roi_curr = small[y:y+h, x:x+w]

            # === PURPLE RAMP FILTER ===
            # Skip if EITHER frame shows purple (ramp moved into/out of area)
            if is_purple_ramp(roi_prev) or is_purple_ramp(roi_curr):
                continue

            # === WHITE BACKGROUND FILTER ===
            if is_background(roi_prev) and is_background(roi_curr):
                continue

            # === CLASSIFY BOTH FRAMES ===
            instrument_prev = classify_block_color(roi_prev)
            instrument_curr = classify_block_color(roi_curr)

            # Require at least one side to classify as a known instrument
            if instrument_prev == 'Unknown' and instrument_curr == 'Unknown':
                continue

            # Determine the instrument: prefer the one that classifies
            # For a flash: prev=normal_color, curr=flash_color (or vice versa)
            if instrument_prev != 'Unknown':
                instrument = instrument_prev
            else:
                instrument = instrument_curr

            # Compute confidence
            roi_diff = diff_gray[y:y+h, x:x+w]
            region_diff_mean = np.mean(roi_diff)
            n_changed = np.sum(roi_diff > DIFF_THRESHOLD)

            if n_changed < MIN_CHANGED_PIXELS:
                continue

            confidence = min(1.0, region_diff_mean / 60.0)

            hit_events.append({
                'time': round(timestamp, 4),
                'frame': frame_idx,
                'instrument': instrument,
                'x': int(x / SCALE),
                'y': int(y / SCALE),
                'w': int(w / SCALE),
                'h': int(h / SCALE),
                'area': int(area / (SCALE ** 2)),
                'confidence': round(confidence, 3),
                'diff_mean': round(region_diff_mean, 1),
                'n_changed_px': int(n_changed),
            })

    prev_frame = small.copy()
    frame_idx += 1

    if frame_idx % 500 == 0:
        print(f"  Frame {frame_idx}/{total_frames} ({timestamp:.1f}s)")

cap.release()
print(f"  Done. {frame_idx} frames processed.")
print(f"  Raw hit events detected: {len(hit_events)}")
print(f"  Scene transitions skipped: {len(scene_transition_frames)} frames")


# ============================================================
# Phase 2: Deduplicate / merge events
# ============================================================
print("\nPhase 2: Merging nearby events...")

merged_events = []
used = set()
hit_events.sort(key=lambda e: e['time'])

for i, ev in enumerate(hit_events):
    if i in used:
        continue

    group = [ev]
    used.add(i)

    for j in range(i + 1, len(hit_events)):
        if j in used:
            continue
        ev2 = hit_events[j]
        if ev2['time'] - ev['time'] > MERGE_TIME_WINDOW:
            break
        # Spatial proximity
        dx = abs(ev2['x'] - ev['x'])
        dy = abs(ev2['y'] - ev['y'])
        if dx < MERGE_DIST and dy < MERGE_DIST:
            group.append(ev2)
            used.add(j)

    # Pick highest-confidence event, use earliest time
    best = max(group, key=lambda e: e['confidence'])
    best['time'] = round(min(e['time'] for e in group), 4)

    # If group has mixed instruments, prefer the most common non-Unknown
    from collections import Counter
    inst_votes = Counter(e['instrument'] for e in group if e['instrument'] != 'Unknown')
    if inst_votes:
        best['instrument'] = inst_votes.most_common(1)[0][0]

    merged_events.append(best)

print(f"  Merged events: {len(merged_events)}")


# ============================================================
# Phase 3: Output results
# ============================================================
from collections import Counter
inst_counts = Counter(e['instrument'] for e in merged_events)
print(f"\nHit events by instrument:")
for inst, count in inst_counts.most_common():
    print(f"  {inst:>10s}: {count}")

# Save to JSON
output_path = os.path.join(OUTPUT_DIR, 'ground_truth_hits.json')
with open(output_path, 'w') as f:
    json.dump({
        'video_file': VIDEO_FILE,
        'fps': fps,
        'duration': duration,
        'total_frames': total_frames,
        'n_events': len(merged_events),
        'scene_transitions_skipped': len(scene_transition_frames),
        'events': merged_events,
    }, f, indent=2)
print(f"\nSaved {output_path}")

# Save timeline as text
txt_path = os.path.join(OUTPUT_DIR, 'ground_truth_hits.txt')
with open(txt_path, 'w') as f:
    f.write(f"# Ground truth hit events from {VIDEO_FILE}\n")
    f.write(f"# {len(merged_events)} events (scene transitions filtered)\n")
    f.write(f"# Format: time_seconds  instrument  confidence  x  y\n\n")
    for ev in merged_events:
        f.write(f"{ev['time']:8.4f}  {ev['instrument']:>10s}  {ev['confidence']:.3f}  ({ev['x']:4d},{ev['y']:4d})\n")
print(f"Saved {txt_path}")


# ============================================================
# Phase 4: Visualization
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(2, 1, figsize=(22, 8), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})

# Top: frame diff magnitude over time
ax = axes[0]
ax.plot(frame_times, diff_magnitudes, color='gray', alpha=0.5, linewidth=0.5)
ax.axhline(y=SCENE_TRANSITION_THRESHOLD, color='red', linestyle='--', alpha=0.5,
           label=f'Scene transition threshold ({SCENE_TRANSITION_THRESHOLD})')
ax.set_ylabel('Mean Frame Diff')
ax.set_title(f'Frame-to-Frame Difference + Detected Hit Events ({len(merged_events)} total)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# Overlay hit events as colored markers
for ev in merged_events:
    color = VIZ_COLORS.get(ev['instrument'], (128, 128, 128))
    mpl_color = (color[2]/255, color[1]/255, color[0]/255)
    ax.axvline(x=ev['time'], color=mpl_color, alpha=0.4, linewidth=0.8)

# Bottom: hit event scatter by instrument
ax2 = axes[1]
instruments_seen = sorted(set(e['instrument'] for e in merged_events))
for i, inst in enumerate(instruments_seen):
    events_inst = [e for e in merged_events if e['instrument'] == inst]
    times_inst = [e['time'] for e in events_inst]
    color = VIZ_COLORS.get(inst, (128, 128, 128))
    mpl_color = (color[2]/255, color[1]/255, color[0]/255)
    ax2.scatter(times_inst, [i] * len(times_inst), c=[mpl_color], s=15,
                marker='|', linewidths=1.5, label=inst)

ax2.set_yticks(range(len(instruments_seen)))
ax2.set_yticklabels(instruments_seen, fontsize=9)
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('Instrument')

legend_patches = [mpatches.Patch(
    color=(VIZ_COLORS.get(inst, (128,128,128))[2]/255,
           VIZ_COLORS.get(inst, (128,128,128))[1]/255,
           VIZ_COLORS.get(inst, (128,128,128))[0]/255),
    label=f'{inst} ({inst_counts[inst]})')
    for inst in instruments_seen]
ax2.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=len(instruments_seen))

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, 'ground_truth_timeline.png')
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
print(f"Saved {viz_path}")


# ============================================================
# Phase 5: Save sample diff frames for visual verification
# ============================================================
print("\nPhase 5: Saving sample diff frames for verification...")

cap = cv2.VideoCapture(VIDEO_FILE)
# Pick 10 evenly-spaced events for inspection
n_samples = min(10, len(merged_events))
if n_samples > 0:
    indices = np.linspace(0, len(merged_events) - 1, n_samples, dtype=int)
    sample_events = [merged_events[i] for i in indices]
    sample_times = sorted(set(e['time'] for e in sample_events))
else:
    sample_times = []

prev_frame = None
frame_idx = 0

sample_dir = os.path.join(OUTPUT_DIR, 'diff_samples')
os.makedirs(sample_dir, exist_ok=True)
# Clean old samples
for f in os.listdir(sample_dir):
    os.remove(os.path.join(sample_dir, f))

saved_count = 0
remaining_times = list(sample_times)

while remaining_times:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps

    if prev_frame is not None:
        for st in remaining_times[:]:
            if abs(timestamp - st) < 0.5 / fps:
                diff = cv2.absdiff(frame, prev_frame)
                diff_amplified = cv2.convertScaleAbs(diff, alpha=3.0)

                h_frame, w_frame = frame.shape[:2]
                canvas = np.zeros((h_frame, w_frame * 3, 3), dtype=np.uint8)
                canvas[:, :w_frame] = prev_frame
                canvas[:, w_frame:2*w_frame] = frame
                canvas[:, 2*w_frame:3*w_frame] = diff_amplified

                cv2.putText(canvas, 'Previous', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(canvas, 'Current', (w_frame + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(canvas, f'Diff x3 (t={timestamp:.3f}s)', (2*w_frame + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Mark detected hit regions on the canvas
                for ev in merged_events:
                    if abs(ev['time'] - timestamp) < 1.0 / fps:
                        ex, ey, ew, eh = ev['x'], ev['y'], ev['w'], ev['h']
                        color = VIZ_COLORS.get(ev['instrument'], (0, 255, 255))
                        # Draw on the "Current" panel
                        cv2.rectangle(canvas,
                                      (w_frame + ex, ey),
                                      (w_frame + ex + ew, ey + eh),
                                      color, 2)
                        cv2.putText(canvas, ev['instrument'],
                                    (w_frame + ex, ey - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                canvas_small = cv2.resize(canvas, (w_frame, h_frame // 3))
                sample_path = os.path.join(sample_dir,
                                           f'diff_{saved_count:03d}_t{timestamp:.3f}s.png')
                cv2.imwrite(sample_path, canvas_small)
                saved_count += 1
                remaining_times.remove(st)
                break

    prev_frame = frame.copy()
    frame_idx += 1

cap.release()
print(f"  Saved {saved_count} diff sample frames to {sample_dir}/")

print("\nDone! Ground truth extraction complete.")
