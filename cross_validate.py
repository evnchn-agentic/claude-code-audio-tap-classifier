"""
Cross-validate visual ground truth (from video block flashes) against
audio-based onset detection (from the extracted WAV).

This script:
  1. Loads visual ground truth from output/ground_truth_hits.json
  2. Runs audio onset detection on mysterious_audio.wav (extracted from .webm)
  3. Matches audio onsets to visual events within a tolerance window
  4. Reports precision, recall, and timing offsets
  5. Extracts frames at matched/unmatched hit moments for inspection
  6. Visualizes the alignment between audio and video detections
"""

import json
import os
import sys
import numpy as np
import librosa
import cv2

OUTPUT_DIR = 'output'
AUDIO_FILE = 'mysterious_audio.wav'
VIDEO_FILE = 'mysterious.webm'
GROUND_TRUTH_FILE = os.path.join(OUTPUT_DIR, 'ground_truth_hits.json')

# Matching tolerance: audio onset within this window of a visual event = match
MATCH_TOLERANCE = 0.08  # 80ms — accounts for A/V sync jitter + flash delay
TRIM_END_SECONDS = 10.0  # exclude last N seconds (parody/credits)


# ============================================================
# 1. Load visual ground truth
# ============================================================
print("Loading visual ground truth...")
with open(GROUND_TRUTH_FILE) as f:
    gt_data = json.load(f)

gt_events_raw = gt_data['events']

# Load audio to get duration for trim calculation
print(f"Loading {AUDIO_FILE} for duration...")
y, sr = librosa.load(AUDIO_FILE, sr=22050, mono=True)
duration = librosa.get_duration(y=y, sr=sr)
effective_duration = duration - TRIM_END_SECONDS
print(f"  Duration: {duration:.1f}s, effective: {effective_duration:.1f}s (last {TRIM_END_SECONDS:.0f}s excluded)")

# Trim last N seconds and apply HPSS
trim_sample = int(effective_duration * sr)
y = y[:trim_sample]
y_harm, y_perc = librosa.effects.hpss(y, margin=(1.0, 2.0))

# Filter visual events to exclude last N seconds
gt_events = [e for e in gt_events_raw if e['time'] <= effective_duration]
gt_times = np.array([e['time'] for e in gt_events])
gt_instruments = [e['instrument'] for e in gt_events]
print(f"  {len(gt_events)} visual hit events (of {len(gt_events_raw)} total, after trim)")

# Count by instrument
from collections import Counter
gt_inst_counts = Counter(gt_instruments)
for inst, count in gt_inst_counts.most_common():
    print(f"    {inst:>10s}: {count}")


# ============================================================
# 2. Audio onset detection
# ============================================================
print(f"\nDetecting audio onsets (on trimmed audio, {effective_duration:.1f}s)...")

onset_env_perc = librosa.onset.onset_strength(y=y_perc, sr=sr, hop_length=512)
audio_onsets_perc = librosa.onset.onset_detect(
    onset_envelope=onset_env_perc, sr=sr, hop_length=512,
    backtrack=False, units='time', delta=0.05, wait=3,
)

onset_env_raw = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
audio_onsets_raw = librosa.onset.onset_detect(
    onset_envelope=onset_env_raw, sr=sr, hop_length=512,
    backtrack=False, units='time', delta=0.05, wait=3,
)

# Union of both, deduplicated within 30ms
all_audio_onsets = set()
for t in np.concatenate([audio_onsets_perc, audio_onsets_raw]):
    all_audio_onsets.add(round(t, 3))

sorted_onsets = sorted(all_audio_onsets)
merged_audio_onsets = []
for t in sorted_onsets:
    if merged_audio_onsets and t - merged_audio_onsets[-1] < 0.03:
        merged_audio_onsets[-1] = (merged_audio_onsets[-1] + t) / 2
    else:
        merged_audio_onsets.append(t)

audio_onsets = np.array(merged_audio_onsets)
# Filter to effective duration (already trimmed audio, but belt-and-suspenders)
audio_onsets = audio_onsets[audio_onsets <= effective_duration]
print(f"  Audio onsets detected: {len(audio_onsets)} (perc: {len(audio_onsets_perc)}, raw: {len(audio_onsets_raw)})")


# ============================================================
# 3. Match audio onsets to visual events
# ============================================================
print(f"\nCross-validation (tolerance = {MATCH_TOLERANCE*1000:.0f}ms)...")

# For each visual event, find the closest audio onset
matched_visual = []      # visual events that have a nearby audio onset
unmatched_visual = []    # visual events with no nearby audio onset
timing_offsets = []      # audio_time - visual_time for matched events

visual_matched_set = set()
audio_matched_set = set()

for i, vt in enumerate(gt_times):
    if len(audio_onsets) == 0:
        unmatched_visual.append(i)
        continue

    diffs = np.abs(audio_onsets - vt)
    closest_idx = np.argmin(diffs)
    closest_diff = diffs[closest_idx]

    if closest_diff <= MATCH_TOLERANCE:
        matched_visual.append(i)
        visual_matched_set.add(i)
        audio_matched_set.add(closest_idx)
        timing_offsets.append(audio_onsets[closest_idx] - vt)
    else:
        unmatched_visual.append(i)

# Audio onsets with no nearby visual event
unmatched_audio = [i for i in range(len(audio_onsets)) if i not in audio_matched_set]

n_visual = len(gt_times)
n_audio = len(audio_onsets)
n_matched = len(matched_visual)

# Visual recall: % of visual events matched by an audio onset
visual_recall = n_matched / n_visual if n_visual > 0 else 0
# Audio precision: % of audio onsets that match a visual event
audio_precision = len(audio_matched_set) / n_audio if n_audio > 0 else 0

print(f"\n  Visual events:     {n_visual}")
print(f"  Audio onsets:      {n_audio}")
print(f"  Matched:           {n_matched}")
print(f"  Unmatched visual:  {len(unmatched_visual)} (visual events with no audio onset)")
print(f"  Unmatched audio:   {len(unmatched_audio)} (audio onsets with no visual event)")
print(f"\n  Visual recall:     {visual_recall:.1%} (% visual events found by audio)")
print(f"  Audio precision:   {audio_precision:.1%} (% audio onsets confirmed by video)")

if timing_offsets:
    offsets = np.array(timing_offsets)
    print(f"\n  Timing offsets (audio - visual):")
    print(f"    Mean:   {np.mean(offsets)*1000:+.1f} ms")
    print(f"    Median: {np.median(offsets)*1000:+.1f} ms")
    print(f"    Std:    {np.std(offsets)*1000:.1f} ms")
    print(f"    Range:  [{np.min(offsets)*1000:+.1f}, {np.max(offsets)*1000:+.1f}] ms")

# Per-instrument breakdown
print(f"\n  Per-instrument matching:")
for inst in sorted(gt_inst_counts.keys()):
    inst_indices = [i for i, g in enumerate(gt_instruments) if g == inst]
    inst_matched = [i for i in inst_indices if i in visual_matched_set]
    recall = len(inst_matched) / len(inst_indices) if inst_indices else 0
    print(f"    {inst:>10s}: {len(inst_matched):3d}/{len(inst_indices):3d} matched ({recall:.0%})")


# ============================================================
# 4. Save results
# ============================================================
results = {
    'match_tolerance_ms': MATCH_TOLERANCE * 1000,
    'n_visual_events': n_visual,
    'n_audio_onsets': n_audio,
    'n_matched': n_matched,
    'visual_recall': round(visual_recall, 4),
    'audio_precision': round(audio_precision, 4),
    'timing_offset_mean_ms': round(np.mean(timing_offsets) * 1000, 1) if timing_offsets else None,
    'timing_offset_std_ms': round(np.std(timing_offsets) * 1000, 1) if timing_offsets else None,
    'per_instrument': {
        inst: {
            'total': len([i for i, g in enumerate(gt_instruments) if g == inst]),
            'matched': len([i for i in range(n_visual)
                           if gt_instruments[i] == inst and i in visual_matched_set]),
        }
        for inst in gt_inst_counts.keys()
    },
}

results_path = os.path.join(OUTPUT_DIR, 'cross_validation.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {results_path}")


# ============================================================
# 5. Visualization
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(3, 1, figsize=(22, 10), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1, 1]})

# Top: waveform with audio onsets
ax = axes[0]
librosa.display.waveshow(y_perc, sr=sr, ax=ax, alpha=0.4, color='#555')
for t in audio_onsets:
    ax.axvline(x=t, color='blue', alpha=0.15, linewidth=0.5)
ax.set_ylabel('Amplitude')
ax.set_title(f'Audio vs Visual: {n_matched} matched, '
             f'{len(unmatched_visual)} visual-only, {len(unmatched_audio)} audio-only '
             f'(tolerance={MATCH_TOLERANCE*1000:.0f}ms)',
             fontsize=12, fontweight='bold')

# Middle: visual ground truth events
ax2 = axes[1]
VIZ_COLORS = {
    'Snare': '#2ecc71', 'Kick': '#2c3e50', 'Hat': '#b8b800',
    'Cymbal': '#d68910', 'Crash': '#6e4b26', 'Ride': '#a0522d',
}

instruments_seen = sorted(gt_inst_counts.keys())
for row, inst in enumerate(instruments_seen):
    for i, ev in enumerate(gt_events):
        if ev['instrument'] != inst:
            continue
        color = VIZ_COLORS.get(inst, '#888')
        alpha = 0.9 if i in visual_matched_set else 0.3
        marker = '|' if i in visual_matched_set else 'x'
        ax2.scatter([ev['time']], [row], c=color, s=20, marker=marker,
                    linewidths=1.5, alpha=alpha)

ax2.set_yticks(range(len(instruments_seen)))
ax2.set_yticklabels(instruments_seen, fontsize=9)
ax2.set_ylabel('Visual Events')

legend_patches = [
    mpatches.Patch(color='#2ecc71', label=f'Matched ({n_matched})'),
    mpatches.Patch(color='#ccc', label=f'Unmatched visual ({len(unmatched_visual)})'),
]
ax2.legend(handles=legend_patches, loc='upper right', fontsize=8)

# Bottom: audio onsets (matched vs unmatched)
ax3 = axes[2]
for i, t in enumerate(audio_onsets):
    if i in audio_matched_set:
        ax3.axvline(x=t, color='blue', alpha=0.5, linewidth=0.8)
    else:
        ax3.axvline(x=t, color='red', alpha=0.3, linewidth=0.5)

ax3.set_ylabel('Audio Onsets')
ax3.set_xlabel('Time (s)', fontsize=11)
legend_patches = [
    mpatches.Patch(color='blue', label=f'Audio matched ({len(audio_matched_set)})'),
    mpatches.Patch(color='red', label=f'Audio unmatched ({len(unmatched_audio)})'),
]
ax3.legend(handles=legend_patches, loc='upper right', fontsize=8)

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, 'cross_validation.png')
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
print(f"Saved {viz_path}")

# Timing offset histogram
if timing_offsets:
    fig2, ax_hist = plt.subplots(figsize=(8, 4))
    ax_hist.hist(np.array(timing_offsets) * 1000, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
    ax_hist.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax_hist.axvline(x=np.mean(timing_offsets) * 1000, color='orange', linewidth=2,
                    label=f'Mean: {np.mean(timing_offsets)*1000:+.1f}ms')
    ax_hist.set_xlabel('Audio - Visual offset (ms)')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('Timing Offset Distribution (Audio vs Visual)')
    ax_hist.legend()
    plt.tight_layout()
    offset_path = os.path.join(OUTPUT_DIR, 'timing_offsets.png')
    plt.savefig(offset_path, dpi=150, bbox_inches='tight')
    print(f"Saved {offset_path}")


# ============================================================
# 6. Extract frames at interesting moments
# ============================================================
print("\nExtracting frames at key moments...")

frame_dir = os.path.join('frames', 'cross_val')
os.makedirs(frame_dir, exist_ok=True)
# Clean old files
for f in os.listdir(frame_dir):
    os.remove(os.path.join(frame_dir, f))

cap = cv2.VideoCapture(VIDEO_FILE)
fps = cap.get(cv2.CAP_PROP_FPS)

# Extract frames at:
# a) First 5 matched visual events (to verify correct detection)
# b) First 5 unmatched visual events (to investigate false positives)
sample_matched = matched_visual[:5]
sample_unmatched = unmatched_visual[:5]

target_frames = {}
for i in sample_matched:
    frame_num = int(gt_events[i]['time'] * fps)
    target_frames[frame_num] = f"MATCHED_{gt_events[i]['instrument']}_t{gt_events[i]['time']:.3f}s"
for i in sample_unmatched:
    frame_num = int(gt_events[i]['time'] * fps)
    target_frames[frame_num] = f"UNMATCHED_{gt_events[i]['instrument']}_t{gt_events[i]['time']:.3f}s"

frame_idx = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx in target_frames:
        label = target_frames[frame_idx]
        path = os.path.join(frame_dir, f'{label}.png')
        cv2.imwrite(path, frame)
        saved += 1
        if saved >= len(target_frames):
            break
    frame_idx += 1

cap.release()
print(f"  Saved {saved} frames to {frame_dir}/")

print("\nCross-validation complete!")
