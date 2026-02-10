"""Step 1: Analyze the structure of mysterious.opus to identify segments."""
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load audio
y, sr = librosa.load('mysterious_audio.wav', sr=22050, mono=True)
duration = librosa.get_duration(y=y, sr=sr)
print(f"Duration: {duration:.2f}s, Sample rate: {sr}")

# HPSS to isolate percussion
y_harm, y_perc = librosa.effects.hpss(y, margin=(1.0, 5.0))

# Compute RMS energy to find segments/silences
rms = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=512)[0]
times = librosa.times_like(rms, sr=sr, hop_length=512)

# Find silent regions (below threshold) to segment the audio
threshold = np.median(rms) * 0.3
is_active = rms > threshold

# Find segment boundaries (transitions from silent to active and vice versa)
segments = []
in_segment = False
seg_start = 0
min_segment_duration = 0.5  # minimum segment length in seconds

for i in range(len(is_active)):
    if is_active[i] and not in_segment:
        seg_start = times[i]
        in_segment = True
    elif not is_active[i] and in_segment:
        seg_end = times[i]
        if seg_end - seg_start >= min_segment_duration:
            segments.append((seg_start, seg_end))
        in_segment = False

# Don't forget the last segment if still active
if in_segment:
    seg_end = times[-1]
    if seg_end - seg_start >= min_segment_duration:
        segments.append((seg_start, seg_end))

print(f"\nFound {len(segments)} segments:")
for i, (start, end) in enumerate(segments):
    print(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")

# Merge segments that are very close together (< 0.3s gap)
merged = []
for seg in segments:
    if merged and seg[0] - merged[-1][1] < 0.3:
        merged[-1] = (merged[-1][0], seg[1])
    else:
        merged.append(seg)

print(f"\nAfter merging close segments: {len(merged)} segments:")
for i, (start, end) in enumerate(merged):
    print(f"  Segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")

# Plot overview
fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)

# Waveform
librosa.display.waveshow(y, sr=sr, ax=axes[0], alpha=0.5, color='gray')
librosa.display.waveshow(y_perc, sr=sr, ax=axes[0], alpha=0.7, color='blue')
axes[0].set_title('Waveform (gray=original, blue=percussive)')
for i, (s, e) in enumerate(merged):
    axes[0].axvspan(s, e, alpha=0.15, color=f'C{i}')
    axes[0].text(s, axes[0].get_ylim()[1]*0.9, f'Seg {i+1}', fontsize=8)

# RMS energy
axes[1].plot(times, rms, color='purple')
axes[1].axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
axes[1].set_title('RMS Energy (percussive)')
axes[1].legend()

# Mel spectrogram of percussive component
S = librosa.feature.melspectrogram(y=y_perc, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=axes[2])
axes[2].set_title('Mel Spectrogram (percussive)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'audio_structure.png'), dpi=150)
print(f"\nSaved {os.path.join(OUTPUT_DIR, 'audio_structure.png')}")

# Also estimate tempo for each segment
print("\nTempo estimation per segment:")
for i, (start, end) in enumerate(merged):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    y_seg = y_perc[start_sample:end_sample]
    if len(y_seg) > sr:  # need at least 1 second
        tempo = librosa.beat.tempo(y=y_seg, sr=sr)
        t = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
        print(f"  Segment {i+1}: ~{t:.1f} BPM")
    else:
        print(f"  Segment {i+1}: too short for tempo estimation")
