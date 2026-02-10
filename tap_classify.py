"""
Binary tap-test classifier: solid vs hollow.

Stepping-stone toward scaffolding coupler joint fixity assessment.
Uses floor tile tap recordings as training data.

Approach:
  1. Load audio files, detect tap onsets
  2. Extract PITCH-SENSITIVE features per tap (frequency shifts ARE the signal)
     - Unlike the drum classifier which seeks pitch-invariance, tap-test
       classification needs pitch sensitivity: hollow = lower freq, longer decay
  3. Visualize feature space to find natural solid/hollow split
  4. Binary KMeans clustering (unsupervised), then allow manual label assignment
  5. Train a supervised classifier if labels are provided
  6. Export model for later iOS/CoreML porting

Key acoustic principle (from literature):
  - Solid substrate: short, sharp, high-frequency tap response
  - Hollow/void: longer, resonant, lower-frequency, more damped response
"""

import sys
import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import hilbert
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
AUDIO_FILES = []
for arg in sys.argv[1:]:
    if os.path.isfile(arg):
        AUDIO_FILES.append(arg)

if not AUDIO_FILES:
    # Default: use tile files in current directory
    for f in ['tile1.opus', 'tile2.opus']:
        if os.path.isfile(f):
            AUDIO_FILES.append(f)

if not AUDIO_FILES:
    print("Usage: python tap_classify.py <audio1> [audio2] ...")
    print("No audio files found.")
    sys.exit(1)

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 22050
ONSET_WINDOW = 0.08       # 80ms window after onset for feature extraction
ONSET_DELTA = 0.08        # onset detection sensitivity
ONSET_WAIT = 4            # minimum frames between onsets
MIN_TAP_RMS = 0.004       # skip noise; lowered from 0.015 to catch quieter taps (noise floor ~0.002)


# ============================================================
# 1. Load audio and detect onsets
# ============================================================
print("=" * 60)
print("TAP-TEST BINARY CLASSIFIER")
print("=" * 60)

all_taps = []       # list of dicts: {file, time, features, chunk}
all_chunks = []     # raw audio chunks for each tap

for audio_file in AUDIO_FILES:
    print(f"\nLoading {audio_file}...")
    y, sr = librosa.load(audio_file, sr=SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"  Duration: {duration:.1f}s")

    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=512,
        backtrack=False, units='time',
        delta=ONSET_DELTA, wait=ONSET_WAIT,
    )
    print(f"  Raw onsets: {len(onsets)}")

    # Filter by RMS — skip silence
    kept = 0
    for t in onsets:
        center = int(t * sr)
        end = min(len(y), center + int(ONSET_WINDOW * sr))
        chunk = y[center:end]

        if len(chunk) < 256:
            continue

        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < MIN_TAP_RMS:
            continue

        all_taps.append({
            'file': audio_file,
            'time': float(t),
        })
        all_chunks.append(chunk)
        kept += 1

    print(f"  Kept taps (rms > {MIN_TAP_RMS}): {kept}")

print(f"\nTotal taps across all files: {len(all_taps)}")


# ============================================================
# 2. Extract PITCH-SENSITIVE features per tap
# ============================================================
def extract_tap_features(chunk, sr):
    """Extract features for a single tap event.

    PITCH-SENSITIVE by design — frequency content is the diagnostic signal.

    Feature layout (20D):
      [0]  spectral_centroid_hz    — center of mass of spectrum (Hz)
      [1]  spectral_bandwidth_hz   — spread of spectrum (Hz)
      [2]  spectral_rolloff_hz     — frequency below which 85% energy
      [3]  spectral_flatness       — noise-like (0) vs tonal (1)
      [4]  dominant_freq_hz        — frequency of peak power
      [5]  low_freq_ratio          — energy < 500Hz / total
      [6]  mid_freq_ratio          — energy 500-2000Hz / total
      [7]  high_freq_ratio         — energy > 2000Hz / total
      [8]  log_attack_time         — time to peak amplitude
      [9]  log_decay_time          — time from peak to half amplitude
      [10] temporal_centroid       — center of gravity of envelope (0-1)
      [11] decay_rate              — exponential decay rate estimate
      [12] zcr                     — zero crossing rate
      [13] rms                     — RMS energy
      [14-18] mfcc[0:5]           — first 5 MFCCs (pitch-sensitive, intentionally)
      [19] spectral_contrast_mean  — mean spectral contrast
    """
    n_fft = min(1024, len(chunk))
    features = []

    # --- Spectral features (pitch-sensitive) ---
    S = np.abs(librosa.stft(chunk, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = np.mean(S ** 2, axis=1)
    total_power = np.sum(power) + 1e-10

    # [0] Spectral centroid
    sc = librosa.feature.spectral_centroid(y=chunk, sr=sr, n_fft=n_fft)
    features.append(float(np.mean(sc)))

    # [1] Spectral bandwidth
    sb = librosa.feature.spectral_bandwidth(y=chunk, sr=sr, n_fft=n_fft)
    features.append(float(np.mean(sb)))

    # [2] Spectral rolloff
    sr_feat = librosa.feature.spectral_rolloff(y=chunk, sr=sr, n_fft=n_fft, roll_percent=0.85)
    features.append(float(np.mean(sr_feat)))

    # [3] Spectral flatness
    sf = librosa.feature.spectral_flatness(y=chunk, n_fft=n_fft)
    features.append(float(np.mean(sf)))

    # [4] Dominant frequency
    dominant_idx = np.argmax(power[1:]) + 1  # skip DC
    features.append(float(freqs[dominant_idx]))

    # [5-7] Energy ratios
    low_mask = freqs < 500
    mid_mask = (freqs >= 500) & (freqs < 2000)
    high_mask = freqs >= 2000
    features.append(np.sum(power[low_mask]) / total_power)
    features.append(np.sum(power[mid_mask]) / total_power)
    features.append(np.sum(power[high_mask]) / total_power)

    # --- Temporal envelope features ---
    analytic = hilbert(chunk)
    envelope = np.abs(analytic)
    kernel_size = max(3, int(0.002 * sr))
    envelope_smooth = np.convolve(envelope, np.ones(kernel_size) / kernel_size, mode='same')

    peak_idx = np.argmax(envelope_smooth)
    peak_val = envelope_smooth[peak_idx] + 1e-10

    # [8] Log-attack-time
    onset_thresh = 0.1 * peak_val
    above = np.where(envelope_smooth > onset_thresh)[0]
    onset_idx = above[0] if len(above) > 0 else 0
    attack_samples = max(1, peak_idx - onset_idx)
    features.append(np.log(attack_samples / sr + 1e-6))

    # [9] Log-decay-time
    if peak_idx < len(envelope_smooth) - 1:
        half_val = peak_val * 0.5
        decay_region = envelope_smooth[peak_idx:]
        below_half = np.where(decay_region < half_val)[0]
        decay_samples = below_half[0] if len(below_half) > 0 else len(decay_region)
    else:
        decay_samples = 1
    features.append(np.log(max(1, decay_samples) / sr + 1e-6))

    # [10] Temporal centroid
    t_indices = np.arange(len(envelope_smooth), dtype=float)
    total_env = np.sum(envelope_smooth) + 1e-10
    tc = np.sum(t_indices * envelope_smooth) / total_env / max(1, len(envelope_smooth))
    features.append(tc)

    # [11] Decay rate (fit exponential to post-peak envelope)
    if peak_idx < len(envelope_smooth) - 2:
        post_peak = envelope_smooth[peak_idx:]
        if len(post_peak) > 5 and np.max(post_peak) > 0:
            post_norm = post_peak / (np.max(post_peak) + 1e-10)
            # Find where it drops below 0.1
            below_10pct = np.where(post_norm < 0.1)[0]
            if len(below_10pct) > 0:
                decay_len = below_10pct[0]
            else:
                decay_len = len(post_norm)
            features.append(decay_len / sr)  # decay time in seconds
        else:
            features.append(0.0)
    else:
        features.append(0.0)

    # --- Standard features ---
    # [12] ZCR
    zcr = librosa.feature.zero_crossing_rate(y=chunk)
    features.append(float(np.mean(zcr)))

    # [13] RMS
    features.append(float(np.sqrt(np.mean(chunk ** 2))))

    # [14-18] MFCCs (first 5 — pitch-sensitive, by design)
    mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=6, n_fft=n_fft)
    for i in range(1, 6):  # skip MFCC[0] (energy, redundant with RMS)
        features.append(float(np.mean(mfccs[i])))

    # [19] Spectral contrast mean
    sc_contrast = librosa.feature.spectral_contrast(y=chunk, sr=sr, n_fft=n_fft, n_bands=4)
    features.append(float(np.mean(sc_contrast)))

    return np.array(features)


FEATURE_NAMES = [
    'centroid_hz', 'bandwidth_hz', 'rolloff_hz', 'flatness', 'dominant_hz',
    'low_ratio', 'mid_ratio', 'high_ratio',
    'log_attack', 'log_decay', 'temporal_centroid', 'decay_rate',
    'zcr', 'rms',
    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
    'contrast_mean',
]

print("\nExtracting features...")
all_features = []
for i, chunk in enumerate(all_chunks):
    feat = extract_tap_features(chunk, SR)
    if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
        all_features.append(feat)
    else:
        # Replace NaN/inf with zeros
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        all_features.append(feat)

all_features = np.array(all_features)
print(f"  Feature matrix: {all_features.shape} ({len(FEATURE_NAMES)} features x {len(all_taps)} taps)")


# ============================================================
# 3. Feature statistics per file
# ============================================================
print("\n" + "=" * 60)
print("FEATURE STATISTICS PER FILE")
print("=" * 60)
for audio_file in AUDIO_FILES:
    mask = np.array([t['file'] == audio_file for t in all_taps])
    feats = all_features[mask]
    print(f"\n{audio_file} ({np.sum(mask)} taps):")
    for j, name in enumerate(FEATURE_NAMES):
        vals = feats[:, j]
        print(f"  {name:>20s}: mean={np.mean(vals):8.3f}  std={np.std(vals):8.3f}  "
              f"range=[{np.min(vals):8.3f}, {np.max(vals):8.3f}]")


# ============================================================
# 4. Per-file binary clustering (unsupervised)
# ============================================================
# Key insight: clustering across files separates by tile type (room/material)
# rather than structural condition (solid/hollow). Within-file variance is the
# diagnostic signal, so we cluster EACH FILE INDEPENDENTLY.
print("\n" + "=" * 60)
print("PER-FILE BINARY CLUSTERING")
print("=" * 60)

BANDWIDTH_IDX = 1
FLATNESS_IDX = 3
CLASS_COLORS = {'HOLLOW': '#e74c3c', 'SOLID': '#2ecc71'}

# We'll fill class_labels per-file
class_labels = np.empty(len(all_taps), dtype='U10')
sil_scores = {}

for audio_file in AUDIO_FILES:
    file_mask = np.array([t['file'] == audio_file for t in all_taps])
    file_indices = np.where(file_mask)[0]
    file_features = all_features[file_indices]
    n_file = len(file_indices)

    print(f"\n{audio_file} ({n_file} taps):")

    if n_file < 4:
        print(f"  Too few taps for clustering, marking all HOLLOW")
        class_labels[file_indices] = 'HOLLOW'
        sil_scores[audio_file] = 0.0
        continue

    # Scale features independently per file
    file_scaler = StandardScaler()
    file_scaled = file_scaler.fit_transform(file_features)

    # Binary KMeans
    km = KMeans(n_clusters=2, n_init=20, random_state=42)
    file_labels = km.fit_predict(file_scaled)
    sil = silhouette_score(file_scaled, file_labels)
    sil_scores[audio_file] = sil

    print(f"  k=2 silhouette={sil:.3f}")

    # Assign HOLLOW/SOLID by bandwidth within this file
    # Lower bandwidth = tonal/resonant = HOLLOW
    # Higher bandwidth = broadband/damped = SOLID
    cluster_bw = []
    for c in range(2):
        c_mask = file_labels == c
        median_bw = np.median(file_features[c_mask, BANDWIDTH_IDX])
        cluster_bw.append((c, median_bw))

    cluster_bw.sort(key=lambda x: x[1])
    hollow_c = cluster_bw[0][0]
    solid_c = cluster_bw[1][0]
    file_class_map = {hollow_c: 'HOLLOW', solid_c: 'SOLID'}

    for j, idx in enumerate(file_indices):
        class_labels[idx] = file_class_map[file_labels[j]]

    # Print per-file cluster stats
    for c in range(2):
        c_mask = file_labels == c
        name = file_class_map[c]
        n = np.sum(c_mask)
        median_bw = np.median(file_features[c_mask, BANDWIDTH_IDX])
        median_flat = np.median(file_features[c_mask, FLATNESS_IDX])
        median_cent = np.median(file_features[c_mask, 0])
        median_decay = np.median(file_features[c_mask, 11])
        print(f"  {name:>8s}: {n} taps, bandwidth={median_bw:.0f} Hz, "
              f"flatness={median_flat:.3f}, centroid={median_cent:.0f} Hz, "
              f"decay={median_decay*1000:.1f} ms")

# Also compute a global scaler + silhouette for reference
scaler = StandardScaler()
features_scaled = scaler.fit_transform(all_features)

# Overall summary
print(f"\nPer-file class distribution:")
for audio_file in AUDIO_FILES:
    file_mask = np.array([t['file'] == audio_file for t in all_taps])
    file_labels_named = class_labels[file_mask]
    n_solid = np.sum(file_labels_named == 'SOLID')
    n_hollow = np.sum(file_labels_named == 'HOLLOW')
    total = len(file_labels_named)
    sil = sil_scores.get(audio_file, 0)
    print(f"  {audio_file}: {n_solid} SOLID ({n_solid/total:.0%}), "
          f"{n_hollow} HOLLOW ({n_hollow/total:.0%}), silhouette={sil:.3f}")


# ============================================================
# 5. Visualization
# ============================================================
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (0,0) Spectral centroid vs decay rate — the key diagnostic axes
ax = axes[0, 0]
for cls_name, color in CLASS_COLORS.items():
    mask = class_labels == cls_name
    ax.scatter(
        all_features[mask, 0],  # centroid Hz
        all_features[mask, 11] * 1000,     # decay_rate in ms
        c=color, label=cls_name, alpha=0.6, s=30, edgecolors='white', linewidth=0.5
    )
ax.set_xlabel('Spectral Centroid (Hz)')
ax.set_ylabel('Decay Time (ms)')
ax.set_title('Centroid vs Decay (primary diagnostic)')
ax.legend()

# (0,1) Dominant frequency vs spectral flatness
ax = axes[0, 1]
for cls_name, color in CLASS_COLORS.items():
    mask = class_labels == cls_name
    ax.scatter(
        all_features[mask, 4],   # dominant_freq_hz
        all_features[mask, 3],   # flatness
        c=color, label=cls_name, alpha=0.6, s=30, edgecolors='white', linewidth=0.5
    )
ax.set_xlabel('Dominant Frequency (Hz)')
ax.set_ylabel('Spectral Flatness')
ax.set_title('Dominant Freq vs Flatness')
ax.legend()

# (0,2) Low-freq ratio vs high-freq ratio
ax = axes[0, 2]
for cls_name, color in CLASS_COLORS.items():
    mask = class_labels == cls_name
    ax.scatter(
        all_features[mask, 5],   # low_ratio
        all_features[mask, 7],   # high_ratio
        c=color, label=cls_name, alpha=0.6, s=30, edgecolors='white', linewidth=0.5
    )
ax.set_xlabel('Low-Freq Energy Ratio (<500Hz)')
ax.set_ylabel('High-Freq Energy Ratio (>2000Hz)')
ax.set_title('Energy Distribution')
ax.legend()

# (1,0) MFCC1 vs MFCC2
ax = axes[1, 0]
for cls_name, color in CLASS_COLORS.items():
    mask = class_labels == cls_name
    ax.scatter(
        all_features[mask, 14],  # mfcc1
        all_features[mask, 15],  # mfcc2
        c=color, label=cls_name, alpha=0.6, s=30, edgecolors='white', linewidth=0.5
    )
ax.set_xlabel('MFCC[1]')
ax.set_ylabel('MFCC[2]')
ax.set_title('MFCC Space')
ax.legend()

# (1,1) Timeline: taps colored by class, per file
ax = axes[1, 1]
for file_idx, audio_file in enumerate(AUDIO_FILES):
    file_mask = np.array([t['file'] == audio_file for t in all_taps])
    file_times = np.array([t['time'] for t in all_taps])[file_mask]
    file_cls = class_labels[file_mask]

    for cls_name, color in CLASS_COLORS.items():
        cls_mask = file_cls == cls_name
        ax.scatter(
            file_times[cls_mask],
            np.full(np.sum(cls_mask), file_idx),
            c=color, s=40, marker='|', linewidths=2, alpha=0.8,
        )

ax.set_yticks(range(len(AUDIO_FILES)))
ax.set_yticklabels([os.path.basename(f) for f in AUDIO_FILES], fontsize=9)
ax.set_xlabel('Time (s)')
ax.set_title('Tap Timeline (colored by class)')
legend_patches = [mpatches.Patch(color=c, label=n) for n, c in CLASS_COLORS.items()]
ax.legend(handles=legend_patches, fontsize=8)

# (1,2) Feature importance: mean difference between clusters
ax = axes[1, 2]
solid_mask = class_labels == 'SOLID'
hollow_mask = class_labels == 'HOLLOW'
mean_diff = np.abs(
    np.mean(features_scaled[solid_mask], axis=0) - np.mean(features_scaled[hollow_mask], axis=0)
)
sorted_idx = np.argsort(mean_diff)[::-1]
top_n = min(12, len(FEATURE_NAMES))
ax.barh(
    range(top_n),
    mean_diff[sorted_idx[:top_n]],
    color='steelblue', alpha=0.8
)
ax.set_yticks(range(top_n))
ax.set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx[:top_n]], fontsize=8)
ax.set_xlabel('|Mean Difference| (standardized)')
ax.set_title('Feature Importance (cluster separation)')
ax.invert_yaxis()

avg_sil = np.mean(list(sil_scores.values())) if sil_scores else 0
plt.suptitle(f'Tap-Test Binary Classification (per-file): {len(all_taps)} taps, '
             f'avg silhouette={avg_sil:.3f}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, 'tap_classify.png')
plt.savefig(viz_path, dpi=150, bbox_inches='tight')
print(f"Saved {viz_path}")


# ============================================================
# 6. Spectrogram comparison: sample taps from each class
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(20, 6))

for row, (cls_name, color) in enumerate(CLASS_COLORS.items()):
    cls_indices = np.where(class_labels == cls_name)[0]
    # Pick 5 taps evenly spaced
    sample_indices = cls_indices[np.linspace(0, len(cls_indices) - 1, 5, dtype=int)]

    for col, idx in enumerate(sample_indices):
        ax = axes[row, col]
        chunk = all_chunks[idx]
        # Compute spectrogram
        S = librosa.feature.melspectrogram(y=chunk, sr=SR, n_fft=min(1024, len(chunk)),
                                            hop_length=128, n_mels=64)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=SR, hop_length=128, x_axis='time',
                                  y_axis='mel', ax=ax, cmap='magma')
        tap = all_taps[idx]
        ax.set_title(f'{cls_name} t={tap["time"]:.2f}s', fontsize=9, color=color)
        if col > 0:
            ax.set_ylabel('')
        if row == 0:
            ax.set_xlabel('')

plt.suptitle('Sample Mel Spectrograms: HOLLOW (top) vs SOLID (bottom)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
spec_path = os.path.join(OUTPUT_DIR, 'tap_spectrograms.png')
plt.savefig(spec_path, dpi=150, bbox_inches='tight')
print(f"Saved {spec_path}")


# ============================================================
# 7. Waveform comparison
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(20, 4))

for row, (cls_name, color) in enumerate(CLASS_COLORS.items()):
    cls_indices = np.where(class_labels == cls_name)[0]
    sample_indices = cls_indices[np.linspace(0, len(cls_indices) - 1, 5, dtype=int)]

    for col, idx in enumerate(sample_indices):
        ax = axes[row, col]
        chunk = all_chunks[idx]
        t_axis = np.arange(len(chunk)) / SR * 1000  # ms
        ax.plot(t_axis, chunk, color=color, linewidth=0.5, alpha=0.8)
        ax.set_ylim(-0.8, 0.8)
        if col == 0:
            ax.set_ylabel(cls_name, fontsize=10, fontweight='bold', color=color)
        if row == 1:
            ax.set_xlabel('ms')
        ax.set_title(f't={all_taps[idx]["time"]:.2f}s', fontsize=8)

plt.suptitle('Sample Waveforms: HOLLOW (top) vs SOLID (bottom)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
wave_path = os.path.join(OUTPUT_DIR, 'tap_waveforms.png')
plt.savefig(wave_path, dpi=150, bbox_inches='tight')
print(f"Saved {wave_path}")


# ============================================================
# 8. Save results
# ============================================================
results = {
    'audio_files': AUDIO_FILES,
    'n_taps': len(all_taps),
    'per_file_silhouettes': {f: round(s, 4) for f, s in sil_scores.items()},
    'classes': {},
    'per_file': {},
    'feature_names': FEATURE_NAMES,
    'taps': [],
}

for cls_name in CLASS_COLORS:
    mask = class_labels == cls_name
    results['classes'][cls_name] = {
        'count': int(np.sum(mask)),
        'median_centroid_hz': round(float(np.median(all_features[mask, 0])), 1),
        'median_bandwidth_hz': round(float(np.median(all_features[mask, BANDWIDTH_IDX])), 1),
        'median_flatness': round(float(np.median(all_features[mask, FLATNESS_IDX])), 4),
        'median_decay_ms': round(float(np.median(all_features[mask, 11]) * 1000), 1),
    }

for audio_file in AUDIO_FILES:
    file_mask = np.array([t['file'] == audio_file for t in all_taps])
    file_labels_out = class_labels[file_mask]
    results['per_file'][audio_file] = {
        'n_solid': int(np.sum(file_labels_out == 'SOLID')),
        'n_hollow': int(np.sum(file_labels_out == 'HOLLOW')),
        'silhouette': round(sil_scores.get(audio_file, 0), 4),
    }

for i, tap in enumerate(all_taps):
    results['taps'].append({
        'file': tap['file'],
        'time': tap['time'],
        'class': str(class_labels[i]),
        'centroid_hz': round(float(all_features[i, 0]), 1),
        'decay_ms': round(float(all_features[i, 11]) * 1000, 1),
    })

results_path = os.path.join(OUTPUT_DIR, 'tap_classify.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved {results_path}")

print(f"\n{'='*60}")
print("DONE. Review the visualizations to verify solid/hollow assignment.")
print("If cluster assignment is inverted, the labels can be swapped.")
print(f"{'='*60}")
