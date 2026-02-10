"""
Unsupervised percussive pattern extraction with pitch-invariant features.

Approach:
  1. HPSS to isolate percussive content
  2. Segment audio into active regions
  3. Detect onsets globally
  4. Extract PITCH-INVARIANT features per onset:
     - Temporal envelope: log-attack-time, log-decay-rate, temporal centroid, onset sharpness
     - Spectral ratios: spectral flatness, low/mid/high energy ratios
     - Spectral contrast: 6 sub-band peak-vs-valley measures
     - Retained: ZCR, RMS, MFCC[1-2] (lowest-order only)
  5. Cluster onsets with KMeans (auto-select k via silhouette score, or user-specified)
  6. Label clusters by low-frequency energy ratio (bass-heavy → treble-heavy)
  7. Quantize to a tempo grid and visualize

Features are designed to be robust to tonal/pitch variation — the same instrument
type clusters together even when different scenes use different sample pitches.
"""

import sys
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import hilbert
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else 'mysterious_audio.wav'
N_CLUSTERS = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = auto-detect
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
MIN_SEGMENT_DURATION = 0.08  # seconds — catch even single hits
MERGE_GAP = 0.15             # merge segments closer than this
ONSET_WINDOW = 0.05          # seconds around each onset for feature extraction
GRID_SUBDIVISION = 4         # 4 = 16th notes, 3 = triplets, 2 = 8th notes
TRIM_END_SECONDS = 10.0      # exclude last N seconds (parody/credits)

# Distinct colors for up to 10 clusters
CLUSTER_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#3498db', '#e91e63', '#00bcd4',
]


# ============================================================
# 1. Load & isolate percussion
# ============================================================
print(f"Loading {AUDIO_FILE}...")
y, sr = librosa.load(AUDIO_FILE, sr=22050, mono=True)
duration = librosa.get_duration(y=y, sr=sr)

# Trim last N seconds (parody/credits)
effective_duration = duration - TRIM_END_SECONDS
trim_sample = int(effective_duration * sr)
y = y[:trim_sample]
print(f"  Full duration: {duration:.1f}s, trimmed to {effective_duration:.1f}s (last {TRIM_END_SECONDS:.0f}s excluded)")
duration = effective_duration

y_harm, y_perc = librosa.effects.hpss(y, margin=(1.0, 2.0))
print(f"  Sample rate: {sr}")


# ============================================================
# 2. Segment detection
# ============================================================
rms = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=512)[0]
times_rms = librosa.times_like(rms, sr=sr, hop_length=512)
seg_threshold = np.median(rms) * 0.1
is_active = rms > seg_threshold

raw_segments = []
in_segment = False
seg_start = 0
for i in range(len(is_active)):
    if is_active[i] and not in_segment:
        seg_start = times_rms[i]
        in_segment = True
    elif not is_active[i] and in_segment:
        seg_end = times_rms[i]
        if seg_end - seg_start >= MIN_SEGMENT_DURATION:
            raw_segments.append((seg_start, seg_end))
        in_segment = False
if in_segment:
    seg_end = times_rms[-1]
    if seg_end - seg_start >= MIN_SEGMENT_DURATION:
        raw_segments.append((seg_start, seg_end))

segments = []
for seg in raw_segments:
    if segments and seg[0] - segments[-1][1] < MERGE_GAP:
        segments[-1] = (segments[-1][0], seg[1])
    else:
        segments.append(seg)

print(f"  Found {len(segments)} segments\n")


# ============================================================
# 3. Collect all onsets + spectral features across the file
# ============================================================
def extract_onset_features(y_perc, sr, onset_time, window=ONSET_WINDOW):
    """Extract a PITCH-INVARIANT feature vector for a single onset.

    Feature layout (18D):
      [0]  log_attack_time     — temporal envelope
      [1]  log_decay_rate      — temporal envelope
      [2]  temporal_centroid    — temporal envelope (0-1)
      [3]  onset_sharpness     — temporal envelope
      [4]  spectral_flatness   — spectral ratio (0-1, noise-like vs tonal)
      [5]  low_freq_ratio      — energy < 300Hz / total
      [6]  mid_freq_ratio      — energy 300-3000Hz / total
      [7]  high_freq_ratio     — energy > 3000Hz / total
      [8-13] spectral_contrast — 6 sub-band contrast values
      [14] zcr                 — zero crossing rate
      [15] rms                 — RMS energy
      [16] mfcc_1              — lowest-order MFCC (broad shape)
      [17] mfcc_2              — second MFCC
    """
    center = int(onset_time * sr)
    half_win = int(window * sr)
    start = max(0, center - half_win // 4)  # slightly before onset
    end = min(len(y_perc), center + half_win)
    chunk = y_perc[start:end]

    if len(chunk) < 256:
        chunk = np.pad(chunk, (0, 256 - len(chunk)))

    n_fft = min(512, len(chunk))
    features = []

    # --- Temporal envelope features (4) ---
    analytic = hilbert(chunk)
    envelope = np.abs(analytic)
    # Smooth envelope to reduce noise
    kernel_size = max(3, int(0.001 * sr))  # ~1ms smoothing
    envelope_smooth = np.convolve(envelope, np.ones(kernel_size) / kernel_size, mode='same')

    peak_idx = np.argmax(envelope_smooth)
    peak_val = envelope_smooth[peak_idx] + 1e-10

    # [0] Log-attack-time
    onset_thresh = 0.1 * peak_val
    above_thresh = np.where(envelope_smooth > onset_thresh)[0]
    onset_idx = above_thresh[0] if len(above_thresh) > 0 else 0
    attack_samples = max(1, peak_idx - onset_idx)
    features.append(np.log(attack_samples / sr + 1e-6))

    # [1] Log-decay-rate (time from peak to half amplitude)
    if peak_idx < len(envelope_smooth) - 1:
        half_val = peak_val * 0.5
        decay_region = envelope_smooth[peak_idx:]
        below_half = np.where(decay_region < half_val)[0]
        decay_samples = below_half[0] if len(below_half) > 0 else len(decay_region)
    else:
        decay_samples = 1
    features.append(np.log(max(1, decay_samples) / sr + 1e-6))

    # [2] Temporal centroid (normalized 0-1)
    t_indices = np.arange(len(envelope_smooth), dtype=float)
    total_env = np.sum(envelope_smooth) + 1e-10
    tc = np.sum(t_indices * envelope_smooth) / total_env / max(1, len(envelope_smooth))
    features.append(tc)

    # [3] Onset sharpness (max gradient / peak, normalized)
    env_grad = np.gradient(envelope_smooth)
    features.append(np.max(env_grad) / peak_val)

    # --- Spectral ratio features (4) ---
    S = np.abs(librosa.stft(chunk, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = np.mean(S ** 2, axis=1)  # mean power per frequency bin
    total_power = np.sum(power) + 1e-10

    # [4] Spectral flatness
    sf = librosa.feature.spectral_flatness(y=chunk, n_fft=n_fft)
    features.append(float(np.mean(sf)))

    # [5-7] Energy ratios
    low_mask = freqs < 300
    mid_mask = (freqs >= 300) & (freqs < 3000)
    high_mask = freqs >= 3000
    features.append(np.sum(power[low_mask]) / total_power)
    features.append(np.sum(power[mid_mask]) / total_power)
    features.append(np.sum(power[high_mask]) / total_power)

    # --- Spectral contrast (6) ---
    sc = librosa.feature.spectral_contrast(y=chunk, sr=sr, n_fft=n_fft, n_bands=5)
    features.extend(np.mean(sc, axis=1).tolist())  # 6 values (5 bands + 1)

    # --- Retained features (4) ---
    # [14] ZCR
    zcr = librosa.feature.zero_crossing_rate(y=chunk)
    features.append(float(np.mean(zcr)))

    # [15] RMS
    features.append(float(np.sqrt(np.mean(chunk ** 2))))

    # [16-17] MFCC[1] and MFCC[2] only (broad timbral shape, pitch-robust)
    mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=3, n_fft=n_fft)
    features.append(float(np.mean(mfccs[1])))  # MFCC[1]
    features.append(float(np.mean(mfccs[2])))  # MFCC[2]

    return np.array(features)


print("Detecting onsets and extracting features...")
all_onset_times = []
all_onset_features = []
all_onset_seg_idx = []  # which segment each onset belongs to

for seg_i, (seg_start, seg_end) in enumerate(segments):
    s_sample = int(seg_start * sr)
    e_sample = int(seg_end * sr)
    y_seg_perc = y_perc[s_sample:e_sample]
    y_seg_raw = y[s_sample:e_sample]

    if len(y_seg_perc) < 512:
        continue

    # Detect onsets on BOTH percussive and raw signals, union the results
    detected_times = set()

    for y_det, label in [(y_seg_perc, 'perc'), (y_seg_raw, 'raw')]:
        onset_env = librosa.onset.onset_strength(y=y_det, sr=sr, hop_length=512)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=512,
            backtrack=False, units='time',
            delta=0.05, wait=3,
        )
        for t in onset_frames:
            # Round to ~1ms to deduplicate near-identical detections
            detected_times.add(round(t, 3))

    # Merge onsets that are within 30ms of each other
    sorted_times = sorted(detected_times)
    merged_times = []
    for t in sorted_times:
        if merged_times and t - merged_times[-1] < 0.03:
            merged_times[-1] = (merged_times[-1] + t) / 2  # average
        else:
            merged_times.append(t)

    for t_local in merged_times:
        t_global = seg_start + t_local
        feat = extract_onset_features(y_perc, sr, t_global)
        if not np.any(np.isnan(feat)):
            all_onset_times.append(t_global)
            all_onset_features.append(feat)
            all_onset_seg_idx.append(seg_i)

all_onset_times = np.array(all_onset_times)
all_onset_features = np.array(all_onset_features)
all_onset_seg_idx = np.array(all_onset_seg_idx)

print(f"  Total onsets detected: {len(all_onset_times)}")

if len(all_onset_times) < 3:
    print("ERROR: Too few onsets detected. Try a different audio file or lower thresholds.")
    sys.exit(1)


# ============================================================
# 4. Cluster onsets
# ============================================================
scaler = StandardScaler()
features_scaled = scaler.fit_transform(all_onset_features)

if N_CLUSTERS > 0:
    n_clusters = N_CLUSTERS
    print(f"\n  Using user-specified k={n_clusters}")
else:
    # Auto-detect optimal k via silhouette score
    print("\n  Auto-detecting number of clusters...")
    best_k = 2
    best_score = -1
    max_k = min(8, len(all_onset_times) - 1)

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(features_scaled)
        score = silhouette_score(features_scaled, labels)
        print(f"    k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_k = k

    n_clusters = best_k
    print(f"  Selected k={n_clusters} (silhouette={best_score:.3f})")

km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
onset_labels = km.fit_predict(features_scaled)

# Sort clusters by median low-frequency energy ratio (index 5) → high to low
# High low_freq_ratio = bass-heavy (kick), low = treble-heavy (hihat/cymbal)
# Feature layout: [attack, decay, tc, sharpness, flatness, lo_ratio, mid_ratio, hi_ratio,
#                  contrast*6, zcr, rms, mfcc1, mfcc2]
LO_FREQ_IDX = 5
FLATNESS_IDX = 4
ZCR_IDX = 14
cluster_sort_vals = []
for c in range(n_clusters):
    mask = onset_labels == c
    median_lo = np.median(all_onset_features[mask, LO_FREQ_IDX])
    cluster_sort_vals.append((c, median_lo))

cluster_sort_vals.sort(key=lambda x: -x[1])  # descending: bass-heavy first
label_remap = {}
for new_label, (old_label, _) in enumerate(cluster_sort_vals):
    label_remap[old_label] = new_label

onset_labels_sorted = np.array([label_remap[l] for l in onset_labels])

# Generate cluster names using pitch-invariant descriptors
cluster_names = []
for new_label in range(n_clusters):
    old_label = cluster_sort_vals[new_label][0]
    mask = onset_labels == old_label
    med_lo = np.median(all_onset_features[mask, LO_FREQ_IDX])
    med_flat = np.median(all_onset_features[mask, FLATNESS_IDX])
    count = np.sum(mask)
    cluster_names.append(f"C{new_label+1} (lo={med_lo:.2f}, flat={med_flat:.2f}, n={count})")

print(f"\nCluster summary (sorted bass-heavy → treble-heavy):")
for i, name in enumerate(cluster_names):
    print(f"  {CLUSTER_COLORS[i % len(CLUSTER_COLORS)]:>7s}  {name}")


# ============================================================
# 5. Print onset summary per segment
# ============================================================
print(f"\n{'='*70}")
for seg_i, (seg_start, seg_end) in enumerate(segments):
    mask = all_onset_seg_idx == seg_i
    seg_times = all_onset_times[mask]
    seg_labels = onset_labels_sorted[mask]
    n_hits = len(seg_times)
    print(f"Segment {seg_i+1}: {seg_start:.2f}s - {seg_end:.2f}s ({seg_end-seg_start:.2f}s) | {n_hits} onsets")
    for c in range(n_clusters):
        c_mask = seg_labels == c
        times_c = seg_times[c_mask]
        if len(times_c) > 0:
            t_str = ', '.join([f'{t:.3f}' for t in times_c])
            print(f"  {cluster_names[c]:>25s}: [{t_str}]")
    print()


# ============================================================
# 6. Timeline visualization — full audio (normal + 5x zoom)
# ============================================================
def render_timeline(filename, width_scale=1.0):
    """Render the full-audio timeline at a given horizontal scale."""
    base_w = 22
    fig_w = base_w * width_scale
    fig_h = 2.0 + n_clusters * 1.5
    fig, axes = plt.subplots(n_clusters + 1, 1, figsize=(fig_w, fig_h),
                             sharex=True,
                             gridspec_kw={'height_ratios': [1.5] + [1] * n_clusters})

    # Top panel: waveform with segment shading
    ax_wave = axes[0]
    librosa.display.waveshow(y_perc, sr=sr, ax=ax_wave, alpha=0.5, color='#555')
    for seg_i, (s, e) in enumerate(segments):
        ax_wave.axvspan(s, e, alpha=0.08, color='blue')
    ax_wave.set_ylabel('Amplitude')
    suffix = '' if width_scale == 1.0 else f' [{width_scale:.0f}x zoom]'
    ax_wave.set_title(
        f'Percussive Waveform + Clustered Onsets '
        f'({n_clusters} clusters, {len(all_onset_times)} onsets){suffix}',
        fontsize=12, fontweight='bold')

    # Per-cluster onset timeline rows
    lw = 1.5 if width_scale <= 2 else 2.0
    dot_s = 12 if width_scale <= 2 else 20
    for c in range(n_clusters):
        ax = axes[c + 1]
        mask = onset_labels_sorted == c
        times_c = all_onset_times[mask]
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]

        ax.vlines(times_c, 0, 1, colors=color, linewidth=lw, alpha=0.8)
        ax.scatter(times_c, np.full_like(times_c, 0.5), c=color, s=dot_s, zorder=3, alpha=0.9)

        for seg_i, (s, e) in enumerate(segments):
            ax.axvspan(s, e, alpha=0.05, color='blue')

        ax.set_ylim(0, 1)
        ax.set_yticks([0.5])
        ax.set_yticklabels([cluster_names[c]], fontsize=9)
        ax.tick_params(left=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

render_timeline(os.path.join(OUTPUT_DIR, 'timeline.png'), width_scale=1.0)


def render_timeline_square(filename, n_strips=5, target_px=2000):
    """Render timeline as a square image: split into horizontal strips stacked vertically.

    Each strip has a tall waveform row + short cluster rows (waveform 5x each cluster).
    Waveform is shown as a symmetrical mirrored envelope (both halves).
    Final image is resized to target_px x target_px.
    """
    from PIL import Image
    import matplotlib.gridspec as gridspec

    strip_duration = duration / n_strips
    rows_per_strip = 1 + n_clusters  # waveform + one row per cluster
    total_rows = n_strips * rows_per_strip

    # Height ratios: [5, 1, 1, 1] per strip (waveform 5x each cluster row)
    height_ratios = []
    for _ in range(n_strips):
        height_ratios.append(5)
        height_ratios.extend([1] * n_clusters)

    fig_inches = 16
    dpi_render = 160

    fig = plt.figure(figsize=(fig_inches, fig_inches))
    gs = gridspec.GridSpec(total_rows, 1, height_ratios=height_ratios,
                           hspace=0.15)

    all_axes = []
    for row in range(total_rows):
        ax = fig.add_subplot(gs[row])
        all_axes.append(ax)

    for strip_i in range(n_strips):
        t_start = strip_i * strip_duration
        t_end = min((strip_i + 1) * strip_duration, duration)
        base_row = strip_i * rows_per_strip

        # --- Waveform row (tall) ---
        ax_wave = all_axes[base_row]

        # Segment shading
        for seg_s, seg_e in segments:
            if seg_e > t_start and seg_s < t_end:
                ax_wave.axvspan(max(seg_s, t_start), min(seg_e, t_end),
                                alpha=0.07, color='blue')

        # Symmetrical mirrored envelope
        s_samp = int(t_start * sr)
        e_samp = min(int(t_end * sr), len(y_perc))
        if e_samp > s_samp:
            chunk = y_perc[s_samp:e_samp]
            hop = max(1, len(chunk) // 1200)
            env = np.array([np.sqrt(np.mean(chunk[j:j+hop]**2))
                            for j in range(0, len(chunk) - hop, hop)])
            if len(env) > 0 and np.max(env) > 0:
                env_norm = env / np.max(env)
                env_t = np.linspace(t_start, t_end, len(env_norm))
                ax_wave.fill_between(env_t, -env_norm, env_norm,
                                     color='#555', alpha=0.25)

        ax_wave.set_xlim(t_start, t_end)
        ax_wave.set_ylim(-1.1, 1.1)
        ax_wave.set_yticks([])
        ax_wave.tick_params(axis='x', labelsize=5, labelbottom=False)
        ax_wave.spines['top'].set_visible(False)
        ax_wave.spines['right'].set_visible(False)
        ax_wave.spines['left'].set_visible(False)

        # Time range label
        ax_wave.text(0.005, 0.92, f'{t_start:.0f}s – {t_end:.0f}s',
                     transform=ax_wave.transAxes, ha='left', va='top',
                     fontsize=6, color='#444', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor='none', alpha=0.7))

        # --- Cluster rows (short) ---
        for c in range(n_clusters):
            ax = all_axes[base_row + 1 + c]
            mask = onset_labels_sorted == c
            times_c = all_onset_times[mask]
            in_range = (times_c >= t_start) & (times_c <= t_end)
            times_c_strip = times_c[in_range]

            # Segment shading
            for seg_s, seg_e in segments:
                if seg_e > t_start and seg_s < t_end:
                    ax.axvspan(max(seg_s, t_start), min(seg_e, t_end),
                               alpha=0.05, color='blue')

            color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            if len(times_c_strip) > 0:
                ax.vlines(times_c_strip, 0.1, 0.9,
                          colors=color, linewidth=1.5, alpha=0.85)

            ax.set_xlim(t_start, t_end)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.5])
            ax.set_yticklabels([f'C{c+1}'], fontsize=5, color=color,
                               fontweight='bold')
            ax.tick_params(axis='x', labelsize=5, labelbottom=False)
            ax.tick_params(axis='y', length=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

        # Show x-axis labels only on the last cluster row of last strip
        if strip_i == n_strips - 1:
            last_ax = all_axes[base_row + n_clusters]
            last_ax.tick_params(axis='x', labelbottom=True)
            last_ax.set_xlabel('Time (s)', fontsize=7)

    # Title and legend on the first waveform axis
    all_axes[0].set_title(
        f'5x Timeline ({n_clusters} clusters, {len(all_onset_times)} onsets)',
        fontsize=10, fontweight='bold')
    legend_patches = [mpatches.Patch(color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                                     label=cluster_names[c])
                      for c in range(n_clusters)]
    all_axes[0].legend(handles=legend_patches, loc='upper right', fontsize=5.5,
                       ncol=n_clusters, framealpha=0.8)

    temp_path = filename + '.tmp.png'
    plt.savefig(temp_path, dpi=dpi_render, bbox_inches='tight')
    plt.close(fig)

    # Resize to exact square target
    img = Image.open(temp_path)
    img = img.resize((target_px, target_px), Image.LANCZOS)
    img.save(filename)
    os.remove(temp_path)
    print(f"Saved {filename} ({target_px}x{target_px}px)")


render_timeline_square(os.path.join(OUTPUT_DIR, 'timeline_5x.png'))


# ============================================================
# 7. Zoomed timeline for each segment with enough onsets
# ============================================================
sig_segments = [(i, s, e) for i, (s, e) in enumerate(segments)
                if np.sum(all_onset_seg_idx == i) >= 3]
n_sig = len(sig_segments)

if n_sig > 0:
    fig, axes = plt.subplots(n_sig, 1, figsize=(20, n_sig * 1.8), squeeze=False)

    for ax_idx, (seg_i, seg_start, seg_end) in enumerate(sig_segments):
        ax = axes[ax_idx, 0]
        mask = all_onset_seg_idx == seg_i
        seg_times = all_onset_times[mask]
        seg_labels = onset_labels_sorted[mask]

        # Plot each cluster as a separate row within the axis
        for c in range(n_clusters):
            c_mask = seg_labels == c
            times_c = seg_times[c_mask]
            color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            y_pos = n_clusters - 1 - c

            ax.scatter(times_c, np.full_like(times_c, y_pos), c=color,
                       s=60, marker='|', linewidths=2.5, zorder=3)
            ax.scatter(times_c, np.full_like(times_c, y_pos), c=color,
                       s=25, marker='o', zorder=4, alpha=0.8)

        ax.set_xlim(seg_start - 0.05, seg_end + 0.05)
        ax.set_ylim(-0.5, n_clusters - 0.5)
        ax.set_yticks(list(range(n_clusters)))
        ax.set_yticklabels([cluster_names[n_clusters - 1 - i] for i in range(n_clusters)],
                           fontsize=7)
        n_onsets = np.sum(mask)
        ax.set_title(f'Seg {seg_i+1}: {seg_start:.2f}s - {seg_end:.2f}s  '
                     f'({seg_end - seg_start:.2f}s, {n_onsets} onsets)',
                     fontsize=9, fontweight='bold', loc='left')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.grid(axis='x', alpha=0.3, linewidth=0.5)

    plt.suptitle(f'Per-Segment Onset Timeline ({n_clusters} clusters)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    seg_path = os.path.join(OUTPUT_DIR, 'segment_timelines.png')
    plt.savefig(seg_path, dpi=150, bbox_inches='tight')
    print(f"Saved {seg_path} ({n_sig} segments)")


# ============================================================
# 8. Cluster feature visualization (for understanding)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Scatter: spectral flatness vs low-freq energy ratio
for c in range(n_clusters):
    mask = onset_labels_sorted == c
    axes[0].scatter(
        all_onset_features[mask, FLATNESS_IDX],
        all_onset_features[mask, LO_FREQ_IDX],
        c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
        label=cluster_names[c], alpha=0.6, s=20
    )
axes[0].set_xlabel('Spectral Flatness (0=tonal, 1=noisy)')
axes[0].set_ylabel('Low-Freq Energy Ratio')
axes[0].set_title('Flatness vs Bass Energy')
axes[0].legend(fontsize=7)

# Scatter: log-attack-time vs log-decay-rate
for c in range(n_clusters):
    mask = onset_labels_sorted == c
    axes[1].scatter(
        all_onset_features[mask, 0],  # log_attack_time
        all_onset_features[mask, 1],  # log_decay_rate
        c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
        label=cluster_names[c], alpha=0.6, s=20
    )
axes[1].set_xlabel('Log Attack Time')
axes[1].set_ylabel('Log Decay Rate')
axes[1].set_title('Attack vs Decay (temporal envelope)')

# Scatter: ZCR vs high-freq energy ratio
for c in range(n_clusters):
    mask = onset_labels_sorted == c
    axes[2].scatter(
        all_onset_features[mask, ZCR_IDX],
        all_onset_features[mask, 7],  # high_freq_ratio
        c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
        label=cluster_names[c], alpha=0.6, s=20
    )
axes[2].set_xlabel('Zero Crossing Rate')
axes[2].set_ylabel('High-Freq Energy Ratio')
axes[2].set_title('ZCR vs Treble Energy')
axes[2].legend(fontsize=7)

plt.tight_layout()
cluster_path = os.path.join(OUTPUT_DIR, 'cluster_analysis.png')
plt.savefig(cluster_path, dpi=150, bbox_inches='tight')
print(f"Saved {cluster_path}")
