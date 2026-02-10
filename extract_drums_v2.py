"""Drum pattern extraction v2: global onset detection + per-hit classification."""
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.signal import butter, sosfilt
import warnings

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load & isolate percussion
# ============================================================
print("Loading audio...")
y, sr = librosa.load('mysterious_audio.wav', sr=22050, mono=True)
y_harm, y_perc = librosa.effects.hpss(y, margin=(1.0, 5.0))

# ============================================================
# 2. Segment detection
# ============================================================
rms = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=512)[0]
times_rms = librosa.times_like(rms, sr=sr, hop_length=512)
threshold = np.median(rms) * 0.3
is_active = rms > threshold

raw_segments = []
in_segment = False
seg_start = 0
for i in range(len(is_active)):
    if is_active[i] and not in_segment:
        seg_start = times_rms[i]
        in_segment = True
    elif not is_active[i] and in_segment:
        seg_end = times_rms[i]
        if seg_end - seg_start >= 0.5:
            raw_segments.append((seg_start, seg_end))
        in_segment = False
if in_segment:
    seg_end = times_rms[-1]
    if seg_end - seg_start >= 0.5:
        raw_segments.append((seg_start, seg_end))

segments = []
for seg in raw_segments:
    if segments and seg[0] - segments[-1][1] < 0.3:
        segments[-1] = (segments[-1][0], seg[1])
    else:
        segments.append(seg)

print(f"Found {len(segments)} segments\n")

# ============================================================
# 3. Helpers
# ============================================================
def bandpass(y, sr, lo, hi, order=4):
    nyq = sr / 2
    sos = butter(order, [max(lo/nyq, 0.001), min(hi/nyq, 0.999)], btype='band', output='sos')
    return sosfilt(sos, y)

def rms_at_time(y_band, sr, t, window=0.03):
    """Compute RMS energy of a band around a time point."""
    center = int(t * sr)
    half_win = int(window * sr)
    start = max(0, center - half_win)
    end = min(len(y_band), center + half_win)
    chunk = y_band[start:end]
    if len(chunk) == 0:
        return 0.0
    return np.sqrt(np.mean(chunk**2))


def extract_pattern_v2(y_seg, sr, seg_duration):
    """
    1. Detect onsets globally on the percussive signal
    2. For each onset, measure energy in kick/snare/hihat bands
    3. Classify by dominant band
    4. Quantize to grid
    """
    # Band-filtered versions
    y_kick  = bandpass(y_seg, sr, 30, 180)
    y_snare = bandpass(y_seg, sr, 180, 3500)
    y_hihat = bandpass(y_seg, sr, 3500, 11000)

    # Global onset detection on percussive signal
    onset_env = librosa.onset.onset_strength(y=y_seg, sr=sr, hop_length=512)
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=512,
        backtrack=False, units='time'
    )

    # Classify each onset by dominant frequency band energy
    kicks, snares, hihats = [], [], []

    for t in onset_times:
        e_kick  = rms_at_time(y_kick,  sr, t)
        e_snare = rms_at_time(y_snare, sr, t)
        e_hihat = rms_at_time(y_hihat, sr, t)

        total = e_kick + e_snare + e_hihat
        if total < 1e-8:
            continue

        # Normalize to ratios
        r_kick  = e_kick / total
        r_snare = e_snare / total
        r_hihat = e_hihat / total

        # Classify: a hit can be assigned to one or two drums
        # Use thresholds relative to the dominant band
        dominant = max(r_kick, r_snare, r_hihat)

        if r_kick == dominant or r_kick > 0.4:
            kicks.append(t)
        if r_snare == dominant or r_snare > 0.4:
            snares.append(t)
        if r_hihat == dominant or r_hihat > 0.35:
            hihats.append(t)

    kicks = np.array(kicks)
    snares = np.array(snares)
    hihats = np.array(hihats)

    # Tempo estimation
    tempo_arr = librosa.beat.tempo(y=y_seg, sr=sr)
    tempo = float(tempo_arr[0]) if hasattr(tempo_arr, '__len__') else float(tempo_arr)

    # Build 16th-note grid
    beat_dur = 60.0 / tempo
    grid_res = beat_dur / 4
    n_steps = max(1, int(seg_duration / grid_res))
    grid_times = np.arange(n_steps) * grid_res

    # Quantize
    pattern = np.zeros((3, n_steps), dtype=int)
    for row, onsets in enumerate([kicks, snares, hihats]):
        for t in onsets:
            if len(grid_times) > 0:
                idx = np.argmin(np.abs(grid_times - t))
                if abs(grid_times[idx] - t) < grid_res * 0.6:
                    pattern[row, idx] = 1

    return {
        'tempo': tempo,
        'grid_res': grid_res,
        'n_steps': n_steps,
        'pattern': pattern,
        'grid_times': grid_times,
        'onset_times': onset_times,
        'kicks': kicks, 'snares': snares, 'hihats': hihats,
    }


# ============================================================
# 4. Process all segments
# ============================================================
all_patterns = []
for i, (start, end) in enumerate(segments):
    dur = end - start
    s_sample = int(start * sr)
    e_sample = int(end * sr)
    y_seg = y_perc[s_sample:e_sample]

    info = extract_pattern_v2(y_seg, sr, dur)
    info['seg_idx'] = i + 1
    info['start'] = start
    info['end'] = end
    info['duration'] = dur
    all_patterns.append(info)

    n = info['n_steps']

    kick_str  = ''.join(['x' if info['pattern'][0, j] else '.' for j in range(n)])
    snare_str = ''.join(['x' if info['pattern'][1, j] else '.' for j in range(n)])
    hihat_str = ''.join(['x' if info['pattern'][2, j] else '.' for j in range(n)])

    def add_sep(s, every=4):
        return '|'.join([s[j:j+every] for j in range(0, len(s), every)])

    print(f"{'='*60}")
    print(f"Segment {i+1}: {start:.2f}s - {end:.2f}s ({dur:.2f}s) | ~{info['tempo']:.0f} BPM")
    print(f"  {n} steps ({n/4:.1f} beats)")
    print(f"  HH: {add_sep(hihat_str)}")
    print(f"  SN: {add_sep(snare_str)}")
    print(f"  KK: {add_sep(kick_str)}")


# ============================================================
# 5. Detailed visualization for the long segment (#24)
# ============================================================
long_seg = all_patterns[-1]
n = long_seg['n_steps']
steps_per_bar = 16
n_bars = int(np.ceil(n / steps_per_bar))

drum_colors = ['#e74c3c', '#3498db', '#2ecc71']
drum_labels = ['Kick', 'Snare', 'HiHat']

# Show 4 bars per row for readability
bars_per_row = 4
n_rows = int(np.ceil(n_bars / bars_per_row))

fig, axes = plt.subplots(n_rows, 1, figsize=(20, n_rows * 3.0))
if n_rows == 1:
    axes = [axes]

for row_idx in range(n_rows):
    ax = axes[row_idx]
    bar_offset = row_idx * bars_per_row
    step_offset = bar_offset * steps_per_bar
    steps_in_row = min(bars_per_row * steps_per_bar, n - step_offset)

    for drum_row in range(3):
        for col in range(steps_in_row):
            global_step = step_offset + col
            if global_step >= n:
                break
            hit = long_seg['pattern'][drum_row, global_step]
            c = drum_colors[drum_row] if hit else '#f0f0f0'
            ec = '#444' if hit else '#ddd'
            lw = 1.2 if hit else 0.5
            rect = plt.Rectangle((col, 2 - drum_row), 0.9, 0.85,
                                 facecolor=c, edgecolor=ec, linewidth=lw)
            ax.add_patch(rect)

    ax.set_xlim(-0.5, bars_per_row * steps_per_bar + 0.5)
    ax.set_ylim(-0.4, 3.2)
    ax.set_yticks([0.42, 1.42, 2.42])
    ax.set_yticklabels(['HiHat', 'Snare', 'Kick'], fontsize=10)

    # Bar and beat lines
    for b in range(0, bars_per_row * steps_per_bar + 1):
        if b % 16 == 0:
            ax.axvline(x=b, color='black', linewidth=2.5, alpha=0.8)
        elif b % 4 == 0:
            ax.axvline(x=b, color='gray', linewidth=0.8, alpha=0.4)

    # Bar labels
    for bi in range(bars_per_row):
        bar_num = bar_offset + bi + 1
        if bar_num <= n_bars:
            ax.text(bi * 16 + 8, -0.3, f'Bar {bar_num}', ha='center',
                    fontsize=9, fontweight='bold')

    # Beat number labels within bars
    for bi in range(bars_per_row):
        for beat in range(4):
            step = bi * 16 + beat * 4 + 2
            if step < steps_in_row:
                ax.text(step, 3.05, str(beat + 1), ha='center', fontsize=7, color='gray')

legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(drum_colors, drum_labels)]
axes[0].legend(handles=legend_patches, loc='upper right', fontsize=9)
axes[0].set_title(
    f"Segment {long_seg['seg_idx']} - Full Drum Pattern | "
    f"{long_seg['tempo']:.0f} BPM | {n_bars} bars | {long_seg['duration']:.1f}s",
    fontsize=13, fontweight='bold'
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'drum_pattern_v2_long.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved drum_pattern_v2_long.png")


# ============================================================
# 6. Summary grid for ALL segments (compact overview)
# ============================================================
sig_patterns = [p for p in all_patterns if p['duration'] >= 1.0]
n_sig = len(sig_patterns)

fig, axes = plt.subplots(n_sig, 1, figsize=(20, n_sig * 2.0), squeeze=False)

for ax_idx, info in enumerate(sig_patterns):
    ax = axes[ax_idx, 0]
    n_show = min(info['n_steps'], 64)

    for drum_row in range(3):
        for col in range(n_show):
            hit = info['pattern'][drum_row, col]
            c = drum_colors[drum_row] if hit else '#f5f5f5'
            ec = '#444' if hit else '#e0e0e0'
            lw = 1.0 if hit else 0.4
            rect = plt.Rectangle((col, 2 - drum_row), 0.9, 0.85,
                                 facecolor=c, edgecolor=ec, linewidth=lw)
            ax.add_patch(rect)

    ax.set_xlim(-0.3, n_show + 0.3)
    ax.set_ylim(-0.1, 3.1)
    ax.set_yticks([0.42, 1.42, 2.42])
    ax.set_yticklabels(['HH', 'SN', 'KK'], fontsize=8)
    ax.set_title(
        f"Seg {info['seg_idx']}: {info['start']:.1f}s-{info['end']:.1f}s | "
        f"{info['tempo']:.0f} BPM | {info['duration']:.1f}s",
        fontsize=9, fontweight='bold', loc='left'
    )

    for b in range(0, n_show + 1):
        if b % 16 == 0:
            ax.axvline(x=b, color='black', linewidth=2.0, alpha=0.7)
        elif b % 4 == 0:
            ax.axvline(x=b, color='gray', linewidth=0.6, alpha=0.4)

plt.suptitle('All Significant Drum Patterns (v2: onset-classify method)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'drum_patterns_v2_all.png'), dpi=150, bbox_inches='tight')
print(f"Saved drum_patterns_v2_all.png ({n_sig} segments)")


# ============================================================
# 7. Text summary of long segment bar by bar
# ============================================================
print(f"\n{'='*60}")
print(f"DETAILED: Segment {long_seg['seg_idx']} | {long_seg['tempo']:.0f} BPM | {n_bars} bars")
print(f"{'='*60}")
print(f"Legend: x=hit, .=rest | Groups of 4 = one beat | 4 beats = one bar")

for bar in range(n_bars):
    bar_start = bar * steps_per_bar
    bar_end = min(bar_start + steps_per_bar, n)
    steps = bar_end - bar_start

    k  = ''.join(['x' if long_seg['pattern'][0, j] else '.' for j in range(bar_start, bar_end)])
    s  = ''.join(['x' if long_seg['pattern'][1, j] else '.' for j in range(bar_start, bar_end)])
    h  = ''.join(['x' if long_seg['pattern'][2, j] else '.' for j in range(bar_start, bar_end)])

    def fmt(st):
        return '|'.join([st[j:j+4] for j in range(0, len(st), 4)])

    t_sec = long_seg['start'] + long_seg['grid_times'][bar_start]
    print(f"\n  Bar {bar+1:2d} (t={t_sec:.1f}s):")
    print(f"    HH: {fmt(h)}")
    print(f"    SN: {fmt(s)}")
    print(f"    KK: {fmt(k)}")
