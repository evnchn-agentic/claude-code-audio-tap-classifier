"""Diagnose onset detection issues — compare parameters and find missed hits."""
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

y, sr = librosa.load('mysterious_audio.wav', sr=22050, mono=True)
y_harm, y_perc = librosa.effects.hpss(y, margin=(1.0, 5.0))

# Also try less aggressive HPSS
y_harm2, y_perc2 = librosa.effects.hpss(y, margin=(1.0, 2.0))

# And raw signal (no HPSS)
y_raw = y

# --- Compare onset detection on 3 signals with varying parameters ---
signals = {
    'perc_aggressive (margin=5)': y_perc,
    'perc_mild (margin=2)': y_perc2,
    'raw (no HPSS)': y_raw,
}

# Also compare onset_detect parameters
detect_configs = {
    'default': {},
    'sensitive (delta=0.05)': {'delta': 0.05},
    'very_sensitive (delta=0.02, wait=1)': {'delta': 0.02, 'wait': 1},
}

print("Onset count comparison:\n")
print(f"{'Signal':<30s} {'Config':<35s} {'Onsets':>6s}")
print("-" * 75)

results = {}
for sig_name, sig in signals.items():
    for cfg_name, cfg in detect_configs.items():
        onset_env = librosa.onset.onset_strength(y=sig, sr=sr, hop_length=512)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=512,
            backtrack=False, units='time', **cfg
        )
        key = f"{sig_name} + {cfg_name}"
        results[key] = onsets
        print(f"{sig_name:<30s} {cfg_name:<35s} {len(onsets):>6d}")

# --- Visual comparison on a few key regions ---
# Region 1: short segments area (34-55s)
# Region 2: long segment (121-152s)
regions = [
    ("Short segments (34-55s)", 34, 55),
    ("Long segment start (121-135s)", 121, 135),
    ("Long segment end (140-152s)", 140, 152),
    ("Quiet area (0-10s)", 0, 10),
]

for region_name, t_start, t_end in regions:
    fig, axes = plt.subplots(4, 1, figsize=(20, 8), sharex=True)
    fig.suptitle(f'Onset Detection Comparison: {region_name}', fontsize=13, fontweight='bold')

    # Waveforms
    s, e = int(t_start * sr), int(t_end * sr)
    t = np.linspace(t_start, t_end, e - s)

    axes[0].plot(t, y[s:e], alpha=0.4, color='gray', linewidth=0.5)
    axes[0].set_ylabel('Raw')

    axes[1].plot(t, y_perc[s:e], alpha=0.6, color='blue', linewidth=0.5)
    axes[1].set_ylabel('Perc (m=5)')

    axes[2].plot(t, y_perc2[s:e], alpha=0.6, color='green', linewidth=0.5)
    axes[2].set_ylabel('Perc (m=2)')

    # Onset envelopes + detected onsets
    ax = axes[3]
    for sig_name, color, sig in [
        ('perc(m=5)', 'blue', y_perc),
        ('perc(m=2)', 'green', y_perc2),
        ('raw', 'gray', y_raw),
    ]:
        env = librosa.onset.onset_strength(y=sig, sr=sr, hop_length=512)
        env_times = librosa.times_like(env, sr=sr, hop_length=512)
        mask = (env_times >= t_start) & (env_times <= t_end)
        # Normalize for comparison
        if env.max() > 0:
            env_norm = env / env.max()
        else:
            env_norm = env
        ax.plot(env_times[mask], env_norm[mask], color=color, alpha=0.6, label=sig_name, linewidth=0.8)

    # Show detected onsets (most sensitive config on mild HPSS)
    key = "perc_mild (margin=2) + very_sensitive (delta=0.02, wait=1)"
    onsets_best = results[key]
    onsets_in_range = onsets_best[(onsets_best >= t_start) & (onsets_best <= t_end)]
    for ot in onsets_in_range:
        ax.axvline(x=ot, color='red', alpha=0.4, linewidth=0.8)

    key_default = "perc_aggressive (margin=5) + default"
    onsets_default = results[key_default]
    onsets_default_range = onsets_default[(onsets_default >= t_start) & (onsets_default <= t_end)]
    for ot in onsets_default_range:
        ax.axvline(x=ot, color='black', alpha=0.7, linewidth=1.5, linestyle='--')

    ax.set_ylabel('Onset Env')
    ax.legend(fontsize=8)
    ax.set_xlabel('Time (s)')

    # Add onset count annotation
    ax.text(0.99, 0.95,
            f'Default (black dashed): {len(onsets_default_range)}\n'
            f'Sensitive (red): {len(onsets_in_range)}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    safe_name = region_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    diag_path = os.path.join(OUTPUT_DIR, f'diag_{safe_name}.png')
    plt.savefig(diag_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved {diag_path}")

# --- Also check: are there hits outside detected segments? ---
print("\n\nChecking for hits outside segments...")
# Use the most sensitive detection on raw signal
onset_env_raw = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
all_onsets_raw = librosa.onset.onset_detect(
    onset_envelope=onset_env_raw, sr=sr, hop_length=512,
    backtrack=False, units='time', delta=0.02, wait=1
)

# Check which onsets fall outside segments
seg_ranges = [(s, e) for s, e in [(s, e) for s, e in
    # rebuild segments with lower threshold
    [(0, 999)]  # placeholder
]]

# Actually, let's check with different segment thresholds
rms = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=512)[0]
times_rms = librosa.times_like(rms, sr=sr, hop_length=512)

for thresh_mult in [0.3, 0.15, 0.05]:
    threshold = np.median(rms) * thresh_mult
    is_active = rms > threshold

    segs = []
    in_seg = False
    for i in range(len(is_active)):
        if is_active[i] and not in_seg:
            seg_s = times_rms[i]
            in_seg = True
        elif not is_active[i] and in_seg:
            seg_e = times_rms[i]
            if seg_e - seg_s >= 0.1:
                segs.append((seg_s, seg_e))
            in_seg = False
    if in_seg:
        segs.append((seg_s, times_rms[-1]))

    # Merge
    merged = []
    for seg in segs:
        if merged and seg[0] - merged[-1][1] < 0.2:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    total_covered = sum(e - s for s, e in merged)
    print(f"  thresh_mult={thresh_mult}: {len(merged)} segments, covering {total_covered:.1f}s / {librosa.get_duration(y=y, sr=sr):.1f}s")

print("\nDone.")
