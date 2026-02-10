# Audio Classification Pipeline

A stepping-stone project toward a **multi-modal scaffolding joint fixity assessment system**. This repo develops and validates audio-based percussive event classification techniques, starting with an unsupervised drum hit classifier and progressing toward a supervised tap-test binary classifier for structural inspection.

## Project Context

This work feeds into an FYP (Final Year Project) on automated scaffolding inspection. The end goal is an iPad-based system where an inspector taps scaffolding couplers and gets an instant tight/loose classification. This repo prototypes the audio classification pipeline in Python before porting to CoreML/iOS.

See `Automated scaffolding inspection_ a multi-modal strategy for joint fixity verification.md` for the full FYP research brief.

## Phase 1: Unsupervised Drum Hit Classification

Uses a DoodleChaos Algodoo physics simulation video (`mysterious.webm`) where balls hit labeled instrument blocks (Snare, Kick, Hat, Cymbal). Two parallel detection pipelines are cross-validated:

- **Visual ground truth** (`extract_ground_truth.py`): Detects block color flashes via frame differencing, with scene transition rejection and purple ramp filtering.
- **Audio onset detection + clustering** (`extract_drums.py`): HPSS isolation, onset detection, pitch-invariant feature extraction (18D), KMeans clustering.
- **Cross-validation** (`cross_validate.py`): Matches audio onsets to visual events within 80ms tolerance.

### Pitch-Invariant Features (18D)

Designed to cluster the same instrument type together regardless of tonal variation across scenes:

| Features | Count | Purpose |
|----------|-------|---------|
| Temporal envelope (attack, decay, centroid, sharpness) | 4 | Strike mechanics |
| Spectral ratios (flatness, low/mid/high energy) | 4 | Dimensionless frequency balance |
| Spectral contrast (5 bands + valley) | 6 | Sub-band peak-vs-valley |
| ZCR, RMS, MFCC[1-2] | 4 | Broad timbral shape |

### Results

- 476 onsets detected, auto-k selects k=2 (bass/treble split, silhouette=0.230)
- Forced k=4 gives plausible instrument grouping (kick/snare/mid/hat)
- Cross-validation: 49.7% visual recall, 35.3% audio precision, +3.5ms mean offset

## Phase 2: Tap-Test Binary Classifier

Binary classification of tap sounds as "solid" vs "hollow/empty" using floor tile recordings as initial training data. This is the stepping stone toward scaffolding coupler joint fixity assessment.

### Key Difference from Phase 1

Phase 1 (drums) uses **pitch-invariant** features — same instrument clusters together regardless of pitch. Phase 2 (tap-test) uses **pitch-sensitive** features — frequency shifts ARE the diagnostic signal (hollow = resonant/tonal, solid = broadband/damped).

### Pitch-Sensitive Features (20D)

| Features | Count | Purpose |
|----------|-------|---------|
| Spectral (centroid, bandwidth, rolloff, flatness, dominant freq) | 5 | Absolute frequency content — shifts indicate structural condition |
| Energy ratios (low/mid/high) | 3 | Frequency balance |
| Temporal envelope (attack, decay, centroid, decay rate) | 4 | Strike response dynamics |
| ZCR, RMS | 2 | Basic signal characteristics |
| MFCCs [1-5] | 5 | Full spectral detail (pitch-sensitive, intentionally) |
| Spectral contrast mean | 1 | Peak-vs-valley structure |

### Approach

- **Per-file clustering**: Each file is clustered independently (KMeans k=2) to prevent between-file material differences from masking within-file solid/hollow variance
- **Label assignment**: Bandwidth-based heuristic — lower bandwidth cluster = HOLLOW (tonal resonance), higher = SOLID (broadband damped)
- **Hardcoded exclusion zones** for known non-tap events (e.g., tile2 ~20-22s walking thuds) — see Limitations below
- **Prediction overlay videos**: Visual verification via `tap_video_overlay.py` — composites classification banners onto source video with 1s persistence

### Results

- 178 audible taps across 2 files (MIN_TAP_RMS=0.004, noise floor ~0.002)
- Primary discriminator: **bandwidth** (tonal resonance vs broadband thud)
- HOLLOW: bandwidth=2150 Hz, flatness=0.082 (narrow, tonal — resonant void)
- SOLID: bandwidth=2556 Hz, flatness=0.139 (wide, noisy — damped by substrate)
- tile1: 83 taps — 14 SOLID (17%), 69 HOLLOW (83%), silhouette=0.340
- tile2: 95 taps — 36 SOLID (38%), 59 HOLLOW (62%), silhouette=0.251
- Both files correctly show hollow-majority, matching ground truth

### Limitations

- **Non-tap events are indistinguishable from taps using audio alone.** Walking thuds, footsteps, and accidental impacts produce broadband transients similar to tap events. The current system uses hardcoded exclusion zones (e.g., tile2 ~20-22s) to reject known non-tap events, but this does not generalise.
- **Accelerometer/IMU data is likely needed** for robust tap detection in field conditions. A multi-modal approach combining audio classification with accelerometer-based gesture recognition (detecting the deliberate tap motion) would allow automatic rejection of incidental impacts without manual exclusion zones.

## Scripts

| Script | Purpose |
|--------|---------|
| `extract_drums.py` | Unsupervised drum classifier (pitch-invariant features + KMeans) |
| `tap_classify.py` | Binary tap-test classifier (pitch-sensitive features + per-file KMeans) |
| `tap_video_overlay.py` | Overlay prediction banners onto source video for visual verification |
| `extract_ground_truth.py` | Visual ground truth from video frame differencing |
| `cross_validate.py` | Audio vs visual cross-validation |
| `analyze_structure.py` | Audio structure analysis (segments, tempo) |
| `diagnose_onsets.py` | Onset detection diagnostics |
| `extract_hit_frames.py` | Extract video frames at hit moments |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install librosa numpy scipy scikit-learn matplotlib opencv-python Pillow
```

## Usage

```bash
# Phase 1: Drum classification
ffmpeg -i mysterious.webm -vn -acodec pcm_s16le -ar 22050 -ac 1 mysterious_audio.wav -y
python extract_ground_truth.py
python extract_drums.py                    # auto-k
python extract_drums.py mysterious_audio.wav 4  # forced k=4
python cross_validate.py

# Phase 2: Tap-test classification
python tap_classify.py                     # uses tile1.opus, tile2.opus by default
python tap_classify.py myfile1.opus myfile2.opus  # custom files
python tap_video_overlay.py                # overlay predictions on video (needs tile*-video.webm)

# Outputs go to output/ and frames/
```

## Configuration

All scripts exclude the last 10 seconds of audio/video (`TRIM_END_SECONDS = 10.0`) to skip a parody ending.
