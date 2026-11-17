# Audit: what the Phase 2 SOLID/HOLLOW split measures

**Verdict.** The tap recordings do contain real structure: both files beat a no-cluster null, and
the labels come in runs over time. But the evidence does not show that the split tracks hollowness.

- **Loudness drives much of the split.** It sorts taps by how hard they were struck almost as much
  as by what they sound like.
- **Most taps fall back to HOLLOW when struck hard.** Quiet taps are labelled SOLID 75 to 100% of the
  time; taps with rms of 0.015 or more are labelled SOLID 0 to 19% of the time.
- **Quieting a tap is enough to relabel it.** Taking a HOLLOW tap, scaling it down and mixing in the
  room's own background noise moves 20 to 50% of them to the SOLID side, though nothing about the
  tile changed.
- **The two files are not split along the same axis.** In tile1, the label rule contradicts two of
  the three cues the README gives for hollowness.

The README's line "Both files correctly show hollow-majority, matching ground truth" has no per-tap
labels behind it. A per-file majority is a weak check: any rule that splits unevenly gets it right
for both files about one time in four.

Everything below comes from `python audit_tap_split.py` on the files in this repo. It is seeded and
deterministic, and `--selfcheck` confirms the null test separates a one-blob case from a two-blob
case. Versions used: Python 3.12.11, librosa 1.0.1, scikit-learn 1.9.1, numpy 2.5.5, scipy 1.18.1.

## Findings

| # | Check | tile1 (83 taps) | tile2 (95 taps) | Reading |
|---|---|---|---|---|
| B | k=2 silhouette against a Gaussian null with the same covariance (95th percentile) | 0.340 vs 0.233 | 0.251 vs 0.198 | Some structure exists. The null is lenient: a skewed single blob would beat it as well. |
| B | Label runs in time order against a shuffled expectation | 13 vs 24.3 | 33 vs 45.7 | Labels cluster in time. This fits the operator moving tile to tile, and also fits a change in tapping style. |
| B | Bootstrap resamples whose split disagrees with the full split (ARI below 0.5) | 36% | 34% | About a third of resamples find a different partition. |
| A | Cues that agree the HOLLOW cluster is hollow (lower bandwidth / flatness / centroid) | 1 of 3 | 3 of 3 | In tile1 only bandwidth agrees, and it is the weakest separator (d = −0.73). The HOLLOW cluster there is *brighter* (centroid +2.27 d) and noisier, which the README attributes to SOLID. |
| F | Correlation of the two files' 20-D effect-size vectors | r = +0.38 | | The single label covers two largely different splits. |
| C | SOLID share by tap rms, low to high: [0.004, 0.008), [0.008, 0.015), [0.015, 0.03), ≥ 0.03 | 100 / 30 / 0 / 4% | 100 / 75 / 19 / 5% | Strongly tied to strike strength. |
| C | HOLLOW taps rescaled to the SOLID median rms plus background, then to rms 0.006 plus background | 12/69, then 24/69 move to SOLID | 15/59, then 31/59 | Signal-to-background ratio alone can flip the label. Scaling without the background flips none (0/69, 0/59), so the flips need the background, not just a lower level. |
| D | Taps whose envelope never drops below 10% of peak within the 80 ms window | 42/83 | 79/95 | The `decay_rate` feature mostly measures the window length. Hollow tiles ring longer, which is the cue this feature should capture, and the window cuts it off. |
| E | SOLID tap times | scattered | bursts at 11.1–11.7 s, 35.0–35.7 s and 38.8–39.3 s, about 5 taps per second | The bursts are rapid light tapping, a different technique from single strikes. |

## What the videos show

The source videos (kept outside this repo) are a flat-handover inspection with a telescopic
tapping rod. Tiles the inspector judged hollow are marked with a blue **X**:

- **tile1:** a balcony floor with many X marks.
- **tile2:** a room floor and bathroom walls.

That means per-tap ground truth exists in principle: for each onset, whether the rod tip is on an
X-marked tile.

## What this does not show

- **It does not show the split is wrong.** Hollow tiles may well ring louder, and some of the
  loudness link could be physical. Only labelled taps can separate the two explanations.
- **Nothing in `tap_classify.py` was changed.**

## Route from here

1. **Label the taps.** For each onset time in `output/tap_classify.json`, record whether the rod is
   on an X-marked tile. It takes one pass through each video, and `extract_hit_frames.py` can
   export the frames. About 180 labels turn every number above into an accuracy figure.
2. **Normalise tap strength before clustering.** Then report the split only for taps above a
   signal-to-background floor, since quiet taps are where the label is least reliable.
3. **Lengthen the analysis window, or measure decay without censoring.** The resonance cue needs
   more than 80 ms.
4. **Then train supervised.** With labels, the unsupervised k=2 step and the bandwidth naming rule
   can go.
