"""
Audit of tap_classify.py's SOLID/HOLLOW split. Prints numbers; it re-runs
tap_classify.py, which regenerates output/ as usual.
See AUDIT.md for what the numbers mean.

    python audit_tap_split.py              # run the audit on tile1.opus / tile2.opus
    python audit_tap_split.py --selfcheck  # confirm the null test tells one blob from two
"""
import contextlib, io, runpy, sys
import numpy as np
import librosa
from scipy.signal import hilbert
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def km(Z, seed=42):
    return KMeans(2, n_init=20, random_state=seed).fit_predict(Z)


def null_silhouette(Z, rng, n=30):
    """k=2 silhouette on Gaussian data with Z's covariance: what 'no clusters' scores."""
    cov = np.cov(Z.T)
    return [silhouette_score(G := rng.multivariate_normal(np.zeros(Z.shape[1]), cov, len(Z)), km(G, s))
            for s in range(n)]


def selfcheck():
    rng = np.random.default_rng(0)
    one = rng.normal(size=(90, 20))
    two = np.vstack([rng.normal(size=(45, 20)), rng.normal(size=(45, 20)) + 1.5])
    for name, Z, want in [('one blob', one, False), ('two blobs', two, True)]:
        Z = StandardScaler().fit_transform(Z)
        beats = silhouette_score(Z, km(Z)) > np.percentile(null_silhouette(Z, rng), 95)
        print(f'{name}: beats null = {beats}')
        assert beats == want, name
    print('selfcheck OK')


def main():
    sys.argv = ['tap_classify.py']
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path('tap_classify.py')
    X, taps, lab, names = g['all_features'], g['all_taps'], g['class_labels'], g['FEATURE_NAMES']
    chunks, SR, feat = g['all_chunks'], g['SR'], g['extract_tap_features']
    rng = np.random.default_rng(0)
    dvec = []
    for f in g['AUDIO_FILES']:
        idx = np.array([i for i, t in enumerate(taps) if t['file'] == f])
        F, h = X[idx], lab[idx] == 'HOLLOW'
        t = np.array([taps[i]['time'] for i in idx])
        rms = F[:, 13]
        Z = StandardScaler().fit_transform(F)
        print(f'\n=== {f}: {len(idx)} taps, {h.sum()} HOLLOW / {(~h).sum()} SOLID')

        # A. what separates the clusters
        d = (F[h].mean(0) - F[~h].mean(0)) / (np.sqrt((F[h].var(0) + F[~h].var(0)) / 2) + 1e-12)
        dvec.append(d)
        print('  A. largest effect sizes, HOLLOW minus SOLID (Cohen d):',
              ', '.join(f'{names[j]} {d[j]:+.2f}' for j in np.argsort(-abs(d))[:5]))
        cues = {'bandwidth lower': np.median(F[h, 1]) < np.median(F[~h, 1]),
                'flatness lower': np.median(F[h, 3]) < np.median(F[~h, 3]),
                'centroid lower': np.median(F[h, 0]) < np.median(F[~h, 0])}
        print('  A. stated cues that agree the HOLLOW cluster is the hollow one:',
              ', '.join(f'{k}={bool(v)}' for k, v in cues.items()), f'(bandwidth d={d[1]:+.2f})')

        # B. is there structure at all, and is the split stable
        sil, null = silhouette_score(Z, km(Z)), null_silhouette(Z, rng)
        print(f'  B. silhouette {sil:.3f} vs no-cluster Gaussian null mean {np.mean(null):.3f}, 95th pct {np.percentile(null, 95):.3f}')
        base, aris = km(Z), []
        for b in range(50):
            bi = rng.choice(len(Z), len(Z))
            Zb = StandardScaler().fit(F[bi]).transform(F)
            aris.append(adjusted_rand_score(base, KMeans(2, n_init=20, random_state=b).fit(Zb[bi]).predict(Zb)))
        print(f'  B. bootstrap ARI vs full split: median {np.median(aris):.2f}, share of resamples below 0.5: {np.mean(np.array(aris) < 0.5):.0%}')
        o = np.argsort(t); s = h[o]; p = h.mean()
        print(f'  B. label runs in time order: {1 + np.sum(s[1:] != s[:-1])} (shuffled expectation {1 + 2*len(s)*p*(1-p):.1f})')

        # C. loudness
        print('  C. SOLID share by tap rms:', ', '.join(
            f'[{lo},{hi}) {np.mean(~h[m]):.0%} of {m.sum()}' for lo, hi in [(0.004, 0.008), (0.008, 0.015), (0.015, 0.03), (0.03, 1)]
            if (m := (rms >= lo) & (rms < hi)).any()))
        y, _ = librosa.load(f, sr=SR, mono=True)
        on = librosa.onset.onset_detect(y=y, sr=SR, units='time', delta=0.02)
        bg = [s for s in np.arange(0, len(y) / SR - 0.1, 0.01)
              if np.min(np.abs(on - s)) > 0.15 and np.min(np.abs(t - s)) > 0.15]
        noise = [y[int(s * SR):int(s * SR) + len(chunks[idx[0]])] for s in bg]
        sc = StandardScaler().fit(F); Zf = sc.transform(F)
        ch_, cs_ = Zf[h].mean(0), Zf[~h].mean(0)
        for target, add_bg in ((np.median(rms[~h]), False), (np.median(rms[~h]), True), (0.006, True)):
            flips = 0
            for k in np.where(h)[0]:
                c = chunks[idx[k]] * (target / rms[k])
                if add_bg:
                    c = c + noise[rng.integers(len(noise))][:len(c)]
                z = sc.transform(np.nan_to_num(feat(c, SR))[None])[0]
                flips += np.linalg.norm(z - cs_) < np.linalg.norm(z - ch_)
            print(f'  C. HOLLOW taps rescaled to rms {target:.4f}{" plus this file background" if add_bg else ", no background"}: '
                  f'{flips}/{h.sum()} move nearer the SOLID centroid')

        # D. decay feature vs the 80 ms window
        cens = 0
        for k in idx:
            env = np.convolve(np.abs(hilbert(chunks[k])), np.ones(44) / 44, mode='same')
            pk = np.argmax(env)
            cens += not np.any(env[pk:] < 0.1 * env[pk])
        print(f'  D. taps whose envelope never falls below 10% of peak inside the window: {cens}/{len(idx)}')
        print(f'  E. SOLID tap times (s): {np.round(np.sort(t[~h]), 2).tolist()}')

    print(f'\nF. correlation of the two files\' effect-size vectors: r = {np.corrcoef(*dvec)[0, 1]:+.2f}')


if __name__ == '__main__':
    selfcheck() if '--selfcheck' in sys.argv else main()
