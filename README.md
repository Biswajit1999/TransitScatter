# TransitScatter

A small, fully reproducible **injection–recovery benchmark** comparing classical
**Box Least Squares (BLS)** transit search against a **wavelet-scattering-based**
alternative, on synthetic exoplanet light curves. No external data or
downloads required — every light curve is simulated from a documented noise
model, so every number in this repo can be regenerated from a random seed in
a few minutes.

This project was built as a focused, single-session research exercise: pick a
concrete, testable hypothesis about detection algorithms, build the smallest
faithful simulation needed to test it, run the experiment, and report what
actually happened — including when the "novel" idea loses to the baseline.

## Motivation

Exoplanet transit surveys (Kepler, TESS, and ground-based programs like
EXOhSPEC) hunt for a periodic, shallow, box-shaped dip in stellar brightness
buried in correlated ("red") noise from stellar granulation/activity, plus
white photon noise, plus occasional flare-like outliers. BLS is the workhorse
algorithm: it phase-folds the light curve at trial periods and searches for
the box-shaped window that best separates in-transit from out-of-transit flux.

Wavelet scattering transforms are popular in time-series/audio ML for
building noise-robust, multi-scale representations. The question this repo
asks: **does a scattering-style multiscale decomposition help transit
detection, either in raw sensitivity or in robustness to non-Gaussian
contamination (flares)?**

## Method

- `transitscatter/lightcurve.py` — simulates a light curve: a trapezoidal
  transit model injected into white Gaussian noise + a smooth
  Ornstein-Uhlenbeck-like red-noise component, with an optional Poisson-rate
  stream of exponential-decay flare events.
- `transitscatter/detectors.py` — three detectors, all built on the same
  vectorized BLS box-statistic core:
  - `bls_search` — standard BLS on raw flux (the baseline).
  - `scattering_search` — 1st-order wavelet scattering (Morlet filter bank →
    modulus → low-pass) used to build a denoised detection proxy, searched
    the same way as BLS.
  - `flare_robust_search` — a two-scale wavelet outlier detector
    (`wavelet_flare_clean`) that flags and clips flare-like fast positive
    spikes via a robust MAD threshold, then runs plain BLS on the cleaned flux.
- `transitscatter/benchmark.py` — two injection-recovery experiments over
  many random trials, with recovery defined as the top-detected period
  matching the injected period (or a low-order harmonic) within 1%.
- `transitscatter/plotting.py` — regenerates every plot in `results/`.

Run it yourself:

```bash
pip install -r requirements.txt
python -m transitscatter.benchmark    # ~5-6 min, writes results/*.csv
python -m transitscatter.plotting     # writes results/*.png
pytest tests/ -q
```

## Results

### Experiment 1 — clean noise, varying SNR

![Recovery curve](results/recovery_curve.png)

| Injected SNR | BLS recovery | Scattering recovery |
|---:|---:|---:|
| 1.2 | 4% | 4% |
| 2.0 | 4% | 0% |
| 3.2 | 8% | 4% |
| 4.6 | 24% | 0% |
| 6.9 | 88% | 4% |
| 10.4 | 100% | 4% |

**Finding: BLS decisively outperforms the wavelet-scattering search in
clean Gaussian (white + red) noise.** The scattering proxy sits near the
~4% chance level (harmonic-tolerance false-positive rate) across the whole
SNR range tested. The likely reason: BLS's box statistic already *is* close
to an optimal matched filter for a box-shaped signal — it directly averages
flux within a candidate transit window. The scattering transform's
oscillatory (zero-mean) Morlet wavelets respond to the *edges* of a
box-shaped dip (ingress/egress), not its flat interior, so taking the
modulus and using it as a signed "denoised flux" proxy discards most of the
depth information that BLS is built to use directly. This is a legitimate,
if humbling, negative result — not every noise-robust representation from
the ML literature transfers cleanly to a box-search problem it wasn't
designed for.

### Experiment 2 — flare-contaminated light curves

![Flare robustness](results/flare_robustness.png)

| Flare rate (events/day) | BLS recovery | Wavelet-clean + BLS recovery |
|---:|---:|---:|
| 0.0 | 100% | 100% |
| 0.2 | 7% | 7% |
| 0.5 | 3% | 0% |
| 1.0 | 3% | 3% |

**Finding: even sparse flare contamination collapses recovery for both
methods, and wavelet-based flare cleaning does not reliably rescue it** at
the settings tested here (depth = 150 ppm, flare amplitude 800–2500 ppm).
Diagnostic runs show why: a single strong flare, or two flares that happen
to fall in phase at some trial period, can produce a box-statistic peak that
outweighs the true (much shallower) transit signal anywhere in a 350-point
period grid — a spurious phase-alignment false alarm, not a
locally-elevated-noise problem. Clipping outlier points with the MAD-based
wavelet filter reduces the height of that spurious peak somewhat, but does
not reliably remove it before the period search sees it, since the affected
points are relatively few (~1% of samples) against a 350-period grid dense
enough for occasional coincidental alignment.

## What I'd do next

- Score periods with a **matched-filter template of the real trapezoidal
  shape** (a proper transit-least-squares / TLS-style detector) as a stronger
  scattering-adjacent baseline than raw-modulus scattering.
- For flare robustness: **iteratively re-fit** (identify best period → mask
  points near that period's transit window → recheck for remaining outliers)
  instead of one static pre-pass, and/or penalize periods whose "detection"
  is driven by very few points.
- Sweep the period grid resolution and formalize a false-alarm-probability
  correction (e.g. bootstrap) rather than a fixed top-1 recovery criterion,
  since some of the apparent "collapse" is standard look-elsewhere-effect
  behavior in a 350-trial period grid.

The period-grid and empirical-null portion of this roadmap is now implemented
in `python scripts/run_validation.py`; see
[`docs/BLS_VALIDATION.md`](docs/BLS_VALIDATION.md). Its probability remains a
simulation-conditional diagnostic, not a survey false-alarm rate.

## Repository layout

```
transitscatter/
  lightcurve.py    synthetic light-curve + flare simulator
  detectors.py     BLS, wavelet-scattering search, flare-robust search
  benchmark.py     injection-recovery experiments
  plotting.py      figure generation
tests/
  test_core.py     sanity checks (high-SNR recovery, transit sign/shape)
results/           generated CSVs + PNGs (checked in for reproducibility)
```

## License

MIT — see [LICENSE](LICENSE).
