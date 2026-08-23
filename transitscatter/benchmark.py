"""Injection-recovery benchmark: BLS vs wavelet-scattering transit search.

For a grid of injected SNR levels, simulate many light curves, run both
detectors, and measure the recovery rate (fraction where the recovered
period matches the injected period within tolerance). This is the standard
methodology used in the transit-search literature (e.g. Kepler/TESS
pipeline validation) for comparing detection algorithms.
"""
from __future__ import annotations

import csv
import time as _time
import numpy as np

from .lightcurve import simulate_light_curve
from .detectors import bls_search, scattering_search, flare_robust_search


def _period_matches(recovered, injected, tol=0.01):
    ratio = recovered / injected
    for harmonic in (1, 2, 0.5, 3, 1 / 3):
        if abs(ratio - harmonic) < tol * harmonic:
            return True
    return False


def run_benchmark(
    depths_ppm=(150, 250, 400, 600, 900, 1300),
    n_trials=25,
    period_days=3.2,
    seed=42,
    out_csv="results/benchmark_results.csv",
):
    rng = np.random.default_rng(seed)
    period_grid = np.linspace(1.0, 8.0, 350)

    rows = []
    t0 = _time.time()
    for depth in depths_ppm:
        bls_hits, scat_hits = 0, 0
        snrs = []
        for trial in range(n_trials):
            t, flux, params, snr = simulate_light_curve(
                depth_ppm=depth, period_days=period_days, rng=rng
            )
            snrs.append(snr)

            bls_res = bls_search(t, flux, period_grid)
            scat_res = scattering_search(t, flux, period_grid)

            if _period_matches(bls_res.best_period, params.period_days):
                bls_hits += 1
            if _period_matches(scat_res.best_period, params.period_days):
                scat_hits += 1

        row = {
            "depth_ppm": depth,
            "mean_snr": float(np.mean(snrs)),
            "bls_recovery_rate": bls_hits / n_trials,
            "scattering_recovery_rate": scat_hits / n_trials,
            "n_trials": n_trials,
        }
        rows.append(row)
        print(f"depth={depth:>5} ppm  SNR~{row['mean_snr']:.2f}  "
              f"BLS={row['bls_recovery_rate']:.2f}  "
              f"Scattering={row['scattering_recovery_rate']:.2f}")

    elapsed = _time.time() - t0
    print(f"Benchmark finished in {elapsed:.1f}s")

    import os
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def run_flare_robustness_benchmark(
    flare_rates=(0.0, 0.2, 0.5, 1.0),
    n_trials=30,
    depth_ppm=150.0,
    period_days=3.2,
    flare_amp_ppm_range=(800.0, 2500.0),
    seed=123,
    out_csv="results/flare_robustness_results.csv",
):
    """Second experiment: fix a moderate, easily-recovered transit depth and
    vary the rate of injected stellar-flare-like outliers per day. Compares
    plain BLS (vulnerable to outliers inflating/distorting the box statistic)
    against wavelet-based flare cleaning + BLS.
    """
    rng = np.random.default_rng(seed)
    period_grid = np.linspace(1.0, 8.0, 350)

    rows = []
    t0 = _time.time()
    for rate in flare_rates:
        bls_hits, robust_hits = 0, 0
        for trial in range(n_trials):
            t, flux, params, snr = simulate_light_curve(
                depth_ppm=depth_ppm, period_days=period_days,
                flare_rate_per_day=rate, flare_amp_ppm_range=flare_amp_ppm_range,
                rng=rng,
            )

            bls_res = bls_search(t, flux, period_grid)
            robust_res, _ = flare_robust_search(t, flux, period_grid)

            if _period_matches(bls_res.best_period, params.period_days):
                bls_hits += 1
            if _period_matches(robust_res.best_period, params.period_days):
                robust_hits += 1

        row = {
            "flare_rate_per_day": rate,
            "bls_recovery_rate": bls_hits / n_trials,
            "flare_robust_recovery_rate": robust_hits / n_trials,
            "n_trials": n_trials,
        }
        rows.append(row)
        print(f"flare_rate={rate:.2f}/day  BLS={row['bls_recovery_rate']:.2f}  "
              f"WaveletClean+BLS={row['flare_robust_recovery_rate']:.2f}")

    elapsed = _time.time() - t0
    print(f"Flare-robustness benchmark finished in {elapsed:.1f}s")

    import os
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    run_benchmark()
    run_flare_robustness_benchmark()
