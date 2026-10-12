"""Numerical convergence and empirical-null validation for BLS searches."""

from __future__ import annotations

import numpy as np

from .detectors import bls_search
from .lightcurve import simulate_light_curve


def period_grid_convergence(*, seed: int = 20261012):
    rng = np.random.default_rng(seed)
    time, flux, params, _ = simulate_light_curve(
        n_points=1200,
        depth_ppm=900,
        white_noise_ppm=180,
        red_noise_amp_ppm=60,
        rng=rng,
    )
    rows = []
    for trial_count in (80, 160, 320, 640):
        periods = np.linspace(1.0, 8.0, trial_count)
        result = bls_search(time, flux, periods)
        rows.append(
            {
                "trial_count": trial_count,
                "grid_spacing_days": float(periods[1] - periods[0]),
                "injected_period_days": params.period_days,
                "recovered_period_days": result.best_period,
                "absolute_period_error_days": abs(
                    result.best_period - params.period_days
                ),
                "best_power": result.best_power,
            }
        )
    return rows


def empirical_null_calibration(
    *,
    replicates: int = 120,
    period_trials: int = 120,
    seed: int = 20261013,
):
    """Compare an injected BLS maximum with maxima from depth-zero simulations."""
    rng = np.random.default_rng(seed)
    periods = np.linspace(1.0, 8.0, period_trials)
    time, flux, params, _ = simulate_light_curve(
        n_points=900,
        depth_ppm=500,
        white_noise_ppm=200,
        red_noise_amp_ppm=80,
        rng=rng,
    )
    injected = bls_search(time, flux, periods)
    maxima = []
    for _ in range(replicates):
        null_time, null_flux, _, _ = simulate_light_curve(
            n_points=900,
            depth_ppm=0,
            white_noise_ppm=200,
            red_noise_amp_ppm=80,
            rng=rng,
        )
        maxima.append(bls_search(null_time, null_flux, periods).best_power)
    maxima = np.asarray(maxima)
    exceedances = int(np.sum(maxima >= injected.best_power))
    return {
        "replicates": replicates,
        "period_trials": period_trials,
        "seed": seed,
        "injected_period_days": params.period_days,
        "recovered_period_days": injected.best_period,
        "injected_max_power": injected.best_power,
        "null_median_max_power": float(np.median(maxima)),
        "null_p95_max_power": float(np.percentile(maxima, 95)),
        "exceedances": exceedances,
        "empirical_p_upper": float((exceedances + 1) / (replicates + 1)),
    }


def trials_factor_sweep(*, replicates: int = 50, seed: int = 20261014):
    rng = np.random.default_rng(seed)
    rows = []
    for trial_count in (40, 80, 160, 320):
        periods = np.linspace(1.0, 8.0, trial_count)
        maxima = []
        for _ in range(replicates):
            time, flux, _, _ = simulate_light_curve(
                n_points=600,
                depth_ppm=0,
                white_noise_ppm=200,
                red_noise_amp_ppm=80,
                rng=rng,
            )
            maxima.append(bls_search(time, flux, periods).best_power)
        rows.append(
            {
                "period_trials": trial_count,
                "replicates": replicates,
                "median_null_max_power": float(np.median(maxima)),
                "p95_null_max_power": float(np.percentile(maxima, 95)),
            }
        )
    return rows
