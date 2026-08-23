"""Two transit-detection algorithms benchmarked against each other:

1. `bls_search`      -- classical Box Least Squares (BLS) on raw flux.
2. `scattering_search` -- our method: a 1st-order wavelet scattering transform
   is applied to the light curve first (Morlet filter bank -> modulus ->
   low-pass), producing a smoothed multi-scale "energy" representation that
   suppresses red/correlated noise far more effectively than a simple box
   filter, because red noise power is concentrated at scales the scattering
   transform explicitly separates out. BLS-style period search is then run on
   the resulting scattering coefficients instead of the raw flux.

Both return a `DetectionResult` so they can be compared apples-to-apples.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class DetectionResult:
    best_period: float
    best_power: float
    periods: np.ndarray
    power: np.ndarray


def _phase_fold_box_power(time, signal, period, duration_grid_hours, n_steps=40):
    """Core BLS-style statistic: for a trial period, find the phase/duration
    box that maximizes the standardized depth (in-transit mean vs out-of-transit
    mean, scaled by uncertainty). Returns the best power for this period.

    Vectorized via prefix sums on the phase-sorted signal: for a sorted
    array, a contiguous phase window's sum is `prefix[end] - prefix[start]`,
    found with `searchsorted` in O(log n) instead of O(n) per window, which
    is what makes scanning duration x phase-offset grids tractable.
    """
    n = len(signal)
    phase = (time % period) / period
    order = np.argsort(phase)
    phase_sorted = phase[order]
    signal_sorted = signal[order]

    mean_all = np.mean(signal_sorted)
    var_all = np.var(signal_sorted) + 1e-12

    # duplicate the sorted arrays with phase+1 appended so windows that wrap
    # around phase=1 -> 0 can still be read as one contiguous slice
    phase_ext = np.concatenate([phase_sorted, phase_sorted + 1.0])
    prefix_sum = np.concatenate([[0.0], np.cumsum(signal_sorted)])
    prefix_sum_ext = np.concatenate([prefix_sum, prefix_sum + prefix_sum[-1]])

    best_power = 0.0
    for dur_hours in duration_grid_hours:
        width = (dur_hours / 24.0) / period
        if width <= 0 or width >= 0.5:
            continue
        starts = np.linspace(0, 1 - width, n_steps)
        ends = starts + width

        idx_start = np.searchsorted(phase_ext, starts, side="left")
        idx_end = np.searchsorted(phase_ext, ends, side="left")
        n_in = idx_end - idx_start
        valid = (n_in >= 3) & (n_in <= n * 0.5)
        if not np.any(valid):
            continue

        sums_in = prefix_sum_ext[idx_end[valid]] - prefix_sum_ext[idx_start[valid]]
        means_in = sums_in / n_in[valid]
        depths = mean_all - means_in
        powers = (depths ** 2) * n_in[valid] / var_all
        window_best = np.max(powers)
        if window_best > best_power:
            best_power = window_best
    return best_power


def bls_search(time, flux, period_grid, duration_grid_hours=(1.0, 2.0, 3.0, 4.0)):
    signal = flux - 1.0
    power = np.array([
        _phase_fold_box_power(time, signal, p, duration_grid_hours)
        for p in period_grid
    ])
    best_idx = int(np.argmax(power))
    return DetectionResult(period_grid[best_idx], power[best_idx], period_grid, power)


def _morlet_filter_bank(n_points, dt_days, scales_days):
    """Build a small bank of real-valued Morlet-like wavelet filters at the
    given time scales (in days), used for the 1st-order scattering transform.
    """
    filters = []
    for scale in scales_days:
        sigma = scale / dt_days  # in samples
        half = int(4 * sigma)
        x = np.arange(-half, half + 1)
        envelope = np.exp(-0.5 * (x / sigma) ** 2)
        carrier = np.cos(2 * np.pi * x / (2 * sigma))
        wavelet = envelope * carrier
        wavelet -= wavelet.mean()  # zero-mean so it rejects the DC / smooth trend
        filters.append(wavelet / (np.sum(np.abs(wavelet)) + 1e-12))
    return filters


def scattering_transform(time, flux, scales_days):
    """1st-order wavelet scattering: |x * psi_scale| for each scale, then
    a light Gaussian low-pass (the scattering "averaging" step) to build a
    denoised multi-scale energy representation of the light curve.
    """
    dt = time[1] - time[0]
    x = flux - 1.0
    filters = _morlet_filter_bank(len(x), dt, scales_days)

    coeffs = []
    for wavelet in filters:
        conv = np.convolve(x, wavelet, mode="same")
        modulus = np.abs(conv)
        # low-pass (scattering averaging) with a window matched to the scale
        win = max(3, int(len(wavelet) / 2))
        kernel = np.ones(win) / win
        smoothed = np.convolve(modulus, kernel, mode="same")
        coeffs.append(smoothed)
    return np.array(coeffs)  # shape (n_scales, n_points)


def wavelet_flare_clean(time, flux, fine_scale_days=0.03, coarse_scale_days=0.4,
                         outlier_sigma=4.0):
    """Detect and clip flare-like outliers using a two-scale wavelet channel,
    then hand back a cleaned flux array for a downstream detector.

    This is the piece where multiscale decomposition earns its keep: a
    stellar flare is a *fast* (sub-hour rise), *strongly positive* deviation,
    so it lights up the fine-scale band-pass channel (flux minus a fine
    Gaussian smoothing) far more than the correlated red-noise/transit
    signal does, which lives at coarser scales. Points whose fine-scale
    residual exceeds `outlier_sigma` robust-MAD-sigmas are replaced by the
    coarse-scale trend, which removes flare contamination without touching
    the (much smaller, much slower) transit dip.
    """
    from scipy.ndimage import gaussian_filter1d

    dt = time[1] - time[0]
    fine_sigma = max(fine_scale_days / dt, 1.0)
    coarse_sigma = max(coarse_scale_days / dt, 1.0)

    fine_trend = gaussian_filter1d(flux, sigma=fine_sigma)
    coarse_trend = gaussian_filter1d(flux, sigma=coarse_sigma)
    residual = flux - fine_trend

    mad = np.median(np.abs(residual - np.median(residual))) + 1e-12
    robust_sigma = 1.4826 * mad
    outliers = residual > outlier_sigma * robust_sigma  # one-sided: flares are positive spikes

    cleaned = flux.copy()
    cleaned[outliers] = coarse_trend[outliers]
    return cleaned, outliers


def flare_robust_search(time, flux, period_grid, duration_grid_hours=(1.0, 2.0, 3.0, 4.0),
                         fine_scale_days=0.03, coarse_scale_days=0.4, outlier_sigma=4.0):
    """Our proposed method for contaminated data: wavelet-based flare cleaning
    followed by standard BLS on the cleaned light curve.
    """
    cleaned, outliers = wavelet_flare_clean(time, flux, fine_scale_days, coarse_scale_days, outlier_sigma)
    result = bls_search(time, cleaned, period_grid, duration_grid_hours)
    return result, outliers


def scattering_search(time, flux, period_grid, scales_days=(0.05, 0.125, 0.25),
                       duration_grid_hours=(1.0, 2.0, 3.0, 4.0)):
    """Transit search in the scattering domain. We sum scattering energy
    across scales (a simple, interpretable combination) and hunt for a
    periodic *dip in local energy relative to baseline*, since a transit
    momentarily reduces high-frequency structure less than it reduces the
    smooth flux level being tracked at the finest, transit-matched scale.
    Concretely: we use the finest-scale scattering coefficient as a denoised
    proxy for the transit signal and run the same BLS box statistic on it.
    """
    coeffs = scattering_transform(time, flux, scales_days)
    finest = coeffs[0]
    # Convert energy (always positive) back into a signed proxy by
    # subtracting local baseline so a transit dip in flux shows as a
    # detectable deviation, matching the sign convention BLS expects.
    proxy = -(finest - np.median(finest))

    power = np.array([
        _phase_fold_box_power(time, proxy, p, duration_grid_hours)
        for p in period_grid
    ])
    best_idx = int(np.argmax(power))
    return DetectionResult(period_grid[best_idx], power[best_idx], period_grid, power)
