import numpy as np
from transitscatter import simulate_light_curve, bls_search, scattering_search


def test_high_snr_recovery():
    """At high SNR both detectors should recover the injected period exactly."""
    rng = np.random.default_rng(0)
    t, flux, params, snr = simulate_light_curve(
        depth_ppm=1500, white_noise_ppm=100, red_noise_amp_ppm=30, rng=rng
    )
    grid = np.linspace(1.0, 8.0, 300)

    bls_res = bls_search(t, flux, grid)
    scat_res = scattering_search(t, flux, grid)

    assert abs(bls_res.best_period - params.period_days) < 0.05
    assert abs(scat_res.best_period - params.period_days) < 0.05


def test_transit_is_a_dip():
    """Injected transit must lower flux, not raise it, and only near mid-transit."""
    rng = np.random.default_rng(1)
    t, flux, params, snr = simulate_light_curve(
        depth_ppm=2000, white_noise_ppm=1, red_noise_amp_ppm=1, rng=rng
    )
    mid_transit_idx = np.argmin(np.abs(
        ((t - params.epoch_days + 0.5 * params.period_days) % params.period_days) - 0.5 * params.period_days
    ))
    assert flux[mid_transit_idx] < 1.0 - 0.5 * params.depth_ppm * 1e-6


if __name__ == "__main__":
    test_high_snr_recovery()
    test_transit_is_a_dip()
    print("All tests passed.")
