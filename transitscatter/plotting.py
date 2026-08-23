"""Plot generation for benchmark results and example light curves."""
from __future__ import annotations

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .lightcurve import simulate_light_curve
from .detectors import scattering_transform


def _read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_recovery_curve(csv_path="results/benchmark_results.csv", out_path="results/recovery_curve.png"):
    rows = _read_csv(csv_path)
    snr = [r["mean_snr"] for r in rows]
    bls = [r["bls_recovery_rate"] for r in rows]
    scat = [r["scattering_recovery_rate"] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(snr, bls, "o-", label="BLS (raw flux)", color="#d95f02", linewidth=2)
    plt.plot(snr, scat, "s-", label="Wavelet-scattering search", color="#1b9e77", linewidth=2)
    plt.xlabel("Injected transit SNR (folded, MES-like estimate)")
    plt.ylabel("Recovery rate")
    plt.title("Injection-Recovery: BLS vs Wavelet-Scattering Search\n(clean red+white Gaussian noise)")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_flare_robustness(csv_path="results/flare_robustness_results.csv", out_path="results/flare_robustness.png"):
    rows = _read_csv(csv_path)
    rate = [r["flare_rate_per_day"] for r in rows]
    bls = [r["bls_recovery_rate"] for r in rows]
    robust = [r["flare_robust_recovery_rate"] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(rate, bls, "o-", label="BLS (raw flux)", color="#d95f02", linewidth=2)
    plt.plot(rate, robust, "^-", label="Wavelet flare-clean + BLS", color="#7570b3", linewidth=2)
    plt.xlabel("Injected flare rate (events / day)")
    plt.ylabel("Recovery rate")
    plt.title("Robustness to Flare Contamination\n(fixed transit depth=150 ppm)")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_example_light_curve(out_path="results/example_light_curve.png", depth_ppm=350, seed=7):
    rng = np.random.default_rng(seed)
    t, flux, params, snr = simulate_light_curve(depth_ppm=depth_ppm, flare_rate_per_day=0.4, rng=rng)
    coeffs = scattering_transform(t, flux, scales_days=(0.05, 0.125, 0.25))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, flux, ".", color="#377eb8", markersize=2, alpha=0.6)
    axes[0].set_ylabel("Relative flux")
    axes[0].set_title(f"Simulated light curve (depth={depth_ppm} ppm, SNR~{snr:.1f}, with injected flares)")

    proxy = -(coeffs[0] - np.median(coeffs[0]))
    axes[1].plot(t, proxy, color="#1b9e77", linewidth=1)
    axes[1].set_ylabel("Scattering proxy\n(finest scale)")
    axes[1].set_xlabel("Time (days)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_example_light_curve()
    plot_recovery_curve()
    plot_flare_robustness()
