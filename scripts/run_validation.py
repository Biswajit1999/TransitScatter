"""Generate BLS convergence and false-alarm validation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transitscatter.validation import (  # noqa: E402
    empirical_null_calibration,
    period_grid_convergence,
    trials_factor_sweep,
)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    convergence = period_grid_convergence()
    null = empirical_null_calibration()
    trials = trials_factor_sweep()
    write_csv(ROOT / "results/bls-grid-convergence.csv", convergence)
    write_csv(ROOT / "results/bls-trials-factor.csv", trials)
    payload = {
        "data_kind": "synthetic injection-recovery and depth-zero null simulations",
        "grid_convergence": convergence,
        "empirical_null": null,
        "trials_factor": trials,
        "limitations": "The null follows this repository's white-plus-smoothed-red-noise generator and fixed BLS implementation. It is not a survey false-alarm probability and omits real gaps, systematics, stellar populations, and search tuning.",
    }
    (ROOT / "results/bls-validation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with plt.style.context(["science", "no-latex", "bright"]):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
        spacings = [r["grid_spacing_days"] for r in convergence]
        errors = [r["absolute_period_error_days"] for r in convergence]
        axes[0].loglog(spacings, errors, "o-", lw=2, ms=6, label="Recovered period")
        axes[0].loglog(spacings, spacings, "--", lw=1.2, label="One grid spacing")
        axes[0].invert_xaxis()
        axes[0].set(
            xlabel="Period-grid spacing [d]",
            ylabel="Absolute period error [d]",
            title="(a) Grid convergence",
        )
        axes[0].legend(frameon=False)
        period_trials = [r["period_trials"] for r in trials]
        p50 = [r["median_null_max_power"] for r in trials]
        p95 = [r["p95_null_max_power"] for r in trials]
        axes[1].plot(period_trials, p50, "o-", lw=2, ms=6, label="Median")
        axes[1].plot(period_trials, p95, "s-", lw=2, ms=5, label="95th percentile")
        axes[1].fill_between(period_trials, p50, p95, alpha=0.16)
        axes[1].set(
            xlabel="Number of searched periods",
            ylabel="Maximum BLS power under null",
            title="(b) Look-elsewhere growth",
        )
        axes[1].legend(frameon=False)
        fig.suptitle("TransitScatter: numerical validation of the BLS search")
        fig.savefig(ROOT / "results/bls-validation.png", dpi=240)
        fig.savefig(ROOT / "results/bls-validation.pdf")
        plt.close(fig)
    print(json.dumps(null, indent=2))


if __name__ == "__main__":
    main()
