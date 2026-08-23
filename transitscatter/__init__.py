from .lightcurve import simulate_light_curve, trapezoidal_transit, TransitParams
from .detectors import (
    bls_search, scattering_search, scattering_transform, DetectionResult,
    wavelet_flare_clean, flare_robust_search,
)
from .benchmark import run_benchmark, run_flare_robustness_benchmark

__all__ = [
    "simulate_light_curve",
    "trapezoidal_transit",
    "TransitParams",
    "bls_search",
    "scattering_search",
    "scattering_transform",
    "DetectionResult",
    "wavelet_flare_clean",
    "flare_robust_search",
    "run_benchmark",
    "run_flare_robustness_benchmark",
]
