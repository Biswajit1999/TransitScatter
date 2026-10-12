# BLS convergence and empirical-null validation

This extension uses labelled synthetic light curves from the existing
white-plus-correlated-noise generator. It adds three analyses requested by the
original research roadmap:

1. period-grid convergence from 80 to 640 trial periods;
2. an empirical maximum-power null from depth-zero simulations; and
3. a trials-factor sweep showing how the 95th-percentile null maximum changes
   as more periods are searched.

The empirical upper-tail estimate uses `(exceedances + 1)/(replicates + 1)` so
finite simulation sets never report a false zero probability. Run
`python scripts/run_validation.py` to regenerate the JSON, CSV, and figure.

These are synthetic pipeline tests, not observations. The null distribution is
conditional on this simulator, cadence, BLS implementation, duration grid and
period range. It omits realistic gaps, non-stationary systematics, stellar
populations and analyst search tuning, so it is not a calibrated Kepler/TESS
false-alarm probability.

