from transitscatter.validation import (
    empirical_null_calibration,
    period_grid_convergence,
    trials_factor_sweep,
)


def test_period_grid_refinement_reduces_final_error():
    rows = period_grid_convergence(seed=3)
    assert (
        rows[-1]["absolute_period_error_days"] < rows[0]["absolute_period_error_days"]
    )


def test_null_calibration_returns_bounded_probability():
    result = empirical_null_calibration(replicates=5, period_trials=20, seed=4)
    assert 0 < result["empirical_p_upper"] <= 1


def test_trials_factor_sweep_shapes():
    rows = trials_factor_sweep(replicates=3, seed=5)
    assert [row["period_trials"] for row in rows] == [40, 80, 160, 320]
