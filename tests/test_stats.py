import math

import pytest

from sentiment.stats import (
    bootstrap_mean_diff_ci,
    mean_absolute_error,
    ols_fit,
    oos_r_squared,
    pearson_r,
    pearson_with_ci,
    permutation_p_value,
    r_squared,
)


def test_pearson_r_perfect_positive_correlation():
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 6, 8, 10]
    assert pearson_r(xs, ys) == pytest.approx(1.0)


def test_pearson_r_perfect_negative_correlation():
    xs = [1, 2, 3, 4, 5]
    ys = [10, 8, 6, 4, 2]
    assert pearson_r(xs, ys) == pytest.approx(-1.0)


def test_pearson_r_no_variance_in_one_series_is_zero_not_a_crash():
    xs = [1, 1, 1, 1]
    ys = [1, 2, 3, 4]
    assert pearson_r(xs, ys) == 0.0


def test_pearson_r_requires_equal_length_and_at_least_two_points():
    with pytest.raises(ValueError):
        pearson_r([1, 2], [1])
    with pytest.raises(ValueError):
        pearson_r([1], [1])


def test_pearson_with_ci_contains_r_and_widens_at_small_n():
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    ys = [1.1, 2.0, 2.9, 4.2, 4.8, 6.1, 6.9, 8.0]
    result = pearson_with_ci(xs, ys)
    assert result.ci_low <= result.r <= result.ci_high
    assert result.n == 8

    small = pearson_with_ci(xs[:3], ys[:3])
    assert (small.ci_high - small.ci_low) >= (result.ci_high - result.ci_low)


def test_pearson_with_ci_below_four_points_returns_full_range():
    result = pearson_with_ci([1, 2, 3], [1, 2, 3])
    assert result.ci_low == -1.0
    assert result.ci_high == 1.0


def test_bootstrap_mean_diff_ci_is_deterministic_given_a_seed():
    a = [0.01, 0.02, -0.01, 0.03]
    b = [-0.02, 0.00, -0.03, 0.01]
    first = bootstrap_mean_diff_ci(a, b, seed=42)
    second = bootstrap_mean_diff_ci(a, b, seed=42)
    assert first == second


def test_bootstrap_mean_diff_ci_diff_matches_plain_mean_difference():
    a = [1.0, 2.0, 3.0]
    b = [0.0, 0.0, 0.0]
    result = bootstrap_mean_diff_ci(a, b, seed=1)
    assert result.diff == pytest.approx(2.0)
    assert result.ci_low <= result.diff <= result.ci_high


def test_bootstrap_mean_diff_ci_rejects_empty_groups():
    with pytest.raises(ValueError):
        bootstrap_mean_diff_ci([], [1.0])
    with pytest.raises(ValueError):
        bootstrap_mean_diff_ci([1.0], [])


def test_bootstrap_mean_diff_ci_identical_groups_center_near_zero():
    a = [0.01, -0.02, 0.03, -0.01, 0.02]
    result = bootstrap_mean_diff_ci(a, list(a), seed=7, n_boot=4000)
    assert result.diff == pytest.approx(0.0, abs=1e-9)
    assert result.ci_low < 0 < result.ci_high or math.isclose(result.ci_low, 0, abs_tol=1e-6)


def test_ols_fit_recovers_a_known_line():
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 3.0, 5.0, 7.0, 9.0]  # y = 1 + 2x, no noise
    fit = ols_fit(xs, ys)
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.predict(10.0) == pytest.approx(21.0)


def test_ols_fit_zero_variance_x_returns_zero_slope_and_mean_intercept():
    # Day 6's own committed-fixture finding: every usable headline shares the
    # identical compound score, so there is no slope to fit at all - this
    # must return a defined answer (slope=0), not divide by zero.
    xs = [0.296, 0.296, 0.296, 0.296]
    ys = [0.01, -0.02, 0.03, -0.04]
    fit = ols_fit(xs, ys)
    assert fit.slope == 0.0
    assert fit.intercept == pytest.approx(sum(ys) / len(ys))


def test_ols_fit_requires_equal_length_and_at_least_two_points():
    with pytest.raises(ValueError):
        ols_fit([1, 2], [1])
    with pytest.raises(ValueError):
        ols_fit([1], [1])


def test_r_squared_perfect_fit_is_one():
    y_true = [1.0, 2.0, 3.0]
    assert r_squared(y_true, y_true) == pytest.approx(1.0)


def test_r_squared_predicting_the_mean_everywhere_is_zero():
    y_true = [1.0, 2.0, 3.0]
    mean_pred = [2.0, 2.0, 2.0]
    assert r_squared(y_true, mean_pred) == pytest.approx(0.0)


def test_oos_r_squared_positive_when_predictions_beat_the_training_mean():
    y_test = [1.0, 2.0, 3.0]
    y_pred = [1.1, 2.1, 2.9]  # close to actual
    assert oos_r_squared(y_test, y_pred, train_mean=2.0) > 0


def test_oos_r_squared_is_zero_when_prediction_equals_the_training_mean():
    # exactly what sentiment.regress reports on the committed fixture: a
    # zero-variance predictor makes every prediction equal the training mean.
    y_test = [1.0, -3.0, 5.0]
    y_pred = [2.0, 2.0, 2.0]
    assert oos_r_squared(y_test, y_pred, train_mean=2.0) == pytest.approx(0.0)


def test_oos_r_squared_negative_when_worse_than_the_training_mean_baseline():
    y_test = [1.0, 2.0, 3.0]
    y_pred = [10.0, -8.0, 15.0]  # wild misses
    assert oos_r_squared(y_test, y_pred, train_mean=2.0) < 0


def test_mean_absolute_error_basic():
    assert mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(2.0 / 3.0)


def test_mean_absolute_error_requires_at_least_one_point():
    with pytest.raises(ValueError):
        mean_absolute_error([], [])


def test_permutation_p_value_is_small_when_real_r_is_an_outlier():
    null_rs = [0.01, -0.02, 0.03, 0.0, -0.01] * 20  # 100 draws clustered near 0
    p = permutation_p_value(real_r=0.95, null_rs=null_rs)
    assert p < 0.05


def test_permutation_p_value_is_large_when_real_r_looks_like_the_null():
    null_rs = [0.2, -0.2, 0.18, -0.19, 0.21, -0.17] * 10
    p = permutation_p_value(real_r=0.19, null_rs=null_rs)
    assert p > 0.5


def test_permutation_p_value_uses_absolute_value_two_sided():
    # a real_r of -0.95 is just as extreme against this null as +0.95
    null_rs = [0.0] * 50
    assert permutation_p_value(0.95, null_rs) == permutation_p_value(-0.95, null_rs)


def test_permutation_p_value_never_reports_impossible_from_a_finite_sample():
    # even a real_r nothing in the null comes close to still gets p > 0,
    # not 0 - a finite number of shuffles can only make a result "rare",
    # never "proven impossible under the null".
    null_rs = [0.0] * 50
    p = permutation_p_value(1.0, null_rs)
    assert p > 0.0


def test_permutation_p_value_rejects_empty_null():
    with pytest.raises(ValueError):
        permutation_p_value(0.5, [])
