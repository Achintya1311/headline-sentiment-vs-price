import math

import pytest

from sentiment.stats import (
    bootstrap_mean_diff_ci,
    mean_absolute_error,
    minimum_detectable_r,
    ols_fit,
    oos_r_squared,
    pearson_r,
    pearson_with_ci,
    PermutationResult,
    percentile_ci,
    permutation_p_value,
    r_squared,
    required_n_for_r,
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


def test_minimum_detectable_r_at_n_100_matches_the_prereg_gate_rationale():
    # docs/POWER.md and README's "Pre-registered follow-up test" section
    # both cite this: the old 100-headline gate could only ever detect a
    # correlation of about 0.28 or larger at 80% power.
    assert minimum_detectable_r(100) == pytest.approx(0.277, abs=0.001)


def test_required_n_for_r_010_matches_the_prereg_gate_rationale():
    # r = 0.10 (a plausible small real effect) needs roughly 790 headlines -
    # the reason Day 84 raised the data gate from 100 to 1,000.
    assert required_n_for_r(0.10) == pytest.approx(782.65, abs=1.0)


def test_minimum_detectable_r_and_required_n_for_r_are_inverses():
    for n in (50, 150, 500, 2000):
        r = minimum_detectable_r(n)
        assert required_n_for_r(r) == pytest.approx(n, abs=0.05)


def test_minimum_detectable_r_shrinks_as_n_grows():
    assert minimum_detectable_r(100) > minimum_detectable_r(1000) > minimum_detectable_r(10000)


def test_minimum_detectable_r_rejects_n_too_small_for_the_approximation():
    with pytest.raises(ValueError):
        minimum_detectable_r(3)


def test_required_n_for_r_rejects_out_of_range_r():
    with pytest.raises(ValueError):
        required_n_for_r(0.0)
    with pytest.raises(ValueError):
        required_n_for_r(1.0)


def test_power_md_table_matches_the_functions_it_cites():
    # docs/POWER.md quotes these numbers by hand; pin them here so the doc
    # can't silently drift from the code that is supposed to back it.
    for n, expected in [(23, 0.556), (42, 0.421), (50, 0.387), (100, 0.277), (250, 0.176), (500, 0.125), (1000, 0.088), (2000, 0.063)]:
        assert minimum_detectable_r(n) == pytest.approx(expected, abs=0.001)
    for r, expected in [(0.05, 3137), (0.10, 783), (0.15, 347), (0.20, 194), (0.28, 98), (0.30, 85)]:
        assert required_n_for_r(r) == pytest.approx(expected, abs=1.0)


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


def test_permutation_p_value_small_when_real_r_is_extreme_outlier():
    # The real statistic sits far outside a null that never gets near it -
    # Day 8's audit flags exactly this shape as leak-like.
    perm_rs = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.015]
    assert permutation_p_value(0.9, perm_rs) == pytest.approx(1 / 8)


def test_permutation_p_value_large_when_real_r_looks_like_the_null():
    perm_rs = [0.2, -0.25, 0.18, -0.19, 0.22, -0.21, 0.17, -0.2]
    p = permutation_p_value(0.19, perm_rs)
    assert p > 0.5


def test_permutation_p_value_never_reports_exactly_zero():
    perm_rs = [0.01] * 1000
    p = permutation_p_value(0.99, perm_rs)
    assert p > 0.0
    assert p == pytest.approx(1 / 1001)


def test_permutation_p_value_rejects_empty_null():
    with pytest.raises(ValueError):
        permutation_p_value(0.5, [])

# Tests recovered from parallel Day 8 runs (percentile_ci, permutation_p_value edge cases).

def test_percentile_ci_contains_the_middle_of_a_known_distribution():
    values = list(range(1, 101))  # 1..100
    lo, hi = percentile_ci(values, ci=0.95)
    # 95% interval should trim roughly the bottom/top 2.5 values each side.
    assert lo == pytest.approx(3, abs=1)
    assert hi == pytest.approx(98, abs=1)

def test_percentile_ci_excludes_zero_when_the_distribution_never_crosses_it():
    values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    lo, hi = percentile_ci(values)
    assert lo > 0.0
    assert hi > 0.0

def test_percentile_ci_includes_zero_when_the_distribution_straddles_it():
    values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    lo, hi = percentile_ci(values)
    assert lo <= 0.0 <= hi

def test_percentile_ci_requires_at_least_one_value():
    with pytest.raises(ValueError):
        percentile_ci([])

def test_permutation_p_value_requires_at_least_one_null_sample():
    with pytest.raises(ValueError):
        permutation_p_value(1.0, [])

def test_permutation_p_value_real_stat_equal_to_every_null_value_gives_one():
    null = [0.3, 0.3, 0.3]
    assert permutation_p_value(0.3, null) == pytest.approx(1.0)

def test_permutation_p_value_requires_at_least_one_null_sample_v2():
    with pytest.raises(ValueError):
        permutation_p_value(0.1, [])

def test_permutation_p_value_is_two_sided():
    # a strongly negative observed value should be treated the same as the
    # equally strongly positive one - this is a two-sided test by design.
    null_samples = [0.0, 0.1, -0.1, 0.2, -0.2]
    assert permutation_p_value(-0.9, null_samples) == permutation_p_value(0.9, null_samples)

def test_permutation_p_value_requires_at_least_one_null_sample_v3():
    with pytest.raises(ValueError):
        permutation_p_value(0.5, [])

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


def test_permutation_result_null_mean_and_std():
    r = PermutationResult(observed=1.0, null_samples=[0.0, 2.0], p_value=0.5)
    assert r.null_mean == pytest.approx(1.0)
    assert r.null_std == pytest.approx(1.0)
