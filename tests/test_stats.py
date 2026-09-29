import math

import pytest

from sentiment.stats import bootstrap_mean_diff_ci, pearson_r, pearson_with_ci


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
