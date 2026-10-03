from pathlib import Path

import pytest

from sentiment.audit import (
    ShuffleTestResult,
    correlation_shuffle_test,
    regression_shuffle_test,
    run,
    shuffle,
    synthetic_signal_rows,
)
from sentiment.stats import pearson_r


def test_shuffle_is_a_permutation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    shuffled = shuffle(xs, seed=1)
    assert sorted(shuffled) == sorted(xs)
    assert shuffled != xs  # vanishingly unlikely to be a fixed point at this seed/length


def test_shuffle_is_deterministic_given_a_seed():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert shuffle(xs, seed=42) == shuffle(xs, seed=42)


def test_shuffle_does_not_mutate_its_input():
    xs = [1.0, 2.0, 3.0]
    shuffle(xs, seed=1)
    assert xs == [1.0, 2.0, 3.0]


def test_synthetic_signal_rows_is_deterministic_given_a_seed():
    a = synthetic_signal_rows(10, slope=0.1, noise_std=0.01, seed=5)
    b = synthetic_signal_rows(10, slope=0.1, noise_std=0.01, seed=5)
    assert a == b


def test_synthetic_signal_rows_has_the_requested_shape():
    rows = synthetic_signal_rows(15, slope=0.1, noise_std=0.01, seed=5)
    assert len(rows) == 15
    assert all(set(r) == {"compound", "lagged_return"} for r in rows)


def test_correlation_shuffle_test_observed_matches_plain_pearson_r():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [5.0, 3.0, 4.0, 1.0, 2.0]
    result = correlation_shuffle_test(xs, ys, n_perm=200, seed=0)
    assert result.observed == pytest.approx(pearson_r(xs, ys))


def test_correlation_shuffle_test_p_value_is_bounded_and_deterministic():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ys = [5.0, 1.0, 4.0, 2.0, 6.0, 3.0]
    first = correlation_shuffle_test(xs, ys, n_perm=300, seed=7)
    second = correlation_shuffle_test(xs, ys, n_perm=300, seed=7)
    assert first == second
    assert 0.0 < first.p_value <= 1.0


def test_correlation_shuffle_test_detects_a_known_relationship():
    # A real, noise-free linear relationship: shuffling it apart should be
    # obviously significant - this is the sanity check behind the synthetic
    # positive control, at a scale small enough to reason about by hand.
    xs = [float(i) for i in range(20)]
    ys = [2.0 * x for x in xs]
    result = correlation_shuffle_test(xs, ys, n_perm=500, seed=0)
    assert result.observed == pytest.approx(1.0)
    assert result.significant
    assert abs(result.null_mean) < 0.3  # shuffled pairs should average out near zero


def test_correlation_shuffle_test_on_pure_noise_is_not_significant():
    # Fixed, hand-picked values with no real relationship - deterministic
    # given the seed, so this assertion is not a flaky statistical claim.
    xs = [0.1, -0.2, 0.3, -0.1, 0.05, -0.3, 0.2, -0.05]
    ys = [0.01, 0.02, -0.01, 0.03, -0.02, 0.0, 0.01, -0.03]
    result = correlation_shuffle_test(xs, ys, n_perm=1000, seed=3)
    assert not result.significant


def test_regression_shuffle_test_returns_none_for_a_zero_variance_predictor():
    # Exactly the committed fixture's Day 6 finding (see README): every
    # usable headline shares the same compound score - nothing to shuffle.
    rows = [{"compound": 0.296, "lagged_return": r} for r in [0.01, -0.02, 0.03, -0.01, 0.02, 0.04, 0.0]]
    assert regression_shuffle_test(rows, n_perm=200, seed=0) is None


def test_regression_shuffle_test_detects_a_known_relationship():
    rows = synthetic_signal_rows(23, slope=0.08, noise_std=0.03, seed=20260930)
    result = regression_shuffle_test(rows, n_perm=500, seed=0)
    assert result is not None
    assert result.observed > 0
    assert result.significant


def test_regression_shuffle_test_is_deterministic_given_a_seed():
    rows = synthetic_signal_rows(23, slope=0.08, noise_std=0.03, seed=1)
    first = regression_shuffle_test(rows, n_perm=200, seed=9)
    second = regression_shuffle_test(rows, n_perm=200, seed=9)
    assert first == second


def test_shuffle_test_result_significant_property():
    sig = ShuffleTestResult(observed=0.9, n_perm=100, p_value=0.01, null_mean=0.0, null_std=0.1)
    not_sig = ShuffleTestResult(observed=0.1, n_perm=100, p_value=0.5, null_mean=0.0, null_std=0.1)
    assert sig.significant
    assert not not_sig.significant


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, n_perm=50, seed=0, live=False)
    assert exit_code == 1


def test_run_against_committed_fixture_passes_and_finds_no_leak():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    exit_code = run(fixture, n_perm=200, seed=0, live=False)
    assert exit_code == 0
