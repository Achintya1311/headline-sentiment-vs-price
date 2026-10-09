import random

from sentiment.audit import (
    contemporaneous_r,
    permutation_test,
    positive_control_headlines,
    run,
    shuffle_published_at,
)
from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import read_csv

# Keep CI fast: the CLI's own default (500) is for a stable, reportable
# number: these just need enough replicates to make the collapse and the
# null-consistency checks reliable.
TEST_N_SHUFFLES = 150


def test_shuffle_published_at_permutes_timestamps_but_keeps_titles_in_place():
    headlines = positive_control_headlines()
    rng = random.Random(1)
    shuffled = shuffle_published_at(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # a real shuffle of >100 distinct timestamps essentially never lands
    # back on the identity permutation.
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_positive_control_headlines_resolve_with_correctly_signed_compound():
    headlines = positive_control_headlines()
    rows, unresolved = build_rows_from_headlines(headlines)

    assert unresolved == []
    assert len(rows) > 50  # six tickers x ~22 fixture sessions, minus a few flat days
    for row in rows:
        # the construction: up-sentiment title iff that session's own
        # contemporaneous return was positive.
        assert (row["compound"] > 0) == (row["contemporaneous_return"] > 0)


def test_contemporaneous_r_on_positive_control_is_strong():
    r = contemporaneous_r(positive_control_headlines())
    assert r is not None
    assert r > 0.5


def test_permutation_test_shows_the_engineered_signal_collapses_under_shuffling():
    headlines = positive_control_headlines()
    result = permutation_test(headlines, n_shuffles=TEST_N_SHUFFLES, seed=0)

    assert result.real_r > 0.5
    assert result.p_value < 0.05
    # the shuffled null should be far weaker than the real-pairing signal -
    # this is what proves the audit has power to catch a leak, not just
    # that it passes when there is nothing to catch.
    assert result.mean_abs_shuffled_r < result.real_r / 2


def test_permutation_test_on_real_fixture_is_consistent_with_days_5_and_6_null_result():
    real_headlines = read_csv(DEFAULT_IN)
    result = permutation_test(real_headlines, n_shuffles=TEST_N_SHUFFLES, seed=0)

    # Day 5 found r=-0.185 with a 95% CI comfortably containing zero.
    assert result.real_r == -0.185 or abs(result.real_r - (-0.185)) < 1e-3
    # a real-pairing result this weak should not be an outlier against its
    # own shuffled null - that is the "no leak detected" case.
    assert result.p_value > 0.05


def test_run_passes_both_checks_and_exits_zero():
    exit_code = run(n_shuffles=TEST_N_SHUFFLES, seed=0, live=False)
    assert exit_code == 0
