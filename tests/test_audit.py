from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sentiment.audit import (
    NOTABLE_EFFECT,
    SIGNIFICANCE,
    AuditResult,
    contemporaneous_r,
    permutation_test,
    run,
    shuffle_timestamps,
)
from sentiment.headline import Headline, read_csv


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def make_headlines(n: int) -> list[Headline]:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [make_headline(f"headline {i}", f"link-{i}", base + timedelta(hours=i)) for i in range(n)]


# --- shuffle_timestamps -------------------------------------------------


def test_shuffle_timestamps_preserves_the_same_pool_of_timestamps():
    import random

    headlines = make_headlines(20)
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links/sources untouched - only WHO published WHEN changed
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.source for h in shuffled] == [h.source for h in headlines]


def test_shuffle_timestamps_actually_permutes_given_enough_headlines():
    import random

    headlines = make_headlines(30)
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    # Not a strict requirement of a random shuffle, but with n=30 and a fixed
    # seed the identity permutation is astronomically unlikely - this guards
    # against a shuffle_timestamps that accidentally no-ops.
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_timestamps_is_deterministic_given_the_same_seed():
    import random

    headlines = make_headlines(15)
    a = shuffle_timestamps(headlines, random.Random(42))
    b = shuffle_timestamps(headlines, random.Random(42))

    assert [h.published_at for h in a] == [h.published_at for h in b]


# --- contemporaneous_r ---------------------------------------------------


def test_contemporaneous_r_needs_at_least_two_rows():
    assert contemporaneous_r([{"compound": 0.1, "contemporaneous_return": 0.01}]) is None
    assert contemporaneous_r([]) is None


def test_contemporaneous_r_matches_pearson_r_on_two_rows():
    rows = [
        {"compound": 0.1, "contemporaneous_return": 0.01},
        {"compound": -0.2, "contemporaneous_return": -0.02},
    ]
    assert contemporaneous_r(rows) == pytest.approx(1.0)


# --- permutation_test: the audit's actual discriminating power -----------


def test_permutation_test_catches_a_pipeline_that_ignores_the_shuffled_timestamp():
    """Positive control: a ``build_rows`` that silently ignores whatever
    ``published_at`` it is handed (keying the return only off the headline's
    own identity) is exactly the leak NEXT_STEPS.md warns about - shuffling
    timestamps should have no effect on its output, and this test proves the
    audit actually notices that rather than rubber-stamping it."""
    headlines = make_headlines(20)
    true_compound = {h.title: float(i) for i, h in enumerate(headlines)}
    true_return = {h.title: float(i) for i, h in enumerate(headlines)}

    def leaky_build_rows(hs: list[Headline]) -> list[dict]:
        # Return values keyed by title (headline identity), never by the
        # published_at actually passed in - a stand-in for a pipeline bug
        # that doesn't really route through the timestamp it claims to.
        return [
            {"compound": true_compound[h.title], "contemporaneous_return": true_return[h.title]} for h in hs
        ]

    result = permutation_test(headlines, leaky_build_rows, iterations=50, seed=1)

    assert result.real_stat == pytest.approx(1.0)
    assert abs(result.real_stat) >= NOTABLE_EFFECT
    # every shuffled run reproduces the identical perfect correlation, since
    # shuffling the input timestamps changed nothing about the output
    assert result.p_value == pytest.approx(1.0)
    assert not result.passes


def test_permutation_test_passes_a_genuinely_timestamp_dependent_signal():
    """Negative control: a ``build_rows`` whose output genuinely depends on
    the timestamp it is handed (return looked up BY published_at, not by
    headline identity) should have its real correlation collapse under most
    reshuffles - proving the audit does not just always fail notable
    effects, only ones that survive shuffling."""
    headlines = make_headlines(20)
    compound_by_title = {h.title: float(i) for i, h in enumerate(headlines)}
    return_by_timestamp = {h.published_at: float(i) for i, h in enumerate(headlines)}

    def genuine_build_rows(hs: list[Headline]) -> list[dict]:
        return [
            {"compound": compound_by_title[h.title], "contemporaneous_return": return_by_timestamp[h.published_at]}
            for h in hs
        ]

    result = permutation_test(headlines, genuine_build_rows, iterations=200, seed=2)

    assert result.real_stat == pytest.approx(1.0)
    assert abs(result.real_stat) >= NOTABLE_EFFECT
    # a random reassignment of 20 timestamps almost never reproduces the one
    # true pairing that made the real correlation perfect
    assert result.p_value < SIGNIFICANCE
    assert result.passes


def test_permutation_test_raises_when_real_statistic_is_undefined():
    headlines = make_headlines(1)

    def empty_build_rows(hs: list[Headline]) -> list[dict]:
        return []

    with pytest.raises(ValueError):
        permutation_test(headlines, empty_build_rows, iterations=5, seed=0)


def test_permutation_test_reports_how_many_shuffles_produced_usable_rows():
    headlines = make_headlines(20)
    compound_by_title = {h.title: float(i) for i, h in enumerate(headlines)}
    return_by_timestamp = {h.published_at: float(i) for i, h in enumerate(headlines)}

    def flaky_build_rows(hs: list[Headline]) -> list[dict]:
        rows = [
            {"compound": compound_by_title[h.title], "contemporaneous_return": return_by_timestamp[h.published_at]}
            for h in hs
        ]
        return rows if len(rows) >= 3 else []

    result = permutation_test(headlines, flaky_build_rows, iterations=30, seed=3)
    assert result.n_shuffles_used <= 30
    assert result.n_shuffles_used == len(result.shuffled_stats)


# --- AuditResult.passes threshold logic -----------------------------------


def test_small_real_effect_passes_regardless_of_p_value():
    result = AuditResult(real_stat=0.1, n_shuffles_used=10, shuffled_stats=[0.9] * 10, p_value=1.0)
    assert result.passes


def test_notable_effect_with_low_p_value_passes():
    result = AuditResult(real_stat=0.9, n_shuffles_used=10, shuffled_stats=[0.1] * 10, p_value=0.01)
    assert result.passes


def test_notable_effect_with_high_p_value_fails():
    result = AuditResult(real_stat=0.9, n_shuffles_used=10, shuffled_stats=[0.9] * 10, p_value=1.0)
    assert not result.passes


# --- run() / CLI, including against the real committed fixture -----------


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, iterations=5, seed=0, live=False)

    assert exit_code == 1


def test_run_against_committed_fixture_is_honest_about_the_null_result():
    # The real committed fixture is exactly Day 5/6/7's low-signal case
    # (contemporaneous r ~ -0.185, see README Findings) - the audit should
    # find nothing notable to flag, deterministically, since `passes` for a
    # sub-threshold real_stat doesn't depend on the random shuffled runs.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, iterations=25, seed=0, live=False)

    assert exit_code == 0

    headlines = read_csv(fixture)
    from sentiment.correlate import build_rows_from_headlines

    rows, _ = build_rows_from_headlines(headlines, live=False)
    real_r = contemporaneous_r(rows)
    assert real_r is not None
    assert abs(real_r) < NOTABLE_EFFECT
