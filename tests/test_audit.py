import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sentiment.audit import audit_statistic, run, shuffle_timestamps
from sentiment.headline import Headline, write_csv


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def _not_significant_rows(n: int) -> list[dict]:
    # Constant compound -> pearson_r's zero-variance guard returns r=0.0,
    # whose Fisher-transform CI is symmetric around 0 and never excludes it.
    return [{"compound": 0.2, "contemporaneous_return": 0.01 * i, "lagged_return": 0.01 * i} for i in range(n)]


def _significant_rows(n: int) -> list[dict]:
    # Perfectly collinear, noise-free -> r=1.0, whose CI sits nowhere near 0.
    return [
        {"compound": 0.1 * (i + 1), "contemporaneous_return": 0.01 * (i + 1), "lagged_return": 0.01 * (i + 1)}
        for i in range(n)
    ]


def test_shuffle_timestamps_preserves_the_multiset_and_keeps_titles_with_their_headline():
    headlines = [
        make_headline(f"title-{i}", f"link-{i}", datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(8)
    ]
    rng = random.Random(0)

    shuffled = shuffle_timestamps(headlines, rng)

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # With 8 distinct timestamps, some reassignment should actually happen -
    # this isn't guaranteed in general, but is overwhelmingly likely here and
    # pins the RNG's actual behaviour for this seed.
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_timestamps_is_deterministic_given_the_same_seed():
    headlines = [
        make_headline(f"title-{i}", f"link-{i}", datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(8)
    ]

    first = shuffle_timestamps(headlines, random.Random(42))
    second = shuffle_timestamps(headlines, random.Random(42))

    assert [h.published_at for h in first] == [h.published_at for h in second]


def test_audit_statistic_passes_when_shuffled_control_rarely_significant():
    real_rows = _not_significant_rows(5)
    # 1 "significant" shuffle out of 20 = 5%, within 2x tolerance of alpha=0.05.
    shuffled_rows_list = [_not_significant_rows(5) for _ in range(19)] + [_significant_rows(5)]

    result = audit_statistic(
        real_rows, shuffled_rows_list, "contemporaneous_return", "contemporaneous", alpha=0.05, tolerance_multiplier=2.0
    )

    assert result.n_shuffles_usable == 20
    assert result.significant_rate == pytest.approx(0.05)
    assert result.passed
    assert result.real_r == pytest.approx(0.0)
    assert not result.real_significant


def test_audit_statistic_fails_when_shuffled_control_too_often_significant():
    real_rows = _not_significant_rows(5)
    # 10/20 = 50%, far above the 10% tolerance - the leak signature.
    shuffled_rows_list = [_significant_rows(5) for _ in range(10)] + [_not_significant_rows(5) for _ in range(10)]

    result = audit_statistic(
        real_rows, shuffled_rows_list, "contemporaneous_return", "contemporaneous", alpha=0.05, tolerance_multiplier=2.0
    )

    assert result.significant_rate == pytest.approx(0.5)
    assert not result.passed


def test_audit_statistic_skips_shuffles_below_the_pearson_ci_floor():
    real_rows = _not_significant_rows(5)
    too_few = _not_significant_rows(2)  # below MIN_N_FOR_CI=4
    shuffled_rows_list = [too_few, too_few, _significant_rows(5)]

    result = audit_statistic(
        real_rows, shuffled_rows_list, "contemporaneous_return", "contemporaneous", alpha=0.05, tolerance_multiplier=2.0
    )

    assert result.n_shuffles_usable == 1
    assert result.significant_rate == pytest.approx(1.0)
    assert not result.passed


def test_audit_statistic_excludes_rows_with_no_lagged_return():
    real_rows = [
        {"compound": 0.1, "contemporaneous_return": 0.01, "lagged_return": None},
        {"compound": 0.2, "contemporaneous_return": 0.02, "lagged_return": 0.01},
    ]

    result = audit_statistic(real_rows, [], "lagged_return", "lagged", alpha=0.05, tolerance_multiplier=2.0)

    assert result.real_n == 1  # only the row with a non-None lagged_return counts
    assert result.real_r is None  # 1 point isn't enough for a correlation


def test_audit_statistic_handles_no_usable_real_or_shuffled_rows():
    result = audit_statistic([], [], "contemporaneous_return", "contemporaneous", alpha=0.05, tolerance_multiplier=2.0)

    assert result.real_r is None
    assert result.real_n == 0
    assert not result.real_significant
    assert result.n_shuffles_usable == 0
    assert result.significant_rate == 0.0
    assert result.passed  # nothing to measure is not a leak


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, live=False, n_shuffles=10, seed=0, alpha=0.05)

    assert exit_code == 1


def test_run_reports_failure_when_nothing_resolves_to_a_ticker(tmp_path: Path):
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Cyient among 4 stocks showing White Marubozu Pattern",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            )
        ],
        in_path,
    )

    exit_code = run(in_path, live=False, n_shuffles=10, seed=0, alpha=0.05)

    assert exit_code == 1


def test_run_against_committed_fixture_passes(capsys):
    # The real end-to-end gate: this repo's own 50-headline fixture, the same
    # one sentiment.correlate's Day 5 null result (r=-0.185, r=+0.000, both
    # CIs containing zero) already reports on in the README. A small shuffle
    # count keeps the test fast; sentiment.audit's own CLI default is larger.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, n_shuffles=50, seed=0, alpha=0.05)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "contemporaneous" in out
    assert "lagged" in out
    assert "PASS" in out
