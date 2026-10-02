import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture

from sentiment.audit import (
    contemporaneous_r,
    permutation_p_value,
    run,
    run_shuffle_trials,
    shuffle_published_at,
)


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_published_at_preserves_the_multiset_of_timestamps():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(5)
    ]
    rng = random.Random(0)

    shuffled = shuffle_published_at(headlines, rng)

    assert len(shuffled) == len(headlines)
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links stay put - only the timestamp assignment moves.
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]


def test_shuffle_published_at_is_deterministic_given_the_same_rng_state():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]

    shuffled_a = shuffle_published_at(headlines, random.Random(7))
    shuffled_b = shuffle_published_at(headlines, random.Random(7))

    assert [h.published_at for h in shuffled_a] == [h.published_at for h in shuffled_b]


def test_shuffle_published_at_actually_moves_timestamps_around(monkeypatch):
    # With 20 distinct timestamps, an unshuffled-looking result is
    # vanishingly unlikely - this guards against a no-op shuffle bug.
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, tzinfo=timezone.utc) + timedelta(hours=i))
        for i in range(20)
    ]
    shuffled = shuffle_published_at(headlines, random.Random(3))
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_permutation_p_value_is_zero_when_real_statistic_is_the_most_extreme():
    assert permutation_p_value(real_r=0.9, null_rs=[0.1, -0.2, 0.3, -0.1]) == 0.0


def test_permutation_p_value_counts_at_least_as_extreme_on_both_sides():
    # |real_r| = 0.2: two of four null draws (-0.5, 0.3) are at least as extreme.
    assert permutation_p_value(real_r=0.2, null_rs=[0.1, -0.5, 0.3, -0.1]) == 0.5


def test_permutation_p_value_is_one_when_null_distribution_is_empty():
    assert permutation_p_value(real_r=0.5, null_rs=[]) == 1.0


def test_contemporaneous_r_is_none_with_fewer_than_two_resolved_headlines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=__import__("datetime").date(2026, 9, 28), open=100.0, close=105.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]

    assert contemporaneous_r(headlines) is None


def test_run_shuffle_trials_returns_one_r_per_successful_trial(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0),
            Bar(date=dt.date(2026, 9, 29), open=105.0, close=102.0),
        ],
    )
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        ),
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "2",
            datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),
        ),
    ]

    null_rs = run_shuffle_trials(headlines, n_shuffles=10, seed=0)

    # Every shuffle here still has both headlines resolve (same ticker,
    # only 2 possible session dates, both covered by the fixture), so all
    # 10 trials should produce a usable r.
    assert len(null_rs) == 10
    assert all(-1.0 <= r <= 1.0 for r in null_rs)


def test_run_against_committed_fixture_passes_the_leakage_audit():
    # The real deal: Day 5 found contemporaneous r=-0.185, n=23 on the
    # committed fixture, with a 95% CI that comfortably contains zero.
    # The shuffled-timestamp null should contain that r too - this is
    # exactly the "Done when" gate NEXT_STEPS.md describes, run as an
    # assertion instead of read off a printout by hand.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, n_shuffles=500, seed=0)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, live=False, n_shuffles=10, seed=0)

    assert exit_code == 1
