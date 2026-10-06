import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import sentiment.correlate as correlate
import sentiment.prices as prices
from sentiment.audit import run, run_audit, shuffle_published_at
from sentiment.headline import Headline
from sentiment.prices import Bar, save_fixture


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_published_at_keeps_titles_but_redistributes_timestamps():
    headlines = [
        make_headline("a", "1", datetime(2026, 9, 28, 1, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 9, 28, 2, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 9, 28, 3, tzinfo=timezone.utc)),
    ]

    shuffled = shuffle_published_at(headlines, random.Random(0))

    assert [h.title for h in shuffled] == ["a", "b", "c"]
    assert {h.published_at for h in shuffled} == {h.published_at for h in headlines}
    # with this seed, the permutation actually moves at least one timestamp -
    # otherwise the "shuffle" wouldn't be testing anything
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_published_at_is_deterministic_for_a_given_seed():
    headlines = [
        make_headline("a", "1", datetime(2026, 9, 28, 1, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 9, 28, 2, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 9, 28, 3, tzinfo=timezone.utc)),
        make_headline("d", "4", datetime(2026, 9, 28, 4, tzinfo=timezone.utc)),
    ]

    first = shuffle_published_at(headlines, random.Random(42))
    second = shuffle_published_at(headlines, random.Random(42))

    assert [h.published_at for h in first] == [h.published_at for h in second]


def _trading_days(start: date, n: int) -> list[date]:
    """n weekday dates starting from start (inclusive), skipping weekends -
    mirrors sentiment.market_hours.is_trading_day's weekday-only rule."""
    days: list[date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def test_audit_detects_a_genuine_timing_dependent_signal(tmp_path: Path, monkeypatch):
    """A synthetic case where the sentiment score is a literal, deterministic
    function of the return of the session the headline's *real* timestamp
    aligns to. Shuffling timestamps reassigns each headline to a different
    session's (different) return while its score stays fixed - exactly the
    leak-free-by-construction setup this audit exists to tell apart from the
    real fixture's null result. Proves the mechanism can actually catch a
    real effect, not just confirm an absence of one."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    days = _trading_days(date(2026, 9, 28), 6)
    returns = [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]
    save_fixture(
        "INFY.NS",
        [Bar(date=d, open=100.0, close=100.0 * (1 + r)) for d, r in zip(days, returns)],
    )

    headlines = [
        make_headline(
            f"Infosys Share Price Highlights day {i}",
            str(i),
            datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).replace(hour=2),  # pre-open IST
        )
        for i, d in enumerate(days)
    ]
    # the score is pinned to the headline's *own* true session's return, by title
    compound_by_title = {h.title: r for h, r in zip(headlines, returns)}
    monkeypatch.setattr(
        correlate, "score_headline", lambda h: SimpleNamespace(compound=compound_by_title[h.title])
    )

    result = run_audit(headlines, live=False, event_threshold=0.3, n_iterations=200, seed=0)

    assert result.n_real_rows == 6
    assert result.contemporaneous is not None
    # perfectly constructed correlation under the real, leak-free alignment
    assert result.contemporaneous.observed == pytest.approx(1.0, abs=1e-9)
    # shuffling the timestamps reassigns each headline to a different
    # session's return while its score stays put, so the real value should
    # stand out sharply from the shuffled null
    assert result.contemporaneous.p_value < 0.05
    assert result.contemporaneous.null_mean < 0.9


def test_run_against_committed_fixture_reports_the_known_null_result(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, event_threshold=0.3, n_iterations=50, seed=0)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, live=False, event_threshold=0.3, n_iterations=50, seed=0)

    assert exit_code == 1
