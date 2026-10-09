"""Day 8: the shuffled-timestamp leakage control.

Two synthetic scenarios prove the mechanism actually works, not just that
it reports "no signal" on this repo's own already-null fixture (which it
would do even if the permutation logic were broken):

- A **ticker-level confound** (a ticker's return is the same sign on every
  session in its fixture, so which specific day a headline lands on after
  shuffling doesn't matter) must be flagged as a leak: the real correlation
  is strong *and* survives shuffling.
- A **genuinely day-specific effect** (a ticker's return is nonzero on only
  one session, and the headline is correctly aligned to exactly that
  session) must pass: shuffling almost always reassigns the headline to a
  session with no real relationship, and the correlation collapses.
"""

from datetime import date, datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import (
    audit_correlation,
    audit_event_diff,
    run,
    shuffle_timestamps,
)
from sentiment.headline import Headline, write_csv
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


def pre_open(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 2, 0, 0, tzinfo=timezone.utc)  # 07:30 IST


POS_SUFFIX = "stock surges after blowout results, beats estimates, raises outlook to excellent"
NEG_SUFFIX = "stock plunges after disappointing results, misses estimates, outlook turns terrible"


def test_shuffle_timestamps_permutes_but_preserves_the_set():
    headlines = [
        make_headline(f"Infosys Share Price Highlights {i}", str(i), pre_open(date(2026, 1, 5 + i)))
        for i in range(5)
    ]
    import random

    shuffled = shuffle_timestamps(headlines, random.Random(1))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def _ticker_confound_headlines(tmp_path: Path) -> list[Headline]:
    """A ticker-level confound: INFY.NS's open-to-close return is +5% on
    every one of 10 fixture sessions, WIPRO.NS's is -5% on every one. Five
    strongly-positive headlines are pinned to INFY.NS on five different
    days, five strongly-negative headlines to WIPRO.NS on the other five -
    the correlation this produces has nothing to do with *which* day a
    headline lands on, only which ticker it is about, so it should survive
    timestamp shuffling undiminished.
    """
    days = [date(2026, 1, d) for d in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16)]
    save_fixture("INFY.NS", [Bar(date=d, open=100.0, close=105.0) for d in days])
    save_fixture("WIPRO.NS", [Bar(date=d, open=100.0, close=95.0) for d in days])

    headlines = []
    for i, d in enumerate(days[:5]):
        title = f"Infosys Share Price Highlights: {POS_SUFFIX}"
        headlines.append(make_headline(title, f"infy-{i}", pre_open(d)))
    for i, d in enumerate(days[5:]):
        title = f"Wipro Share Price Highlights: {NEG_SUFFIX}"
        headlines.append(make_headline(title, f"wipro-{i}", pre_open(d)))
    return headlines


def _day_specific_signal_headlines(tmp_path: Path) -> list[Headline]:
    """A genuinely day-specific effect: each of 5 tickers has a nonzero
    open-to-close return on exactly one of 5 fixture sessions (every other
    session is flat), and each headline is correctly aligned to precisely
    that session, with sentiment matching its sign. Shuffle any headline's
    timestamp onto a different session and that ticker's return there is
    0.0 - the relationship is destroyed unless the permutation happens to
    leave that headline's day fixed.
    """
    days = [date(2026, 1, d) for d in (5, 6, 7, 8, 9)]
    tickers = ["INFY.NS", "WIPRO.NS", "TATASTEEL.NS", "HCLTECH.NS", "HDFCBANK.NS"]
    signs = [+1, +1, -1, -1, +1]
    prefixes = {
        "INFY.NS": "Infosys Share Price Highlights:",
        "WIPRO.NS": "Wipro Share Price Highlights:",
        "TATASTEEL.NS": "Tata Steel Share Price Highlights:",
        "HCLTECH.NS": "HCL Tech Share Price Highlights:",
        "HDFCBANK.NS": "HDFC Bank",
    }

    headlines = []
    for k, (ticker, sign) in enumerate(zip(tickers, signs)):
        bars = [
            Bar(date=d, open=100.0, close=100.0 * (1 + (sign * 0.05 if d_idx == k else 0.0)))
            for d_idx, d in enumerate(days)
        ]
        save_fixture(ticker, bars)
        suffix = POS_SUFFIX if sign > 0 else NEG_SUFFIX
        title = f"{prefixes[ticker]} {suffix}"
        headlines.append(make_headline(title, f"{ticker}-{k}", pre_open(days[k])))
    return headlines


def test_audit_correlation_fails_when_a_ticker_level_confound_survives_shuffling(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = _ticker_confound_headlines(tmp_path)

    outcome = audit_correlation(headlines, live=False, n_perm=200, seed=0)

    assert outcome.notable is True
    assert abs(outcome.real_stat) > 0.9  # near-perfect: only 2 distinct (x, y) pairs
    assert outcome.passed is False
    assert outcome.p_value > 0.5  # the shuffled null reproduces it almost every time


def _ticker_confound_headlines_for_event_diff(tmp_path: Path) -> list[Headline]:
    """Same ticker-level-confound idea as ``_ticker_confound_headlines``,
    but split by |compound| *magnitude* (what ``event_diff`` actually
    buckets on) rather than sign: INFY.NS headlines score ~0.71 (above the
    0.3 threshold - "high"), SUZLON.NS headlines score ~0.13 (below it -
    "low"). INFY.NS returns +5% and SUZLON.NS returns -5% on every one of
    10 fixture sessions, so the high/low group-mean difference this
    produces has nothing to do with which day a headline lands on."""
    days = [date(2026, 1, d) for d in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16)]
    save_fixture("INFY.NS", [Bar(date=d, open=100.0, close=105.0) for d in days])
    save_fixture("SUZLON.NS", [Bar(date=d, open=100.0, close=95.0) for d in days])

    headlines = []
    for i, d in enumerate(days[:5]):
        title = f"Infosys Share Price Highlights: {POS_SUFFIX}"
        headlines.append(make_headline(title, f"infy-{i}", pre_open(d)))
    for i, d in enumerate(days[5:]):
        title = "Suzlon Energy shares steady amid broader market weakness"
        headlines.append(make_headline(title, f"suzlon-{i}", pre_open(d)))
    return headlines


def test_audit_event_diff_fails_when_a_ticker_level_confound_survives_shuffling(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = _ticker_confound_headlines_for_event_diff(tmp_path)

    outcome = audit_event_diff(headlines, threshold=0.3, live=False, n_perm=200, seed=0)

    assert outcome.notable is True
    assert outcome.real_stat > 0.05  # +5% (INFY, high) vs -5% (SUZLON, low) group means
    assert outcome.passed is False
    assert outcome.p_value == 1.0  # every permutation reproduces the exact same group means


def test_audit_correlation_passes_when_the_signal_is_genuinely_day_specific(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = _day_specific_signal_headlines(tmp_path)

    outcome = audit_correlation(headlines, live=False, n_perm=500, seed=0)

    assert outcome.notable is True
    assert abs(outcome.real_stat) > 0.9  # correctly aligned: near-perfect
    assert outcome.passed is True  # shuffling collapses it often enough to not be a fluke
    assert outcome.p_value <= 0.05


def test_audit_correlation_passes_vacuously_on_too_few_resolved_headlines(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=date(2026, 1, 5), open=100.0, close=105.0)])
    headlines = [make_headline("Infosys Share Price Highlights: ok", "1", pre_open(date(2026, 1, 5)))]

    outcome = audit_correlation(headlines, live=False, n_perm=50, seed=0)

    assert outcome.notable is False
    assert outcome.passed is True
    assert outcome.n_null == 0


def test_audit_event_diff_passes_vacuously_when_one_group_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = _day_specific_signal_headlines(tmp_path)

    # threshold above every compound in this fixture -> "high" group empty
    outcome = audit_event_diff(headlines, threshold=0.99, live=False, n_perm=50, seed=0)

    assert outcome.notable is False
    assert outcome.passed is True


def test_audit_against_the_real_committed_fixture_finds_no_notable_signal():
    """Day 5/6/7 already found no significant correlation or event-study
    effect on the real 23-ticker fixture (every reported CI includes
    zero) - this is the honest result the shuffle control has nothing to
    disprove, exercised end to end via sentiment.headline.read_csv rather
    than a synthetic one."""
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    from sentiment.headline import read_csv

    headlines = read_csv(fixture)

    corr = audit_correlation(headlines, live=False, n_perm=50, seed=0)
    event = audit_event_diff(headlines, live=False, n_perm=50, seed=0)

    assert corr.notable is False
    assert corr.passed is True
    assert round(corr.real_stat, 3) == -0.185  # matches Day 5's README-reported r

    assert event.notable is False
    assert event.passed is True
    assert round(event.real_stat, 4) == -0.0095  # matches Day 5's README-reported diff


def test_run_exits_zero_against_the_real_committed_fixture():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    assert run(fixture, live=False, event_threshold=0.3, n_perm=50, seed=0) == 0


def test_run_exits_one_when_a_leak_is_present(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = _ticker_confound_headlines(tmp_path)
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    assert run(in_path, live=False, event_threshold=0.0, n_perm=200, seed=0) == 1


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    assert run(missing, live=False, event_threshold=0.3, n_perm=50, seed=0) == 1
