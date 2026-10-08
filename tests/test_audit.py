import random
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import AuditResult, run, run_audit, shuffle_published_at
from sentiment.correlate import DEFAULT_IN
from sentiment.headline import Headline, read_csv
from sentiment.prices import Bar, save_fixture
from sentiment.vader_score import score_headline


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_published_at_permutes_timestamps_but_keeps_titles_in_place():
    headlines = [
        make_headline("A", "1", datetime(2026, 9, 21, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("B", "2", datetime(2026, 9, 22, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("C", "3", datetime(2026, 9, 23, 2, 0, 0, tzinfo=timezone.utc)),
    ]

    shuffled = shuffle_published_at(headlines, random.Random(1))

    assert [h.title for h in shuffled] == ["A", "B", "C"]
    assert [h.link for h in shuffled] == ["1", "2", "3"]
    # same multiset of timestamps, just redistributed
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)


def test_shuffle_published_at_is_deterministic_for_a_given_seed():
    headlines = [
        make_headline(str(i), str(i), datetime(2026, 9, 21 + i, 2, 0, 0, tzinfo=timezone.utc)) for i in range(5)
    ]

    first = shuffle_published_at(headlines, random.Random(42))
    second = shuffle_published_at(headlines, random.Random(42))

    assert [h.published_at for h in first] == [h.published_at for h in second]


def test_audit_result_p_value_is_the_fraction_of_shuffled_trials_at_least_as_extreme():
    result = AuditResult(real_r=0.5, real_n=10, shuffled_rs=[0.0, 0.1, 0.6, -0.7, 0.4])
    # |shuffled| >= |real|=0.5: 0.6 and -0.7 -> 2 of 5
    assert result.p_value == 2 / 5
    assert result.trials == 5


def test_run_audit_not_enough_resolved_headlines_returns_none():
    headlines = [make_headline("nothing this table recognises", "1", datetime(2026, 9, 21, 2, 0, 0, tzinfo=timezone.utc))]

    result = run_audit(headlines, live=False, trials=10, seed=0)

    assert result is None


def test_run_audit_detects_a_genuine_timestamp_dependent_signal(tmp_path: Path, monkeypatch):
    """Prove the audit can tell a real, alignment-dependent relationship apart
    from timestamp noise - otherwise a PASS from this module would mean
    nothing (the same reason Day 6's regression tests included a synthetic
    signal, not just the real fixture's null result).

    Five headlines, five tickers, five distinct trading days. Each ticker's
    *own* day has a return deliberately set to a linear function of that
    headline's own VADER compound; every other day for that ticker returns
    exactly 0. Shuffling which day each headline is timed to breaks that one
    correct pairing almost every time, so the correlation should collapse
    while the real, correctly-timed pipeline's correlation stays large.
    """
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    titles_and_tickers = [
        ("Fortis Healthcare shares surge on blockbuster profit beat and stellar outlook", "FORTIS.NS"),
        ("Great Eastern Shipping shares plunge amid fraud allegations and investor panic", "GESHIP.NS"),
        ("PC Jeweller shares soar after outstanding earnings surprise and robust growth", "PCJEWELLER.NS"),
        ("Suzlon Energy shares crash following disastrous losses and harsh downgrade", "SUZLON.NS"),
        ("BSE shares rally on excellent quarterly results and encouraging guidance", "BSE.NS"),
    ]
    days = [21, 22, 23, 24, 25]  # Mon-Fri, 2026-09
    scale = 0.1

    compounds = []
    for title, _ in titles_and_tickers:
        placeholder = make_headline(title, "x", datetime(2026, 9, 21, 2, 0, 0, tzinfo=timezone.utc))
        compounds.append(score_headline(placeholder).compound)

    assert len(set(compounds)) > 1, "fixture titles must produce varied compound scores"

    headlines = []
    for (title, ticker), day, compound in zip(titles_and_tickers, days, compounds):
        headlines.append(make_headline(title, ticker, datetime(2026, 9, day, 2, 0, 0, tzinfo=timezone.utc)))
        bars = []
        for other_day in days:
            if other_day == day:
                bars.append(Bar(date=__import__("datetime").date(2026, 9, day), open=100.0, close=100.0 * (1 + scale * compound)))
            else:
                bars.append(Bar(date=__import__("datetime").date(2026, 9, other_day), open=100.0, close=100.0))
        save_fixture(ticker, bars)

    result = run_audit(headlines, live=False, trials=300, seed=0)

    assert result is not None
    assert abs(result.real_r) > 0.95
    assert result.p_value < 0.05


def test_run_audit_against_the_real_fixture_is_not_distinguishable_from_noise():
    """The real pipeline's Day 5/6/7 finding is a null result (r=-0.185,
    dominated by headlines saturated at one VADER score) - the audit should
    confirm that this real correlation is an unremarkable member of the
    shuffled-timestamp null distribution, not an outlier. A small ``trials``
    keeps this fast; see README Day 8 Findings for a larger run's numbers."""
    headlines = read_csv(DEFAULT_IN)

    result = run_audit(headlines, live=False, trials=100, seed=0)

    assert result is not None
    assert result.real_n == 23
    assert result.p_value >= 0.05


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, live=False, trials=10, seed=0)

    assert exit_code == 1


def test_run_against_committed_fixture_passes(capsys):
    exit_code = run(DEFAULT_IN, live=False, trials=100, seed=0)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PASS" in out
