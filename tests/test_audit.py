import math
import random
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import sentiment.audit as audit
from sentiment.audit import contemporaneous_r, run, run_audit, shuffle_timestamps
from sentiment.headline import Headline
from sentiment.prices import Bar


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


def test_shuffle_timestamps_permutes_without_losing_or_duplicating_any():
    rng = random.Random(0)
    headlines = [
        make_headline(f"H{i}", str(i), datetime(2026, 9, 7 + i, 2, 0, tzinfo=timezone.utc)) for i in range(6)
    ]

    shuffled = shuffle_timestamps(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]  # titles never move
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # with 6 distinct timestamps a fixed seed should actually move at least one
    assert any(a.published_at != b.published_at for a, b in zip(headlines, shuffled))


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_leak_free_correlation_collapses_under_a_fixed_derangement(monkeypatch):
    """Ground truth: compound truly predicts contemporaneous return when each
    headline is paired with its own, correctly aligned session date. A fixed
    derangement of the timestamps (every headline takes the next headline's
    timestamp, cyclically - zero fixed points) must collapse that
    correlation, because each ticker's price-moving bar only sits on its own
    original date.

    This is the test that gives the audit's "signal must disappear" claim
    teeth: the real fixture's own correlation is already null (see
    module docstring and the README's Day 5-7 Findings), so a permutation
    test run only against real data could pass for the wrong reason - there
    being nothing to leak in the first place. This constructs a case where
    there *is* a real, planted relationship and shows shuffling destroys it.
    """
    n = 12
    dates = _weekdays(date(2026, 9, 7), n)  # a Monday, n weekdays forward
    compounds = [-1.0 + 2.0 * i / (n - 1) for i in range(n)]  # evenly spaced -1..+1
    tickers = [f"SYN{i}" for i in range(n)]

    headlines = [
        # 07:30 IST (02:00 UTC) - pre-open on a weekday, so align_headline
        # assigns session_date == the headline's own calendar day.
        make_headline(f"synthetic headline {i}", str(i), datetime(day.year, day.month, day.day, 2, 0, tzinfo=timezone.utc))
        for i, day in enumerate(dates)
    ]

    # Each ticker's bar history spans every one of the n dates: a real,
    # compound-sized move on its OWN date, flat (zero return) everywhere
    # else - the price only moves on the day the headline actually predicted it.
    bars_by_ticker: dict[str, list[Bar]] = {}
    for i, ticker in enumerate(tickers):
        bars_by_ticker[ticker] = [
            Bar(date=day, open=1.0, close=1.0 + (0.05 * compounds[i] if j == i else 0.0))
            for j, day in enumerate(dates)
        ]

    title_to_ticker = {h.title: ("Synthetic Co", t) for h, t in zip(headlines, tickers)}
    title_to_compound = {h.title: c for h, c in zip(headlines, compounds)}

    class FakeScored:
        def __init__(self, compound: float) -> None:
            self.compound = compound

    monkeypatch.setattr(audit, "resolve", lambda title: title_to_ticker.get(title))
    monkeypatch.setattr(audit, "score_headline", lambda h: FakeScored(title_to_compound[h.title]))

    real_r, real_n = contemporaneous_r(headlines, bars_by_ticker)
    assert real_n == n
    assert real_r is not None and real_r > 0.999  # planted to be ~perfectly linear

    derangement = [replace(h, published_at=headlines[(i + 1) % n].published_at) for i, h in enumerate(headlines)]
    deranged_r, deranged_n = contemporaneous_r(derangement, bars_by_ticker)

    assert deranged_n == n  # every ticker still finds a bar, just the wrong (flat) one
    assert deranged_r == 0.0  # every paired return is now the flat 0.0 - no variance left to correlate


def test_run_audit_against_the_real_committed_fixture_matches_day5s_r():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    result = run_audit(fixture, n_shuffles=50, seed=0)

    # Day 5's README Findings: contemporaneous r=-0.185, n=23.
    assert result.real_n == 23
    assert result.real_r is not None and math.isclose(result.real_r, -0.185, abs_tol=0.001)
    assert len(result.shuffled_rs) > 0
    assert result.p_value is not None and 0.0 <= result.p_value <= 1.0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_shuffles=10, seed=0, live=False)

    assert exit_code == 1


def test_run_against_the_real_committed_fixture_passes(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, n_shuffles=50, seed=0, live=False)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "PASS" in out
