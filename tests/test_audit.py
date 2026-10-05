"""Day 8: the ml-pipeline-audit pass.

Two different things need testing here, and they are easy to conflate:

1. That ``shuffle_timestamps`` actually does what it claims (a genuine
   permutation, not a no-op or a resample).
2. That the permutation test built on top of it can tell a real signal from
   noise at all - which the real fixture alone cannot prove, because Day
   5/6 already found the real fixture has no signal to begin with. A test
   that only ever sees "shuffled looks like real" could be hiding a bug
   that makes the shuffle a no-op just as easily as it could be confirming
   a clean pipeline. So a synthetic fixture with a real, injected
   sentiment/return relationship is built below specifically to prove the
   audit can detect a signal collapsing when one genuinely exists.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import sentiment.prices as prices
from sentiment.audit import correlation_for, run_audit, shuffle_timestamps
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture
from sentiment.vader_score import score_text

IST = ZoneInfo("Asia/Kolkata")

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


def test_shuffle_timestamps_is_a_genuine_permutation():
    headlines = [
        make_headline(f"Infosys Share Price Highlights: variant {i}", str(i), datetime(2026, 9, 1 + i, 2, 0, tzinfo=IST))
        for i in range(10)
    ]
    rng = __import__("random").Random(42)
    shuffled = shuffle_timestamps(headlines, rng)

    # same multiset of timestamps, same headline content, in the same order...
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    # ...but not the identity assignment - some headline got a different time.
    assert any(a.published_at != b.published_at for a, b in zip(headlines, shuffled))


def test_correlation_for_matches_day5_committed_fixture_result():
    from sentiment.headline import read_csv

    headlines = read_csv(FIXTURE)
    r, n = correlation_for(headlines, live=False)

    # Day 5's README Findings: contemporaneous r=-0.185, n=23.
    assert n == 23
    assert abs(r - (-0.185)) < 0.001


def test_run_audit_against_committed_fixture_is_deterministic_for_a_seed():
    from sentiment.headline import read_csv

    headlines = read_csv(FIXTURE)
    result_a = run_audit(headlines, live=False, n_shuffles=40, seed=7)
    result_b = run_audit(headlines, live=False, n_shuffles=40, seed=7)

    assert result_a.real_r == result_b.real_r
    assert result_a.shuffle_rs == result_b.shuffle_rs
    assert result_a.p_value == result_b.p_value


# --- synthetic fixture: a real, injected sentiment/return relationship -----
#
# Twelve distinct single-company "Share Price Highlights" headlines (one per
# ticker, matching sentiment/tickers.py's real patterns), each published
# pre-open on its own distinct trading day. Each ticker's price fixture has
# a bar for every one of the twelve days: a non-zero return on *its own*
# headline's true day (set to a fixed multiple of that headline's real VADER
# compound score, so the real/unshuffled relationship is exactly linear) and
# a flat zero return on every other day. A shuffle reassigns which day's bar
# a ticker's return is read from - on its own day the true relationship
# still holds, on any other day the return is a flat zero, so a permutation
# should, on average, erase most of the real correlation.

_SYNTHETIC_COMPANIES = [
    ("SBI Life", "SBILIFE.NS", "soars as blockbuster profit beats guidance by a wide margin, upgrade to strong buy"),
    ("Nestle India", "NESTLEIND.NS", "jumps as strong results beat estimates, analysts raise target price"),
    ("Sun Pharma", "SUNPHARMA.NS", "rises as results beat expectations, outlook raised"),
    ("Grasim Inds", "GRASIM.NS", "edges higher on a steady quarter, outlook unchanged"),
    ("Tech Mahindra", "TECHM.NS", "ticks up slightly as margins hold steady"),
    ("Wipro", "WIPRO.NS", "trades flat as results meet expectations"),
    ("Bharti Airtel", "BHARTIARTL.NS", "slips slightly as margins come in soft"),
    ("Tata Steel", "TATASTEEL.NS", "dips as demand softens, outlook trimmed"),
    ("HUL", "HINDUNILVR.NS", "falls as results miss expectations, outlook cut"),
    ("Infosys", "INFY.NS", "drops as profit warning spooks investors, downgrade"),
    ("HCL Tech", "HCLTECH.NS", "slides as weak guidance disappoints investors, downgrade to sell"),
    ("HDFC Life", "HDFCLIFE.NS", "plunges as fraud probe deepens, outlook turns bearish, downgrade to sell"),
]

RETURN_SCALE = 0.3


def _weekdays_from(start: date, count: int) -> list[date]:
    days: list[date] = []
    d = start
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _build_synthetic_signal_fixture(tmp_path: Path, monkeypatch) -> list[Headline]:
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    session_days = _weekdays_from(date(2026, 9, 1), len(_SYNTHETIC_COMPANIES))
    headlines: list[Headline] = []
    compounds: list[float] = []

    for (company, ticker, suffix), day in zip(_SYNTHETIC_COMPANIES, session_days):
        title = f"{company} Share Price Highlights: {company} {suffix}"
        compound = score_text(title)["compound"]
        compounds.append(compound)
        published_at = datetime(day.year, day.month, day.day, 8, 0, tzinfo=IST)
        headlines.append(make_headline(title, f"{ticker}-link", published_at))

    # every ticker needs a bar on every session day, so a shuffle always
    # finds a bar (just not the signal-bearing one) rather than dropping the
    # row entirely - that would test "shuffle breaks ticker resolution",
    # not "shuffle breaks the sentiment/return link".
    for (_, ticker, _), own_day, own_compound in zip(_SYNTHETIC_COMPANIES, session_days, compounds):
        bars = []
        for day in session_days:
            if day == own_day:
                bars.append(Bar(date=day, open=100.0, close=100.0 * (1 + RETURN_SCALE * own_compound)))
            else:
                bars.append(Bar(date=day, open=100.0, close=100.0))
        save_fixture(ticker, sorted(bars, key=lambda b: b.date))

    assert len({round(c, 6) for c in compounds}) > 1, "need varying compound scores for a real correlation"
    return headlines


def test_real_relationship_is_near_perfect_before_shuffling(tmp_path: Path, monkeypatch):
    headlines = _build_synthetic_signal_fixture(tmp_path, monkeypatch)
    r, n = correlation_for(headlines, live=False)

    assert n == len(_SYNTHETIC_COMPANIES)
    assert abs(r) > 0.95, f"expected a near-perfect injected correlation, got r={r}"


def test_audit_detects_the_injected_signal_collapsing_under_shuffling(tmp_path: Path, monkeypatch):
    headlines = _build_synthetic_signal_fixture(tmp_path, monkeypatch)
    result = run_audit(headlines, live=False, n_shuffles=200, seed=0)

    assert abs(result.real_r) > 0.95
    assert len(result.shuffle_rs) == 200  # every ticker has a bar for every day - nothing drops

    mean_abs_shuffled = sum(abs(r) for r in result.shuffle_rs) / len(result.shuffle_rs)
    # The real statistic massively outweighs what random timestamp
    # reassignment produces - this is what "the pipeline would catch a real
    # leak" looks like, in contrast to the committed fixture's p=0.1-ish
    # result where there was never a signal to catch.
    assert mean_abs_shuffled < 0.5 * abs(result.real_r)
    assert result.p_value < 0.05
