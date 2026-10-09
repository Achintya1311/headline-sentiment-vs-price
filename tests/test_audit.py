from __future__ import annotations

import datetime as dt
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from random import Random
from types import SimpleNamespace

import sentiment.correlate as correlate
import sentiment.prices as prices
from sentiment.audit import run, run_audit, shuffle_timestamps
from sentiment.headline import Headline
from sentiment.market_hours import is_trading_day
from sentiment.prices import Bar, save_fixture
from sentiment.stats import pearson_r

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_timestamps_keeps_the_same_multiset_of_timestamps():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 21 + i, 2, 0, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]

    shuffled = shuffle_timestamps(headlines, Random(0))

    assert len(shuffled) == len(headlines)
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links are untouched - only which timestamp each is paired with moves
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # with 6 distinct timestamps the identity permutation is astronomically
    # unlikely; this shuffle actually changed at least one pairing
    assert any(a.published_at != b.published_at for a, b in zip(shuffled, headlines))


def test_shuffle_timestamps_is_deterministic_given_the_same_seed():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 21 + i, 2, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]

    a = shuffle_timestamps(headlines, Random(42))
    b = shuffle_timestamps(headlines, Random(42))

    assert [h.published_at for h in a] == [h.published_at for h in b]


def test_real_fixture_shuffled_timestamp_audit_finds_no_signal(tmp_path: Path):
    # Day 5/6 already found this fixture's contemporaneous correlation
    # statistically indistinguishable from zero. The audit should agree:
    # the real pairing's |r| should look unremarkable against the null
    # distribution a random headline-to-session pairing produces.
    out_path = tmp_path / "audit_shuffle.csv"

    exit_code = run(FIXTURE, out_path, live=False, n_shuffles=200, seed=0)

    assert exit_code == 0
    assert out_path.exists()

    from sentiment.headline import read_csv

    headlines = read_csv(FIXTURE)
    result = run_audit(headlines, live=False, n_shuffles=200, seed=0)

    assert result.real_n == 23
    assert len(result.shuffled_rs) > 150  # most shuffles still resolve >= 2 priced headlines
    assert result.p_value > 0.05  # no detectable signal, consistent with README Findings


def _trading_days(start: dt.date, n: int) -> list[dt.date]:
    days: list[dt.date] = []
    d = start
    while len(days) < n:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def test_shuffle_destroys_a_real_injected_leak(tmp_path, monkeypatch):
    """Prove the audit has teeth: build a synthetic fixture where sentiment
    score and contemporaneous return are *deliberately* wired together
    through the real session_date each headline's own true timestamp
    produces, confirm the unshuffled pipeline finds that signal (r ~= 1),
    then confirm shuffling the timestamps - which reassigns each headline to
    a different session's return without touching its score - collapses it.

    This is the case the real committed fixture can never exercise (its own
    correlation is already null - see README Findings), so it has to be
    built rather than observed, the same way earlier days built synthetic
    cases to prove a mechanism works when the real data can't reach the
    interesting branch (e.g. Day 6's synthetic-signal regression test).
    """
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    n = 20
    days = _trading_days(dt.date(2026, 9, 7), n)  # 2026-09-07 is a Monday
    compounds = [-0.95 + 0.1 * i for i in range(n)]
    tickers = [f"SYN{i}.NS" for i in range(n)]
    k = 0.05
    noise = [Random(999).uniform(-0.01, 0.01) for _ in range(n)]

    headlines = []
    title_to_ticker: dict[str, str] = {}
    title_to_compound: dict[str, float] = {}
    for i in range(n):
        title = f"Synthetic Co {i} headline"
        title_to_ticker[title] = tickers[i]
        title_to_compound[title] = compounds[i]
        published_at = datetime.combine(days[i], time(2, 0), tzinfo=timezone.utc)  # pre-open IST
        headlines.append(make_headline(title, str(i), published_at))

        bars = []
        for j in range(n):
            ret = k * compounds[i] if j == i else noise[j]
            bars.append(Bar(date=days[j], open=100.0, close=100.0 * (1 + ret)))
        save_fixture(tickers[i], bars)

    def fake_resolve(title: str):
        return "Synthetic Co", title_to_ticker[title]

    def fake_score_headline(headline: Headline):
        return SimpleNamespace(compound=title_to_compound[headline.title])

    monkeypatch.setattr(correlate, "resolve", fake_resolve)
    monkeypatch.setattr(correlate, "score_headline", fake_score_headline)

    real_rows, _ = correlate.build_rows_from_headlines(headlines, live=False)
    assert len(real_rows) == n
    real_r = pearson_r([r["compound"] for r in real_rows], [r["contemporaneous_return"] for r in real_rows])
    assert real_r > 0.999  # the injected dependency is exactly linear and noiseless

    result = run_audit(headlines, live=False, n_shuffles=300, seed=0)

    assert result.real_r > 0.999
    mean_abs_shuffled = sum(abs(r) for r in result.shuffled_rs) / len(result.shuffled_rs)
    assert mean_abs_shuffled < 0.4  # the shuffled null is nowhere near the real signal
    assert result.p_value < 0.05  # the gate correctly flags this as a real, non-noise signal
