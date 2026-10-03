import math
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import sentiment.correlate as correlate
from sentiment.audit import (
    contemporaneous_r,
    lagged_r,
    percentile_rank,
    permutation_pvalue,
    run_audit,
    shuffle_timestamps,
)
from sentiment.correlate import build_rows_from_headlines
from sentiment.headline import Headline
from sentiment.prices import Bar

IST = ZoneInfo("Asia/Kolkata")


def make_headline(title: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=f"http://example/{title}",
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


# ---------------------------------------------------------------------------
# shuffle_timestamps
# ---------------------------------------------------------------------------


def test_shuffle_timestamps_permutes_the_same_multiset_of_timestamps():
    import random

    headlines = [
        make_headline(f"h{i}", datetime(2026, 1, i + 1, 8, 0, tzinfo=IST)) for i in range(10)
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)


def test_shuffle_timestamps_is_deterministic_given_the_same_seed():
    import random

    headlines = [
        make_headline(f"h{i}", datetime(2026, 1, i + 1, 8, 0, tzinfo=IST)) for i in range(10)
    ]
    a = shuffle_timestamps(headlines, random.Random(42))
    b = shuffle_timestamps(headlines, random.Random(42))
    assert [h.published_at for h in a] == [h.published_at for h in b]


def test_shuffle_timestamps_keeps_title_source_link_attached_to_their_row():
    import random

    headlines = [
        make_headline(f"h{i}", datetime(2026, 1, i + 1, 8, 0, tzinfo=IST)) for i in range(5)
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(1))
    for original, s in zip(headlines, shuffled):
        assert s.title == original.title
        assert s.link == original.link
        assert s.source == original.source


# ---------------------------------------------------------------------------
# permutation_pvalue / percentile_rank
# ---------------------------------------------------------------------------


def test_permutation_pvalue_is_small_when_real_r_is_extreme_relative_to_null():
    null_rs = [0.01, -0.02, 0.03, -0.01, 0.02] * 20  # 100 near-zero null draws
    p = permutation_pvalue(0.95, null_rs)
    assert p == pytest.approx(1 / 101)


def test_permutation_pvalue_is_large_when_real_r_sits_inside_the_null_bulk():
    null_rs = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3] * 10
    p = permutation_pvalue(-0.15, null_rs)
    assert p > 0.5


def test_permutation_pvalue_is_none_without_a_real_r_or_without_any_null_draws():
    assert permutation_pvalue(None, [0.1, 0.2]) is None
    assert permutation_pvalue(0.5, []) is None


def test_percentile_rank_orders_low_to_high():
    null_rs = [0.0, 0.1, 0.2, 0.3, 0.4]
    assert percentile_rank(-1.0, null_rs) == 0
    assert percentile_rank(10.0, null_rs) == 100
    assert percentile_rank(0.2, null_rs) == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# contemporaneous_r / lagged_r helpers
# ---------------------------------------------------------------------------


def test_contemporaneous_r_needs_at_least_two_rows():
    assert contemporaneous_r([{"compound": 0.1, "contemporaneous_return": 0.01}]) is None


def test_lagged_r_only_uses_rows_with_a_computed_lagged_return():
    rows = [
        {"compound": 0.1, "lagged_return": 0.01},
        {"compound": 0.2, "lagged_return": 0.02},
        {"compound": 0.3, "lagged_return": None},
    ]
    assert lagged_r(rows) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Real fixture: confirms the refactor behind audit.py didn't change Day 5's
# numbers, and reproduces this project's actual, already-null finding via an
# independent method (permutation, not the Fisher-z CI Day 5 used).
# ---------------------------------------------------------------------------


def test_audit_on_the_real_fixture_matches_day5s_documented_null_result():
    in_path = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    from sentiment.headline import read_csv

    headlines = read_csv(in_path)
    result = run_audit(headlines, n_perm=200, seed=0, live=False)

    # README Day 5 Findings: contemporaneous r=-0.185 n=23, lagged r=+0.000 n=15.
    assert result.real_contemp_r == pytest.approx(-0.185, abs=0.001)
    assert result.real_contemp_n == 23
    assert result.real_lagged_r == pytest.approx(0.0, abs=0.001)
    assert result.real_lagged_n == 15

    # Neither real correlation is remarkable next to 200 random timestamp
    # shuffles of the same fixture - consistent with "no signal here", not
    # "a signal that a shuffle destroyed" (see module docstring/README).
    assert permutation_pvalue(result.real_contemp_r, result.null_contemp_rs) > 0.05
    assert permutation_pvalue(result.real_lagged_r, result.null_lagged_rs) > 0.05


# ---------------------------------------------------------------------------
# Synthetic planted signal: proves the audit mechanism actually detects (and
# a shuffle actually destroys) a real sentiment/session-return relationship,
# since the real fixture above has none to demonstrate that on. Without
# this, "no signal survived shuffling" on real data would be indistinguishable
# from "this audit can't catch anything."
# ---------------------------------------------------------------------------


def test_permutation_audit_flags_a_planted_signal_and_shuffling_destroys_it(monkeypatch):
    n = 20
    # Pre-open IST timestamps on consecutive weekdays: align_headline keeps
    # each one on its own calendar day, no roll-forward ambiguity to reason
    # about separately.
    day = date(2026, 1, 5)  # a Monday
    published_ats = []
    while len(published_ats) < n:
        if day.weekday() < 5:
            published_ats.append(datetime.combine(day, datetime.min.time(), tzinfo=IST).replace(hour=8))
        day += timedelta(days=1)

    titles = [f"synthetic-{i}" for i in range(n)]
    headlines = [make_headline(titles[i], published_ats[i]) for i in range(n)]

    # Each headline is "about" its own, otherwise-unused ticker - a trivial
    # 1:1 resolve() so no real tickers.py rule needs to match these titles.
    tickers = {titles[i]: f"SYN{i}.NS" for i in range(n)}
    monkeypatch.setattr(correlate, "resolve", lambda title: (title, tickers[title]))

    # compound is a deterministic function of the title only - realistic,
    # since a real VADER/FinBERT score depends on content, not timing.
    compounds = {titles[i]: (i - (n - 1) / 2) * 0.1 for i in range(n)}
    monkeypatch.setattr(correlate, "score_headline", lambda h: SimpleNamespace(compound=compounds[h.title]))

    # The session_date each headline aligns to under its ORIGINAL timestamp.
    from sentiment.market_hours import align_headline

    true_session_date = {titles[i]: align_headline(published_ats[i]).session_date for i in range(n)}
    all_session_dates = sorted(true_session_date.values())

    # Each ticker has a bar on every session date in play: the correct
    # (unshuffled) date carries a return that exactly equals that headline's
    # compound (a planted r=1.0 relationship); every other date carries a
    # fixed, content-independent "noise" return. A shuffle that reassigns
    # headline i to a different session date therefore picks up unrelated
    # noise instead of the planted match.
    noise_by_date = {d: math.sin(idx) * 0.01 for idx, d in enumerate(all_session_dates)}
    bars_by_ticker = {}
    for i in range(n):
        ticker = tickers[titles[i]]
        bars = []
        for d in all_session_dates:
            ret = compounds[titles[i]] if d == true_session_date[titles[i]] else noise_by_date[d]
            bars.append(Bar(date=d, open=1.0, close=1.0 + ret))
        bars_by_ticker[ticker] = bars
    monkeypatch.setattr(correlate, "load_bars", lambda ticker, live=False: bars_by_ticker[ticker])

    real_rows, _ = build_rows_from_headlines(headlines, live=False)
    assert len(real_rows) == n
    real_r = contemporaneous_r(real_rows)
    assert real_r == pytest.approx(1.0, abs=1e-6)

    result = run_audit(headlines, n_perm=200, seed=0, live=False)
    assert result.real_contemp_r == pytest.approx(1.0, abs=1e-6)

    # The planted signal only exists under the TRUE timestamp pairing - a
    # random shuffle should essentially never reproduce it.
    assert max(abs(r) for r in result.null_contemp_rs) < 0.99
    assert permutation_pvalue(result.real_contemp_r, result.null_contemp_rs) <= 0.05
