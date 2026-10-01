from datetime import datetime
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    PermutationResult,
    leak_free_session_date,
    naive_same_day_session_date,
    permutation_test,
    run,
    shuffle_timestamps,
)
from sentiment.headline import Headline, write_csv
from sentiment.market_hours import IST
from sentiment.prices import Bar, save_fixture


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


def test_shuffle_timestamps_preserves_the_multiset_of_times_and_reorders_them():
    headlines = [
        make_headline(f"Fortis Healthcare shares headline {i}", str(i), datetime(2026, 9, 7 + i, 12, 0, tzinfo=IST))
        for i in range(6)
    ]
    import random

    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.title for h in shuffled] == [h.title for h in headlines]  # titles never move
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]  # times do


def test_shuffle_timestamps_is_deterministic_given_a_seed():
    import random

    headlines = [
        make_headline(f"Fortis Healthcare shares headline {i}", str(i), datetime(2026, 9, 7 + i, 12, 0, tzinfo=IST))
        for i in range(8)
    ]
    a = shuffle_timestamps(headlines, random.Random(42))
    b = shuffle_timestamps(headlines, random.Random(42))
    assert [h.published_at for h in a] == [h.published_at for h in b]


def test_permutation_test_raises_when_fewer_than_two_resolved_headlines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    headlines = [make_headline("nothing this table resolves", "1", datetime(2026, 9, 7, 2, 0, tzinfo=IST))]

    with pytest.raises(ValueError, match="fewer than 2 resolved headlines"):
        permutation_test(headlines, leak_free_session_date, n_perm=10, seed=0)


# --- synthetic leak: proves the control actually has power (see README Day 8 Findings) ---
#
# 8 headlines, each published INTRADAY on its own trading day, with VADER
# sentiment sign deliberately matching that SAME day's rigged open-to-close
# return (positive text -> +5% that day, negative text -> -5%). Day 4's
# leak-free alignment rolls every intraday headline to the *next* trading
# day instead, which is a separate "buffer" day rigged flat (0% return) for
# every headline - no sentiment-driven signal reaches the leak-free pairing
# at all. The naive same-calendar-day alignment (the pre-Day-4 bug) reads
# the rigged same-day return directly, so it should show a strong, real
# correlation that a timestamp shuffle collapses - that's the behaviour the
# audit CLI exists to catch.
_POSITIVE_TEXTS = [
    "Fortis Healthcare shares soar on blockbuster earnings beat",
    "Fortis Healthcare shares rally as profit surges past estimates",
    "Fortis Healthcare shares jump on positive outlook upgrade",
    "Fortis Healthcare shares climb after strong guidance raise",
]
_NEGATIVE_TEXTS = [
    "Fortis Healthcare shares plunge on disappointing earnings miss",
    "Fortis Healthcare shares sink on dismal profit collapse",
    "Fortis Healthcare shares slump on weak outlook downgrade",
    "Fortis Healthcare shares fall sharply on terrible quarterly loss",
]


def _trading_days(start, count):
    import datetime as dt

    from sentiment.market_hours import is_trading_day

    days = []
    cur = start
    while len(days) < count:
        if is_trading_day(cur):
            days.append(cur)
        cur += dt.timedelta(days=1)
    return days


def _build_synthetic_leak(tmp_path: Path, monkeypatch):
    import datetime as dt

    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    days = _trading_days(dt.date(2026, 9, 7), 16)
    own_days = days[0::2]  # 8 days each headline is published on
    buffer_days = days[1::2]  # the very next trading day each one rolls to

    texts = _POSITIVE_TEXTS + _NEGATIVE_TEXTS
    headlines = [
        make_headline(text, str(i), datetime(d.year, d.month, d.day, 12, 0, tzinfo=IST))
        for i, (text, d) in enumerate(zip(texts, own_days))
    ]

    bars = []
    for text, d in zip(texts, own_days):
        close = 105.0 if text in _POSITIVE_TEXTS else 95.0
        bars.append(Bar(date=d, open=100.0, close=close))
    for d in buffer_days:
        bars.append(Bar(date=d, open=100.0, close=100.0))  # flat - no signal to leak forward

    save_fixture("FORTIS.NS", bars)
    return headlines


def test_naive_alignment_shows_a_strong_spurious_correlation(tmp_path: Path, monkeypatch):
    headlines = _build_synthetic_leak(tmp_path, monkeypatch)

    result = permutation_test(headlines, naive_same_day_session_date, n_perm=300, seed=0)

    assert abs(result.real_r) > 0.8
    assert result.p_value < 0.05  # shuffling makes this look like an extreme outlier, correctly


def test_leak_free_alignment_does_not_pick_up_the_same_synthetic_leak(tmp_path: Path, monkeypatch):
    headlines = _build_synthetic_leak(tmp_path, monkeypatch)

    result = permutation_test(headlines, leak_free_session_date, n_perm=300, seed=0)

    # every headline rolls forward to a flat (0% return) buffer day under
    # leak-free alignment, so there is no sentiment-driven signal left for
    # the correlation to find, with or without shuffling.
    assert result.real_r == pytest.approx(0.0)
    assert result.p_value >= 0.05


def test_shuffling_the_naive_alignment_collapses_the_null_toward_zero(tmp_path: Path, monkeypatch):
    headlines = _build_synthetic_leak(tmp_path, monkeypatch)

    result = permutation_test(headlines, naive_same_day_session_date, n_perm=300, seed=0)

    assert abs(result.null_mean) < abs(result.real_r)


def test_run_against_committed_fixture_passes_and_exits_zero(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, n_perm=50, seed=0, live=False)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, n_perm=50, seed=0, live=False)

    assert exit_code == 1


def test_permutation_result_null_mean_property():
    result = PermutationResult(real_r=0.5, real_n=5, null_rs=[0.1, 0.2, 0.3], p_value=0.2)
    assert result.null_mean == pytest.approx(0.2)
