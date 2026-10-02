from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import (
    _LEAK_TICKER_DATES,
    audit_real_fixture,
    audit_synthetic_leak,
    build_synthetic_leak_headlines,
    run,
    run_shuffle_audit,
    shuffle_timestamps,
)
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture
from sentiment.tickers import resolve


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_timestamps_preserves_the_same_multiset_of_timestamps():
    import random

    headlines = [
        make_headline(f"headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links untouched - only the clock moved
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # with 10 distinct timestamps a fixed non-trivial seed should reassign
    # at least one of them away from its original headline
    assert any(a.published_at != b.published_at for a, b in zip(headlines, shuffled))


def test_shuffle_timestamps_is_deterministic_for_a_given_seed():
    import random

    headlines = [
        make_headline(f"headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]

    shuffled_a = shuffle_timestamps(headlines, random.Random(42))
    shuffled_b = shuffle_timestamps(headlines, random.Random(42))

    assert [h.published_at for h in shuffled_a] == [h.published_at for h in shuffled_b]


def test_synthetic_leak_headlines_resolve_to_the_tickers_they_were_built_for():
    headlines = build_synthetic_leak_headlines()
    assert len(headlines) == len(_LEAK_TICKER_DATES)
    for headline, (ticker, _session_date, _sign) in zip(headlines, _LEAK_TICKER_DATES):
        match = resolve(headline.title)
        assert match is not None, headline.title
        assert match[1] == ticker


def test_run_shuffle_audit_detects_the_constructed_leak():
    # Small n_shuffles keeps this fast while still being a meaningful
    # permutation test - the leak is large and should be obvious well
    # before 100 shuffles.
    result = audit_synthetic_leak(n_shuffles=100, seed=0)

    assert result.real_r is not None
    assert abs(result.real_r) > 0.5  # the constructed leak correlates strongly, unshuffled
    assert result.p_value is not None
    assert result.p_value < 0.05  # and is a clear outlier against the shuffled null
    # the shuffled null itself should be centred far closer to zero than
    # the real, leaky correlation - shuffling actually destroyed the leak
    assert result.shuffled_mean is not None
    assert abs(result.shuffled_mean) < abs(result.real_r) / 2


def test_run_shuffle_audit_against_real_fixture_is_not_flagged():
    result = audit_real_fixture(n_shuffles=100, seed=0)

    assert result.real_r is not None
    # matches Day 5's own reported null result (see README Findings) - this
    # pins that number down so a future change to the pipeline that moves
    # it is caught here, not just noticed by eye in a CLI run
    assert round(result.real_r, 3) == -0.185
    assert result.real_n == 23
    assert result.p_value is not None
    # not an outlier against its own shuffled-timestamp null - the honest
    # expected result given Day 5 already found r's CI crosses zero
    assert result.p_value >= 0.05


def test_run_shuffle_audit_with_fewer_than_two_resolved_rows_returns_none(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]

    result = run_shuffle_audit(headlines, n_shuffles=10, seed=0)

    assert result.real_r is None
    assert result.real_n == 1
    assert result.p_value is None


def test_cli_run_reports_both_audits_and_exits_zero(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(n_shuffles=20, seed=0, in_path=fixture)

    assert exit_code == 0
