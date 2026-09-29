import datetime as dt
from datetime import date
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.overlay import (
    average_car_by_offset,
    cumulative_by_offset,
    cumulative_return_series,
    event_window_abnormal_returns,
    run,
    split_by_magnitude,
)
from sentiment.prices import Bar, save_fixture


def test_cumulative_return_series_compounds_not_sums():
    bars = [
        Bar(date=date(2026, 9, 1), open=100.0, close=110.0),  # +10%
        Bar(date=date(2026, 9, 2), open=110.0, close=99.0),  # -10%
    ]
    dates, cum = cumulative_return_series(bars)
    assert dates == [date(2026, 9, 1), date(2026, 9, 2)]
    assert cum[0] == pytest.approx(0.10)
    assert cum[1] == pytest.approx(1.10 * 0.90 - 1.0)  # -0.01, not 0.0


def test_event_window_abnormal_returns_clips_to_available_bars():
    bars = [
        Bar(date=date(2026, 9, 25), open=100.0, close=101.0),
        Bar(date=date(2026, 9, 28), open=101.0, close=102.0),
        Bar(date=date(2026, 9, 29), open=102.0, close=100.0),
    ]
    market = {
        date(2026, 9, 25): 0.0,
        date(2026, 9, 28): 0.0,
        date(2026, 9, 29): 0.0,
    }

    # event is the last bar - window_after=1 has nothing to clip to
    result = event_window_abnormal_returns(bars, date(2026, 9, 29), market, window_before=2, window_after=1)

    offsets = [o for o, _ in result]
    assert offsets == [-2, -1, 0]  # +1 clipped: no bar after the last one


def test_event_window_abnormal_returns_skips_dates_missing_from_the_market_proxy():
    bars = [
        Bar(date=date(2026, 9, 28), open=100.0, close=101.0),
        Bar(date=date(2026, 9, 29), open=101.0, close=102.0),
    ]
    market = {date(2026, 9, 28): 0.0}  # the 29th has no market-proxy entry

    result = event_window_abnormal_returns(bars, date(2026, 9, 28), market, window_before=0, window_after=1)

    assert result == [(0, 0.01)]


def test_event_window_abnormal_returns_empty_when_event_date_not_in_bars():
    bars = [Bar(date=date(2026, 9, 28), open=100.0, close=101.0)]
    result = event_window_abnormal_returns(bars, date(2026, 9, 30), {}, window_before=1, window_after=1)
    assert result == []


def test_cumulative_by_offset_is_a_running_sum_in_offset_order():
    # deliberately out of order - the function must sort by offset first
    offset_returns = [(1, 0.02), (-1, 0.01), (0, -0.03)]
    car = cumulative_by_offset(offset_returns)
    assert car == [(-1, 0.01), (0, 0.01 - 0.03), (1, 0.01 - 0.03 + 0.02)]


def test_split_by_magnitude_uses_strict_greater_than():
    rows = [{"compound": 0.30}, {"compound": 0.31}, {"compound": -0.40}]
    high, low = split_by_magnitude(rows, event_threshold=0.30)
    assert high == [{"compound": -0.40}, {"compound": 0.31}] or high == [{"compound": 0.31}, {"compound": -0.40}]
    assert low == [{"compound": 0.30}]


def test_average_car_by_offset_averages_only_where_data_exists():
    bars_by_ticker = {
        "AAA.NS": [
            Bar(date=date(2026, 9, 28), open=100.0, close=101.0),
            Bar(date=date(2026, 9, 29), open=101.0, close=102.0),
        ],
        # this ticker's event is the last bar in its own series - offset +1
        # has no data, so it must not drag down the average at offset +1
        "BBB.NS": [
            Bar(date=date(2026, 9, 28), open=50.0, close=49.0),
            Bar(date=date(2026, 9, 29), open=49.0, close=48.0),
        ],
    }
    market_returns = {date(2026, 9, 28): 0.0, date(2026, 9, 29): 0.0}
    rows = [
        {"ticker": "AAA.NS", "session_date": "2026-09-28"},
        {"ticker": "BBB.NS", "session_date": "2026-09-29"},
    ]

    result = average_car_by_offset(rows, bars_by_ticker, market_returns, window_before=1, window_after=1)

    # offset +1: only AAA.NS contributes (BBB.NS's event is its last bar)
    assert result[1][1] == 1
    mean_1, n_1 = result[1]
    assert n_1 == 1


def test_run_end_to_end_against_a_small_synthetic_universe(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    from sentiment.headline import Headline, write_csv
    from datetime import datetime, timezone

    bars_aaa = [
        Bar(date=dt.date(2026, 9, d), open=100.0 + d, close=100.0 + d + (1 if d % 2 else -1))
        for d in range(20, 30)
    ]
    save_fixture("FORTIS.NS", bars_aaa)
    bars_bbb = [Bar(date=dt.date(2026, 9, d), open=50.0, close=50.5) for d in range(20, 30)]
    save_fixture("HDFCBANK.NS", bars_bbb)

    headlines = [
        Headline(
            source="feed",
            title="Fortis Healthcare shares jump 6% on strong outlook",
            link="1",
            published_at=datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            published_raw="x",
            scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
        ),
        Headline(
            source="feed",
            title="HDFC Bank Share Price Highlights: HDFC Bank Stock Price History",
            link="2",
            published_at=datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            published_raw="x",
            scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
        ),
    ]
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)
    out_dir = tmp_path / "outputs"

    exit_code = run(in_path, out_dir, live=False, event_threshold=0.3, window_before=5, window_after=1)

    assert exit_code == 0
    assert (out_dir / "sentiment_price_overlay.png").exists()
    assert (out_dir / "cumulative_abnormal_return.png").exists()


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, tmp_path / "outputs", live=False, event_threshold=0.3, window_before=5, window_after=1)
    assert exit_code == 1


def test_run_against_committed_fixture_produces_both_charts(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_dir = tmp_path / "outputs"

    exit_code = run(fixture, out_dir, live=False, event_threshold=0.3, window_before=5, window_after=1)

    assert exit_code == 0
    assert (out_dir / "sentiment_price_overlay.png").exists()
    assert (out_dir / "cumulative_abnormal_return.png").exists()
