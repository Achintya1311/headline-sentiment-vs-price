import datetime as dt
from pathlib import Path

import sentiment.prices as prices
from sentiment.market import equal_weighted_daily_returns, load_universe_bars
from sentiment.prices import Bar, save_fixture


def test_load_universe_bars_reads_every_fixture_in_the_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("AAA.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0)])
    save_fixture("BBB.NS", [Bar(date=dt.date(2026, 9, 28), open=50.0, close=45.0)])

    universe = load_universe_bars()

    assert set(universe) == {"AAA.NS", "BBB.NS"}
    assert universe["AAA.NS"][0].close == 110.0


def test_equal_weighted_daily_returns_averages_across_tickers_present_on_a_date():
    universe = {
        # +10% this day
        "AAA.NS": [Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0)],
        # -10% this day
        "BBB.NS": [Bar(date=dt.date(2026, 9, 28), open=100.0, close=90.0)],
        # only this ticker has a bar the next day
        "CCC.NS": [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=100.0),
            Bar(date=dt.date(2026, 9, 29), open=100.0, close=104.0),
        ],
    }

    returns = equal_weighted_daily_returns(universe)

    assert returns[dt.date(2026, 9, 28)] == 0.0  # (+0.10 - 0.10 + 0.0) / 3
    assert returns[dt.date(2026, 9, 29)] == 0.04  # CCC.NS alone that day
