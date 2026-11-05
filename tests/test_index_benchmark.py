"""P163: abnormal return against NIFTY 50, beside the equal-weighted proxy.

The proxy is the mean of the same headline-selected tickers; the index comes
from the same free Yahoo chart endpoint, stored as fixtures/prices/^NSEI.json
(the price workflow commits that directory) and kept out of the proxy.
"""

import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.headline import Headline, write_csv
from sentiment.market import INDEX_TICKER, equal_weighted_daily_returns, index_daily_returns, load_universe_bars
from sentiment.overlay import mean_ar_0_1, run
from sentiment.prices import Bar, save_fixture


def test_index_fixture_is_not_part_of_the_proxy(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0)])
    save_fixture(INDEX_TICKER, [Bar(date=dt.date(2026, 9, 28), open=100.0, close=90.0)])
    universe = load_universe_bars()
    assert list(universe) == ["INFY.NS"]
    assert abs(equal_weighted_daily_returns(universe)[dt.date(2026, 9, 28)] - 0.10) < 1e-12


def test_index_daily_returns_none_without_a_fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    assert index_daily_returns() is None
    save_fixture(INDEX_TICKER, [Bar(date=dt.date(2026, 9, 28), open=100.0, close=101.0)])
    assert abs(index_daily_returns()[dt.date(2026, 9, 28)] - 0.01) < 1e-12


def test_mean_ar_0_1_against_a_benchmark():
    bars = {"A.NS": [Bar(dt.date(2026, 9, 28), 100.0, 103.0), Bar(dt.date(2026, 9, 29), 100.0, 101.0)]}
    bench = {dt.date(2026, 9, 28): 0.01, dt.date(2026, 9, 29): 0.0}
    rows = [{"ticker": "A.NS", "session_date": "2026-09-28"}, {"ticker": "A.NS", "session_date": "2026-09-29"}]
    mean, n = mean_ar_0_1(rows, bars, bench)
    # only the first row has a +1 session: (0.03 - 0.01) + (0.01 - 0.0)
    assert n == 1 and abs(mean - 0.03) < 1e-12


def _universe(tmp_path, monkeypatch, with_index):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    days = range(21, 30)
    save_fixture("FORTIS.NS", [Bar(dt.date(2026, 9, d), 100.0, 102.0) for d in days])
    save_fixture("HDFCBANK.NS", [Bar(dt.date(2026, 9, d), 50.0, 50.5) for d in days])
    if with_index:
        save_fixture(INDEX_TICKER, [Bar(dt.date(2026, 9, d), 100.0, 100.5) for d in days])
    hl = lambda t, link: Headline(
        source="feed", title=t, link=link, published_at=datetime(2026, 9, 28, 2, 0, tzinfo=timezone.utc),
        published_raw="x", scraped_at=datetime(2026, 9, 28, 9, 0, tzinfo=timezone.utc),
    )
    in_path = tmp_path / "raw.csv"
    write_csv([hl("Fortis Healthcare shares jump 6% on strong outlook", "1"),
               hl("HDFC Bank Share Price Highlights: HDFC Bank Stock Price History", "2")], in_path)
    return in_path


def test_run_prints_both_benchmarks(tmp_path: Path, monkeypatch, capsys):
    in_path = _universe(tmp_path, monkeypatch, with_index=True)
    assert run(in_path, tmp_path / "out", live=False, event_threshold=0.3, window_before=1, window_after=1) == 0
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if "[0, +1] abnormal return" in l]
    assert any("equal-weighted proxy" in l for l in lines)
    assert any("NIFTY 50" in l for l in lines)


def test_run_says_when_the_index_fixture_is_missing(tmp_path: Path, monkeypatch, capsys):
    in_path = _universe(tmp_path, monkeypatch, with_index=False)
    assert run(in_path, tmp_path / "out", live=False, event_threshold=0.3, window_before=1, window_after=1) == 0
    assert "no NIFTY 50 fixture" in capsys.readouterr().out


def test_fetch_prices_also_refreshes_the_index(tmp_path: Path, monkeypatch):
    from sentiment import fetch_prices

    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    seen = []
    monkeypatch.setattr(fetch_prices, "load_bars", lambda t, live=False: seen.append(t) or [])
    fetch_prices.run()
    assert INDEX_TICKER in seen


def test_fetch_live_quotes_the_caret(monkeypatch):
    urls = []

    def fake(req, timeout=None):
        urls.append(req.full_url)
        raise prices.urllib.error.URLError("offline")

    monkeypatch.setattr(prices.urllib.request, "urlopen", fake)
    monkeypatch.setattr(prices.time, "sleep", lambda _: None)
    try:
        prices.fetch_live(INDEX_TICKER, retries=1)
    except prices.PriceFetchError:
        pass
    assert "/chart/%5ENSEI?" in urls[0]
