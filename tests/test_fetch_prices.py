from datetime import date
from pathlib import Path

import sentiment.prices as prices
from sentiment.fetch_prices import run
from sentiment.prices import Bar, load_fixture, save_fixture


def _chart_payload(rows: list[tuple[int, float, float]]) -> dict:
    return {
        "chart": {
            "result": [
                {
                    "timestamp": [ts for ts, _, _ in rows],
                    "indicators": {"quote": [{"open": [o for _, o, _ in rows], "close": [c for _, _, c in rows]}]},
                }
            ]
        }
    }


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_run_refreshes_every_known_ticker_and_merges_with_existing_fixtures(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    # One known ticker already has a committed fixture outside the live
    # fetch's own trailing window - must survive the refresh (D26).
    save_fixture("INFY.NS", [Bar(date=date(2026, 8, 31), open=1.0, close=2.0)])

    payload = _chart_payload([(1790000000, 5.0, 6.0)])  # 2026-09-21
    monkeypatch.setattr(prices.json, "load", lambda resp: payload)
    monkeypatch.setattr(prices.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))

    exit_code = run()

    assert exit_code == 0
    bars = load_fixture("INFY.NS")
    assert {b.date for b in bars} == {date(2026, 8, 31), date(2026, 9, 21)}
    out = capsys.readouterr().out
    assert "tickers refreshed" in out


def test_run_reports_a_failed_ticker_and_continues(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    def boom(*args, **kwargs):
        raise prices.urllib.error.URLError("no network")

    monkeypatch.setattr(prices.urllib.request, "urlopen", boom)
    monkeypatch.setattr(prices.time, "sleep", lambda _: None)

    exit_code = run()

    assert exit_code == 0
    err = capsys.readouterr().err
    assert "ticker(s) failed to fetch" in err
