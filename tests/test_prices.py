import json
import urllib.error
from datetime import date
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.prices import (
    Bar,
    PriceFetchError,
    bar_on,
    fetch_live,
    load_bars,
    load_fixture,
    next_session_bar,
    save_fixture,
)


def test_bar_session_return():
    bar = Bar(date=date(2026, 9, 28), open=100.0, close=110.0)
    assert bar.session_return == pytest.approx(0.10)


def test_bar_on_finds_exact_date():
    bars = [Bar(date=date(2026, 9, 28), open=1, close=2), Bar(date=date(2026, 9, 29), open=3, close=4)]
    assert bar_on(bars, date(2026, 9, 29)) is bars[1]
    assert bar_on(bars, date(2026, 9, 30)) is None


def test_next_session_bar_returns_first_bar_strictly_after():
    bars = [
        Bar(date=date(2026, 9, 26), open=1, close=1),
        Bar(date=date(2026, 9, 28), open=2, close=2),
        Bar(date=date(2026, 9, 29), open=3, close=3),
    ]
    assert next_session_bar(bars, date(2026, 9, 26)) is bars[1]
    assert next_session_bar(bars, date(2026, 9, 29)) is None


def test_save_and_load_fixture_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    bars = [Bar(date=date(2026, 9, 28), open=100.0, close=101.5), Bar(date=date(2026, 9, 29), open=101.5, close=99.0)]
    save_fixture("TEST.NS", bars)
    loaded = load_fixture("TEST.NS")
    assert loaded == bars


def test_load_fixture_missing_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    with pytest.raises(PriceFetchError):
        load_fixture("NOPE.NS")


def test_load_bars_offline_reads_fixture_without_network(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    bars = [Bar(date=date(2026, 9, 28), open=1.0, close=2.0)]
    save_fixture("OFFLINE.NS", bars)

    def boom(*args, **kwargs):
        raise AssertionError("load_bars(live=False) must not touch the network")

    monkeypatch.setattr(prices.urllib.request, "urlopen", boom)
    assert load_bars("OFFLINE.NS", live=False) == bars


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _chart_payload(ticker: str, rows: list[tuple[int, float, float]]) -> dict:
    return {
        "chart": {
            "result": [
                {
                    "timestamp": [ts for ts, _, _ in rows],
                    "indicators": {
                        "quote": [
                            {
                                "open": [o for _, o, _ in rows],
                                "close": [c for _, _, c in rows],
                            }
                        ]
                    },
                }
            ]
        }
    }


def test_fetch_live_parses_bars_and_skips_null_days(monkeypatch):
    payload = _chart_payload(
        "FAKE.NS",
        [
            (1790000000, 100.0, 101.0),
            (1790100000, None, None),
        ],
    )

    def fake_json_load(resp):
        return payload

    monkeypatch.setattr(prices.json, "load", fake_json_load)
    monkeypatch.setattr(prices.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))

    bars = fetch_live("FAKE.NS")
    assert len(bars) == 1
    assert bars[0].open == 100.0
    assert bars[0].close == 101.0


def test_fetch_live_raises_on_unknown_symbol(monkeypatch):
    payload = {"chart": {"result": None, "error": {"code": "Not Found"}}}
    monkeypatch.setattr(prices.json, "load", lambda resp: payload)
    monkeypatch.setattr(prices.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))

    with pytest.raises(PriceFetchError):
        fetch_live("NOPE.NS")


def test_fetch_live_retries_then_raises_when_unreachable(monkeypatch):
    calls = []

    def always_fails(*a, **k):
        calls.append(1)
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(prices.urllib.request, "urlopen", always_fails)
    monkeypatch.setattr(prices.time, "sleep", lambda _: None)

    with pytest.raises(PriceFetchError):
        fetch_live("FAKE.NS", retries=2)
    assert len(calls) == 2


def test_load_bars_live_fetches_and_saves_fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    payload = _chart_payload("LIVE.NS", [(1790000000, 5.0, 6.0)])
    monkeypatch.setattr(prices.json, "load", lambda resp: payload)
    monkeypatch.setattr(prices.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload))

    bars = load_bars("LIVE.NS", live=True)
    assert bars[0].open == 5.0
    assert load_fixture("LIVE.NS") == bars
