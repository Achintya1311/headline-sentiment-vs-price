"""Daily OHLC price bars for the tickers Day 5's correlation work needs.

Offline by default: reads the committed snapshots under ``fixtures/prices/``
(one JSON file per ticker, ``{"ticker": ..., "bars": [{"date", "open",
"close"}, ...]}``), fetched once from Yahoo Finance's public chart endpoint
(free, no API key - the same source the rest of this portfolio's satellites
use via yfinance) and committed so nothing here blocks on network access.
``--live`` re-fetches and overwrites those snapshots instead.

A "session return" is open-to-close for one trading day, not close-to-close:
the headline that names a ticker is, by Day 4's construction, already known
to predate that session's open, so open-to-close is the return the market
had available to react with, not a return that partly precedes the news.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "prices"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1mo&interval=1d"


class PriceFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bar:
    date: date
    open: float
    close: float

    @property
    def session_return(self) -> float:
        """Open-to-close return for this one session."""
        return (self.close - self.open) / self.open


def fixture_path(ticker: str) -> Path:
    return FIXTURE_DIR / f"{ticker}.json"


def fetch_live(ticker: str, retries: int = 3) -> list[Bar]:
    """Fetch ``ticker``'s last month of daily bars from Yahoo Finance. Raises
    ``PriceFetchError`` if the symbol is not found or every retry fails."""
    url = CHART_URL.format(ticker=ticker)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                payload = json.load(resp)
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    else:
        raise PriceFetchError(f"{ticker}: unreachable after {retries} attempts ({last_error})")

    result = payload["chart"]["result"]
    if not result:
        err = payload["chart"].get("error")
        raise PriceFetchError(f"{ticker}: {err}")

    r = result[0]
    timestamps = r["timestamp"]
    quote = r["indicators"]["quote"][0]
    bars = []
    for i, ts in enumerate(timestamps):
        o, c = quote["open"][i], quote["close"][i]
        if o is None or c is None:
            continue
        d = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        bars.append(Bar(date=d, open=round(o, 4), close=round(c, 4)))
    return bars


def save_fixture(ticker: str, bars: list[Bar]) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "bars": [{"date": b.date.isoformat(), "open": b.open, "close": b.close} for b in bars],
    }
    fixture_path(ticker).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_fixture(ticker: str) -> list[Bar]:
    path = fixture_path(ticker)
    if not path.exists():
        raise PriceFetchError(f"{ticker}: no fixture at {path}; run with --live first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Bar(date=date.fromisoformat(b["date"]), open=b["open"], close=b["close"]) for b in payload["bars"]]


def load_bars(ticker: str, live: bool = False) -> list[Bar]:
    if live:
        bars = fetch_live(ticker)
        save_fixture(ticker, bars)
        return bars
    return load_fixture(ticker)


def bar_on(bars: list[Bar], on: date) -> Bar | None:
    for b in bars:
        if b.date == on:
            return b
    return None


def next_session_bar(bars: list[Bar], after: date) -> Bar | None:
    """The first bar strictly after ``after`` (bars are already date-ordered
    by Yahoo's chart response)."""
    for b in bars:
        if b.date > after:
            return b
    return None
