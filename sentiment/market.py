"""Day 7: an equal-weighted market proxy for abnormal-return charts.

Day 7 had no committed NSE index, so the "market" here is the equal-weighted mean
daily open-to-close return, on each date, across every ticker already
fetched into ``fixtures/prices/`` - the same 23-name universe Day 5 resolved
headlines against. An abnormal return is then a ticker's return in excess of
that proxy, not in excess of the real NSE market. See the README Findings
and Limitations for what that substitution costs.

Day 223 (hub CRITIQUES.md P163): NIFTY 50 (``^NSEI``) is now fetched from the
same free Yahoo chart endpoint into ``fixtures/prices/^NSEI.json``, kept out
of the proxy universe (an index is not a constituent), and offered as a
second benchmark by ``index_daily_returns``. The pre-registered statistic
still uses the proxy; changing it is a method decision, not a fix.
"""

from __future__ import annotations

from datetime import date

from sentiment import prices
from sentiment.prices import Bar, PriceFetchError

INDEX_TICKER = "^NSEI"  # NIFTY 50 on Yahoo Finance


def load_universe_bars() -> dict[str, list[Bar]]:
    """Every ticker with a committed (or freshly fetched) price fixture.

    Reads ``sentiment.prices.FIXTURE_DIR`` at call time (not import time) so
    tests can ``monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)`` the
    same way ``sentiment.prices.load_bars`` itself is tested.
    """
    universe: dict[str, list[Bar]] = {}
    for path in sorted(prices.FIXTURE_DIR.glob("*.json")):
        ticker = path.stem
        if ticker.startswith("^"):  # an index fixture, not a stock (P163)
            continue
        universe[ticker] = prices.load_fixture(ticker)
    return universe


def index_daily_returns() -> dict[date, float] | None:
    """NIFTY 50's open-to-close return per date from its committed fixture,
    or None when the fixture has not been fetched yet (P163)."""
    try:
        bars = prices.load_fixture(INDEX_TICKER)
    except PriceFetchError:
        return None
    return {b.date: b.session_return for b in bars}


def equal_weighted_daily_returns(universe: dict[str, list[Bar]]) -> dict[date, float]:
    """For each date any ticker has a bar on, the mean session_return across
    every ticker that has one - an equal-weighted proxy portfolio return."""
    by_date: dict[date, list[float]] = {}
    for bars in universe.values():
        for b in bars:
            by_date.setdefault(b.date, []).append(b.session_return)
    return {d: sum(rs) / len(rs) for d, rs in by_date.items()}
