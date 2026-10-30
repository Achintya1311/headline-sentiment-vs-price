"""P135: a headline aligned to an NSE holiday rolls to the ticker's next bar.

``market_hours`` has no holiday calendar, so a headline before a holiday's
open gets that holiday as its ``session_date``. The ticker's own fixture has
no bar there; the first later bar is the first session that actually opened
after the headline, so it is used (still leak-free). A date past the last
bar is still "not traded yet" and stays unpaired.
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import _Resolved, _returns_for_times
from sentiment.correlate import build_rows_from_headlines
from sentiment.headline import Headline
from sentiment.market_hours import IST, MARKET_OPEN
from sentiment.prices import Bar, save_fixture, session_bar

ROOT = Path(__file__).resolve().parent.parent

# Mon 19 Oct, (Tue 20 Oct holiday, no bar), Wed 21 Oct, Thu 22 Oct
BARS = [
    Bar(date=date(2026, 10, 19), open=100.0, close=101.0),
    Bar(date=date(2026, 10, 21), open=102.0, close=104.0),
    Bar(date=date(2026, 10, 22), open=104.0, close=103.0),
]


def _headline(published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title="Infosys Share Price Highlights: Infosys Stock Price History",
        link="1",
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


def test_session_bar_exact_date():
    assert session_bar(BARS, date(2026, 10, 19)) is BARS[0]


def test_session_bar_rolls_past_a_gap_to_the_next_bar():
    assert session_bar(BARS, date(2026, 10, 20)) is BARS[1]


def test_session_bar_past_the_last_bar_is_none():
    assert session_bar(BARS, date(2026, 10, 23)) is None


def test_correlate_pairs_holiday_headline_with_next_session(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", BARS)
    # 19 Oct 18:00 IST, after the close -> weekday rule says 20 Oct (holiday)
    h = _headline(datetime(2026, 10, 19, 12, 30, tzinfo=timezone.utc))
    rows, _, price_errors = build_rows_from_headlines([h])
    assert price_errors == []
    assert len(rows) == 1
    row = rows[0]
    assert row["session_date"] == "2026-10-21"
    assert row["contemporaneous_return"] == (104.0 - 102.0) / 102.0
    assert row["lagged_return"] == (103.0 - 104.0) / 104.0
    # leak-free: the session used opens after the headline
    opened = datetime.combine(date.fromisoformat(row["session_date"]), MARKET_OPEN, tzinfo=IST)
    assert opened > h.published_at


def test_audit_pipeline_rolls_the_same_way():
    r = _Resolved(ticker="INFY.NS", compound=0.3, bars=BARS)
    compounds, returns = _returns_for_times([r], [datetime(2026, 10, 19, 12, 30, tzinfo=timezone.utc)])
    assert returns == [(104.0 - 102.0) / 102.0]


def test_real_fixtures_have_no_bar_on_20_oct_and_the_headlines_now_pair():
    # The 2026-10-20 case from the critique: NSE was shut for Dussehra.
    for t in ("BAJAJFINSV.NS", "PCJEWELLER.NS", "HDFCBANK.NS"):
        bars = json.loads((ROOT / "fixtures" / "prices" / f"{t}.json").read_text())
        dates = {b["date"] for b in (bars["bars"] if isinstance(bars, dict) else bars)}
        assert "2026-10-20" not in dates
        assert "2026-10-21" in dates
    loaded = prices.load_bars("HDFCBANK.NS")
    assert session_bar(loaded, date(2026, 10, 20)).date == date(2026, 10, 21)
