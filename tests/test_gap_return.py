"""P159: the overnight gap (previous close to the session's open) for headlines
published outside market hours.

Open-to-close starts after the open, so for a headline published after the
previous close or before the pre-open auction, the first price reaction (the
gap) is not in it. The gap is leak-free only when the headline came after the
previous bar's 15:30 IST close and before 09:00 IST on the session date (the
pre-open call auction that sets the open starts at 09:00).
"""

from datetime import date, datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.correlate import build_rows_from_headlines, gap_return, run
from sentiment.headline import Headline, write_csv
from sentiment.market_hours import IST
from sentiment.prices import Bar, save_fixture

BARS = [
    Bar(date=date(2026, 10, 16), open=100.0, close=110.0),  # Fri
    Bar(date=date(2026, 10, 19), open=121.0, close=120.0),  # Mon
    # Tue 20 Oct 2026: NSE holiday, no bar
    Bar(date=date(2026, 10, 21), open=114.0, close=118.0),  # Wed
]


def at(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=IST)


def test_pre_open_before_auction_gets_the_gap():
    # Mon 08:59 IST: after Fri's close, before the 09:00 auction -> 121/110 - 1
    assert abs(gap_return(at(2026, 10, 19, 8, 59), BARS, BARS[1]) - 0.1) < 1e-12


def test_weekend_and_post_close_headlines_get_the_gap():
    assert gap_return(at(2026, 10, 17, 12, 0), BARS, BARS[1]) is not None
    assert gap_return(at(2026, 10, 16, 15, 31), BARS, BARS[1]) is not None


def test_auction_window_headline_gets_none():
    # 09:00 to 09:15: the auction may already have set the open
    assert gap_return(at(2026, 10, 19, 9, 0), BARS, BARS[1]) is None
    assert gap_return(at(2026, 10, 19, 9, 10), BARS, BARS[1]) is None


def test_headline_before_previous_close_gets_none():
    # Fri 15:30 is still inside Fri's session: the close is not after it
    assert gap_return(at(2026, 10, 16, 15, 30), BARS, BARS[1]) is None
    assert gap_return(at(2026, 10, 16, 14, 0), BARS, BARS[1]) is None


def test_holiday_rolls_to_the_next_bar_and_gaps_from_the_last_close():
    # Mon 19 evening -> aligned to Tue 20 (holiday) -> rolled to Wed 21;
    # the gap runs from Mon's close (120) to Wed's open (114)
    assert abs(gap_return(at(2026, 10, 19, 18, 0), BARS, BARS[2]) - (114.0 / 120.0 - 1)) < 1e-12


def test_first_bar_has_no_previous_close():
    assert gap_return(at(2026, 10, 16, 8, 0), BARS, BARS[0]) is None


def _headline(title, link, published_at):
    return Headline(
        source="feed", title=title, link=link, published_at=published_at,
        published_raw=published_at.isoformat(), scraped_at=datetime(2026, 10, 22, tzinfo=timezone.utc),
    )


def test_build_rows_carries_gap_return(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", BARS)
    rows, _, _ = build_rows_from_headlines(
        [
            _headline("Infosys shares rise 2% on deal win", "1", at(2026, 10, 19, 7, 0)),
            _headline("Infosys shares fall 1% after guidance cut", "2", at(2026, 10, 19, 11, 0)),
        ]
    )
    by_title = {r["title"]: r for r in rows}
    assert abs(by_title["Infosys shares rise 2% on deal win"]["gap_return"] - 0.1) < 1e-12
    assert by_title["Infosys shares fall 1% after guidance cut"]["gap_return"] is None  # intraday


def test_run_prints_a_labelled_gap_line(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", BARS)
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            _headline("Infosys shares rise 2% on deal win", "1", at(2026, 10, 19, 7, 0)),
            _headline("Infosys shares fall 1% after guidance cut", "2", at(2026, 10, 19, 18, 0)),
        ],
        in_path,
    )
    assert run(in_path, tmp_path / "c.csv", live=False, event_threshold=0.3) == 0
    out = capsys.readouterr().out
    gap_line = [l for l in out.splitlines() if l.startswith("overnight gap")]
    assert len(gap_line) == 1
    assert "n=2" in gap_line[0] and "exploratory" in gap_line[0]
