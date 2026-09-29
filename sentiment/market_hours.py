"""NSE market-hours alignment (Day 4).

The point of this module: given a headline's ``published_at``, decide which
trading session's close-to-close return can honestly be attributed to it.
Get this wrong and later days' correlation work is measuring look-ahead, not
a real reaction - the exact trap NEXT_STEPS.md calls out ("a headline
timestamped 09:20 IST is not tradable at the 09:15 open").

NSE cash session: 09:15-15:30 IST, Monday-Friday. Three cases for a
headline published on a trading day:

- **before the open** - the whole session's return happens after the news,
  so that same day is clean to attribute the headline to.
- **during the session** - part of that day's move already happened before
  the headline (the open-to-headline stretch), so attributing the same
  day's full return would partly be reacting to something that happened
  before the news existed. Roll forward to the next trading day instead.
- **after the close, or on a non-trading day at all** - nothing left to
  react to today; roll forward to the next trading day.

That collapses to one rule: the aligned session is today if and only if
today is a trading day and the headline arrived before today's open,
otherwise the next trading day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


class Timing(str, Enum):
    """Where ``published_at`` falls relative to the session it landed on."""

    PRE_OPEN = "pre_open"
    INTRADAY = "intraday"
    POST_CLOSE = "post_close"


def is_trading_day(day: date) -> bool:
    """Weekday check only - no NSE holiday calendar. See README limitations:
    a headline published on an actual NSE holiday is treated as a normal
    weekday and rolled forward by one calendar day too few.
    """
    return day.weekday() < 5


def next_trading_day(day: date) -> date:
    """The first trading day strictly after ``day``."""
    candidate = day + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def classify_timing(published_at: datetime) -> Timing:
    """Where ``published_at`` (any timezone) falls in IST, for diagnostics.

    A headline on a non-trading day (weekend) is classified ``POST_CLOSE``:
    there is no session left to be "pre-open" or "intraday" relative to, and
    like a genuine post-close headline it rolls forward to the next trading
    day.
    """
    local = published_at.astimezone(IST)
    if is_trading_day(local.date()):
        if local.time() < MARKET_OPEN:
            return Timing.PRE_OPEN
        if local.time() <= MARKET_CLOSE:
            return Timing.INTRADAY
    return Timing.POST_CLOSE


@dataclass(frozen=True)
class Alignment:
    published_at: datetime
    local_time: datetime
    timing: Timing
    session_date: date

    def leak_free(self) -> bool:
        """True if the session's own open happens strictly after the headline.

        This is the actual look-ahead guarantee, not just a label: it holds
        by construction for every ``Alignment`` this module produces (see
        ``align_headline``), and exists so a test can assert the invariant
        directly rather than trusting the classification logic that
        produced it.
        """
        session_open = datetime.combine(self.session_date, MARKET_OPEN, tzinfo=IST)
        return session_open > self.local_time


def align_headline(published_at: datetime) -> Alignment:
    """Return the leak-free trading session for ``published_at``.

    Raises ``ValueError`` if ``published_at`` is naive - a timestamp with no
    timezone cannot be safely converted to IST and would silently mislocate
    the headline in time, exactly the failure mode this module exists to
    prevent.
    """
    if published_at.tzinfo is None:
        raise ValueError(f"published_at must be timezone-aware: {published_at!r}")

    local = published_at.astimezone(IST)
    timing = classify_timing(published_at)

    if timing is Timing.PRE_OPEN:
        session_date = local.date()
    else:
        session_date = next_trading_day(local.date())

    return Alignment(
        published_at=published_at,
        local_time=local,
        timing=timing,
        session_date=session_date,
    )
