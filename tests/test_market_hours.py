from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from sentiment.market_hours import (
    IST,
    Timing,
    align_headline,
    classify_timing,
    is_trading_day,
    next_trading_day,
)


def ist(y, m, d, hh, mm, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=IST)


def test_is_trading_day_weekday_vs_weekend():
    monday = date(2026, 9, 28)
    saturday = date(2026, 10, 3)
    sunday = date(2026, 10, 4)
    assert is_trading_day(monday)
    assert not is_trading_day(saturday)
    assert not is_trading_day(sunday)


def test_next_trading_day_skips_weekend():
    friday = date(2026, 10, 2)
    assert next_trading_day(friday) == date(2026, 10, 5)  # Monday


def test_next_trading_day_from_saturday_is_monday():
    saturday = date(2026, 10, 3)
    assert next_trading_day(saturday) == date(2026, 10, 5)


@pytest.mark.parametrize(
    "when,expected",
    [
        (ist(2026, 9, 28, 6, 0), Timing.PRE_OPEN),
        (ist(2026, 9, 28, 9, 14, 59), Timing.PRE_OPEN),
        (ist(2026, 9, 28, 9, 15, 0), Timing.INTRADAY),  # trap: 09:15 open itself
        (ist(2026, 9, 28, 9, 20, 0), Timing.INTRADAY),  # trap example from NEXT_STEPS.md
        (ist(2026, 9, 28, 15, 30, 0), Timing.INTRADAY),
        (ist(2026, 9, 28, 15, 30, 1), Timing.POST_CLOSE),
        (ist(2026, 9, 28, 20, 47, 16), Timing.POST_CLOSE),
        (ist(2026, 10, 3, 12, 0, 0), Timing.POST_CLOSE),  # Saturday: no session at all
    ],
)
def test_classify_timing(when, expected):
    assert classify_timing(when) == expected


def test_pre_open_headline_aligns_to_same_day():
    headline = ist(2026, 9, 28, 8, 25, 19)  # real fixture timestamp
    alignment = align_headline(headline)
    assert alignment.timing is Timing.PRE_OPEN
    assert alignment.session_date == date(2026, 9, 28)
    assert alignment.leak_free()


def test_intraday_headline_rolls_forward_one_trading_day():
    """The exact trap NEXT_STEPS.md names: 09:20 is not tradable at 09:15 open."""
    headline = ist(2026, 9, 28, 9, 20, 57)
    alignment = align_headline(headline)
    assert alignment.timing is Timing.INTRADAY
    assert alignment.session_date == date(2026, 9, 29)  # Tuesday, not the 28th
    assert alignment.leak_free()


def test_post_close_headline_rolls_forward_one_trading_day():
    headline = ist(2026, 9, 28, 20, 47, 16)  # real fixture timestamp
    alignment = align_headline(headline)
    assert alignment.timing is Timing.POST_CLOSE
    assert alignment.session_date == date(2026, 9, 29)
    assert alignment.leak_free()


def test_weekend_headline_rolls_forward_to_monday():
    headline = ist(2026, 10, 3, 12, 0, 0)  # Saturday
    alignment = align_headline(headline)
    assert alignment.session_date == date(2026, 10, 5)
    assert alignment.leak_free()


def test_friday_post_close_rolls_to_monday_not_saturday():
    headline = ist(2026, 10, 2, 18, 0, 0)  # Friday evening
    alignment = align_headline(headline)
    assert alignment.session_date == date(2026, 10, 5)


def test_utc_input_is_converted_before_classifying():
    # 03:50 UTC == 09:20 IST on the same calendar day: still intraday.
    headline = datetime(2026, 9, 28, 3, 50, 0, tzinfo=timezone.utc)
    alignment = align_headline(headline)
    assert alignment.timing is Timing.INTRADAY
    assert alignment.session_date == date(2026, 9, 29)


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        align_headline(datetime(2026, 9, 28, 9, 20, 0))


def test_leak_free_holds_for_every_timing_bucket():
    """The alignment's own invariant, not just the classification labels."""
    samples = [
        ist(2026, 9, 28, 6, 0),
        ist(2026, 9, 28, 9, 20),
        ist(2026, 9, 28, 20, 47),
        ist(2026, 10, 3, 12, 0),
    ]
    for when in samples:
        assert align_headline(when).leak_free()
