"""Day 8: the ml-pipeline-audit pass.

Two things have to be true for the "shuffle headline timestamps and the
signal must disappear" gate (NEXT_STEPS.md / README "Correctness gate") to
mean anything:

1. On the real, committed fixture, the correlation Day 5 already found is
   not distinguishable from its own shuffled-timestamp null distribution -
   consistent with Day 5/6's own finding of no detectable sentiment signal,
   not with a leak.
2. The test actually has the power to say otherwise: given a fixture with a
   manufactured, genuinely timestamp-dependent relationship, the same
   machinery must flag the real result as inconsistent with its shuffled
   null. Without this, (1) would be indistinguishable from the test always
   passing no matter what it's given.

This is what ``sentiment.audit`` runs in CI, not a check run once by hand.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import contemporaneous_r, permutation_test, shuffle_timestamps
from sentiment.correlate import DEFAULT_IN
from sentiment.headline import Headline, read_csv
from sentiment.prices import Bar, save_fixture


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


def test_shuffle_timestamps_preserves_the_multiset_of_timestamps():
    headlines = [
        make_headline("a", "1", datetime(2026, 1, 5, 2, 0, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 1, 6, 2, 0, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 1, 7, 2, 0, tzinfo=timezone.utc)),
    ]
    import random

    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    # published_raw is rewritten to match the reassigned timestamp, not left
    # pointing at a publish time that no longer belongs to this headline.
    assert all(h.published_raw == h.published_at.isoformat() for h in shuffled)


def test_permutation_test_on_the_real_fixture_passes_the_day8_gate():
    """The real committed fixture's contemporaneous r (-0.185, n=23 - see
    README Day 5 Findings) must not look extreme next to its own
    shuffled-timestamp null distribution. n_permutations and seed are fixed
    so this is a deterministic assertion, not a flaky one."""
    headlines = read_csv(DEFAULT_IN)

    result = permutation_test(headlines, n_permutations=200, seed=0)

    assert result.real_n == 23
    assert result.real_r == contemporaneous_r(headlines)[0]
    # Not an extreme draw against its own null: the correlation does not
    # survive as something the shuffle fails to explain away.
    assert result.p_value > 0.05


def _weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    d = start
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _build_manufactured_signal_fixture(tmp_path: Path, monkeypatch) -> list[Headline]:
    """A synthetic fixture where a headline's sentiment and its own session's
    return are deterministically linked through *when* the headline was
    published, nothing more - alternating strongly-positive/strongly-negative
    VADER text on alternating strongly-up/strongly-down trading days, all
    pre-open so each headline aligns to exactly the day it names. This is
    not a claim that such a relationship exists anywhere in this repo's real
    data (it doesn't, per README Findings) - it exists only to prove the
    permutation test can catch a timestamp-dependent effect when there is
    one, same as a leak would be."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    days = _weekdays(date(2026, 1, 5), 10)
    pos_text = "Infosys Share Price Highlights: fantastic outstanding excellent profit surge success"
    neg_text = "Infosys Share Price Highlights: terrible disastrous awful crash failure loss"

    bars = []
    headlines = []
    for i, day in enumerate(days):
        positive = i % 2 == 0
        ret = 0.08 if positive else -0.08
        bars.append(Bar(date=day, open=100.0, close=100.0 * (1 + ret)))
        published_at = datetime(day.year, day.month, day.day, 2, 0, tzinfo=timezone.utc)  # 07:30 IST, pre-open
        headlines.append(make_headline(pos_text if positive else neg_text, f"link-{i}", published_at))

    save_fixture("INFY.NS", bars)
    return headlines


def test_permutation_test_detects_a_manufactured_timestamp_dependent_signal(tmp_path, monkeypatch):
    """Proves the gate has power: given a fixture engineered so sentiment and
    return line up only because of *when* each headline landed, the real
    (true-timestamp) result must come back as an outlier against its own
    shuffled-timestamp null - the opposite verdict from the real-fixture
    test above, and the thing that makes that PASS mean something."""
    headlines = _build_manufactured_signal_fixture(tmp_path, monkeypatch)

    result = permutation_test(headlines, n_permutations=300, seed=0)

    assert result.real_n == 10
    assert abs(result.real_r) > 0.9
    assert result.p_value < 0.05
