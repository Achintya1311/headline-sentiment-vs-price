import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    contemporaneous_r,
    event_study_diff,
    run_audit,
    shuffle_published_at,
)
from sentiment.correlate import DEFAULT_EVENT_THRESHOLD, DEFAULT_IN
from sentiment.headline import Headline, read_csv
from sentiment.prices import Bar, save_fixture


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_published_at_preserves_the_multiset_of_timestamps():
    headlines = [
        make_headline("a", "1", datetime(2026, 9, 1, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 9, 2, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 9, 3, tzinfo=timezone.utc)),
    ]
    shuffled = shuffle_published_at(headlines, random.Random(0))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links stay pinned to the original headline - only the timestamp moves
    assert [h.title for h in shuffled] == [h.title for h in headlines]


def test_shuffle_published_at_actually_reassigns_timestamps():
    headlines = [make_headline(f"h{i}", str(i), datetime(2026, 9, i + 1, tzinfo=timezone.utc)) for i in range(10)]
    shuffled = shuffle_published_at(headlines, random.Random(1))

    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_contemporaneous_r_needs_at_least_two_rows():
    assert contemporaneous_r([{"compound": 0.1, "contemporaneous_return": 0.01}]) is None
    rows = [
        {"compound": 0.1, "contemporaneous_return": 0.02},
        {"compound": -0.1, "contemporaneous_return": -0.01},
        {"compound": 0.3, "contemporaneous_return": 0.03},
    ]
    result = contemporaneous_r(rows)
    assert result is not None
    r, ci_low, ci_high, n = result
    assert n == 3
    assert -1.0 <= ci_low <= r <= ci_high <= 1.0


def test_event_study_diff_needs_both_groups_nonempty():
    rows = [{"compound": 0.9, "contemporaneous_return": 0.05}]
    assert event_study_diff(rows, event_threshold=0.3) is None

    rows = [
        {"compound": 0.9, "contemporaneous_return": 0.05},
        {"compound": 0.1, "contemporaneous_return": -0.01},
        {"compound": 0.1, "contemporaneous_return": -0.02},
    ]
    result = event_study_diff(rows, event_threshold=0.3)
    assert result is not None
    diff, ci_low, ci_high, n_high, n_low = result
    assert n_high == 1
    assert n_low == 2
    assert diff == pytest.approx(0.05 - (-0.015))


def test_run_audit_recomputes_rows_per_shuffle(tmp_path: Path, monkeypatch):
    """A fast, synthetic end-to-end check of the plumbing: two headlines
    whose aligned session - and therefore whose return - depends on
    ``published_at``, paired so that swapping their timestamps flips the
    sign of the correlation. If shuffling the timestamps left the rows (and
    so the statistic) unchanged, the "leakage" test below would be auditing
    nothing."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=__import__("datetime").date(2026, 9, 28), open=100.0, close=110.0),
            Bar(date=__import__("datetime").date(2026, 9, 29), open=100.0, close=90.0),
        ],
    )
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: profit jumps, beats estimates, raises guidance",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        ),
        make_headline(
            "Infosys Share Price Highlights: fraud probe, losses mount, outlook cut",
            "2",
            datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),
        ),
    ]

    result = run_audit(headlines, event_threshold=DEFAULT_EVENT_THRESHOLD, n_perm=20, seed=0)

    assert result.real_contemporaneous is not None
    assert len(result.shuffles) == 20
    # the positive-sentiment headline is paired with the +10% session and the
    # negative one with the -10% session, so at least one of the 20 shuffles
    # swapping their timestamps must flip the sign of the correlation
    real_r = result.real_contemporaneous[0]
    shuffled_rs = [run.contemporaneous[0] for run in result.shuffles if run.contemporaneous]
    assert any(r != pytest.approx(real_r) for r in shuffled_rs)


def test_leakage_audit_shuffled_timestamps_do_not_manufacture_significance():
    """The actual Day 8 'done when' check from NEXT_STEPS.md, run against the
    committed fixture: shuffle headline timestamps many times and confirm a
    shuffled-timestamp control does not come back 'significant' (95% CI
    excluding zero) far more often than chance would allow. If it did, some
    part of this pipeline would be deriving its apparent signal from
    something other than correctly-timed news - i.e. leaking - and the
    result would be an artifact, exactly what this test exists to catch.

    The threshold is deliberately generous (chance alone is ~5%; this allows
    up to 20%) because event_study_diff's bootstrap CI is already known to
    be under-calibrated at n_high=3 (see README Day 8 Findings) - the point
    of this test is to catch a pipeline bug that makes the shuffle meaningless,
    not to demand textbook-perfect calibration from a 3-headline sample.
    """
    headlines = read_csv(DEFAULT_IN)
    result = run_audit(headlines, event_threshold=DEFAULT_EVENT_THRESHOLD, n_perm=200, seed=0)

    # sanity: the real, correctly-timed data is not itself a wild outlier -
    # if it were already "significant", a shuffle failing to kill it would
    # not even be surprising.
    assert result.real_contemporaneous is not None
    r, ci_low, ci_high, _ = result.real_contemporaneous
    assert ci_low <= 0 <= ci_high, "real contemporaneous r is already 'significant' - audit assumptions below don't hold"

    sig_count, computable = result.contemporaneous_false_positive_rate()
    assert computable > 0
    assert sig_count / computable <= 0.20

    sig_count, computable = result.event_false_positive_rate()
    assert computable > 0
    assert sig_count / computable <= 0.20
