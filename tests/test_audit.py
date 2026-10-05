import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import contemporaneous_r, run, run_shuffle_audit, shuffle_timestamps
from sentiment.headline import Headline
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


def test_shuffle_timestamps_preserves_the_set_of_timestamps_but_not_the_pairing():
    headlines = [
        make_headline(f"Infosys Share Price Highlights #{i}", str(i), datetime(2026, 9, 1 + i, 2, 0, 0, tzinfo=timezone.utc))
        for i in range(8)
    ]
    import random

    shuffled = shuffle_timestamps(headlines, random.Random(0))

    # Same content, same order, every field but the timestamp untouched.
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # Same multiset of timestamps...
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # ...but not the same pairing (astronomically unlikely to shuffle back to
    # itself with 8 items and this seed - if this ever flakes, the seed or
    # headline count needs to change, not this assertion).
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_contemporaneous_r_needs_at_least_two_resolved_headlines():
    headlines = [make_headline("not a resolvable headline at all", "1", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc))]
    r, n = contemporaneous_r(headlines, live=False)
    assert r is None
    assert n == 0


def test_contemporaneous_r_against_the_committed_fixture_matches_day5_findings():
    from sentiment.headline import read_csv

    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)

    r, n = contemporaneous_r(headlines, live=False)

    # README Findings, Day 5: contemporaneous r = -0.185, n=23.
    assert n == 23
    assert r == pytest.approx(-0.185, abs=1e-3)


def test_run_shuffle_audit_on_the_real_fixture_does_not_flag_a_leak():
    # The committed fixture's own honest finding (Day 5/6/7) is a null
    # result - no detectable signal at all. A shuffled-timestamp control
    # over data that already shows no real relationship should, if
    # anything, be even less likely to look like one.
    from sentiment.headline import read_csv

    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)

    result = run_shuffle_audit(headlines, live=False, n_shuffles=200, seed=0)

    assert result.real_r == pytest.approx(-0.185, abs=1e-3)
    assert result.real_n == 23
    assert not result.leaking
    lo, hi = result.ci
    assert lo < 0 < hi


def test_run_against_committed_fixture_passes_and_returns_zero():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, n_shuffles=200, seed=0)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, live=False, n_shuffles=50, seed=0)

    assert exit_code == 1


def _weekdays(start: dt.date, count: int) -> list[dt.date]:
    days: list[dt.date] = []
    d = start
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def test_shuffle_audit_catches_a_timing_independent_identity_confound(tmp_path: Path, monkeypatch):
    """The audit's whole point: a correlation that survives shuffling the
    timestamps is not coming from genuine timing at all. Build exactly that
    case directly - one ticker whose price is up on *every* shared date,
    paired with uniformly glowing headlines, and one ticker down on every
    shared date, paired with uniformly bleak headlines. Shuffling timestamps
    only moves a headline to a different date *within that shared pool*; it
    can never move a headline onto a different ticker. So the identity-level
    confound (positive words <-> the always-up ticker) survives every
    shuffle, and the audit must flag it.
    """
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    days = _weekdays(dt.date(2026, 9, 7), 10)
    save_fixture("INFY.NS", [Bar(date=d, open=100.0, close=104.0) for d in days])  # always +4%
    save_fixture("WIPRO.NS", [Bar(date=d, open=100.0, close=96.0) for d in days])  # always -4%

    headlines = []
    for i, d in enumerate(days):
        pre_open = datetime(d.year, d.month, d.day, 2, 0, 0, tzinfo=timezone.utc)
        headlines.append(
            make_headline(
                "Infosys Share Price Highlights: wonderful fantastic great amazing excellent earnings beat",
                f"infy-{i}",
                pre_open,
            )
        )
        headlines.append(
            make_headline(
                "Wipro Share Price Highlights: terrible horrible awful disaster dreadful crisis decline",
                f"wipro-{i}",
                pre_open,
            )
        )

    result = run_shuffle_audit(headlines, live=False, n_shuffles=200, seed=0)

    assert result.real_n == 20
    assert result.real_r > 0.9  # positive words always paired with the always-up ticker
    assert result.leaking  # and shuffling timestamps cannot fix an identity-level confound


def test_shuffle_audit_passes_a_genuine_timing_dependent_signal():
    """The mirror image of the identity-confound test: a *real*, correctly
    timed relationship (same single ticker, content genuinely matched to
    that day's actual move) should look strong with real timestamps but
    collapse toward zero once shuffling breaks the content-to-date pairing -
    exactly what NEXT_STEPS.md's Done when section asks for, and the
    opposite outcome from the identity-confound case above. Uses the real
    committed INFY.NS price fixture, no monkeypatching.
    """
    up_days = [dt.date(2026, 9, 1), dt.date(2026, 9, 11), dt.date(2026, 9, 17), dt.date(2026, 9, 25), dt.date(2026, 9, 29)]
    down_days = [dt.date(2026, 9, 3), dt.date(2026, 9, 7), dt.date(2026, 9, 9), dt.date(2026, 9, 16), dt.date(2026, 9, 22)]

    headlines = []
    for i, d in enumerate(up_days):
        pre_open = datetime(d.year, d.month, d.day, 2, 0, 0, tzinfo=timezone.utc)
        headlines.append(
            make_headline(
                "Infosys Share Price Highlights: wonderful fantastic great amazing excellent earnings beat",
                f"up-{i}",
                pre_open,
            )
        )
    for i, d in enumerate(down_days):
        pre_open = datetime(d.year, d.month, d.day, 2, 0, 0, tzinfo=timezone.utc)
        headlines.append(
            make_headline(
                "Infosys Share Price Highlights: terrible horrible awful disaster dreadful crisis decline",
                f"down-{i}",
                pre_open,
            )
        )

    result = run_shuffle_audit(headlines, live=False, n_shuffles=300, seed=0)

    assert result.real_n == 10
    assert result.real_r > 0.8  # content genuinely matches each specific day's move
    assert not result.leaking  # but it's one ticker - shuffling the dates among these
    lo, hi = result.ci  # headlines breaks the pairing, and the null interval shows it
    assert lo < 0 < hi
