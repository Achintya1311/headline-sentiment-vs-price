import itertools
import random
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import sentiment.correlate as correlate
import sentiment.prices as prices
from sentiment.audit import (
    SIGNIFICANT_FRACTION_ALARM,
    is_significant,
    run,
    run_shuffle_trials,
    shuffle_timestamps,
)
from sentiment.headline import Headline, write_csv
from sentiment.market_hours import Alignment, Timing
from sentiment.prices import Bar, save_fixture
from sentiment.stats import PearsonResult


def make_headline(i: int, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=f"Synthetic headline {i}",
        link=f"link-{i}",
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


def test_shuffle_timestamps_keeps_the_same_multiset_in_the_same_slots():
    headlines = [
        make_headline(i, datetime(2026, 9, 28, i % 23, 0, 0, tzinfo=timezone.utc)) for i in range(12)
    ]
    rng = random.Random(1)

    shuffled = shuffle_timestamps(headlines, rng)

    # Same headlines, same order (titles/links untouched) - only published_at moves.
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # And it actually permuted something, rather than silently returning the input order.
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_timestamps_does_not_mutate_the_input():
    headlines = [make_headline(i, datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc)) for i in range(5)]
    original_times = [h.published_at for h in headlines]

    shuffle_timestamps(headlines, random.Random(3))

    assert [h.published_at for h in headlines] == original_times


def test_is_significant_matches_whether_the_ci_excludes_zero():
    assert is_significant(PearsonResult(r=0.8, n=10, ci_low=0.2, ci_high=0.95)) is True
    assert is_significant(PearsonResult(r=-0.8, n=10, ci_low=-0.95, ci_high=-0.2)) is True
    assert is_significant(PearsonResult(r=0.3, n=10, ci_low=-0.2, ci_high=0.6)) is False


def test_run_shuffle_trials_against_the_committed_fixture_shows_no_leak():
    # The real, documented Day 5 finding is a null result (r=-0.185, CI
    # crosses zero). The leakage control adds to that: shuffling timestamps
    # should not turn up "significant" results any more often than the ~5%
    # pure chance would produce at a 95% confidence level - this is the
    # concrete, CI-checkable form of NEXT_STEPS.md's "Done when" bar.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    from sentiment.headline import read_csv

    headlines = read_csv(fixture)

    result = run_shuffle_trials(headlines, trials=200, seed=0)

    assert result.real is not None
    assert is_significant(result.real) is False
    assert len(result.shuffled) > 0
    assert result.fraction_significant <= SIGNIFICANT_FRACTION_ALARM
    assert result.leak_suspected is False


def test_run_cli_against_the_committed_fixture_passes_the_audit():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, trials=200, seed=0)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, live=False, trials=50, seed=0)

    assert exit_code == 1


def test_run_shuffle_trials_catches_a_leak_that_ignores_real_timestamps(monkeypatch, tmp_path: Path):
    """Prove the control has teeth: build a pipeline where the alignment step
    is buggy in a specific, realistic way - it picks each headline's session
    by its position in the list rather than by its real timestamp, so it
    produces the exact same (compound, return) pairing on every single
    shuffle trial. A real look-ahead bug like this is invisible to a single
    offline run; shuffling timestamps and seeing the "signal" survive anyway
    is exactly what should catch it.
    """
    n = 10
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    bar_up = Bar(date=date(2026, 9, 28), open=100.0, close=110.0)  # +10%
    bar_down = Bar(date=date(2026, 9, 29), open=100.0, close=90.0)  # -10%
    save_fixture("FAKE.NS", [bar_up, bar_down])

    headlines = [
        make_headline(i, datetime(2026, 9, 28, i % 23, 0, 0, tzinfo=timezone.utc)) for i in range(n)
    ]
    # Odd/even compound, perfectly matched (by construction) to bar_up/bar_down
    # below, regardless of which headline's real timestamp a given call sees.
    compound_by_index = {i: (0.9 if i % 2 == 0 else -0.9) for i in range(n)}
    monkeypatch.setattr(correlate, "resolve", lambda title: ("Fake Co", "FAKE.NS"))
    monkeypatch.setattr(
        correlate,
        "score_headline",
        lambda h: SimpleNamespace(compound=compound_by_index[int(h.link.split("-")[1])]),
    )

    fixed_alignments = [
        Alignment(
            published_at=headlines[i].published_at,
            local_time=headlines[i].published_at,
            timing=Timing.PRE_OPEN,
            session_date=bar_up.date if compound_by_index[i] > 0 else bar_down.date,
        )
        for i in range(n)
    ]
    # itertools.cycle over exactly n alignments realigns to the start of the
    # list every time a full pass of n headlines is consumed - i.e. every
    # trial sees the same alignment-by-position, however published_at was
    # shuffled, because this stub never looks at its argument at all.
    positional_alignment = itertools.cycle(fixed_alignments)
    monkeypatch.setattr(correlate, "align_headline", lambda published_at: next(positional_alignment))

    result = run_shuffle_trials(headlines, trials=20, seed=0)

    assert result.real is not None
    assert is_significant(result.real) is True
    assert result.fraction_significant == 1.0
    assert result.leak_suspected is True
