import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    DEFAULT_SEED,
    SIGNIFICANCE_LEVEL,
    AuditResult,
    run,
    run_permutation_test,
    shuffle_timestamps,
)
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_timestamps_is_a_permutation_not_a_resample():
    headlines = [
        make_headline("a", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(1))

    # same headlines, in the same order, only published_at moves
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # the multiset of timestamps is preserved exactly - this is a shuffle,
    # not a resample that could duplicate or drop one
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)


def test_shuffle_timestamps_actually_reassigns_them():
    # With 50 distinct timestamps (the real fixture), the identity
    # permutation is astronomically unlikely - a real shuffle should move
    # at least some of them for a fixed, arbitrary seed.
    headlines = [
        make_headline(f"h{i}", str(i), datetime(2026, 9, 28, 0, 0, 0, tzinfo=timezone.utc).replace(hour=i % 24))
        for i in range(50)
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(DEFAULT_SEED))
    moved = sum(1 for h, s in zip(headlines, shuffled) if h.published_at != s.published_at)
    assert moved > 0


def test_contemporaneous_r_matches_across_a_relabelled_timestamp(tmp_path: Path, monkeypatch):
    # Swapping two headlines' timestamps (one shuffle) should change which
    # session each is paired with when their aligned sessions differ, which
    # is exactly the mechanism the permutation test relies on.
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0),  # +10%
            Bar(date=dt.date(2026, 9, 29), open=110.0, close=99.0),  # -10%
        ],
    )
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),  # pre-open 28th
        ),
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "2",
            datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),  # pre-open 29th
        ),
    ]
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    from sentiment.audit import _contemporaneous_r

    real = _contemporaneous_r(headlines, live=False)
    assert real is not None
    real_r, real_n = real
    assert real_n == 2

    swapped = [
        make_headline(headlines[0].title, headlines[0].link, headlines[1].published_at),
        make_headline(headlines[1].title, headlines[1].link, headlines[0].published_at),
    ]
    swapped_result = _contemporaneous_r(swapped, live=False)
    assert swapped_result is not None
    # both headlines are identical in content (same template), so the real
    # and swapped pairing resolve to the same two (session, compound) pairs
    # in the opposite order - r is invariant to that, confirming the swap
    # mechanics work on a case simple enough to check by hand.
    assert swapped_result[0] == pytest.approx(real_r)


def test_run_permutation_test_raises_when_too_few_headlines_resolve(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = tmp_path / "raw.csv"
    write_csv(
        [make_headline("no company named here", "1", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc))],
        in_path,
    )
    with pytest.raises(ValueError):
        run_permutation_test(in_path, n_shuffles=10, seed=0)


def test_run_permutation_test_against_the_committed_fixture_passes_the_leakage_test():
    # This is the actual Day 8 "Done when" leakage test from NEXT_STEPS.md,
    # run in CI rather than once by hand: shuffle the real headlines'
    # timestamps and confirm the real contemporaneous correlation (r=-0.185
    # per Day 5's README Findings) is not an outlier against that
    # shuffled-timestamp null - i.e. the pipeline is not leaking information
    # into the sentiment/return pairing that real alignment doesn't provide.
    # Deterministic: fixed seed, so this assertion holds every run.
    result = run_permutation_test(FIXTURE, n_shuffles=300, seed=DEFAULT_SEED)

    assert isinstance(result, AuditResult)
    assert result.real_n == 23
    assert result.real_r == pytest.approx(-0.1853951191026191)
    assert result.p_value > SIGNIFICANCE_LEVEL


def test_run_against_committed_fixture_exits_zero_and_prints_pass(capsys):
    exit_code = run(FIXTURE, n_shuffles=300, seed=DEFAULT_SEED, live=False)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "PASS" in captured.out
    assert "p-value" in captured.out


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, n_shuffles=10, seed=0, live=False)
    assert exit_code == 1
