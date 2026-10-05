import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.audit as audit
import sentiment.prices as prices
from sentiment.audit import run, run_audit, shuffle_timestamps
from sentiment.headline import Headline, write_csv
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


def test_shuffle_timestamps_is_a_permutation_not_a_resample():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    # same multiset of timestamps, just reassigned - never an invented time.
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # every other field stays attached to its original headline.
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    # with 10 distinct timestamps, a real shuffle almost never leaves every
    # headline's own timestamp in place.
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_shuffle_timestamps_is_deterministic_given_a_seed():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    first = shuffle_timestamps(headlines, random.Random(42))
    second = shuffle_timestamps(headlines, random.Random(42))
    assert [h.published_at for h in first] == [h.published_at for h in second]


def test_run_audit_too_few_resolved_headlines_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = tmp_path / "raw.csv"
    write_csv(
        [make_headline("Cyient among 4 stocks showing White Marubozu Pattern", "1", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc))],
        in_path,
    )

    with pytest.raises(ValueError, match="not enough to audit"):
        run_audit(in_path, n_shuffles=10, seed=0, tolerance=0.1)


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    assert run(missing, n_shuffles=10, seed=0, tolerance=0.1) == 1


def test_run_audit_fails_when_shuffled_mean_exceeds_tolerance(monkeypatch):
    """Exercise the gate directly against a controlled fake pipeline, rather
    than relying on a real-world fixture to happen to produce a bias bigger
    than the tolerance - proves the audit *can* fail, not just that it
    happens to pass on the one fixture this repo has."""
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]
    calls = {"n": 0}

    def fake_contemporaneous_r(hs):
        calls["n"] += 1
        # real run (the first call) is unremarkable; every shuffled rerun
        # reports the same strong correlation regardless of the scrambled
        # timestamps - exactly the "predicts returns no matter what" leak
        # signature the audit exists to catch.
        return (0.05, 6) if calls["n"] == 1 else (0.9, 6)

    monkeypatch.setattr(audit, "contemporaneous_r", fake_contemporaneous_r)
    monkeypatch.setattr(audit, "read_csv", lambda path: headlines)

    result = run_audit(Path("unused.csv"), n_shuffles=20, seed=0, tolerance=0.1)

    assert result.real_r == 0.05
    assert result.shuffled_mean == pytest.approx(0.9)
    assert result.passed is False


def test_run_audit_passes_when_shuffled_mean_stays_within_tolerance(monkeypatch):
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]
    # a small, alternating near-zero sequence - averages out well inside
    # the default 0.1 tolerance.
    shuffled_values = [0.02, -0.03, 0.01, -0.02, 0.03, -0.01, 0.0, 0.01]
    calls = {"n": 0}

    def fake_contemporaneous_r(hs):
        calls["n"] += 1
        if calls["n"] == 1:
            return -0.185, 23
        return shuffled_values[(calls["n"] - 2) % len(shuffled_values)], 6

    monkeypatch.setattr(audit, "contemporaneous_r", fake_contemporaneous_r)
    monkeypatch.setattr(audit, "read_csv", lambda path: headlines)

    result = run_audit(Path("unused.csv"), n_shuffles=len(shuffled_values), seed=0, tolerance=0.1)

    assert result.passed is True
    assert abs(result.shuffled_mean) < 0.1


def test_run_audit_drops_shuffles_with_too_few_resolved_rows(monkeypatch):
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]
    calls = {"n": 0}

    def fake_contemporaneous_r(hs):
        calls["n"] += 1
        if calls["n"] == 1:
            return 0.1, 23
        # half the shuffles "lose" rows (e.g. landed on a session_date with
        # no price bar) and must be excluded from the null distribution.
        return (0.0, 2) if calls["n"] % 2 == 0 else (0.05, 6)

    monkeypatch.setattr(audit, "contemporaneous_r", fake_contemporaneous_r)
    monkeypatch.setattr(audit, "read_csv", lambda path: headlines)

    result = run_audit(Path("unused.csv"), n_shuffles=10, seed=0, tolerance=0.1)

    assert len(result.shuffled_rs) == 5
    assert all(r == 0.05 for r in result.shuffled_rs)


def test_run_audit_against_committed_real_fixture_passes():
    """Integration check against the real 50-headline fixture (23 resolved
    headlines, see sentiment/tickers.py). A small shuffle count keeps this
    fast; sentiment/audit.py's own docstring documents the exact nonzero-but-
    small shuffled mean a full 500-shuffle run against this fixture finds,
    and why it is not read as a leak."""
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    result = run_audit(fixture, n_shuffles=200, seed=0, tolerance=0.1)

    assert result.real_n == 23
    assert len(result.shuffled_rs) == 200
    assert result.passed is True
    assert abs(result.shuffled_mean) < 0.1


def test_run_against_committed_real_fixture_cli_exit_code():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    assert run(fixture, n_shuffles=50, seed=0, tolerance=0.1) == 0
