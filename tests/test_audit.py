"""Day 8: shuffle-timestamp leakage audit.

Two different things are tested here, and they must not be confused:

1. Against this repo's *real* fixture, the correct outcome is "not
   significant" - not because the audit found the pipeline clean, but
   because Day 5/6 already found no real correlation to leak in the first
   place (see ``sentiment/audit.py``'s module docstring and the README's Day
   8 Findings/Limitations). A test that only ever ran against real data
   where the right answer is always "nothing to see" would not prove the
   audit can catch anything.
2. ``test_run_shuffle_audit_detects_a_fabricated_leak`` is the positive
   control that proves it can: a synthetic pipeline with a manufactured,
   perfect timestamp-to-return mapping, where shuffling timestamps *must*
   destroy the correlation if the statistics are implemented correctly. This
   is the only place a "leak" is injected on purpose, and it is never run
   against the real fixture - doing that would require fabricating a result
   this repo's own data does not support.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import sentiment.audit as audit
from sentiment.audit import run, run_shuffle_audit, shuffle_timestamps
from sentiment.headline import Headline


def make_headline(title: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=title,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_timestamps_preserves_the_multiset_but_can_reassign_it():
    base = datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)
    headlines = [make_headline(f"h{i}", base + timedelta(minutes=i)) for i in range(8)]

    import random

    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # the whole point of the shuffle: at least one headline's title is now
    # paired with a timestamp it didn't originally have.
    assert any(a.published_at != b.published_at for a, b in zip(shuffled, headlines))
    # original list must be untouched - shuffle_timestamps returns new objects.
    assert [h.published_at for h in headlines] == [base + timedelta(minutes=i) for i in range(8)]


def test_run_shuffle_audit_against_the_real_fixture_finds_nothing_significant():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    from sentiment.headline import read_csv

    headlines = read_csv(fixture)
    result = run_shuffle_audit(headlines, n_shuffles=200, seed=0)

    assert result.real_n == 23
    assert result.real_r == pytest.approx(-0.185, abs=0.01)
    # not significant - matches Day 5/6's own null finding. This is the
    # "signal-free, so nothing to leak" case, not a clean pass.
    assert result.p_value > 0.05
    assert result.n_usable_shuffles == 200
    assert all(-1.0 <= r <= 1.0 for r in result.shuffled_rs)


def test_run_shuffle_audit_raises_when_too_few_headlines_resolve():
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]
    with pytest.raises(ValueError, match="need >="):
        run_shuffle_audit(headlines)


def test_run_shuffle_audit_detects_a_fabricated_leak(monkeypatch):
    """Positive control: prove the shuffle audit has the power to catch a
    real leak, using a synthetic pipeline where the answer is known by
    construction. Never run against this repo's real fixture - see module
    docstring."""
    n = 12
    base = datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)
    headlines = [make_headline(f"h{i}", base + timedelta(minutes=i)) for i in range(n)]

    # Fixed per headline (by title), unaffected by shuffling - standing in
    # for VADER's compound score, which is computed from title text alone.
    compound_by_title = {h.title: -1.0 + 2.0 * i / (n - 1) for i, h in enumerate(headlines)}
    # Fixed per *timestamp* - standing in for "which session's return this
    # headline gets paired with," the thing alignment (and therefore the
    # shuffle) actually controls.
    return_by_timestamp = {
        h.published_at: -0.05 + 0.10 * i / (n - 1) for i, h in enumerate(headlines)
    }

    def fake_rows_from_headlines(hs, live=False, quiet=False):
        rows = [
            {
                "compound": compound_by_title[h.title],
                "contemporaneous_return": return_by_timestamp[h.published_at],
            }
            for h in hs
        ]
        return rows, []

    monkeypatch.setattr(audit, "rows_from_headlines", fake_rows_from_headlines)

    result = run_shuffle_audit(headlines, n_shuffles=300, seed=0)

    # By construction, the real pairing (title i's compound with title i's
    # own original timestamp's return) is a perfect line.
    assert result.real_r == pytest.approx(1.0, abs=1e-9)
    # Shuffling reassigns timestamps - and therefore returns - at random
    # relative to the fixed compounds, so the null distribution should sit
    # well below the real statistic almost always.
    mean_abs_shuffled = sum(abs(r) for r in result.shuffled_rs) / len(result.shuffled_rs)
    assert mean_abs_shuffled < 0.6
    assert result.p_value <= 0.05


def test_run_against_real_fixture_succeeds_and_reports_not_significant(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    exit_code = run(fixture, live=False, n_shuffles=50, seed=0)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "real timestamps:" in out
    assert "permutation p-value" in out


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, live=False, n_shuffles=50, seed=0)
    assert exit_code == 1
