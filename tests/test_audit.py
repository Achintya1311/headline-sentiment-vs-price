from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from sentiment.audit import (
    Resolved,
    ShuffleAuditResult,
    audit_resolved,
    permutation_p_value,
    resolve_headlines,
    run,
    run_shuffle_audit,
)
from sentiment.correlate import build_rows
from sentiment.headline import read_csv
from sentiment.market_hours import IST
from sentiment.prices import Bar

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"


# ---------------------------------------------------------------------------
# permutation_p_value: pure function, no fixtures needed.
# ---------------------------------------------------------------------------


def test_permutation_p_value_counts_shuffles_at_least_as_extreme():
    # none of the shuffled stats reach the real one -> smallest possible
    # p-value for 10 shuffles, add-one smoothed.
    assert permutation_p_value(0.5, [0.1] * 10) == pytest.approx(1 / 11)


def test_permutation_p_value_is_one_when_every_shuffle_matches_or_exceeds():
    assert permutation_p_value(0.0, [0.0] * 5) == pytest.approx(1.0)


def test_permutation_p_value_with_no_shuffles_is_conservative():
    assert permutation_p_value(0.9, []) == 1.0


# ---------------------------------------------------------------------------
# ShuffleAuditResult.passes(): both conditions must hold.
# ---------------------------------------------------------------------------


def test_passes_requires_both_a_high_p_value_and_a_sane_significance_rate():
    not_an_outlier_and_well_behaved = ShuffleAuditResult(
        real_r=-0.2, real_n=23, shuffled_rs=[0.1, -0.1], p_value=0.5, pct_shuffles_significant=0.02
    )
    assert not_an_outlier_and_well_behaved.passes() is True

    outlier = ShuffleAuditResult(
        real_r=0.95, real_n=23, shuffled_rs=[0.1, -0.1], p_value=0.01, pct_shuffles_significant=0.02
    )
    assert outlier.passes() is False

    manufactures_significance = ShuffleAuditResult(
        real_r=-0.2, real_n=23, shuffled_rs=[0.1, -0.1], p_value=0.5, pct_shuffles_significant=0.3
    )
    assert manufactures_significance.passes() is False


# ---------------------------------------------------------------------------
# resolve_headlines: same scope as sentiment.correlate.build_rows.
# ---------------------------------------------------------------------------


def test_resolve_headlines_matches_correlate_scope_on_the_real_fixture():
    headlines = read_csv(FIXTURE)
    resolved = resolve_headlines(headlines)
    rows, _unresolved = build_rows(FIXTURE)

    assert len(resolved) == len(rows) == 23
    assert {r.ticker for r in resolved} == {r["ticker"] for r in rows}


# ---------------------------------------------------------------------------
# Real fixture: the already-established null result should pass the audit.
# ---------------------------------------------------------------------------


def test_real_fixture_audit_passes_and_matches_days_5_and_6_null_result():
    headlines = read_csv(FIXTURE)
    result = run_shuffle_audit(headlines, n_shuffles=200, seed=0)

    assert result.real_n == 23
    # Day 5's README reports r=-0.185 for this exact fixture.
    assert result.real_r == pytest.approx(-0.185, abs=0.001)
    assert result.passes() is True


def test_cli_run_against_committed_fixture_exits_zero():
    exit_code = run(FIXTURE, n_shuffles=100, seed=0)
    assert exit_code == 0


def test_cli_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    assert run(missing, n_shuffles=100, seed=0) == 1


# ---------------------------------------------------------------------------
# Positive control: a deliberately leaky pipeline must fail this audit, or
# a "pass" on the real (already-null) fixture above would mean nothing - see
# sentiment/audit.py's module docstring.
# ---------------------------------------------------------------------------


def _weekdays(start: date, n: int) -> list[date]:
    days: list[date] = []
    d = start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def test_shuffle_audit_catches_a_synthetic_leak():
    """Build a case where the 'sentiment' score is wired directly to its own
    correctly-aligned session's return (open=1.0, close=1.0+compound) - the
    kind of look-ahead bug this whole audit exists to catch. Every headline
    is timestamped pre-open on a distinct trading day, so under the real
    (unshuffled) timestamps the pipeline recovers the leak exactly (r=1.0);
    shuffling timestamps scrambles which day's manufactured return a
    headline's compound gets compared to, and the leak should collapse.
    """
    n = 16
    dates = _weekdays(date(2026, 9, 1), n)
    compounds = [((-1) ** i) * (i + 1) / (n + 1) for i in range(n)]

    resolved = [
        Resolved(
            title=f"synthetic headline {i}",
            ticker="SYN.TEST",
            compound=compounds[i],
            # 07:00 IST is well before the 09:15 IST open -> pre-open,
            # aligns to this same calendar date.
            published_at=datetime(dates[i].year, dates[i].month, dates[i].day, 7, 0, 0, tzinfo=IST),
        )
        for i in range(n)
    ]
    bar_cache = {
        "SYN.TEST": [Bar(date=dates[i], open=1.0, close=1.0 + compounds[i]) for i in range(n)]
    }

    result = audit_resolved(resolved, bar_cache, n_shuffles=300, seed=0)

    assert result.real_r == pytest.approx(1.0)
    assert result.p_value < 0.05
    assert result.passes() is False
    # the shuffled null should be nowhere near the real, leak-driven r=1.0
    mean_abs_shuffled = sum(abs(r) for r in result.shuffled_rs) / len(result.shuffled_rs)
    assert mean_abs_shuffled < 0.5


def test_shuffle_audit_on_a_genuinely_leak_free_synthetic_case_passes():
    """Same shape as the leak case above, but the return is independent of
    the compound score (a fixed small daily return unrelated to sentiment) -
    there is nothing to leak, and shuffling should leave the (already near
    zero) correlation looking the same."""
    n = 16
    dates = _weekdays(date(2026, 9, 1), n)
    compounds = [((-1) ** i) * (i + 1) / (n + 1) for i in range(n)]
    # Alternating small return, uncorrelated with `compounds` by construction.
    returns = [0.001 if i % 3 == 0 else -0.0015 if i % 3 == 1 else 0.0005 for i in range(n)]

    resolved = [
        Resolved(
            title=f"synthetic headline {i}",
            ticker="SYN.TEST",
            compound=compounds[i],
            published_at=datetime(dates[i].year, dates[i].month, dates[i].day, 7, 0, 0, tzinfo=IST),
        )
        for i in range(n)
    ]
    bar_cache = {
        "SYN.TEST": [Bar(date=dates[i], open=1.0, close=1.0 + returns[i]) for i in range(n)]
    }

    result = audit_resolved(resolved, bar_cache, n_shuffles=300, seed=0)

    assert result.passes() is True
