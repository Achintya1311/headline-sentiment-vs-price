from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.headline import Headline, write_csv
from sentiment.leakage_audit import (
    compute_metrics,
    percentile_rank,
    run,
    run_audit,
    shuffle_timestamps,
)
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
    headlines = [make_headline(f"title {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc)) for i in range(10)]

    shuffled = shuffle_timestamps(headlines, seed=0)

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)


def test_shuffle_timestamps_is_deterministic_for_a_fixed_seed():
    headlines = [make_headline(f"title {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc)) for i in range(10)]

    a = shuffle_timestamps(headlines, seed=7)
    b = shuffle_timestamps(headlines, seed=7)

    assert [h.published_at for h in a] == [h.published_at for h in b]


def test_shuffle_timestamps_actually_moves_at_least_one_headline():
    # Not a mathematical certainty for an arbitrary seed, but true for this
    # fixed one - if this ever broke it would mean shuffle_timestamps had
    # regressed into an identity permutation, which would make the whole
    # leakage control a no-op.
    headlines = [make_headline(f"title {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc)) for i in range(10)]

    shuffled = shuffle_timestamps(headlines, seed=0)

    assert any(a.published_at != b.published_at for a, b in zip(headlines, shuffled))


def test_compute_metrics_reports_none_when_too_few_lagged_rows():
    rows = [{"compound": 0.5, "lagged_return": None}]
    metrics = compute_metrics(rows, train_frac=0.7)
    assert metrics.lagged_r is None
    assert metrics.test_r2 is None


def test_compute_metrics_reports_none_test_r2_on_a_degenerate_split_not_a_crash():
    # Mirrors sentiment.regress's own MIN_TRAIN/MIN_TEST guard: 2 usable rows
    # can't support a 0.7 train/test split (time_split raises ValueError) -
    # the audit must treat that as "not computable this trial", not crash
    # the whole run partway through hundreds of trials.
    rows = [
        {"compound": 0.1, "lagged_return": 0.01},
        {"compound": 0.2, "lagged_return": -0.01},
    ]
    metrics = compute_metrics(rows, train_frac=0.7)
    assert metrics.test_r2 is None
    assert metrics.lagged_r is not None  # 2 rows is enough for a correlation, just not a split


def test_compute_metrics_recovers_a_real_relationship():
    rows = [{"compound": x, "lagged_return": 0.01 * x} for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]
    metrics = compute_metrics(rows, train_frac=0.7)
    assert metrics.lagged_r == 1.0
    assert metrics.test_r2 == 1.0


def test_percentile_rank_counts_at_or_above():
    assert percentile_rank(0.5, [0.1, 0.4, 0.5, 0.6, 0.9]) == 3 / 5
    assert percentile_rank(10.0, [0.1, 0.2]) == 0.0
    assert percentile_rank(-10.0, [0.1, 0.2]) == 1.0


def test_run_audit_against_the_real_fixture_matches_the_documented_degenerate_case():
    # Same honest finding Day 5/6 already recorded: the real run's lagged
    # correlation and out-of-sample R^2 are both exactly 0.0, because every
    # usable headline shares one VADER score. The audit must reproduce that,
    # not a different number, since it reruns the identical Day 5/6 pipeline
    # on the unshuffled headlines before it ever shuffles anything.
    result = run_audit(FIXTURE, trials=200, seed=0, train_frac=0.7)

    assert result.real.n_resolved == 23
    assert result.real.n_lagged == 15
    assert result.real.lagged_r == 0.0
    assert result.real.test_r2 == 0.0
    assert len(result.null) == 200


def test_leakage_gate_the_real_result_is_not_an_outlier_against_its_shuffled_null():
    """The automated version of NEXT_STEPS.md's 'Done when': shuffle the
    headline timestamps and the signal must disappear. Operationalised here
    as a two-sided outlier check at the 95% level - if the *true*-timestamp
    run scored outside the middle 95% of its own shuffled-timestamp null
    distribution, that would mean real alignment is doing something a random
    reassignment of the same timestamps can't replicate, which is exactly
    what 'the pipeline is leaking' would look like. This runs in CI on every
    test run, not once by hand, per NEXT_STEPS.md's own requirement.
    """
    result = run_audit(FIXTURE, trials=200, seed=0, train_frac=0.7)
    real, null = result.real, result.null

    r2_null = [m.test_r2 for m in null if m.test_r2 is not None]
    lagged_null = [m.lagged_r for m in null if m.lagged_r is not None]
    assert len(r2_null) > 50  # enough shuffled trials produced a usable split to judge against
    assert len(lagged_null) > 50

    r2_frac = percentile_rank(real.test_r2, r2_null)
    lagged_frac = percentile_rank(real.lagged_r, lagged_null)

    assert 0.025 < r2_frac < 0.975, (
        f"real out-of-sample R^2 sits at the {r2_frac:.1%} mark of its shuffled null - "
        "an extreme outlier here would mean the result depends on something other than "
        "real timestamp alignment"
    )
    assert 0.025 < lagged_frac < 0.975, (
        f"real lagged correlation sits at the {lagged_frac:.1%} mark of its shuffled null - "
        "an extreme outlier here would mean the result depends on something other than "
        "real timestamp alignment"
    )


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, trials=10, seed=0, train_frac=0.7, live=False)
    assert exit_code == 1


def test_run_against_committed_fixture_succeeds_and_prints_a_verdict(tmp_path: Path, capsys):
    exit_code = run(FIXTURE, trials=20, seed=0, train_frac=0.7, live=False)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "leakage check" in out
    assert "real test R^2" in out


def test_run_audit_tolerates_a_single_resolved_headline_without_crashing(tmp_path: Path, monkeypatch):
    # A pathological case the real fixture never hits but a smaller or
    # future scrape could: too few resolved headlines for any correlation or
    # split, in both the real run and every shuffled trial.
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=__import__("datetime").date(2026, 9, 28), open=100.0, close=105.0)])
    in_path = tmp_path / "raw.csv"
    write_csv(
        [make_headline("Infosys Share Price Highlights: Infosys Stock Price History", "1", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc))],
        in_path,
    )

    result = run_audit(in_path, trials=10, seed=0, train_frac=0.7)

    assert result.real.lagged_r is None
    assert result.real.test_r2 is None
    assert result.skipped_r2 == 10
