from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.audit as audit
from sentiment.audit import (
    PipelineStat,
    make_synthetic_world,
    placebo_p_value,
    run,
    run_synthetic_power_check,
    shuffle_published_at,
    stat_from_rows,
    synthetic_rows,
)
from sentiment.headline import Headline


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
    headlines = [make_headline(f"h{i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc)) for i in range(6)]

    shuffled = shuffle_published_at(headlines, seed=1)

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert sorted(h.link for h in shuffled) == sorted(h.link for h in headlines)
    # Re-sorted oldest-first, matching sentiment.headline.write_csv's own ordering.
    assert [h.published_at for h in shuffled] == sorted(h.published_at for h in shuffled)


def test_shuffle_published_at_actually_reassigns_at_least_one_pairing():
    headlines = [make_headline(f"h{i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc)) for i in range(10)]

    shuffled = shuffle_published_at(headlines, seed=1)

    original_pairs = {(h.link, h.published_at) for h in headlines}
    shuffled_pairs = {(h.link, h.published_at) for h in shuffled}
    assert original_pairs != shuffled_pairs


def test_shuffle_published_at_is_deterministic_for_the_same_seed():
    headlines = [make_headline(f"h{i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc)) for i in range(10)]

    a = shuffle_published_at(headlines, seed=5)
    b = shuffle_published_at(headlines, seed=5)

    assert [h.published_at for h in a] == [h.published_at for h in b]


def test_placebo_p_value_computes_two_sided_fraction():
    # 2 of 5 null values are at least as extreme (|.| >= 0.5) as the real one.
    assert placebo_p_value(0.5, [0.0, 0.1, -0.6, 0.9, 0.2]) == pytest.approx(0.4)


def test_placebo_p_value_returns_none_when_real_value_is_none():
    assert placebo_p_value(None, [0.0, 0.1]) is None


def test_placebo_p_value_ignores_none_entries_in_the_null_population():
    assert placebo_p_value(0.5, [None, None, 1.0]) == pytest.approx(1.0)


def test_make_synthetic_world_encodes_compound_as_a_linear_function_of_true_return():
    world = make_synthetic_world(n_dates=4, headlines_per_date=2, true_slope=10.0)

    assert len(world.headlines) == 8
    for h in world.headlines:
        d = h.published_at.date()
        assert world.compound_by_link[h.link] == pytest.approx(10.0 * world.true_return_by_date[d])


def test_synthetic_rows_recovers_the_built_in_relationship_unshuffled():
    world = make_synthetic_world(n_dates=4, headlines_per_date=3, true_slope=10.0)

    rows = synthetic_rows(world, world.headlines)

    assert len(rows) == len(world.headlines)
    for row in rows:
        assert row["compound"] == pytest.approx(10.0 * row["lagged_return"])


def test_run_synthetic_power_check_passes_with_a_real_signal():
    result = run_synthetic_power_check(
        n_dates=6, headlines_per_date=4, true_slope=20.0, trials=30, seed=0, train_frac=0.7
    )

    assert result.unshuffled_r2 == pytest.approx(1.0)
    assert result.shuffled_mean_r2 is not None
    assert abs(result.shuffled_mean_r2) < 0.3
    assert result.passed is True


def test_run_synthetic_power_check_fails_with_no_signal_to_find():
    # true_slope=0 means compound carries no information about lagged_return
    # at all - the control has nothing to detect unshuffled, so the power
    # check itself must say so rather than reporting a false pass.
    result = run_synthetic_power_check(
        n_dates=6, headlines_per_date=4, true_slope=0.0, trials=10, seed=0, train_frac=0.7
    )

    assert result.passed is False


def test_stat_from_rows_handles_too_few_rows_without_crashing():
    stat = stat_from_rows([{"compound": 0.1, "contemporaneous_return": 0.01, "lagged_return": None}], train_frac=0.7)

    assert isinstance(stat, PipelineStat)
    assert stat.contemporaneous_r is None  # fewer than 2 rows
    assert stat.oos_r2 is None  # no usable (lagged) rows


def test_run_against_committed_fixture_passes_and_is_fast(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, trials=20, seed=0, train_frac=0.7)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, trials=20, seed=0, train_frac=0.7)

    assert exit_code == 1


def test_run_fails_when_the_synthetic_power_check_itself_fails(tmp_path: Path, monkeypatch):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    def broken_power_check(*args, **kwargs):
        return audit.SyntheticPowerCheck(unshuffled_r2=0.0, shuffled_r2s=[0.0], shuffled_mean_r2=0.0)

    monkeypatch.setattr(audit, "run_synthetic_power_check", broken_power_check)

    exit_code = run(fixture, trials=5, seed=0, train_frac=0.7)

    assert exit_code == 1


def test_run_fails_when_the_real_result_is_an_extreme_outlier(monkeypatch):
    # Simulate what a genuine leak would look like: the real statistic sits
    # far outside its own shuffled-timestamp null distribution. The audit
    # must flag this as a failure, not wave it through. The synthetic power
    # check is stubbed to a trivial pass so this isolates the real-data path.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    def fake_power_check(*args, **kwargs):
        return audit.SyntheticPowerCheck(unshuffled_r2=1.0, shuffled_r2s=[0.0], shuffled_mean_r2=0.0)

    def fake_stat_from_rows(rows, train_frac):
        return PipelineStat(contemporaneous_r=0.99, oos_r2=0.99)

    def fake_real_data_trials(headlines, trials, seed, train_frac):
        return [PipelineStat(contemporaneous_r=0.0, oos_r2=0.0) for _ in range(trials)]

    monkeypatch.setattr(audit, "run_synthetic_power_check", fake_power_check)
    monkeypatch.setattr(audit, "stat_from_rows", fake_stat_from_rows)
    monkeypatch.setattr(audit, "run_real_data_trials", fake_real_data_trials)

    exit_code = run(fixture, trials=20, seed=0, train_frac=0.7)

    assert exit_code == 1
