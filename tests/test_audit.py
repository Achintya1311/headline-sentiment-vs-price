from datetime import datetime

from sentiment.audit import (
    DEFAULT_IN,
    planted_return_table,
    planted_rows,
    planted_signal_passed,
    run_planted_signal_control,
    run_real_fixture_control,
    shuffle_timestamps,
    synthetic_headlines,
)
from sentiment.market_hours import IST


def test_shuffle_timestamps_is_a_derangement():
    # Every position must get a *different* element's value - a plain
    # random.shuffle could by chance leave some elements in place.
    original = [datetime(2026, 1, i, 9, 0, tzinfo=IST) for i in range(1, 21)]

    shuffled = shuffle_timestamps(original, seed=3)

    assert sorted(shuffled) == sorted(original)
    assert all(a != b for a, b in zip(original, shuffled))


def test_shuffle_timestamps_is_deterministic_for_a_given_seed():
    original = [datetime(2026, 1, i, 9, 0, tzinfo=IST) for i in range(1, 11)]

    assert shuffle_timestamps(original, seed=5) == shuffle_timestamps(original, seed=5)


def test_synthetic_headlines_each_align_pre_open_to_their_own_day():
    from sentiment.market_hours import align_headline

    timestamps, compounds = synthetic_headlines(n=10, seed=0)

    assert len(timestamps) == len(compounds) == 10
    for ts in timestamps:
        alignment = align_headline(ts)
        assert alignment.session_date == ts.date()
        assert alignment.leak_free()


def test_planted_rows_recovers_the_planted_relationship_before_shuffling():
    timestamps, compounds = synthetic_headlines(n=30, seed=0)
    table = planted_return_table(timestamps, compounds, seed=1)

    rows = planted_rows(timestamps, compounds, table)

    assert len(rows) == 30
    # Unshuffled, each row's return is keyed to its own compound - the two
    # must be nearly identical up to the construction's small noise term.
    for compound, ret in rows:
        assert abs(ret - 0.05 * compound) < 0.02


def test_run_planted_signal_control_detects_pre_shuffle_and_collapses_post_shuffle():
    result = run_planted_signal_control()

    assert result.unshuffled is not None
    assert result.shuffled is not None
    assert abs(result.unshuffled.r) > 0.9
    assert abs(result.shuffled.r) < 0.4
    assert planted_signal_passed(result)


def test_planted_signal_passed_is_false_when_shuffle_does_not_collapse_the_signal():
    from sentiment.audit import ControlResult
    from sentiment.stats import PearsonResult

    still_strong = ControlResult(
        label="x",
        unshuffled=PearsonResult(r=0.95, n=40, ci_low=0.9, ci_high=0.98),
        shuffled=PearsonResult(r=0.8, n=40, ci_low=0.6, ci_high=0.9),
    )
    assert planted_signal_passed(still_strong) is False


def test_planted_signal_passed_is_false_when_not_enough_rows():
    from sentiment.audit import ControlResult

    too_few = ControlResult(label="x", unshuffled=None, shuffled=None)
    assert planted_signal_passed(too_few) is False


def test_run_real_fixture_control_against_the_committed_fixture():
    result = run_real_fixture_control(in_path=DEFAULT_IN)

    assert result.unshuffled is not None
    assert result.shuffled is not None
    assert result.unshuffled.n == 23
    # Real data has no signal to begin with (Day 5 Findings) - shuffling
    # timestamps should not manufacture one. This is a sanity bound, not a
    # proof: see the module docstring for why this half is informational.
    assert abs(result.shuffled.r) < 0.6


def test_run_real_fixture_control_is_deterministic():
    a = run_real_fixture_control(in_path=DEFAULT_IN, seed=7)
    b = run_real_fixture_control(in_path=DEFAULT_IN, seed=7)

    assert a.unshuffled.r == b.unshuffled.r
    assert a.shuffled.r == b.shuffled.r


def test_main_exits_zero_and_writes_a_report(tmp_path, monkeypatch):
    from sentiment.audit import main

    monkeypatch.chdir(tmp_path)
    exit_code = main(["--output-dir", str(tmp_path / "outputs"), "--date", "2026-01-01"])

    assert exit_code == 0
    report = tmp_path / "outputs" / "audit_2026-01-01.md"
    assert report.exists()
    text = report.read_text()
    assert "planted-signal control" in text
    assert "real-fixture control" in text
