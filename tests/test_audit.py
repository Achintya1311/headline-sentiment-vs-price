from datetime import datetime, timedelta, timezone
from pathlib import Path

import sentiment.audit as audit
from sentiment.audit import run, run_shuffle_audit, shuffle_changes_alignment, shuffle_timestamps
from sentiment.headline import Headline, write_csv


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


# Pre-open UTC times on distinct trading days, spread far enough apart that a
# random pairing essentially never reproduces the identity pairing.
_BASE_TIMESTAMPS = [
    datetime(2026, 9, d, 2, 0, 0, tzinfo=timezone.utc)
    for d in (1, 2, 3, 4, 7, 8, 9, 10, 11, 14)  # weekdays only
]


def make_fixture(n: int = 10) -> list[Headline]:
    return [make_headline(f"Synthetic headline {i}", str(i), _BASE_TIMESTAMPS[i]) for i in range(n)]


def test_shuffle_timestamps_preserves_the_multiset_but_changes_the_pairing():
    headlines = make_fixture()
    rng_fixed = __import__("random").Random(1)
    shuffled = shuffle_timestamps(headlines, rng_fixed)

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]
    # titles/links/scraped_at travel with the headline, only published_at moves
    assert [h.title for h in shuffled] == [h.title for h in headlines]


def test_shuffle_changes_alignment_true_for_a_real_reshuffle():
    import random

    headlines = make_fixture()
    shuffled = shuffle_timestamps(headlines, random.Random(1))
    assert shuffle_changes_alignment(headlines, shuffled) is True


def test_shuffle_changes_alignment_false_when_timestamps_are_all_identical():
    same_time = datetime(2026, 9, 1, 2, 0, 0, tzinfo=timezone.utc)
    headlines = [make_headline(f"H{i}", str(i), same_time) for i in range(5)]
    import random

    shuffled = shuffle_timestamps(headlines, random.Random(1))
    assert shuffle_changes_alignment(headlines, shuffled) is False


def test_vacuous_pass_when_real_result_is_not_significant(tmp_path: Path, monkeypatch):
    """A real result whose CI already contains zero has no claimed signal for
    a leak to have produced - the audit must pass without needing the
    shuffled distribution to look any particular way."""
    headlines = make_fixture()
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    def fake_build(hs, live=False, bars_cache=None, report_warnings=True):
        # No real relationship between compound and return at all.
        values = [0.1, -0.1, 0.05, -0.05, 0.0, 0.02, -0.02, 0.01, -0.01, 0.03]
        return [
            {"compound": v, "contemporaneous_return": -v if i % 2 else v}
            for i, v in enumerate(values[: len(hs)])
        ], []

    monkeypatch.setattr(audit, "build_rows_from_headlines", fake_build)

    result = run_shuffle_audit(in_path, n_shuffles=50, seed=0)

    assert result.leak_suspected is False


def test_flags_leak_when_the_result_does_not_depend_on_timestamp_at_all(tmp_path: Path, monkeypatch):
    """A broken pipeline that keys its output off headline identity (title)
    rather than the published_at it was actually given: shuffling timestamps
    changes nothing, so every shuffle reproduces the real correlation
    exactly. That is precisely the leak this audit exists to catch."""
    headlines = make_fixture()
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    title_to_value = {h.title: float(i) for i, h in enumerate(headlines)}

    def leaking_build(hs, live=False, bars_cache=None, report_warnings=True):
        rows = [
            {"compound": title_to_value[h.title], "contemporaneous_return": title_to_value[h.title]} for h in hs
        ]
        return rows, []

    monkeypatch.setattr(audit, "build_rows_from_headlines", leaking_build)

    result = run_shuffle_audit(in_path, n_shuffles=50, seed=0)

    assert result.real_significant is True
    assert result.p_value == 1.0  # every shuffle reproduces the same r
    assert result.leak_suspected is True


def test_passes_when_a_genuine_signal_degrades_under_shuffling(tmp_path: Path, monkeypatch):
    """A pipeline that correctly keys its return lookup off the (possibly
    shuffled) published_at it is given, while the compound score stays
    intrinsic to the headline: real (true-timestamp) pairing is perfectly
    correlated by construction, and shuffling should wreck it, because the
    sentiment score travels with the headline but the matched return now
    comes from a different, randomly assigned session."""
    headlines = make_fixture()
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    title_to_compound = {h.title: float(i) for i, h in enumerate(headlines)}
    timestamp_to_return = {h.published_at: float(i) for i, h in enumerate(headlines)}

    def genuine_build(hs, live=False, bars_cache=None, report_warnings=True):
        rows = [
            {
                "compound": title_to_compound[h.title],
                "contemporaneous_return": timestamp_to_return[h.published_at],
            }
            for h in hs
        ]
        return rows, []

    monkeypatch.setattr(audit, "build_rows_from_headlines", genuine_build)

    result = run_shuffle_audit(in_path, n_shuffles=200, seed=0)

    assert result.real_r == 1.0
    assert result.real_significant is True
    assert result.p_value is not None and result.p_value < 0.05
    assert result.leak_suspected is False
    # the real pairing is an outlier against the shuffled null, not buried in it
    assert abs(result.mean_shuffled_r) < 0.5


def test_run_against_committed_fixture_is_a_vacuous_pass(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_path = tmp_path / "shuffle_audit.json"

    exit_code = run(fixture, out_path, n_shuffles=100, seed=0, live=False)

    assert exit_code == 0
    assert out_path.exists()
    import json

    report = json.loads(out_path.read_text())
    assert report["real_n"] == 23
    assert report["real_significant"] is False
    assert report["leak_suspected"] is False


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out_path = tmp_path / "shuffle_audit.json"

    exit_code = run(missing, out_path, n_shuffles=10, seed=0, live=False)

    assert exit_code == 1
    assert not out_path.exists()
