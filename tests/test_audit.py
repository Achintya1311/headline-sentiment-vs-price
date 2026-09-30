import random
from datetime import datetime, timezone
from pathlib import Path

from sentiment.audit import (
    AuditResult,
    correlation_r,
    run_audit,
    shuffle_timestamps,
    write_null_distribution_csv,
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


def test_shuffle_timestamps_preserves_the_set_of_timestamps_and_other_fields():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.source for h in shuffled] == [h.source for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert [h.scraped_at for h in shuffled] == [h.scraped_at for h in headlines]
    # a real permutation, not a no-op - with 10 items a fixed seed should not
    # land on the identity permutation
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_correlation_r_needs_at_least_two_usable_points():
    rows = [{"compound": 0.5, "contemporaneous_return": 0.01}]
    assert correlation_r(rows, "contemporaneous_return") is None
    assert correlation_r([], "contemporaneous_return") is None


def test_correlation_r_skips_none_lagged_returns():
    rows = [
        {"compound": 0.1, "lagged_return": 0.01},
        {"compound": 0.2, "lagged_return": None},
        {"compound": 0.3, "lagged_return": 0.02},
    ]
    # only the 2 rows with a non-None lagged_return are usable
    r = correlation_r(rows, "lagged_return")
    assert r is not None


def test_correlation_r_matches_a_hand_computed_perfect_line():
    rows = [
        {"compound": 0.0, "contemporaneous_return": 0.0},
        {"compound": 1.0, "contemporaneous_return": 1.0},
        {"compound": 2.0, "contemporaneous_return": 2.0},
    ]
    assert correlation_r(rows, "contemporaneous_return") == 1.0


def test_p_value_none_when_no_real_result_or_no_null():
    empty = AuditResult(
        real_contemp_r=None,
        real_contemp_n=0,
        real_lagged_r=None,
        real_lagged_n=0,
        shuffled_contemp_r=[],
        shuffled_lagged_r=[],
        n_shuffles=0,
        seed=0,
    )
    assert empty.p_value("contemporaneous") is None
    assert empty.p_value("lagged") is None


def test_p_value_is_one_when_real_r_is_the_smallest_magnitude_in_the_null():
    result = AuditResult(
        real_contemp_r=0.01,
        real_contemp_n=10,
        real_lagged_r=None,
        real_lagged_n=0,
        shuffled_contemp_r=[0.5, -0.4, 0.3, -0.2],
        shuffled_lagged_r=[],
        n_shuffles=4,
        seed=0,
    )
    assert result.p_value("contemporaneous") == 1.0


def test_p_value_is_small_when_real_r_dwarfs_the_null():
    result = AuditResult(
        real_contemp_r=0.99,
        real_contemp_n=10,
        real_lagged_r=None,
        real_lagged_n=0,
        shuffled_contemp_r=[0.01, -0.02, 0.03, -0.01, 0.02] * 20,
        shuffled_lagged_r=[],
        n_shuffles=100,
        seed=0,
    )
    p = result.p_value("contemporaneous")
    assert p == 1 / 101


def test_run_audit_against_committed_fixture_matches_correlate_and_is_reproducible():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    result = run_audit(fixture, n_shuffles=25, seed=0)

    # Day 5's own committed-fixture numbers (README Findings): 23 resolved
    # headlines contemporaneous, 15 with a computable lagged return.
    assert result.real_contemp_n == 23
    assert result.real_lagged_n == 15
    assert result.real_contemp_r is not None
    assert abs(result.real_contemp_r - (-0.185)) < 0.001
    assert result.real_lagged_r == 0.0

    assert len(result.shuffled_contemp_r) == 25
    assert 0.0 <= result.p_value("contemporaneous") <= 1.0
    assert 0.0 <= result.p_value("lagged") <= 1.0

    # deterministic given a fixed seed
    again = run_audit(fixture, n_shuffles=25, seed=0)
    assert result.shuffled_contemp_r == again.shuffled_contemp_r
    assert result.shuffled_lagged_r == again.shuffled_lagged_r


def test_run_audit_correctness_gate_passes_on_this_fixture():
    """The actual leakage/leakage-audit gate: on this fixture, the real
    (leak-free, correctly-timestamped) correlation is not a statistical
    outlier against a timestamp-shuffled null - there is no sign the
    pipeline is exploiting something only the correct timestamp could give
    it, because (per README Findings) there was never a real correlation
    for a look-ahead bug to have inflated in the first place."""
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    result = run_audit(fixture, n_shuffles=500, seed=0)

    assert result.p_value("contemporaneous") >= 0.05
    assert result.p_value("lagged") >= 0.05


def test_write_null_distribution_csv(tmp_path):
    result = AuditResult(
        real_contemp_r=0.1,
        real_contemp_n=5,
        real_lagged_r=0.2,
        real_lagged_n=3,
        shuffled_contemp_r=[0.1, 0.2],
        shuffled_lagged_r=[0.3],
        n_shuffles=2,
        seed=0,
    )
    out = tmp_path / "null.csv"
    write_null_distribution_csv(result, out)

    import csv

    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["contemporaneous_r"] == "0.1"
    assert rows[0]["lagged_r"] == "0.3"
    assert rows[1]["lagged_r"] == ""
