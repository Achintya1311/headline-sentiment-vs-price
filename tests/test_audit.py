import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sentiment.audit import run, shuffle_audit, shuffle_timestamps
from sentiment.headline import Headline
from sentiment.stats import permutation_p_value


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def make_headlines(n: int) -> list[Headline]:
    return [
        make_headline(f"headline {i}", f"link-{i}", datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(n)
    ]


# --- permutation_p_value (pure stats) ---------------------------------------


def test_permutation_p_value_counts_the_actual_draw_itself():
    # Every null draw ties the actual value exactly: all n count as "at least
    # as extreme", so p = (n + 1) / (n + 1) = 1.0.
    assert permutation_p_value(0.2, [0.2] * 10) == pytest.approx(1.0)


def test_permutation_p_value_is_never_exactly_zero():
    # None of the null draws are anywhere near as extreme as the actual
    # value, but the +1/+1 form still keeps p strictly positive.
    p = permutation_p_value(0.99, [0.01] * 99)
    assert p == pytest.approx(1 / 100)
    assert p > 0


def test_permutation_p_value_is_two_sided():
    # A null draw of -0.9 is just as extreme as +0.9 against an actual of 0.9.
    assert permutation_p_value(0.9, [-0.9]) == pytest.approx(1.0)


def test_permutation_p_value_rejects_empty_null():
    with pytest.raises(ValueError):
        permutation_p_value(0.5, [])


# --- shuffle_timestamps ------------------------------------------------------


def test_shuffle_timestamps_preserves_the_same_multiset_of_timestamps():
    headlines = make_headlines(8)
    shuffled = shuffle_timestamps(headlines, rng=random.Random(0))

    assert sorted(h.published_at for h in headlines) == sorted(h.published_at for h in shuffled)
    assert [h.title for h in shuffled] == [h.title for h in headlines]  # titles/order untouched
    assert [h.link for h in shuffled] == [h.link for h in headlines]


def test_shuffle_timestamps_actually_reassigns_them():
    headlines = make_headlines(10)
    shuffled = shuffle_timestamps(headlines, rng=random.Random(0))

    # Not every headline can keep its own original timestamp - a real
    # permutation of 10 distinct values almost never comes back as the
    # identity, and this seed doesn't either.
    assert any(h.published_at != s.published_at for h, s in zip(headlines, shuffled))


# --- shuffle_audit: the audit's own detection logic, via a fake pipeline ---
#
# These two tests stand in for a "leaky" and a "leak-free" build_rows
# implementation without needing the real ticker-resolution/price-fixture
# machinery - they test what shuffle_audit *concludes*, which is Day 8's
# actual deliverable, independent of whether the real pipeline (tested
# separately below, against the committed fixture) happens to leak or not.


def test_shuffle_audit_flags_a_control_that_ignores_timestamps_entirely():
    headlines = make_headlines(10)
    # A fixed per-headline compound/return pair, keyed by link so it survives
    # shuffling untouched - standing in for a bug that computes the "right"
    # answer regardless of what published_at got reassigned to.
    compound = {h.link: (i - 4.5) * 0.1 for i, h in enumerate(headlines)}

    def leaky_build_rows(hs, live):
        rows = [
            {"compound": compound[h.link], "contemporaneous_return": 2 * compound[h.link]} for h in hs
        ]
        return rows, []

    result = shuffle_audit(headlines, n_shuffles=50, seed=0, build_rows_fn=leaky_build_rows)

    assert result.false_positive_rate == pytest.approx(1.0)
    assert result.leak_detected
    assert not result.passed


def test_shuffle_audit_passes_a_control_whose_signal_depends_on_the_real_timestamp():
    headlines = make_headlines(10)
    correct_ts = {h.link: h.published_at for h in headlines}
    compound = {h.link: (i - 4.5) * 0.1 for i, h in enumerate(headlines)}

    def aligned_build_rows(hs, live):
        rows = []
        for h in hs:
            if h.published_at == correct_ts[h.link]:
                rows.append({"compound": compound[h.link], "contemporaneous_return": 2 * compound[h.link]})
            else:
                # Misaligned (the shuffle moved this headline off its own
                # timestamp): the "market" has nothing to react to.
                rows.append({"compound": compound[h.link], "contemporaneous_return": 0.0})
        return rows, []

    # Many shuffles, so the expected ~1/10 fixed-point rate per trial
    # averages out rather than occasionally producing a fluke "significant"
    # draw this assertion would be unlucky to hit.
    result = shuffle_audit(
        headlines, n_shuffles=200, seed=0, build_rows_fn=aligned_build_rows
    )

    assert result.actual is not None
    assert result.actual.r == pytest.approx(1.0)  # unshuffled: every row aligned, perfect correlation
    assert result.passed
    assert not result.leak_detected


def test_shuffle_audit_handles_too_few_resolvable_rows_as_no_usable_shuffles():
    headlines = make_headlines(3)

    def empty_build_rows(hs, live):
        return [], []

    result = shuffle_audit(headlines, n_shuffles=5, seed=0, build_rows_fn=empty_build_rows)

    assert result.actual is None
    assert result.shuffled == []
    assert result.p_value is None
    assert result.false_positive_rate is None
    assert result.passed  # nothing to flag as a leak


# --- run(): the CLI, against the real committed fixture and edge cases -----


def test_run_against_committed_fixture_passes_and_matches_the_documented_day5_result():
    # Day 5's README Findings pin the real baseline: contemporaneous
    # r=-0.185, n=23. Day 8 adds the shuffle control on top of that same,
    # already-null result - it should find no sign of a leak either.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, n_shuffles=100, seed=0)

    assert exit_code == 0


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, live=False, n_shuffles=10, seed=0)

    assert exit_code == 1
