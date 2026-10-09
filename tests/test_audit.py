import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    AuditResult,
    contemporaneous_r,
    permutation_p_value,
    run,
    run_audit,
    shuffle_timestamps,
)
from sentiment.correlate import DEFAULT_IN
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


# --- shuffle_timestamps -------------------------------------------------


def test_shuffle_timestamps_preserves_the_multiset_of_times():
    import random

    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # every title is still present - only timestamps moved, nothing dropped
    assert {h.title for h in shuffled} == {h.title for h in headlines}


def test_shuffle_timestamps_result_is_sorted_oldest_first():
    import random

    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i in range(6)
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(42))

    times = [h.published_at for h in shuffled]
    assert times == sorted(times)


def test_shuffle_timestamps_is_deterministic_for_a_given_rng_state():
    import random

    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i in range(8)
    ]
    first = shuffle_timestamps(headlines, random.Random(7))
    second = shuffle_timestamps(headlines, random.Random(7))

    assert [h.title for h in first] == [h.title for h in second]


# --- permutation_p_value -------------------------------------------------


def test_permutation_p_value_is_small_when_real_stat_is_an_outlier():
    null = [0.01, -0.02, 0.03, -0.01, 0.02] * 20  # 100 draws clustered near 0
    p = permutation_p_value(real_stat=0.9, null_stats=null)
    assert p < 0.05


def test_permutation_p_value_is_large_when_real_stat_is_typical():
    null = [0.1, -0.1, 0.15, -0.15, 0.2, -0.2] * 20
    p = permutation_p_value(real_stat=0.12, null_stats=null)
    assert p > 0.5


def test_permutation_p_value_uses_plus_one_correction_never_reports_zero():
    null = [0.0] * 50
    p = permutation_p_value(real_stat=0.9, null_stats=null)
    assert p == pytest.approx(1 / 51)


def test_permutation_p_value_requires_at_least_one_null_draw():
    with pytest.raises(ValueError):
        permutation_p_value(real_stat=0.5, null_stats=[])


# --- contemporaneous_r ---------------------------------------------------


def test_contemporaneous_r_none_below_two_rows():
    assert contemporaneous_r([]) is None
    assert contemporaneous_r([{"compound": 0.5, "contemporaneous_return": 0.01}]) is None


def test_contemporaneous_r_matches_plain_pearson():
    rows = [
        {"compound": 0.1, "contemporaneous_return": 0.01},
        {"compound": 0.5, "contemporaneous_return": 0.03},
        {"compound": -0.3, "contemporaneous_return": -0.02},
    ]
    from sentiment.stats import pearson_r

    expected = pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])
    assert contemporaneous_r(rows) == expected


# --- run_audit: a genuine, timing-dependent signal must be flagged -------


def _twelve_headline_fixture(tmp_path: Path) -> Path:
    """Six strongly-positive and six strongly-negative headlines, each
    naming a distinct real ticker, timed so that - and only so that - the
    real published_at values pair positive sentiment with a +5% session and
    negative sentiment with a -5% session. Every ticker's price fixture
    carries both sessions, so a shuffle changes which return a headline's
    (fixed) sentiment is paired with, not whether a bar exists at all.

    This is a synthetic stress-test fixture, not research data: it exists
    to prove the permutation test in sentiment.audit actually has the power
    to flag a real timing-dependent relationship, not just to pass quietly
    on the real near-null fixture (see test_run_audit_against_... below).
    """
    positive = [
        ("Infosys Share Price Highlights: markets jubilant as profit soars, outlook turns excellent", "INFY.NS"),
        ("Wipro Share Price Highlights: investors delighted as results beat and outlook shines bright", "WIPRO.NS"),
        ("HCL Tech Share Price Highlights: fantastic quarter, strong upgrade and wonderful outlook", "HCLTECH.NS"),
        ("Tech Mahindra Share Price Highlights: brilliant rally as profit surges and guidance improves", "TECHM.NS"),
        ("SBI Life Share Price Highlights: superb results cheer investors, outlook turns optimistic", "SBILIFE.NS"),
        ("Nestle India Share Price Highlights: terrific growth and upgrade lift shares, outlook bright", "NESTLEIND.NS"),
    ]
    negative = [
        ("Sun Pharma Share Price Highlights: disastrous quarter, profit crashes, outlook turns awful and grim", "SUNPHARMA.NS"),
        ("Grasim Inds Share Price Highlights: dismal results and awful guidance cut hurt investors badly", "GRASIM.NS"),
        ("Bharti Airtel Share Price Highlights: disastrous quarter as losses mount and outlook worsens", "BHARTIARTL.NS"),
        ("Tata Steel Share Price Highlights: horrible results, outlook turns grim amid weak demand", "TATASTEEL.NS"),
        ("HUL Share Price Highlights: awful quarter, profit crashes badly, outlook turns grim and dismal", "HINDUNILVR.NS"),
        ("HDFC Life Share Price Highlights: poor results and bleak outlook spook investors badly", "HDFCLIFE.NS"),
    ]

    for _, ticker in positive + negative:
        save_fixture(
            ticker,
            [
                Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0),  # +5%
                Bar(date=dt.date(2026, 9, 29), open=100.0, close=95.0),  # -5%
            ],
        )

    headlines = [
        # pre-open on Mon 28th -> aligns to the 28th's +5% session
        make_headline(title, f"p{i}", datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc))
        for i, (title, _) in enumerate(positive)
    ] + [
        # intraday on Mon 28th -> rolls to Tue 29th's -5% session
        make_headline(title, f"n{i}", datetime(2026, 9, 28, 5, i, 0, tzinfo=timezone.utc))
        for i, (title, _) in enumerate(negative)
    ]

    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)
    return in_path


def test_run_audit_flags_a_genuine_timing_dependent_signal(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = _twelve_headline_fixture(tmp_path)

    result = run_audit(in_path, trials=300, seed=0)

    assert result is not None
    assert result.real_n == 12
    # strong, sign-consistent pairing between sentiment and return
    assert result.real_r > 0.9
    # a shuffled-timestamp control essentially never reproduces that by chance
    assert not result.passes()
    assert result.p_value < 0.05


def test_run_audit_null_distribution_is_centred_near_zero(tmp_path: Path, monkeypatch):
    """Sanity check on the null itself: scrambling which session a fixed
    sentiment score lands on should average out to roughly no relationship,
    not a systematic bias in one direction - a biased null would mean the
    audit's control is not a fair one."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = _twelve_headline_fixture(tmp_path)

    result = run_audit(in_path, trials=300, seed=0)
    assert abs(result.null_mean) < 0.2


# --- run_audit / run against the real committed fixture ------------------


def test_run_audit_against_the_real_fixture_is_this_repos_leakage_gate():
    """This is the "Done when" test NEXT_STEPS.md set before Day 1: the
    real pipeline's contemporaneous correlation (Day 5's r=-0.185, n=23)
    must not be distinguishable from what shuffling headline timestamps
    alone produces. It runs here, in the suite pytest runs every time - not
    once by hand - which is the whole point.

    A future, wider scrape could legitimately make this fail (a lot more
    headlines could surface a real, non-spurious relationship); if it does,
    that is a finding to investigate, not a bug to silence by deleting the
    test - see README "Why this result might still be spurious".
    """
    result = run_audit(DEFAULT_IN, trials=500, seed=0)

    assert result is not None
    assert result.real_n == 23
    assert result.real_r == pytest.approx(-0.185, abs=0.01)
    assert result.passes()
    assert result.p_value > 0.05


def test_run_cli_exits_zero_when_signal_does_not_survive_the_shuffle():
    assert run(DEFAULT_IN, live=False, trials=500, seed=0) == 0


def test_run_cli_exits_nonzero_when_too_few_headlines_to_audit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    in_path = tmp_path / "raw.csv"
    write_csv(
        [make_headline("Infosys Share Price Highlights: Infosys Stock Price History", "1", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc))],
        in_path,
    )

    assert run(in_path, live=False, trials=50, seed=0) == 1


def test_run_cli_reports_missing_input_file(tmp_path: Path):
    assert run(tmp_path / "does_not_exist.csv", live=False, trials=50, seed=0) == 1
