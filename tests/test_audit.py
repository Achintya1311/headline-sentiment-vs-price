from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    MIN_PAIRS,
    Resolved,
    contemporaneous_pairs,
    load_resolved,
    run,
    run_permutation_test,
)
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture

IST = timezone(timedelta(hours=5, minutes=30))


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


# --- contemporaneous_pairs -------------------------------------------------


def test_contemporaneous_pairs_uses_each_headlines_own_ticker_and_timestamp():
    bars = [Bar(date=datetime(2026, 9, 28).date(), open=100.0, close=110.0)]
    resolved = [
        Resolved(
            title="h",
            ticker="T",
            compound=0.5,
            published_at=datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),  # pre-open IST
            bars=bars,
        )
    ]

    compounds, returns = contemporaneous_pairs(resolved, [r.published_at for r in resolved])

    assert compounds == [0.5]
    assert returns == pytest.approx([0.10])


def test_contemporaneous_pairs_drops_a_pair_with_no_bar_for_the_assigned_session():
    bars = [Bar(date=datetime(2026, 9, 28).date(), open=100.0, close=110.0)]
    resolved = [
        Resolved(
            title="h",
            ticker="T",
            compound=0.5,
            published_at=datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            bars=bars,
        )
    ]
    # A shuffled timestamp landing on a session this ticker has no bar for -
    # the pair drops out rather than crashing, same as a real missing bar.
    other_day = datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc)

    compounds, returns = contemporaneous_pairs(resolved, [other_day])

    assert compounds == []
    assert returns == []


def test_contemporaneous_pairs_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        contemporaneous_pairs([], [datetime(2026, 9, 28, tzinfo=timezone.utc)])


# --- run_permutation_test --------------------------------------------------


def _synthetic_resolved_with_designed_signal() -> list[Resolved]:
    """6 headlines, each with its own ticker and its own trading day. Each
    ticker's bar on its *own* day is built so contemporaneous return = 0.1 *
    compound exactly - a perfect, noise-free signal under the real
    (identity) timestamp assignment. Each ticker also carries bars on every
    *other* headline's day, set from a fixed sequence that has no relationship
    to compound - so a shuffle that reassigns timestamps still finds a bar
    (no degenerate pairs) but pairs each ticker with an unrelated return.
    """
    dates = [
        datetime(2026, 9, 7).date(),
        datetime(2026, 9, 8).date(),
        datetime(2026, 9, 9).date(),
        datetime(2026, 9, 10).date(),
        datetime(2026, 9, 11).date(),
        datetime(2026, 9, 14).date(),
    ]
    compounds = [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]
    designed_returns = [0.1 * c for c in compounds]
    unrelated_returns = [0.003, -0.010, 0.007, -0.002, 0.0125, -0.008]

    resolved = []
    for i in range(6):
        bars = []
        for j in range(6):
            r = designed_returns[i] if j == i else unrelated_returns[j]
            bars.append(Bar(date=dates[j], open=100.0, close=100.0 * (1 + r)))
        resolved.append(
            Resolved(
                title=f"headline {i}",
                ticker=f"T{i}",
                compound=compounds[i],
                published_at=datetime(dates[i].year, dates[i].month, dates[i].day, 2, 0, 0, tzinfo=timezone.utc),
                bars=bars,
            )
        )
    return resolved


def test_run_permutation_test_finds_the_real_signal_under_identity_assignment():
    resolved = _synthetic_resolved_with_designed_signal()

    result = run_permutation_test(resolved, n_shuffles=500, seed=0)

    assert result.real_r == pytest.approx(1.0, abs=1e-6)
    assert result.real_n == 6


def test_run_permutation_test_shuffle_mostly_destroys_a_real_signal():
    # The point of the audit: once timestamps are shuffled across headlines,
    # the near-perfect real correlation should not survive - the null
    # distribution of |r| should sit far below the real value.
    resolved = _synthetic_resolved_with_designed_signal()

    result = run_permutation_test(resolved, n_shuffles=500, seed=0)

    mean_abs_shuffled = sum(abs(r) for r in result.shuffled_rs) / len(result.shuffled_rs)
    assert mean_abs_shuffled < 0.5  # far below the real |r| = 1.0
    assert result.p_value < 0.05  # the real result is an outlier against the null distribution
    assert result.n_shuffles_degenerate == 0  # every ticker has a bar for every date by construction


def test_run_permutation_test_raises_below_min_pairs():
    bars = [Bar(date=datetime(2026, 9, 28).date(), open=100.0, close=101.0)]
    resolved = [
        Resolved(title="h", ticker="T", compound=0.1, published_at=datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc), bars=bars)
    ]
    with pytest.raises(ValueError):
        run_permutation_test(resolved, n_shuffles=10, seed=0)


# --- load_resolved / run (CLI) ---------------------------------------------


def test_load_resolved_skips_unresolved_and_untickered_headlines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=datetime(2026, 9, 28).date(), open=100.0, close=105.0)])
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Infosys Share Price Highlights: Infosys Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            ),
            make_headline(
                "Cyient among 4 stocks showing White Marubozu Pattern",  # not a single-company headline
                "2",
                datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc),
            ),
            make_headline(
                "LTIMindtree Share Price Highlights: LTIMindtree Stock Price History",  # no fetchable ticker
                "3",
                datetime(2026, 9, 28, 4, 0, 0, tzinfo=timezone.utc),
            ),
        ],
        in_path,
    )

    resolved = load_resolved(in_path)

    assert len(resolved) == 1
    assert resolved[0].ticker == "INFY.NS"
    assert resolved[0].compound == pytest.approx(0.296, abs=0.01)


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, live=False, n_shuffles=10, seed=0)
    assert exit_code == 1


def test_run_reports_failure_rather_than_crashing_when_too_few_resolved_headlines(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=datetime(2026, 9, 28).date(), open=100.0, close=105.0)])
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Infosys Share Price Highlights: Infosys Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            )
        ],
        in_path,
    )

    exit_code = run(in_path, live=False, n_shuffles=10, seed=0)

    assert exit_code == 1  # only 1 resolved headline, below MIN_PAIRS


def test_run_against_the_committed_fixture_is_not_a_leaking_signal():
    # The real, honest result (see README): Day 5 already found contemporaneous
    # r=-0.185 with a 95% CI comfortably containing zero. The audit's job here
    # is to confirm that result is not an outlier against shuffled-timestamp
    # controls - i.e. there is no detectable leak making the real pairing look
    # artificially strong.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, n_shuffles=500, seed=0)

    assert exit_code == 0


def test_load_resolved_against_committed_fixture_matches_correlates_count():
    # sentiment.correlate.build_rows finds 23 resolved-with-a-bar headlines
    # against this same fixture (see test_correlate.py) - the audit's
    # resolution step should agree, since it uses the same resolve()/load_bars().
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    resolved = load_resolved(fixture)
    assert len(resolved) == 23


def test_run_permutation_test_against_committed_fixture_real_r_matches_day5(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    resolved = load_resolved(fixture)

    result = run_permutation_test(resolved, n_shuffles=500, seed=0)

    assert result.real_r == pytest.approx(-0.185, abs=0.01)
    assert result.real_n == 23
    assert result.n_shuffles_degenerate == 0  # every headline's timestamp lands in-range for every ticker here
    assert result.p_value > 0.05  # not distinguishable from the shuffled null - see README Findings
