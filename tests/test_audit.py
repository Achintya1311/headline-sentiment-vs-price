from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    ResolvedHeadline,
    build_resolved,
    contemporaneous_return,
    correlate_with_timestamps,
    permutation_null,
    permutation_p_value,
    run,
    run_audit,
)
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


# A pre-open UTC timestamp aligns to the same IST calendar date (02:00 UTC
# is 07:30 IST, before the 09:15 open) - see sentiment/market_hours.py.
def pre_open(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 2, 0, 0, tzinfo=timezone.utc)


def test_build_resolved_keeps_only_single_ticker_headlines(tmp_path: Path):
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Infosys Share Price Highlights: Infosys Stock Price History",
                "1",
                pre_open(date(2026, 9, 28)),
            ),
            make_headline("Cyient among 4 stocks showing White Marubozu Pattern", "2", pre_open(date(2026, 9, 28))),
            make_headline(
                "LTIMindtree Share Price Highlights: LTIMindtree Stock Price History",
                "3",
                pre_open(date(2026, 9, 28)),
            ),
        ],
        in_path,
    )

    resolved = build_resolved(in_path)

    assert len(resolved) == 1
    assert resolved[0].ticker == "INFY.NS"


def test_contemporaneous_return_none_when_ticker_has_no_fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    ret = contemporaneous_return("NOFIXTURE.NS", pre_open(date(2026, 9, 28)), {}, live=False)
    assert ret is None


def test_correlate_with_timestamps_reproduces_pearson_r(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    # close = open + compound * 10 for every bar, so return = compound / 10
    # exactly - a perfectly linear relationship regardless of which 4 of
    # these dates/compounds get paired, as long as they're paired correctly.
    compounds = [0.1, 0.2, 0.3, 0.4]
    dates = [date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)]
    save_fixture(
        "A.NS",
        [Bar(date=d, open=100.0, close=100.0 + c * 10) for d, c in zip(dates, compounds)],
    )
    resolved = [
        ResolvedHeadline(title=f"h{i}", ticker="A.NS", compound=c, published_at=pre_open(d))
        for i, (c, d) in enumerate(zip(compounds, dates))
    ]

    result = correlate_with_timestamps(resolved, [rh.published_at for rh in resolved], {}, live=False)

    assert result is not None
    assert result.r == pytest.approx(1.0)  # compound and return move in lockstep by construction
    assert result.n == 4


def test_correlate_with_timestamps_returns_none_below_min_rows(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("A.NS", [Bar(date=date(2026, 9, 21), open=100.0, close=101.0)])
    resolved = [ResolvedHeadline(title="h0", ticker="A.NS", compound=0.5, published_at=pre_open(date(2026, 9, 21)))]

    result = correlate_with_timestamps(resolved, [rh.published_at for rh in resolved], {}, live=False)

    assert result is None


def test_permutation_p_value_is_small_when_observed_is_an_outlier():
    # The real r (1.0) is more extreme than every value in a null clustered near 0.
    null_rs = [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.03, 0.01]
    p = permutation_p_value(1.0, null_rs)
    assert p == (0 + 1) / (len(null_rs) + 1)


def test_permutation_p_value_is_large_when_observed_sits_inside_the_null():
    null_rs = [-0.9, -0.5, 0.0, 0.5, 0.9]
    p = permutation_p_value(0.1, null_rs)
    assert p >= 0.6  # every null value here is at least as extreme as 0.1


def _synthetic_leaky_bars_and_resolved(tmp_path: Path) -> list[ResolvedHeadline]:
    """A ``ResolvedHeadline`` list engineered so compound predicts the
    *correctly-aligned* return almost perfectly - a genuine,
    timestamp-mediated signal the shuffle control must be able to detect,
    not just the real repo's own null case. Each headline is published
    pre-open on its own distinct trading day with a distinct compound that
    tracks that day's bar exactly, so shuffling ``published_at`` across
    headlines reassigns which bar's return a headline's compound is paired
    with - the same mechanism a shuffle exercises against the real
    pipeline, just engineered to contain a real effect."""
    base = date(2026, 9, 7)  # a Monday
    days = [base + timedelta(days=i) for i in range(10) if (base + timedelta(days=i)).weekday() < 5][:8]
    compounds = [-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7]
    bars = [Bar(date=d, open=100.0, close=100.0 + c * 10) for d, c in zip(days, compounds)]
    save_fixture("LEAKY.NS", bars)
    return [
        ResolvedHeadline(title=f"leaky-{i}", ticker="LEAKY.NS", compound=c, published_at=pre_open(d))
        for i, (c, d) in enumerate(zip(compounds, days))
    ]


def test_shuffle_destroys_a_real_timestamp_mediated_signal(tmp_path: Path, monkeypatch):
    """Positive control for the audit itself: if sentiment really does
    predict returns only because the timestamp honestly selects the right
    bar, randomising timestamps must wreck that correlation. This is the
    exact "Done when" claim NEXT_STEPS.md makes, proven against data built
    to contain a real effect rather than the repo's own (already null)
    committed data - see test_audit_on_committed_fixture_finds_no_leak for
    that case."""
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    resolved = _synthetic_leaky_bars_and_resolved(tmp_path)

    bars_cache: dict = {}
    observed = correlate_with_timestamps(resolved, [rh.published_at for rh in resolved], bars_cache, live=False)
    assert observed is not None
    assert observed.r == pytest.approx(1.0)  # the engineered, correctly-aligned signal is exact

    null_results = permutation_null(resolved, bars_cache, live=False, n_permutations=200, seed=0)
    null_rs = [r.r for r in null_results]
    p_value = permutation_p_value(observed.r, null_rs)

    # The real, correctly-aligned r must be an outlier against the shuffled
    # null - i.e. the shuffle can tell a real effect apart from noise.
    assert p_value < 0.05
    # And shuffling itself must have pulled the correlation away from the
    # real value on average - the signal "disappeared" under randomisation.
    assert sum(null_rs) / len(null_rs) < abs(observed.r) / 2


def test_audit_on_committed_fixture_finds_no_leak():
    # The real, honest case: Day 5/6's own findings already show this
    # fixture's contemporaneous correlation is not significant (CI contains
    # zero). The audit's job is to confirm that null is not itself an
    # artifact of a broken shuffle - the real r should sit comfortably
    # inside the shuffled-timestamp null, not be an outlier against it.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    result = run_audit(fixture, live=False, n_permutations=500, seed=0)

    assert result is not None
    assert result.observed.n == 23
    assert not result.leaking
    assert result.p_value >= 0.05


def test_run_prints_pass_and_returns_zero_on_the_committed_fixture(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, n_permutations=200, seed=0)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "PASS" in out
    assert "real alignment" in out


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, live=False, n_permutations=100, seed=0)

    assert exit_code == 1
