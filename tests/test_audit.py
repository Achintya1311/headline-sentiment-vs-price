import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import (
    AuditResult,
    ResolvedHeadline,
    correlation_for_timestamps,
    resolve_headlines,
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


def test_resolve_headlines_matches_correlate_build_rows_count():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    resolved = resolve_headlines(fixture)
    # Same 23 of 24 headlines correlate.build_rows resolves against the
    # committed fixture (LTIMindtree has no fetchable ticker - see tickers.py).
    assert len(resolved) == 23
    assert all(isinstance(r, ResolvedHeadline) for r in resolved)


def test_correlation_for_timestamps_matches_real_alignment(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0),
            Bar(date=dt.date(2026, 9, 29), open=105.0, close=103.0),
        ],
    )
    resolved = [
        ResolvedHeadline(
            title="Infosys Share Price Highlights",
            ticker="INFY.NS",
            compound=0.5,
            published_at=datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]
    # a single point can't correlate; add a second, different-return headline
    save_fixture(
        "WIPRO.NS",
        [Bar(date=dt.date(2026, 9, 28), open=50.0, close=45.0)],
    )
    resolved.append(
        ResolvedHeadline(
            title="Wipro Share Price Highlights",
            ticker="WIPRO.NS",
            compound=-0.5,
            published_at=datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    )

    bars = {"INFY.NS": prices.load_bars("INFY.NS"), "WIPRO.NS": prices.load_bars("WIPRO.NS")}
    result = correlation_for_timestamps(resolved, [r.published_at for r in resolved], bars)

    assert result is not None
    r, n = result
    assert n == 2
    # two points -> r is always exactly +1 or -1; compound and return both
    # move the same direction here (INFY up + positive compound, WIPRO down
    # + negative compound), so it's +1.
    assert r == 1.0


def test_correlation_for_timestamps_skips_tickers_with_no_bars():
    resolved = [
        ResolvedHeadline(
            title="x", ticker="MISSING.NS", compound=0.1, published_at=datetime(2026, 9, 28, tzinfo=timezone.utc)
        )
    ]
    result = correlation_for_timestamps(resolved, [r.published_at for r in resolved], {"MISSING.NS": None})
    assert result is None


def test_run_audit_is_deterministic_given_a_fixed_seed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    tickers = ["INFY.NS", "WIPRO.NS", "TATASTEEL.NS", "HCLTECH.NS"]
    for i, t in enumerate(tickers):
        save_fixture(
            t,
            [
                Bar(date=dt.date(2026, 9, 28), open=100.0, close=100.0 + (i - 1.5) * 2),
                Bar(date=dt.date(2026, 9, 29), open=100.0, close=100.0 - (i - 1.5)),
            ],
        )
    resolved = [
        ResolvedHeadline(
            title=f"headline {i}",
            ticker=t,
            compound=(i - 1.5) / 2,
            published_at=datetime(2026, 9, 28, 2 + i, 0, 0, tzinfo=timezone.utc),
        )
        for i, t in enumerate(tickers)
    ]

    result_a = run_audit(resolved, n_shuffles=200, seed=7)
    result_b = run_audit(resolved, n_shuffles=200, seed=7)

    assert result_a.real_r == result_b.real_r
    assert result_a.null_rs == result_b.null_rs
    assert result_a.p_value == result_b.p_value


def test_run_audit_rejects_too_few_resolved_headlines():
    resolved = [
        ResolvedHeadline(title="x", ticker="INFY.NS", compound=0.1, published_at=datetime(2026, 9, 28, tzinfo=timezone.utc))
    ]
    try:
        run_audit(resolved, n_shuffles=10, seed=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_audit_result_passes_when_real_r_is_unremarkable_against_the_null():
    result = AuditResult(real_r=0.1, n_real=10, null_rs=[-0.3, -0.1, 0.0, 0.1, 0.2, 0.3], p_value=0.5)
    assert result.passes is True
    assert result.null_mean == sum([-0.3, -0.1, 0.0, 0.1, 0.2, 0.3]) / 6


def test_audit_result_fails_when_real_r_is_an_outlier_against_the_null():
    # real_r sits far outside a tight null cluster around 0 - a p-value this
    # small should trip the FAIL branch.
    result = AuditResult(real_r=0.9, n_real=10, null_rs=[0.01, -0.02, 0.0, 0.01, -0.01], p_value=0.01)
    assert result.passes is False


def test_audit_result_null_percentile_is_a_fraction_in_zero_one():
    result = AuditResult(real_r=0.2, n_real=5, null_rs=[0.0, 0.1, 0.2, 0.3, 0.4], p_value=0.6)
    # |0.2| <= |0.0|, |0.1|, |0.2| -> 3 of 5 at or below -> 0.6
    assert result.null_percentile == 0.6


def test_run_against_committed_fixture_passes_with_the_default_seed():
    """The leakage test itself, run in CI rather than only by hand (see
    NEXT_STEPS.md's 'Done when'): the real, leak-free alignment's correlation
    must not be a statistical outlier against shuffled-timestamp controls.
    Fully deterministic given the default seed, so this asserts on the exact
    numbers rather than just "did not crash"."""
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    resolved = resolve_headlines(fixture)

    result = run_audit(resolved, n_shuffles=2000, seed=0)

    assert result.n_real == 23
    assert round(result.real_r, 3) == -0.185
    assert result.passes is True
    assert result.p_value > 0.05


def test_run_cli_exits_zero_on_the_committed_fixture(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, n_shuffles=2000, seed=0)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out
    assert "real (leak-free) alignment: r=-0.185" in captured.out


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, live=False, n_shuffles=10, seed=0)
    assert exit_code == 1


def test_run_with_too_few_resolved_headlines_reports_failure(tmp_path: Path):
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "LTIMindtree Share Price Highlights: LTIMindtree Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            )
        ],
        in_path,
    )

    exit_code = run(in_path, live=False, n_shuffles=10, seed=0)
    assert exit_code == 1
