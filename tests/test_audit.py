import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    Resolved,
    contemporaneous_return,
    correlation_for_timestamps,
    resolve_headlines,
    run,
    run_shuffle_test,
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


def test_resolve_headlines_attaches_compound_ticker_and_bars(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)],
    )
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Infosys Share Price Highlights: Infosys Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            ),
            # unresolvable company -> excluded, same as sentiment.correlate
            make_headline(
                "Cyient among 4 stocks showing White Marubozu Pattern",
                "2",
                datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc),
            ),
        ],
        in_path,
    )

    resolved = resolve_headlines(in_path)

    assert len(resolved) == 1
    r = resolved[0]
    assert r.ticker == "INFY.NS"
    assert r.bars[0].open == 100.0
    assert isinstance(r.compound, float)


def test_contemporaneous_return_uses_the_given_timestamp_not_the_headlines_own(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    bars = [
        Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0),
        Bar(date=dt.date(2026, 9, 29), open=100.0, close=90.0),
    ]
    headline = Resolved(
        title="x",
        ticker="X.NS",
        compound=0.5,
        published_at=datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),  # pre-open 28th IST
        bars=bars,
    )

    # its own timestamp aligns to the 28th session (+10%)
    assert contemporaneous_return(headline, headline.published_at) == pytest.approx(0.10)
    # a different (later, intraday) timestamp aligns to the 29th session (-10%) instead
    later = datetime(2026, 9, 28, 10, 0, 0, tzinfo=timezone.utc)  # 15:30 IST, post-close
    assert contemporaneous_return(headline, later) == pytest.approx(-0.10)


def test_correlation_for_timestamps_matches_plain_pearson_r():
    bars = [Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0)]
    resolved = [
        Resolved(title="a", ticker="A", compound=1.0, published_at=datetime(2026, 9, 28, 2, tzinfo=timezone.utc), bars=bars),
        Resolved(title="b", ticker="B", compound=-1.0, published_at=datetime(2026, 9, 28, 2, tzinfo=timezone.utc), bars=bars),
    ]
    timestamps = [r.published_at for r in resolved]

    result = correlation_for_timestamps(resolved, timestamps)

    assert result is not None
    r, n = result
    assert n == 2
    # both headlines land on the same single-bar session with the same return,
    # so there is no variation in y - pearson_r's documented zero-variance case.
    assert r == 0.0


def test_correlation_for_timestamps_returns_none_below_min_points():
    bars = [Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0)]
    resolved = [
        Resolved(title="a", ticker="A", compound=1.0, published_at=datetime(2026, 9, 28, 2, tzinfo=timezone.utc), bars=bars)
    ]
    assert correlation_for_timestamps(resolved, [r.published_at for r in resolved]) is None


def test_run_shuffle_test_is_deterministic_for_a_fixed_seed():
    bars_up = [Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0), Bar(date=dt.date(2026, 9, 29), open=100.0, close=95.0)]
    bars_down = [Bar(date=dt.date(2026, 9, 28), open=100.0, close=90.0), Bar(date=dt.date(2026, 9, 29), open=100.0, close=105.0)]
    pre_open_28 = datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)
    intraday_28 = datetime(2026, 9, 28, 10, 0, 0, tzinfo=timezone.utc)  # rolls to the 29th
    resolved = [
        Resolved(title="a", ticker="A", compound=0.9, published_at=pre_open_28, bars=bars_up),
        Resolved(title="b", ticker="B", compound=-0.9, published_at=intraday_28, bars=bars_down),
        Resolved(title="c", ticker="C", compound=0.1, published_at=pre_open_28, bars=bars_up),
        Resolved(title="d", ticker="D", compound=-0.1, published_at=intraday_28, bars=bars_down),
    ]

    first = run_shuffle_test(resolved, n_shuffles=200, seed=42)
    second = run_shuffle_test(resolved, n_shuffles=200, seed=42)

    assert first.real_r == second.real_r
    assert first.null_rs == second.null_rs
    assert 0.0 <= first.p_value <= 1.0
    assert first.real_n == 4


def test_run_shuffle_test_raises_when_not_enough_resolved_headlines():
    bars = [Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0)]
    resolved = [
        Resolved(title="a", ticker="A", compound=1.0, published_at=datetime(2026, 9, 28, 2, tzinfo=timezone.utc), bars=bars)
    ]
    with pytest.raises(ValueError):
        run_shuffle_test(resolved, n_shuffles=10, seed=0)


def test_cli_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    exit_code = run(missing, live=False, n_shuffles=10, seed=0)
    assert exit_code == 1


def test_cli_run_prints_real_r_and_p_value(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0),
            Bar(date=dt.date(2026, 9, 29), open=105.0, close=103.0),
        ],
    )
    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            make_headline(
                "Infosys Share Price Highlights: Infosys Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            ),
            make_headline(
                "HDFC Bank shares rally after results",
                "2",
                datetime(2026, 9, 28, 10, 0, 0, tzinfo=timezone.utc),
            ),
        ],
        in_path,
    )
    monkeypatch.setattr(
        "sentiment.audit.resolve", lambda title: ("Infosys", "INFY.NS") if "Infosys" in title else ("HDFC Bank", "INFY.NS")
    )

    exit_code = run(in_path, live=False, n_shuffles=50, seed=0)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "real (unshuffled) contemporaneous: r=" in out
    assert "two-sided p-value" in out


def test_cli_run_against_committed_fixture_reproduces_day5_correlation(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, live=False, n_shuffles=200, seed=0)

    assert exit_code == 0
    out = capsys.readouterr().out
    # Day 5's README-reported contemporaneous correlation on the real fixture.
    assert "r=-0.185" in out
    assert "n=23" in out
