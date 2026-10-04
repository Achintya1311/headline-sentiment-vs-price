import datetime as dt
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.audit as audit
import sentiment.prices as prices
from sentiment.audit import AuditResult, correlation_for, run, run_audit, shuffle_timestamps
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture
from sentiment.stats import PearsonResult


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


# --- shuffle_timestamps -----------------------------------------------------


def test_shuffle_timestamps_preserves_the_multiset_of_timestamps():
    headlines = [
        make_headline("a", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 9, 28, 5, 0, 0, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 9, 29, 10, 0, 0, tzinfo=timezone.utc)),
    ]

    shuffled = shuffle_timestamps(headlines, seed=1)

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links/scraped_at untouched - only the timestamp pairing moves
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert [h.scraped_at for h in shuffled] == [h.scraped_at for h in headlines]


def test_shuffle_timestamps_actually_changes_the_pairing_for_some_seed():
    headlines = [
        make_headline("a", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 9, 28, 5, 0, 0, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 9, 29, 10, 0, 0, tzinfo=timezone.utc)),
        make_headline("d", "4", datetime(2026, 9, 29, 14, 0, 0, tzinfo=timezone.utc)),
    ]
    original = [h.published_at for h in headlines]

    changed = any(
        [h.published_at for h in shuffle_timestamps(headlines, seed=s)] != original for s in range(20)
    )
    assert changed


def test_shuffle_timestamps_is_deterministic_for_a_given_seed():
    headlines = [
        make_headline("a", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 9, 28, 5, 0, 0, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 9, 29, 10, 0, 0, tzinfo=timezone.utc)),
    ]

    first = [h.published_at for h in shuffle_timestamps(headlines, seed=7)]
    second = [h.published_at for h in shuffle_timestamps(headlines, seed=7)]
    assert first == second


# --- correlation_for ---------------------------------------------------------


def test_correlation_for_matches_build_rows_on_synthetic_fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0),
            Bar(date=dt.date(2026, 9, 29), open=105.0, close=95.0),
        ],
    )
    save_fixture(
        "WIPRO.NS",
        [Bar(date=dt.date(2026, 9, 28), open=200.0, close=190.0)],
    )
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        ),
        make_headline(
            "Wipro Share Price Highlights: Wipro Stock Price History",
            "2",
            datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc),
        ),
    ]

    stat = correlation_for(headlines)

    assert stat is not None
    assert stat.n == 2


def test_correlation_for_returns_none_when_fewer_than_two_rows_resolve(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]

    assert correlation_for(headlines) is None


# --- run_audit orchestration (statistics, mocked correlation_for) -----------


def test_run_audit_computes_p_value_and_significant_rate(tmp_path: Path, monkeypatch):
    in_path = tmp_path / "raw.csv"
    write_csv(
        [make_headline("Infosys Share Price Highlights: x", "1", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc))],
        in_path,
    )

    # Real result: r=0.5, not significant. Shuffled controls: half as extreme
    # or more, none individually "significant".
    calls = iter(
        [PearsonResult(r=0.5, n=10, ci_low=-0.2, ci_high=0.8)]
        + [PearsonResult(r=0.5, n=10, ci_low=-0.2, ci_high=0.8)] * 2
        + [PearsonResult(r=0.1, n=10, ci_low=-0.4, ci_high=0.5)] * 2
    )
    monkeypatch.setattr(audit, "correlation_for", lambda headlines, live=False: next(calls))

    result = run_audit(in_path, n_shuffles=4, seed=0)

    assert result.real_r == 0.5
    assert result.n_shuffles_used == 4
    assert result.p_value == pytest.approx(0.5)  # 2 of 4 shuffles have |r| >= 0.5
    assert result.significant_rate == 0.0


def test_run_audit_flags_a_shuffled_control_that_is_significant_far_more_than_chance(tmp_path: Path, monkeypatch):
    in_path = tmp_path / "raw.csv"
    write_csv(
        [make_headline("Infosys Share Price Highlights: x", "1", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc))],
        in_path,
    )

    # Every shuffled control comes back "significant" (CI excludes 0) - the
    # leak this module exists to catch.
    calls = iter(
        [PearsonResult(r=0.1, n=10, ci_low=-0.2, ci_high=0.4)]
        + [PearsonResult(r=0.6, n=10, ci_low=0.1, ci_high=0.9)] * 10
    )
    monkeypatch.setattr(audit, "correlation_for", lambda headlines, live=False: next(calls))

    result = run_audit(in_path, n_shuffles=10, seed=0)

    assert result.significant_rate == 1.0
    assert result.leak_suspected is True


def test_run_audit_raises_when_real_data_never_resolves(tmp_path: Path, monkeypatch):
    in_path = tmp_path / "raw.csv"
    write_csv(
        [make_headline("Infosys Share Price Highlights: x", "1", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc))],
        in_path,
    )
    monkeypatch.setattr(audit, "correlation_for", lambda headlines, live=False: None)

    with pytest.raises(ValueError, match="not enough resolved headlines"):
        run_audit(in_path, n_shuffles=5, seed=0)


# --- leak_suspected threshold -------------------------------------------------


@pytest.mark.parametrize(
    "significant_rate, expected",
    [(0.0, False), (0.05, False), (0.1499, False), (0.1501, True), (0.5, True)],
)
def test_leak_suspected_threshold(significant_rate, expected):
    result = AuditResult(
        real_r=0.0,
        real_n=10,
        real_significant=False,
        null_r=[0.0],
        n_shuffles_used=1,
        n_shuffles_requested=1,
        p_value=1.0,
        null_mean=0.0,
        null_std=0.0,
        significant_rate=significant_rate,
    )
    assert result.leak_suspected is expected


# --- CLI / run() --------------------------------------------------------------


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    assert run(missing, n_shuffles=5, seed=0, live=False) == 1


def test_run_against_committed_fixture_passes(capsys):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, n_shuffles=50, seed=0, live=False)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "real (true timestamps):" in out
    assert "PASS" in out
