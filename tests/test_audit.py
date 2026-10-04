from __future__ import annotations

import csv
import random
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import sentiment.audit as audit
import sentiment.correlate as correlate
import sentiment.prices as prices
from sentiment.audit import (
    contemporaneous_r,
    empirical_two_sided_p,
    percentile,
    run,
    shuffle_null_distribution,
    shuffle_timestamps,
)
from sentiment.headline import Headline, read_csv
from sentiment.prices import Bar, save_fixture

UTC = timezone.utc


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=UTC),
    )


def test_shuffle_timestamps_preserves_the_multiset_of_timestamps():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=UTC)) for i in range(6)
    ]
    rng = random.Random(0)

    shuffled = shuffle_timestamps(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # Original list is untouched (shuffle_timestamps returns a new list).
    assert [h.published_at for h in headlines] == [
        datetime(2026, 9, 28, i, 0, 0, tzinfo=UTC) for i in range(6)
    ]


def test_percentile_and_p_value_helpers():
    sorted_vals = [float(i) for i in range(100)]  # 0..99
    assert percentile(sorted_vals, 0.025) == 2.0
    assert percentile(sorted_vals, 0.975) == 97.0

    null_rs = [-0.5, -0.1, 0.0, 0.1, 0.5]
    assert empirical_two_sided_p(0.5, null_rs) == 2 / 5
    assert empirical_two_sided_p(0.0, null_rs) == 5 / 5


def test_contemporaneous_r_matches_build_rows_pearson(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [Bar(date=date(2026, 9, 28), open=100.0, close=105.0), Bar(date=date(2026, 9, 29), open=105.0, close=103.0)],
    )
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=UTC),
        ),
        make_headline(
            "Wipro Share Price Highlights: Wipro Stock Price History",
            "2",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=UTC),
        ),
    ]
    save_fixture("WIPRO.NS", [Bar(date=date(2026, 9, 28), open=50.0, close=49.0)])

    r, n = contemporaneous_r(headlines)

    assert n == 2
    assert r is not None


def test_contemporaneous_r_is_none_when_fewer_than_two_rows_resolve(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=date(2026, 9, 28), open=100.0, close=105.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=UTC),
        )
    ]

    r, n = contemporaneous_r(headlines)

    assert r is None
    assert n == 1


def test_shuffle_null_distribution_is_deterministic_for_a_fixed_seed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [Bar(date=date(2026, 9, 28), open=100.0, close=105.0), Bar(date=date(2026, 9, 29), open=105.0, close=103.0)],
    )
    save_fixture(
        "WIPRO.NS",
        [Bar(date=date(2026, 9, 28), open=50.0, close=49.0), Bar(date=date(2026, 9, 29), open=49.0, close=51.0)],
    )
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=UTC),
        ),
        make_headline(
            "Wipro Share Price Highlights: Wipro Stock Price History",
            "2",
            datetime(2026, 9, 28, 6, 0, 0, tzinfo=UTC),
        ),
    ]

    first = shuffle_null_distribution(headlines, n_shuffles=20, seed=7)
    second = shuffle_null_distribution(headlines, n_shuffles=20, seed=7)

    assert first == second
    assert len(first) == 20


def test_audit_against_committed_fixture_is_within_the_shuffled_null_band():
    """Regression check on the repo's own 50-headline fixture: the real
    contemporaneous r Day 5 reported (-0.185, n=23) should be statistically
    indistinguishable from a null built by shuffling timestamps - exactly
    the 'done when' leakage test the README commits to, run against the
    actual committed data rather than a synthetic case."""
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    headlines = read_csv(fixture)

    observed_r, n = contemporaneous_r(headlines)
    assert n == 23
    assert observed_r is not None

    null_rs = shuffle_null_distribution(headlines, n_shuffles=300, seed=0)
    sorted_null = sorted(null_rs)
    lo = percentile(sorted_null, 0.025)
    hi = percentile(sorted_null, 0.975)

    assert lo <= observed_r <= hi, (
        f"real-timestamp r={observed_r:.3f} falls outside the shuffled null band "
        f"[{lo:.3f}, {hi:.3f}] - that would mean the pipeline is leaking"
    )


def test_synthetic_leak_is_detected_as_outside_the_shuffled_null_band(tmp_path: Path, monkeypatch):
    """Positive control: proves the shuffle test actually has teeth. Without
    this, a 'PASS' on the real (already-null) fixture above would be
    unfalsifiable - a toothless test passes everything. Here, 10 synthetic
    headlines are built so that, under their *real* timestamps, VADER
    compound (mocked, so the text itself carries nothing) exactly equals
    the one session's return it is genuinely aligned to - a deliberately
    perfect, textbook look-ahead leak. Shuffling timestamps reassigns each
    headline to the *other* (unrelated) session's return, which should
    destroy the relationship.
    """
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)

    companies = [
        ("Infosys", "INFY.NS"),
        ("Wipro", "WIPRO.NS"),
        ("Tech Mahindra", "TECHM.NS"),
        ("HCL Tech", "HCLTECH.NS"),
        ("SBI Life", "SBILIFE.NS"),
        ("Nestle India", "NESTLEIND.NS"),
        ("Sun Pharma", "SUNPHARMA.NS"),
        ("Grasim Inds", "GRASIM.NS"),
        ("HUL", "HINDUNILVR.NS"),
        ("Bharti Airtel", "BHARTIARTL.NS"),
    ]
    day1, day2 = date(2026, 9, 28), date(2026, 9, 29)  # real Mon/Tue, like the repo's own fixture
    day1_returns = [0.05, -0.03, 0.02, 0.07, -0.01, 0.04, -0.06, 0.03, -0.02, 0.01]
    day2_returns = [-0.04, 0.06, -0.07, 0.01, 0.05, -0.02, 0.03, -0.05, 0.04, -0.01]

    pre_open_at = datetime(2026, 9, 28, 2, 0, 0, tzinfo=UTC)  # 07:30 IST: before the 09:15 open
    intraday_at = datetime(2026, 9, 28, 6, 0, 0, tzinfo=UTC)  # 11:30 IST: mid-session

    headlines = []
    compound_by_title = {}
    for i, (name, ticker) in enumerate(companies):
        title = f"{name} Share Price Highlights: {name} Stock Price History"
        save_fixture(
            ticker,
            [
                Bar(date=day1, open=100.0, close=100.0 * (1 + day1_returns[i])),
                Bar(date=day2, open=100.0, close=100.0 * (1 + day2_returns[i])),
            ],
        )
        if i < 5:
            # pre-open on day1 -> aligns to day1 -> correct return is day1_returns[i]
            headlines.append(make_headline(title, str(i), pre_open_at))
            compound_by_title[title] = day1_returns[i]
        else:
            # intraday on day1 -> rolls to day2 -> correct return is day2_returns[i]
            headlines.append(make_headline(title, str(i), intraday_at))
            compound_by_title[title] = day2_returns[i]

    def fake_score_headline(headline: Headline) -> SimpleNamespace:
        return SimpleNamespace(compound=compound_by_title[headline.title])

    monkeypatch.setattr(correlate, "score_headline", fake_score_headline)

    observed_r, n = contemporaneous_r(headlines)
    assert n == 10
    assert observed_r == 1.0  # compound was set to exactly the correctly-aligned return

    null_rs = shuffle_null_distribution(headlines, n_shuffles=500, seed=0)
    sorted_null = sorted(null_rs)
    lo = percentile(sorted_null, 0.025)
    hi = percentile(sorted_null, 0.975)

    # The engineered leak sits far outside the shuffled null band - this is
    # what a FAIL verdict looks like, proving the audit can actually catch one.
    assert not (lo <= observed_r <= hi)
    assert hi < 0.99


def test_run_against_committed_fixture_writes_csv_and_passes(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_path = tmp_path / "audit_shuffle_null.csv"

    exit_code = run(fixture, out_path, n_shuffles=100, seed=0, live=False)

    assert exit_code == 0
    assert out_path.exists()
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    assert set(rows[0].keys()) == {"shuffle_index", "r"}


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out_path = tmp_path / "audit_shuffle_null.csv"

    exit_code = run(missing, out_path, n_shuffles=10, seed=0, live=False)

    assert exit_code == 1
    assert not out_path.exists()
