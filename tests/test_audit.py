import random
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    SIGNIFICANCE,
    contemporaneous_r,
    run,
    run_shuffle_audit,
    shuffle_published_at,
)
from sentiment.headline import Headline, write_csv
from sentiment.prices import Bar, save_fixture

IST = ZoneInfo("Asia/Kolkata")


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_shuffle_published_at_is_a_permutation_that_leaves_content_alone():
    headlines = [
        make_headline(f"Headline {i}", f"link-{i}", datetime(2026, 9, 20 + i, 2, 0, 0, tzinfo=timezone.utc))
        for i in range(5)
    ]
    rng = random.Random(0)

    shuffled = shuffle_published_at(headlines, rng)

    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]
    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # some reassignment actually happened - this seed does not round-trip to
    # the identity permutation on 5 elements.
    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_contemporaneous_r_against_a_small_synthetic_pipeline(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=__import__("datetime").date(2026, 9, 28), open=100.0, close=105.0),
            Bar(date=__import__("datetime").date(2026, 9, 29), open=105.0, close=103.0),
        ],
    )
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        ),
        make_headline(
            "Infosys extends losses amid weak IT sentiment",
            "2",
            datetime(2026, 9, 28, 2, 30, 0, tzinfo=timezone.utc),
        ),
    ]

    # only the first title is one sentiment.tickers.resolve recognises.
    r, n = contemporaneous_r(headlines)

    assert r is None  # a single resolved row isn't enough for a correlation
    assert n == 1


def test_run_shuffle_audit_against_the_committed_fixture_matches_the_documented_null():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    from sentiment.headline import read_csv

    headlines = read_csv(fixture)

    audit = run_shuffle_audit(headlines, trials=200, seed=0)

    # Day 5's own documented finding: contemporaneous r=-0.185, n=23.
    assert audit.real_r == pytest.approx(-0.1854, abs=1e-3)
    assert audit.real_n == 23
    # not distinguishable from a random-timestamp shuffle - no leak-shaped signal.
    assert audit.p_value > SIGNIFICANCE


def test_run_shuffle_audit_detects_a_real_timestamp_driven_association():
    """Positive control: prove the shuffle actually has the power to catch a
    real association, using genuine committed price fixtures (not the real
    headline corpus) paired with headlines deliberately worded to match the
    sign of a specific trading day's move. If this failed to show the real
    correlation collapsing under shuffling, the audit's shuffle plumbing
    would not actually be wired into the pipeline it claims to test."""
    tickers_and_titles = {
        "FORTIS.NS": (
            "Fortis Healthcare shares surge on excellent record profit, outstanding guidance",
            "Fortis Healthcare shares plunge on terrible disastrous awful loss warning",
        ),
        "HDFCBANK.NS": (
            "HDFC Bank shares surge on excellent record profit, outstanding guidance",
            "HDFC Bank shares plunge on terrible disastrous awful loss warning",
        ),
        "BSE.NS": (
            "BSE shares surge on excellent record profit, outstanding guidance",
            "BSE shares plunge on terrible disastrous awful loss warning",
        ),
        "SUZLON.NS": (
            "Suzlon Energy shares surge on excellent record profit, outstanding guidance",
            "Suzlon Energy shares plunge on terrible disastrous awful loss warning",
        ),
        "PCJEWELLER.NS": (
            "PC Jeweller shares surge on excellent record profit, outstanding guidance",
            "PC Jeweller shares plunge on terrible disastrous awful loss warning",
        ),
        "GESHIP.NS": (
            "Great Eastern Shipping shares surge on excellent record profit, outstanding guidance",
            "Great Eastern Shipping shares plunge on terrible disastrous awful loss warning",
        ),
        "MFSL.NS": (
            "Jefferies names Max Financial a top pick after excellent record profit",
            "Jefferies names Max Financial a sell after terrible disastrous awful loss",
        ),
        "POLICYBZR.NS": (
            "PB Fintech shares surge on excellent record profit, outstanding guidance",
            "PB Fintech shares plunge on terrible disastrous awful loss warning",
        ),
    }

    headlines = []
    for ticker, (pos_title, neg_title) in tickers_and_titles.items():
        bars = prices.load_bars(ticker)
        by_magnitude = sorted(bars, key=lambda b: abs(b.session_return), reverse=True)
        pos_bar = next(b for b in by_magnitude if b.session_return > 0)
        neg_bar = next(b for b in by_magnitude if b.session_return < 0)
        for title, bar in [(pos_title, pos_bar), (neg_title, neg_bar)]:
            published = datetime(bar.date.year, bar.date.month, bar.date.day, 8, 0, 0, tzinfo=IST)
            headlines.append(make_headline(title, f"{ticker}-{bar.date.isoformat()}", published))

    audit = run_shuffle_audit(headlines, trials=200, seed=0)

    assert audit.real_n == 16
    assert abs(audit.real_r) > 0.5  # the deliberate sentiment/move pairing is strong
    assert audit.p_value < SIGNIFICANCE  # extreme relative to random-timestamp chance

    mean_abs_shuffled = sum(abs(r) for r in audit.shuffled_rs) / len(audit.shuffled_rs)
    assert mean_abs_shuffled < abs(audit.real_r)  # shuffling the timestamps weakens it


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    exit_code = run(missing, trials=50, seed=0, live=False)

    assert exit_code == 1


def test_run_with_too_few_resolved_headlines_does_not_crash(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=__import__("datetime").date(2026, 9, 28), open=100.0, close=105.0)])
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

    exit_code = run(in_path, trials=50, seed=0, live=False)

    assert exit_code == 0  # one resolved headline is not an error, just nothing to test


def test_run_against_the_committed_fixture_passes():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    exit_code = run(fixture, trials=200, seed=0, live=False)

    assert exit_code == 0
