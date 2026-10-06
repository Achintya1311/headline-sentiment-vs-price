import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.audit import (
    permutation_p_value,
    run,
    run_audit,
    shuffle_published_at,
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


def test_permutation_p_value_counts_at_least_as_extreme_trials():
    # |0.5| is matched or exceeded by 0.6, -0.9 and -0.5 itself (three of five).
    assert permutation_p_value(0.5, [0.1, 0.6, -0.9, 0.2, -0.5]) == 3 / 5


def test_permutation_p_value_is_two_sided():
    # a negative real stat is compared by magnitude, so a positive trial of
    # equal or greater size still counts as "at least as extreme".
    assert permutation_p_value(-0.5, [0.5, 0.9, 0.1]) == 2 / 3


def test_permutation_p_value_requires_at_least_one_trial():
    import pytest

    with pytest.raises(ValueError):
        permutation_p_value(0.5, [])


def test_shuffle_published_at_preserves_the_multiset_of_timestamps():
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    rng = random.Random(0)

    shuffled = shuffle_published_at(headlines, rng)

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links are untouched - only the timestamp pairing changes
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]


def test_shuffle_published_at_actually_reassigns_timestamps():
    # with 10 distinct timestamps, a seeded shuffle should not hand every
    # headline back its own original timestamp - if it did, the whole audit
    # would be comparing the real result to copies of itself.
    headlines = [
        make_headline(f"Headline {i}", str(i), datetime(2026, 9, 28, i, 0, 0, tzinfo=timezone.utc))
        for i in range(10)
    ]
    rng = random.Random(0)

    shuffled = shuffle_published_at(headlines, rng)

    assert [h.published_at for h in shuffled] != [h.published_at for h in headlines]


def test_run_audit_against_committed_fixture_finds_no_detectable_signal():
    # The real pipeline's own contemporaneous correlation (see Day 5 Findings,
    # r=-0.185 n=23) must not be distinguishable from a random relabelling of
    # the same headlines' timestamps - this is the leakage/correctness gate
    # the README's "Correctness gate" section and NEXT_STEPS.md's "Done when"
    # both require to run in CI, not just once by hand.
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"

    report, unresolved = run_audit(fixture, live=False, trials=200, seed=0)

    assert report.real.n_contemporaneous == 23
    assert len(unresolved) == 1  # LTIMindtree - see Day 5 Findings
    assert report.real.contemporaneous_r is not None
    assert len(report.trials) == 200
    assert report.p_value_contemporaneous is not None
    assert report.p_value_lagged is not None
    # not significant at the conventional 5% level in either direction
    assert report.p_value_contemporaneous > 0.05
    assert report.p_value_lagged > 0.05


def test_run_audit_flags_a_genuine_synthetic_signal(tmp_path: Path, monkeypatch):
    # Sanity check on the audit method itself: if a real timing-dependent
    # relationship between sentiment and return actually exists, the
    # permutation test must be able to say so, not report "no signal"
    # unconditionally regardless of input. Eight headlines, four tickers'
    # worth of distinctly-worded positive headlines pre-open on Monday
    # (contemporaneous return +0.05) and three worth of negative headlines
    # intraday on Monday, rolling to Tuesday (contemporaneous return -0.05).
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    tickers = [
        "FORTIS.NS", "BSE.NS", "POLICYBZR.NS", "HDFCBANK.NS", "NSE.BO",
        "SUZLON.NS", "PCJEWELLER.NS", "GESHIP.NS",
    ]
    for ticker in tickers:
        save_fixture(
            ticker,
            [
                Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0),
                Bar(date=dt.date(2026, 9, 29), open=100.0, close=95.0),
            ],
        )

    pos_time = datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)  # pre-open -> Monday (+0.05)
    neg_time = datetime(2026, 9, 28, 6, 0, 0, tzinfo=timezone.utc)  # intraday -> Tuesday (-0.05)

    headlines = [
        make_headline("Fortis Healthcare shares rally as profit beats estimates and outlook raised", "1", pos_time),
        make_headline("BSE shares surge after blockbuster quarter, upgraded by brokerages", "2", pos_time),
        make_headline("PB Fintech shares jump on strong growth and raised guidance", "3", pos_time),
        make_headline("HDFC Bank shares gain as profit beats expectations, outlook upgraded", "4", pos_time),
        make_headline("NSE shares climb after strong listing gains and bullish outlook", "5", pos_time),
        make_headline("Suzlon Energy shares crash after disappointing results and downgrade", "6", neg_time),
        make_headline("PC Jeweller shares plunge on fraud allegations and weak outlook", "7", neg_time),
        make_headline("Great Eastern Shipping shares tumble after losses and warning", "8", neg_time),
    ]
    in_path = tmp_path / "raw.csv"
    write_csv(headlines, in_path)

    report, unresolved = run_audit(in_path, live=False, trials=300, seed=0)

    assert unresolved == []
    assert report.real.n_contemporaneous == 8
    assert report.real.contemporaneous_r > 0.8
    assert report.p_value_contemporaneous is not None
    assert report.p_value_contemporaneous < 0.05


def test_run_writes_output_csv_and_succeeds(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_path = tmp_path / "audit_shuffle.csv"

    exit_code = run(fixture, out_path, live=False, trials=20, seed=0)

    assert exit_code == 0
    assert out_path.exists()
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 21  # 1 real row + 20 shuffled trials
    assert rows[0]["trial"] == "real"
    assert rows[1]["trial"] == "0"


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out_path = tmp_path / "audit_shuffle.csv"

    exit_code = run(missing, out_path, live=False, trials=10, seed=0)

    assert exit_code == 1
    assert not out_path.exists()
