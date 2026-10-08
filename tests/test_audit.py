import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

import sentiment.prices as prices
from sentiment.audit import (
    AuditResult,
    contemporaneous_r,
    run,
    run_audit,
    shuffle_timestamps,
    write_null_csv,
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


def test_shuffle_timestamps_is_a_permutation_not_a_resample():
    headlines = [
        make_headline("a", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc)),
        make_headline("b", "2", datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc)),
        make_headline("c", "3", datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc)),
    ]
    shuffled = shuffle_timestamps(headlines, random.Random(0))

    assert sorted(h.published_at for h in shuffled) == sorted(h.published_at for h in headlines)
    # titles/links stay with their original headline - only the timestamp moves
    assert [h.title for h in shuffled] == [h.title for h in headlines]
    assert [h.link for h in shuffled] == [h.link for h in headlines]


def test_shuffle_timestamps_leaves_everything_but_published_at_untouched():
    h = make_headline("a", "1", datetime(2026, 9, 28, 1, 0, 0, tzinfo=timezone.utc))
    [shuffled] = shuffle_timestamps([h], random.Random(0))
    assert shuffled.source == h.source
    assert shuffled.scraped_at == h.scraped_at
    assert shuffled.published_raw == h.published_raw
    # only one headline -> the "permutation" of one timestamp is itself
    assert shuffled.published_at == h.published_at


def test_contemporaneous_r_needs_at_least_two_rows():
    assert contemporaneous_r([]) is None
    assert contemporaneous_r([{"compound": 0.1, "contemporaneous_return": 0.01}]) is None
    rows = [
        {"compound": 0.1, "contemporaneous_return": 0.01},
        {"compound": -0.1, "contemporaneous_return": -0.02},
    ]
    r = contemporaneous_r(rows)
    assert r is not None
    assert -1.0 <= r <= 1.0


def test_run_audit_rejects_fewer_than_two_resolvable_headlines(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture("INFY.NS", [Bar(date=__import__("datetime").date(2026, 9, 28), open=100.0, close=105.0)])
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            "1",
            datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
        )
    ]
    with pytest.raises(ValueError):
        run_audit(headlines, n_shuffles=10, seed=0)


def test_run_audit_is_deterministic_given_a_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0),
            Bar(date=dt.date(2026, 9, 29), open=110.0, close=90.0),
        ],
    )
    headlines = [
        make_headline(
            "Infosys Share Price Highlights: Infosys Stock Price History",
            str(i),
            datetime(2026, 9, 28, hour, 0, 0, tzinfo=timezone.utc),
        )
        for i, hour in enumerate([1, 2, 3, 4, 10, 11, 12, 13])
    ]

    first = run_audit(headlines, n_shuffles=40, seed=7)
    second = run_audit(headlines, n_shuffles=40, seed=7)

    assert first.real_r == second.real_r
    assert first.null_rs == second.null_rs
    assert first.p_value == second.p_value


def test_run_audit_a_genuine_timestamp_dependent_pattern_is_an_outlier_against_the_shuffled_null(tmp_path, monkeypatch):
    # Construct a fixture where the *only* thing that makes compound and
    # contemporaneous_return line up is which session (Monday vs Tuesday)
    # each headline's real timestamp aligns to - exactly the kind of
    # look-ahead dependency this audit exists to catch. Scrambling which
    # headline gets which timestamp should wreck that alignment-dependent
    # pattern, so the real r should sit far out in the tail of the shuffled
    # null distribution (a small p-value), not blend into it.
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    save_fixture(
        "INFY.NS",
        [
            Bar(date=dt.date(2026, 9, 28), open=100.0, close=110.0),  # +10% session
            Bar(date=dt.date(2026, 9, 29), open=100.0, close=90.0),  # -10% session
        ],
    )

    headlines = []
    # Half the headlines are pre-open on the 28th (-> aligns to the +10% Monday
    # session) and carry a positive VADER-scoring title; the other half are
    # post-close on the 28th (-> rolls to the -10% Tuesday session) and carry
    # a negative-scoring title. The real, leak-free alignment should find a
    # strong positive correlation; a random timestamp has no reason to keep
    # positive titles paired with Monday and negative ones with Tuesday.
    for i in range(6):
        headlines.append(
            make_headline(
                "Infosys Share Price Highlights: great profit surge beats estimates",
                f"pos-{i}",
                datetime(2026, 9, 28, 2, i, 0, tzinfo=timezone.utc),  # pre-open IST
            )
        )
        headlines.append(
            make_headline(
                "Infosys Share Price Highlights: terrible loss crisis fraud",
                f"neg-{i}",
                datetime(2026, 9, 28, 11, i, 0, tzinfo=timezone.utc),  # post-close IST
            )
        )

    result = run_audit(headlines, n_shuffles=300, seed=0)

    assert result.real_r > 0.8
    assert result.p_value < 0.05
    # the null distribution itself should be centered well below the real r -
    # scrambling the pairing destroys most of the correlation on average.
    assert result.null_mean < result.real_r / 2


def test_write_null_csv_writes_one_row_per_null_sample(tmp_path):
    result = AuditResult(real_r=-0.2, real_n=10, null_rs=[0.1, -0.1, 0.05], p_value=0.5)
    out_path = tmp_path / "null.csv"
    write_null_csv(result, out_path)

    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["shuffle_contemporaneous_r"]
    assert [float(r[0]) for r in rows[1:]] == result.null_rs


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out_path = tmp_path / "null.csv"
    exit_code = run(missing, out_path, n_shuffles=10, seed=0, live=False)
    assert exit_code == 1
    assert not out_path.exists()


def test_run_against_committed_fixture_passes_and_writes_the_null_csv(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_path = tmp_path / "null.csv"

    exit_code = run(fixture, out_path, n_shuffles=50, seed=0, live=False)

    assert exit_code == 0  # this fixture's real r should pass (see README Day 8 Findings)
    assert out_path.exists()
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert 0 < len(rows) <= 50
    for row in rows:
        assert -1.0 <= float(row["shuffle_contemporaneous_r"]) <= 1.0
