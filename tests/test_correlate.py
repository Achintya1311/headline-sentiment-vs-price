import csv
from datetime import datetime, timezone
from pathlib import Path

import sentiment.prices as prices
from sentiment.correlate import build_rows, run
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


def test_build_rows_pairs_resolved_headlines_with_returns(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    save_fixture(
        "INFY.NS",
        [
            Bar(date=__import__("datetime").date(2026, 9, 28), open=100.0, close=105.0),
            Bar(date=__import__("datetime").date(2026, 9, 29), open=105.0, close=103.0),
        ],
    )

    in_path = tmp_path / "raw.csv"
    write_csv(
        [
            # pre-open on the 28th -> aligns to session_date 2026-09-28
            make_headline(
                "Infosys Share Price Highlights: Infosys Stock Price History",
                "1",
                datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
            ),
            # a screener-list headline naming no single company -> excluded
            make_headline(
                "Cyient among 4 stocks showing White Marubozu Pattern",
                "2",
                datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc),
            ),
        ],
        in_path,
    )

    rows, unresolved = build_rows(in_path)

    assert unresolved == []
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "INFY.NS"
    assert row["session_date"] == "2026-09-28"
    assert row["contemporaneous_return"] == (105.0 - 100.0) / 100.0
    assert row["lagged_return"] == (103.0 - 105.0) / 105.0


def test_build_rows_reports_lagged_return_none_when_next_session_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)])

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

    rows, _ = build_rows(in_path)
    assert rows[0]["lagged_return"] is None


def test_build_rows_reports_company_with_no_resolvable_ticker(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
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

    rows, unresolved = build_rows(in_path)
    assert rows == []
    assert unresolved == [
        ("LTIMindtree Share Price Highlights: LTIMindtree Stock Price History", "LTIMindtree")
    ]


def test_run_writes_output_csv_and_succeeds(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    save_fixture(
        "INFY.NS",
        [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0), Bar(date=dt.date(2026, 9, 29), open=105.0, close=103.0)],
    )
    in_path = tmp_path / "raw.csv"
    out_path = tmp_path / "correlation.csv"
    write_csv(
        [
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
        ],
        in_path,
    )

    exit_code = run(in_path, out_path, live=False, event_threshold=0.3)

    assert exit_code == 0
    assert out_path.exists()
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # the second title isn't one sentiment.tickers.resolve recognises


def test_run_with_a_single_resolved_headline_does_not_crash(tmp_path: Path, monkeypatch):
    # pearson_r needs at least 2 points - run() must degrade gracefully, not
    # raise, when only one headline resolves to a ticker.
    monkeypatch.setattr(prices, "FIXTURE_DIR", tmp_path)
    import datetime as dt

    save_fixture("INFY.NS", [Bar(date=dt.date(2026, 9, 28), open=100.0, close=105.0)])
    in_path = tmp_path / "raw.csv"
    out_path = tmp_path / "correlation.csv"
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

    exit_code = run(in_path, out_path, live=False, event_threshold=0.3)

    assert exit_code == 0
    with out_path.open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out_path = tmp_path / "correlation.csv"

    exit_code = run(missing, out_path, live=False, event_threshold=0.3)

    assert exit_code == 1
    assert not out_path.exists()


def test_run_against_committed_fixture_produces_the_expected_headline_count(tmp_path: Path):
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    out_path = tmp_path / "correlation.csv"

    exit_code = run(fixture, out_path, live=False, event_threshold=0.3)

    assert exit_code == 0
    with out_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # 24 headlines resolve to a single company, one (LTIMindtree) has no
    # fetchable ticker - see sentiment/tickers.py and the README Findings.
    assert len(rows) == 23
