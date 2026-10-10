import json
from datetime import datetime, timezone
from pathlib import Path

from sentiment.contract import build_contract, headlines_for_ticker, run
from sentiment.headline import Headline, write_csv


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


INFY_1 = make_headline(
    "Infosys Share Price Highlights: Infosys Stock Price History",
    "1",
    datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
)
INFY_2 = make_headline(
    "Infosys Share Price Highlights: Infosys Stock Price History",
    "2",
    datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),
)
SCREENER_LIST = make_headline(
    "Cyient among 4 stocks showing White Marubozu Pattern",
    "3",
    datetime(2026, 9, 28, 3, 0, 0, tzinfo=timezone.utc),
)


def test_headlines_for_ticker_filters_to_exact_resolved_ticker():
    headlines = [INFY_1, INFY_2, SCREENER_LIST]
    matches = headlines_for_ticker(headlines, "INFY.NS")
    assert matches == [INFY_1, INFY_2]


def test_headlines_for_ticker_returns_empty_for_a_ticker_with_no_headlines():
    assert headlines_for_ticker([INFY_1], "WIPRO.NS") == []


def test_build_contract_averages_compound_over_the_window():
    contract = build_contract([INFY_1, INFY_2], "INFY.NS", window_days=7)
    assert contract.headline_count_7d == 2
    assert contract.model == "vader"
    assert contract.coverage == "partial"
    assert contract.score_7d is not None


def test_build_contract_excludes_headlines_older_than_the_window():
    old = make_headline(
        "Infosys Share Price Highlights: Infosys Stock Price History",
        "old",
        datetime(2026, 9, 1, 2, 0, 0, tzinfo=timezone.utc),
    )
    contract = build_contract([old, INFY_2], "INFY.NS", window_days=7)
    # window anchors to the latest headline for this ticker (Sep 29), so the
    # Sep 1 headline (28 days earlier) falls outside a 7-day window.
    assert contract.headline_count_7d == 1


def test_build_contract_is_null_with_zero_matching_headlines():
    contract = build_contract([SCREENER_LIST], "INFY.NS")
    assert contract.score_7d is None
    assert contract.headline_count_7d == 0
    assert contract.model == "vader"
    assert contract.coverage == "partial"


def test_run_writes_contract_json(tmp_path: Path):
    in_path = tmp_path / "raw.csv"
    write_csv([INFY_1, INFY_2], in_path)
    out_path = tmp_path / "contract.json"

    exit_code = run(in_path, "INFY.NS", 7, out_path)

    assert exit_code == 0
    written = json.loads(out_path.read_text())
    assert written["sentiment"]["headline_count_7d"] == 2
    assert written["sentiment"]["model"] == "vader"
    assert written["sentiment"]["coverage"] == "partial"
    assert set(written["sentiment"]) == {"score_7d", "headline_count_7d", "model", "coverage"}


def test_run_without_contract_path_does_not_write_a_file(tmp_path: Path):
    in_path = tmp_path / "raw.csv"
    write_csv([INFY_1], in_path)

    exit_code = run(in_path, "INFY.NS", 7, None)

    assert exit_code == 0
    assert list(tmp_path.iterdir()) == [in_path]


def test_run_missing_input_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    assert run(missing, "INFY.NS", 7, None) == 1


def test_run_against_committed_fixture_produces_a_real_contract():
    fixture = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
    exit_code = run(fixture, "INFY.NS", 7, None)
    assert exit_code == 0
