"""Day 8 integration: the v0.5 ``sentiment`` contract tests."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sentiment.contract import DEFAULT_IN, compute_sentiment_contract, main, run
from sentiment.headline import Headline, write_csv


def make_headline(title: str, link: str, published_at: datetime) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=published_at,
    )


# --- compute_sentiment_contract ------------------------------------------


def test_compute_sentiment_contract_against_the_real_fixture_reports_fortis():
    """FORTIS.NS has exactly one resolved headline in the real fixture -
    the VADER finance-jargon miss Day 7's README documents (compound
    +0.70 on a stock that fell -10.4% that month)."""
    result = compute_sentiment_contract(DEFAULT_IN, "FORTIS.NS")
    assert result.headline_count_7d == 1
    assert result.score_7d == pytest.approx(0.7003, abs=1e-4)
    assert result.model == "vader"
    assert result.coverage == "full"


def test_compute_sentiment_contract_reports_none_coverage_for_a_ticker_with_no_headlines():
    """Like v0.7's GULFOILLUB non-match: a ticker this repo never wrote a
    headline about is an expected, reported outcome, not an error."""
    result = compute_sentiment_contract(DEFAULT_IN, "RELIANCE.NS")
    assert result.score_7d is None
    assert result.headline_count_7d == 0
    assert result.coverage == "none"


def test_compute_sentiment_contract_averages_multiple_headlines_for_the_same_ticker(tmp_path):
    path = tmp_path / "headlines.csv"
    headlines = [
        make_headline(
            "Fortis Healthcare shares rise 2%", "a", datetime(2026, 9, 28, 4, 0, tzinfo=timezone.utc)
        ),
        make_headline(
            "Fortis Healthcare shares in focus as Supreme Court allows forensic audit to proceed",
            "b",
            datetime(2026, 9, 28, 5, 0, tzinfo=timezone.utc),
        ),
    ]
    write_csv(headlines, path)

    result = compute_sentiment_contract(path, "FORTIS.NS")
    assert result.headline_count_7d == 2
    assert result.coverage == "full"


def test_compute_sentiment_contract_window_excludes_headlines_older_than_7_days(tmp_path):
    path = tmp_path / "headlines.csv"
    headlines = [
        make_headline(
            "Fortis Healthcare shares in focus as Supreme Court allows forensic audit to proceed",
            "a",
            datetime(2026, 9, 1, 4, 0, tzinfo=timezone.utc),
        ),
        make_headline(
            "Fortis Healthcare shares rise 2%", "b", datetime(2026, 9, 28, 4, 0, tzinfo=timezone.utc)
        ),
    ]
    write_csv(headlines, path)

    result = compute_sentiment_contract(path, "FORTIS.NS")
    assert result.headline_count_7d == 1
    assert result.coverage == "partial"


# --- to_contract ----------------------------------------------------------


def test_to_contract_shape_matches_nist_steps_schema():
    result = compute_sentiment_contract(DEFAULT_IN, "FORTIS.NS")
    contract = result.to_contract()
    assert set(contract) == {"sentiment"}
    assert set(contract["sentiment"]) == {"score_7d", "headline_count_7d", "model", "coverage"}


def test_to_contract_reports_null_score_for_an_unresolved_ticker():
    result = compute_sentiment_contract(DEFAULT_IN, "RELIANCE.NS")
    contract = result.to_contract()
    assert contract["sentiment"]["score_7d"] is None
    assert contract["sentiment"]["headline_count_7d"] == 0


# --- CLI -------------------------------------------------------------------


def test_run_writes_the_contract_file(tmp_path):
    out_path = tmp_path / "contract.json"
    exit_code = run(DEFAULT_IN, "FORTIS.NS", out_path)
    assert exit_code == 0

    written = json.loads(out_path.read_text())
    assert written["sentiment"]["headline_count_7d"] == 1
    assert written["sentiment"]["model"] == "vader"


def test_run_without_contract_path_does_not_write_a_file(tmp_path, capsys):
    exit_code = run(DEFAULT_IN, "FORTIS.NS", None)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "FORTIS.NS" in out


def test_run_reports_an_error_for_a_missing_headlines_file(tmp_path):
    missing = tmp_path / "nope.csv"
    exit_code = run(missing, "FORTIS.NS", None)
    assert exit_code == 1


def test_main_parses_args_and_writes_the_contract(tmp_path):
    out_path = tmp_path / "contract.json"
    exit_code = main(["--ticker", "FORTIS.NS", "--contract", str(out_path)])
    assert exit_code == 0
    written = json.loads(out_path.read_text())
    assert written["sentiment"]["headline_count_7d"] == 1
