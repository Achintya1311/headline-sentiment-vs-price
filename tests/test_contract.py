import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from sentiment.contract import (
    build_contract,
    coverage_for,
    headlines_in_window,
    latest_session_date,
    resolved_ticker,
    run,
)
from sentiment.headline import Headline, write_csv


def make_headline(title: str, published_at: datetime, link: str | None = None) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link or title,
        published_at=published_at,
        published_raw=published_at.isoformat(),
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


# pre-open (before 09:15 IST / 03:45 UTC) on each calendar day
INFY_HEADLINE = make_headline(
    "Infosys Share Price Highlights: Infosys Stock Price History",
    datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
)
WIPRO_HEADLINE = make_headline(
    "Wipro Share Price Highlights: Wipro Stock Price History",
    datetime(2026, 9, 29, 2, 0, 0, tzinfo=timezone.utc),
)
UNRESOLVED_HEADLINE = make_headline(
    "Cyient among 4 stocks showing White Marubozu Pattern",
    datetime(2026, 9, 28, 2, 0, 0, tzinfo=timezone.utc),
)


def test_resolved_ticker_matches_a_single_company_headline():
    assert resolved_ticker(INFY_HEADLINE) == "INFY.NS"


def test_resolved_ticker_is_none_for_a_screener_list_headline():
    assert resolved_ticker(UNRESOLVED_HEADLINE) is None


def test_latest_session_date_is_the_max_aligned_session():
    assert latest_session_date([INFY_HEADLINE, WIPRO_HEADLINE]) == date(2026, 9, 29)


def test_latest_session_date_of_empty_list_is_none():
    assert latest_session_date([]) is None


def test_headlines_in_window_matches_only_the_requested_ticker():
    matched = headlines_in_window(
        [INFY_HEADLINE, WIPRO_HEADLINE, UNRESOLVED_HEADLINE], "INFY.NS", date(2026, 9, 28), window_days=7
    )
    assert matched == [INFY_HEADLINE]


def test_headlines_in_window_excludes_sessions_before_the_window():
    # window_days=1 ending 9/28 excludes 9/29's headline and includes 9/28's
    matched = headlines_in_window([INFY_HEADLINE], "INFY.NS", date(2026, 9, 28), window_days=1)
    assert matched == [INFY_HEADLINE]
    matched = headlines_in_window([INFY_HEADLINE], "INFY.NS", date(2026, 9, 27), window_days=1)
    assert matched == []


def test_headlines_in_window_is_case_insensitive_on_ticker():
    matched = headlines_in_window([INFY_HEADLINE], "infy.ns", date(2026, 9, 28), window_days=7)
    assert matched == [INFY_HEADLINE]


def test_coverage_for_thresholds():
    assert coverage_for(0) == "none"
    assert coverage_for(1) == "partial"
    assert coverage_for(4) == "partial"
    assert coverage_for(5) == "full"
    assert coverage_for(12) == "full"


def test_build_contract_averages_vader_compound_over_the_window():
    contract = build_contract([INFY_HEADLINE], "INFY.NS", "vader", as_of=date(2026, 9, 28), window_days=7)
    assert contract.ticker == "INFY.NS"
    assert contract.headline_count_7d == 1
    assert contract.coverage == "partial"
    assert contract.model == "vader"
    # "Share Price Highlights" template headlines saturate VADER at 0.296
    # (see README Findings) - a known, documented value, not a magic number.
    assert contract.score_7d == pytest.approx(0.296, abs=1e-3)


def test_build_contract_reports_none_coverage_when_nothing_matches():
    contract = build_contract([INFY_HEADLINE], "RELIANCE.NS", "vader", as_of=date(2026, 9, 28), window_days=7)
    assert contract.score_7d is None
    assert contract.headline_count_7d == 0
    assert contract.coverage == "none"


def test_build_contract_defaults_as_of_to_the_latest_session_in_the_fixture():
    contract = build_contract([INFY_HEADLINE, WIPRO_HEADLINE], "INFY.NS", "vader", window_days=7)
    # as_of defaults to 2026-09-29 (WIPRO's session); INFY's 9/28 session is
    # still inside a 7-day trailing window ending there.
    assert contract.headline_count_7d == 1


def test_build_contract_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        build_contract([INFY_HEADLINE], "INFY.NS", "not-a-real-model")


def test_to_dict_matches_the_next_steps_contract_shape():
    contract = build_contract([INFY_HEADLINE], "INFY.NS", "vader", as_of=date(2026, 9, 28), window_days=7)
    d = contract.to_dict()
    assert set(d) == {"sentiment"}
    assert set(d["sentiment"]) == {"score_7d", "headline_count_7d", "model", "coverage"}


def test_run_writes_a_contract_file_for_a_resolved_ticker(tmp_path):
    in_path = tmp_path / "headlines.csv"
    write_csv([INFY_HEADLINE], in_path)
    out_path = tmp_path / "contract.json"

    exit_code = run(in_path, "INFY.NS", "vader", None, 7, out_path)
    assert exit_code == 0

    written = json.loads(out_path.read_text())
    assert written["sentiment"]["headline_count_7d"] == 1
    assert written["sentiment"]["model"] == "vader"


def test_run_writes_a_none_coverage_contract_for_an_unresolved_ticker(tmp_path):
    in_path = tmp_path / "headlines.csv"
    write_csv([INFY_HEADLINE], in_path)
    out_path = tmp_path / "contract.json"

    exit_code = run(in_path, "GULFOILLUB.NS", "vader", None, 7, out_path)
    assert exit_code == 0

    written = json.loads(out_path.read_text())
    assert written["sentiment"]["headline_count_7d"] == 0
    assert written["sentiment"]["coverage"] == "none"
    assert written["sentiment"]["score_7d"] is None


def test_run_fails_fast_when_headlines_file_is_missing(tmp_path):
    missing = tmp_path / "nope.csv"
    assert run(missing, "INFY.NS", "vader", None, 7, tmp_path / "out.json") == 1


def test_run_against_the_real_committed_fixture_for_infosys(tmp_path):
    # pinned to the frozen 50-row regression snapshot, not the live
    # headlines_raw.csv the daily scrape workflow keeps appending to: the
    # default --as-of (latest session in the fixture) slides forward as
    # real headlines land, pushing this INFY headline's session out of the
    # 7-day window - a live consequence of the corpus-growth automation
    # actually working, not a bug in the default (D10).
    regression_fixture = (
        Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_regression_50.csv"
    )
    out_path = tmp_path / "sentiment_contract_test.json"
    exit_code = run(regression_fixture, "INFY.NS", "vader", None, 7, out_path)
    assert exit_code == 0
    written = json.loads(out_path.read_text())
    assert written["sentiment"]["headline_count_7d"] == 1
    assert written["sentiment"]["coverage"] == "partial"
