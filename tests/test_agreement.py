from datetime import datetime, timezone
from pathlib import Path

from sentiment.agreement import (
    DEFAULT_EVAL_SET,
    DEFAULT_HEADLINES,
    corpus_agreement,
    eval_set_comparison,
    render_report,
    run,
)
from sentiment.headline import Headline
from sentiment.evaluate import load_eval_set


def make_headline(title: str, link: str) -> Headline:
    return Headline(
        source="feed",
        title=title,
        link=link,
        published_at=datetime(2026, 9, 28, 8, 0, 0, tzinfo=timezone.utc),
        published_raw="Mon, 28 Sep 2026 08:00:00 +0000",
        scraped_at=datetime(2026, 9, 28, 9, 0, 0, tzinfo=timezone.utc),
    )


def test_corpus_agreement_counts_matches_and_mismatches():
    headlines = [
        make_headline("Stock soars after strong quarterly results", "1"),  # both positive
        make_headline("Company beats earnings estimates, raises full-year guidance", "2"),  # VADER neutral, FinBERT positive
    ]

    result = corpus_agreement(headlines)

    assert result["n"] == 2
    assert result["rows"][0]["agree"] is True
    assert result["rows"][1]["agree"] is False
    assert result["agreement_rate"] == 0.5
    assert len(result["disagreements"]) == 1
    assert result["disagreements"][0]["title"] == "Company beats earnings estimates, raises full-year guidance"
    assert sum(result["confusion"].values()) == 2


def test_eval_set_comparison_shows_finbert_closing_the_finance_jargon_gap():
    rows = [
        {"id": "1", "text": "Company beats earnings estimates, raises full-year guidance",
         "expected_label": "positive", "category": "finance_jargon", "note": ""},
        {"id": "2", "text": "Manufacturer recalls product over safety defect",
         "expected_label": "negative", "category": "finance_jargon", "note": ""},
    ]

    result = eval_set_comparison(rows)

    assert result["n"] == 2
    # VADER's documented failure mode on this exact category (see
    # sentiment/vader_score.py and tests/test_vader_score.py).
    assert result["vader_overall_accuracy"] == 0.0
    # FinBERT reads both correctly.
    assert result["finbert_overall_accuracy"] == 1.0
    assert result["by_category"]["finance_jargon"]["vader_accuracy"] == 0.0
    assert result["by_category"]["finance_jargon"]["finbert_accuracy"] == 1.0


def test_render_report_includes_both_sections():
    headlines = [make_headline("Stock soars after strong quarterly results", "1")]
    corpus = corpus_agreement(headlines)
    eval_rows = [
        {"id": "1", "text": "Company beats earnings estimates, raises full-year guidance",
         "expected_label": "positive", "category": "finance_jargon", "note": ""},
    ]
    eval_comparison = eval_set_comparison(eval_rows)

    report = render_report(corpus, eval_comparison)

    assert "corpus agreement" in report
    assert "eval-set accuracy" in report
    assert "finance_jargon" in report


def test_run_writes_report_and_returns_zero(tmp_path: Path):
    out = tmp_path / "report.txt"

    exit_code = run(DEFAULT_HEADLINES, DEFAULT_EVAL_SET, out)

    assert exit_code == 0
    assert out.exists()
    text = out.read_text()
    assert "corpus agreement" in text
    assert "eval-set accuracy" in text


def test_run_missing_headlines_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out = tmp_path / "report.txt"

    exit_code = run(missing, DEFAULT_EVAL_SET, out)

    assert exit_code == 1
    assert not out.exists()


def test_committed_eval_set_is_reusable_via_evaluate_module():
    # Sanity check that this module reads the same fixture Day 2's
    # sentiment.evaluate uses, rather than a divergent copy.
    rows = load_eval_set(DEFAULT_EVAL_SET)
    assert len(rows) > 0
