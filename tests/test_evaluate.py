from pathlib import Path

from sentiment.evaluate import DEFAULT_EVAL_SET, evaluate, load_eval_set, render_report, run


def test_committed_eval_set_loads_and_every_row_has_the_expected_columns():
    rows = load_eval_set(DEFAULT_EVAL_SET)
    assert len(rows) > 0
    for row in rows:
        assert set(row) == {"id", "text", "expected_label", "category", "note"}
        assert row["expected_label"] in {"positive", "neutral", "negative"}


def test_evaluate_scores_every_row_and_flags_correctness():
    rows = [
        {"id": "1", "text": "Stock soars after strong quarterly results", "expected_label": "positive",
         "category": "general", "note": ""},
        {"id": "2", "text": "Company beats earnings estimates, raises full-year guidance",
         "expected_label": "positive", "category": "finance_jargon", "note": "known VADER miss"},
    ]

    result = evaluate(rows)

    assert result["n"] == 2
    by_id = {r["id"]: r for r in result["results"]}
    assert by_id["1"]["correct"] is True
    assert by_id["1"]["predicted_label"] == "positive"
    # The finance-jargon trap case: VADER scores it neutral, not positive.
    assert by_id["2"]["correct"] is False
    assert by_id["2"]["predicted_label"] == "neutral"
    assert result["overall_accuracy"] == 0.5


def test_evaluate_breaks_down_accuracy_by_category():
    rows = [
        {"id": "1", "text": "Stock soars after strong quarterly results", "expected_label": "positive",
         "category": "general", "note": ""},
        {"id": "2", "text": "Shares plunge on weak guidance", "expected_label": "negative",
         "category": "general", "note": ""},
        {"id": "3", "text": "Company to report Q2 results on Thursday", "expected_label": "neutral",
         "category": "neutral_factual", "note": ""},
    ]

    result = evaluate(rows)

    assert result["by_category"]["general"] == {"n": 2, "correct": 2, "accuracy": 1.0}
    assert result["by_category"]["neutral_factual"] == {"n": 1, "correct": 1, "accuracy": 1.0}


def test_render_report_lists_mismatches_with_their_note():
    rows = [
        {"id": "1", "text": "Company beats earnings estimates, raises full-year guidance",
         "expected_label": "positive", "category": "finance_jargon", "note": "known VADER miss"},
    ]
    report = render_report(evaluate(rows))
    assert "finance_jargon" in report
    assert "known VADER miss" in report
    assert "expected=positive" in report
    assert "predicted=neutral" in report


def test_run_writes_report_file_and_returns_zero(tmp_path: Path):
    out = tmp_path / "report.txt"
    exit_code = run(DEFAULT_EVAL_SET, out)

    assert exit_code == 0
    assert out.exists()
    text = out.read_text()
    assert "overall accuracy" in text


def test_run_missing_eval_set_reports_failure(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    out = tmp_path / "report.txt"

    exit_code = run(missing, out)

    assert exit_code == 1
    assert not out.exists()
