"""Day 2 CLI: evaluation harness for the VADER baseline scorer.

    python -m sentiment.evaluate
    python -m sentiment.evaluate --eval-set fixtures/eval/vader_eval_set.csv --out outputs/vader_eval_report.txt

Scores a small hand-labeled set of headlines (``fixtures/eval/vader_eval_set.csv``,
columns: id, text, expected_label, category, note) with VADER and reports
accuracy overall and per ``category``. There is no ground truth yet for
whether a headline actually moved a price - that is Day 5/6's job. This
harness only checks whether VADER's own label agrees with a human's reading
of the headline's tone, which is the thing Day 3's FinBERT comparison needs
a documented baseline for.

The ``category`` column exists because the failure mode is not random: see
sentiment/vader_score.py's docstring and this module's own findings in
README.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from sentiment.vader_score import score_text, label_for_compound

DEFAULT_EVAL_SET = Path(__file__).resolve().parent.parent / "fixtures" / "eval" / "vader_eval_set.csv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "vader_eval_report.txt"


def load_eval_set(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(rows: list[dict[str, str]]) -> dict:
    """Score every row and return per-row results plus overall/per-category accuracy."""
    results = []
    for row in rows:
        scores = score_text(row["text"])
        predicted = label_for_compound(scores["compound"])
        results.append(
            {
                "id": row["id"],
                "text": row["text"],
                "category": row["category"],
                "expected_label": row["expected_label"],
                "predicted_label": predicted,
                "compound": scores["compound"],
                "correct": predicted == row["expected_label"],
                "note": row.get("note", ""),
            }
        )

    overall = sum(r["correct"] for r in results) / len(results) if results else 0.0

    by_category: dict[str, dict] = {}
    grouped = defaultdict(list)
    for r in results:
        grouped[r["category"]].append(r)
    for category, rows_in_category in grouped.items():
        correct = sum(r["correct"] for r in rows_in_category)
        by_category[category] = {
            "n": len(rows_in_category),
            "correct": correct,
            "accuracy": correct / len(rows_in_category),
        }

    return {"results": results, "overall_accuracy": overall, "by_category": by_category, "n": len(results)}


def render_report(evaluation: dict) -> str:
    lines = [
        f"VADER evaluation: {evaluation['n']} labeled headlines, "
        f"overall accuracy {evaluation['overall_accuracy']:.1%}",
        "",
        "by category:",
    ]
    for category in sorted(evaluation["by_category"]):
        stats = evaluation["by_category"][category]
        lines.append(f"  {category:<16} {stats['correct']}/{stats['n']}  ({stats['accuracy']:.1%})")
    lines.append("")
    lines.append("mismatches:")
    mismatches = [r for r in evaluation["results"] if not r["correct"]]
    if not mismatches:
        lines.append("  none")
    for r in mismatches:
        lines.append(
            f"  [{r['category']}] {r['text']!r}: expected={r['expected_label']} "
            f"predicted={r['predicted_label']} (compound={r['compound']:+.4f}) - {r['note']}"
        )
    return "\n".join(lines) + "\n"


def run(eval_set_path: Path, out_path: Path) -> int:
    if not eval_set_path.exists():
        print(f"no eval set at {eval_set_path}", file=sys.stderr)
        return 1
    rows = load_eval_set(eval_set_path)
    evaluation = evaluate(rows)
    report = render_report(evaluation)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    print(report, end="")
    print(f"report written to {out_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET, help="labeled CSV to evaluate against")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="report file to write")
    args = parser.parse_args()
    sys.exit(run(args.eval_set, args.out))


if __name__ == "__main__":
    main()
