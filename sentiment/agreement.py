"""Day 3 CLI: VADER vs FinBERT agreement analysis.

    python -m sentiment.agreement
    python -m sentiment.agreement --headlines fixtures/headlines/headlines_raw.csv \\
        --eval-set fixtures/eval/vader_eval_set.csv --out outputs/agreement_report.txt

Two comparisons, because they answer different questions:

1. **Corpus agreement** - score every scraped headline with both models and
   measure how often their labels agree. This is a description of how
   different the two scorers are on real, unlabeled data; it says nothing
   about which one is *right*, since the corpus has no ground truth.
2. **Eval-set accuracy** - score the same 24 hand-labeled headlines
   ``sentiment/evaluate.py`` already uses and compare each model's accuracy
   overall and by category. This is where the finance-jargon gap Day 2
   found either closes or doesn't, against a human label.

NEXT_STEPS.md is explicit that VADER and FinBERT "will disagree, and where
they disagree is interesting" - the corpus disagreements are printed in
full for exactly that reason, not summarised away.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from sentiment.finbert_score import label_for_scores as finbert_label_for_scores
from sentiment.finbert_score import score_text as finbert_score_text
from sentiment.headline import read_csv
from sentiment.vader_score import label_for_compound as vader_label_for_compound
from sentiment.vader_score import score_headlines as vader_score_headlines
from sentiment.vader_score import score_text as vader_score_text
from sentiment.finbert_score import score_headlines as finbert_score_headlines

DEFAULT_HEADLINES = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_EVAL_SET = Path(__file__).resolve().parent.parent / "fixtures" / "eval" / "vader_eval_set.csv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "agreement_report.txt"

LABELS = ("positive", "neutral", "negative")


def corpus_agreement(headlines) -> dict:
    """Score ``headlines`` with both models and compare labels pairwise.

    Relies on both scorers preserving input order (both do, see their own
    tests) rather than re-matching by link, since every row here comes from
    the same in-memory headline list.
    """
    vader_scored = vader_score_headlines(headlines)
    finbert_scored = finbert_score_headlines(headlines)

    rows = []
    confusion: Counter = Counter()
    for v, f in zip(vader_scored, finbert_scored):
        agree = v.label == f.label
        confusion[(v.label, f.label)] += 1
        rows.append(
            {
                "title": v.title,
                "vader_label": v.label,
                "vader_compound": v.compound,
                "finbert_label": f.label,
                "finbert_compound": f.compound,
                "agree": agree,
            }
        )

    n = len(rows)
    agreement_rate = sum(r["agree"] for r in rows) / n if n else 0.0
    disagreements = [r for r in rows if not r["agree"]]
    return {
        "n": n,
        "rows": rows,
        "agreement_rate": agreement_rate,
        "disagreements": disagreements,
        "confusion": dict(confusion),
    }


def eval_set_comparison(rows: list[dict[str, str]]) -> dict:
    """Score the hand-labeled eval set with both models and compare accuracy."""
    results = []
    for row in rows:
        vader_scores = vader_score_text(row["text"])
        vader_predicted = vader_label_for_compound(vader_scores["compound"])

        finbert_scores = finbert_score_text(row["text"])
        finbert_predicted = finbert_label_for_scores(finbert_scores)

        results.append(
            {
                "id": row["id"],
                "text": row["text"],
                "category": row["category"],
                "expected_label": row["expected_label"],
                "vader_predicted": vader_predicted,
                "vader_correct": vader_predicted == row["expected_label"],
                "finbert_predicted": finbert_predicted,
                "finbert_correct": finbert_predicted == row["expected_label"],
            }
        )

    def accuracy(key: str, subset: list[dict]) -> float:
        return sum(r[key] for r in subset) / len(subset) if subset else 0.0

    categories = sorted({r["category"] for r in results})
    by_category = {}
    for category in categories:
        subset = [r for r in results if r["category"] == category]
        by_category[category] = {
            "n": len(subset),
            "vader_accuracy": accuracy("vader_correct", subset),
            "finbert_accuracy": accuracy("finbert_correct", subset),
        }

    return {
        "n": len(results),
        "results": results,
        "vader_overall_accuracy": accuracy("vader_correct", results),
        "finbert_overall_accuracy": accuracy("finbert_correct", results),
        "by_category": by_category,
    }


def render_report(corpus: dict, eval_comparison: dict) -> str:
    lines = [
        f"VADER vs FinBERT corpus agreement: {corpus['n']} headlines, "
        f"{corpus['agreement_rate']:.1%} label agreement",
        "",
        "confusion (rows=VADER, cols=FinBERT):",
        "  " + "".join(f"{label:>10}" for label in ("", *LABELS)),
    ]
    for v_label in LABELS:
        counts = [corpus["confusion"].get((v_label, f_label), 0) for f_label in LABELS]
        lines.append(f"  {v_label:>8}" + "".join(f"{c:>10}" for c in counts))

    lines.append("")
    lines.append(f"disagreements ({len(corpus['disagreements'])}/{corpus['n']}):")
    if not corpus["disagreements"]:
        lines.append("  none")
    for r in corpus["disagreements"]:
        lines.append(
            f"  {r['title']!r}: VADER={r['vader_label']} ({r['vader_compound']:+.3f})  "
            f"FinBERT={r['finbert_label']} ({r['finbert_compound']:+.3f})"
        )

    lines.append("")
    lines.append(
        f"eval-set accuracy ({eval_comparison['n']} hand-labeled headlines): "
        f"VADER {eval_comparison['vader_overall_accuracy']:.1%}  "
        f"FinBERT {eval_comparison['finbert_overall_accuracy']:.1%}"
    )
    lines.append("")
    lines.append("by category:")
    for category in sorted(eval_comparison["by_category"]):
        stats = eval_comparison["by_category"][category]
        lines.append(
            f"  {category:<16} n={stats['n']:<3} VADER {stats['vader_accuracy']:.1%}  "
            f"FinBERT {stats['finbert_accuracy']:.1%}"
        )
    lines.append("")
    lines.append("eval-set mismatches where the two models disagree with each other:")
    cross_mismatches = [
        r
        for r in eval_comparison["results"]
        if r["vader_predicted"] != r["finbert_predicted"]
    ]
    if not cross_mismatches:
        lines.append("  none")
    for r in cross_mismatches:
        lines.append(
            f"  [{r['category']}] {r['text']!r}: expected={r['expected_label']} "
            f"VADER={r['vader_predicted']} FinBERT={r['finbert_predicted']}"
        )

    return "\n".join(lines) + "\n"


def run(headlines_path: Path, eval_set_path: Path, out_path: Path) -> int:
    if not headlines_path.exists():
        print(f"no headlines file at {headlines_path}; run sentiment.scrape first", file=sys.stderr)
        return 1
    if not eval_set_path.exists():
        print(f"no eval set at {eval_set_path}", file=sys.stderr)
        return 1

    headlines = read_csv(headlines_path)
    if not headlines:
        print(f"{headlines_path} has no headlines to score", file=sys.stderr)
        return 1

    with eval_set_path.open(newline="", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))

    corpus = corpus_agreement(headlines)
    eval_comparison = eval_set_comparison(eval_rows)
    report = render_report(corpus, eval_comparison)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    print(report, end="")
    print(f"report written to {out_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headlines", type=Path, default=DEFAULT_HEADLINES, help="scraped headlines CSV")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET, help="hand-labeled CSV to evaluate against")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="report file to write")
    args = parser.parse_args()
    sys.exit(run(args.headlines, args.eval_set, args.out))


if __name__ == "__main__":
    main()
