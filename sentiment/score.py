"""Day 2 CLI: score scraped headlines with the VADER baseline.

    python -m sentiment.score
    python -m sentiment.score --in fixtures/headlines/headlines_raw.csv --out outputs/headlines_scored.csv

Reads the CSV that ``sentiment.scrape`` writes, scores each headline's title
with VADER (see ``sentiment/vader_score.py``), and writes a scored CSV plus a
label distribution summary to stdout. Entirely offline - the lexicon is
vendored, not downloaded.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from sentiment.headline import read_csv
from sentiment.vader_score import score_headlines, write_scored_csv

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "headlines_scored.csv"


def run(in_path: Path, out_path: Path) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    if not headlines:
        print(f"{in_path} has no headlines to score", file=sys.stderr)
        return 1

    scored = score_headlines(headlines)
    write_scored_csv(scored, out_path)

    counts = Counter(s.label for s in scored)
    mean_compound = sum(s.compound for s in scored) / len(scored)
    print(f"scored {len(scored)} headlines from {in_path} -> {out_path}")
    print(
        f"labels: positive={counts['positive']} neutral={counts['neutral']} "
        f"negative={counts['negative']}  mean compound={mean_compound:+.4f}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to score")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="scored CSV to write")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path))


if __name__ == "__main__":
    main()
