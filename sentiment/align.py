"""Day 4 CLI: align scraped headlines to a leak-free trading session.

    python -m sentiment.align
    python -m sentiment.align --in fixtures/headlines/headlines_raw.csv \\
        --out outputs/headlines_aligned.csv

Reads the CSV ``sentiment.scrape`` writes, computes each headline's
``market_hours.Alignment`` (see that module for the rule), and writes a CSV
with the original columns plus ``local_time``, ``timing``, and
``session_date`` - the trading day whose close-to-close return Day 5's
correlation work may attribute to this headline without look-ahead.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from sentiment.headline import read_csv
from sentiment.market_hours import align_headline

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "headlines_aligned.csv"

FIELDNAMES = [
    "source",
    "title",
    "link",
    "published_at",
    "local_time",
    "timing",
    "session_date",
]


def align_rows(headlines) -> list[dict]:
    rows = []
    for h in headlines:
        alignment = align_headline(h.published_at)
        assert alignment.leak_free()
        rows.append(
            {
                "source": h.source,
                "title": h.title,
                "link": h.link,
                "published_at": h.published_at.isoformat(),
                "local_time": alignment.local_time.isoformat(),
                "timing": alignment.timing.value,
                "session_date": alignment.session_date.isoformat(),
            }
        )
    return rows


def write_aligned_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run(in_path: Path, out_path: Path) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    if not headlines:
        print(f"{in_path} has no headlines to align", file=sys.stderr)
        return 1

    rows = align_rows(headlines)
    write_aligned_csv(rows, out_path)

    counts = Counter(r["timing"] for r in rows)
    sessions = sorted({r["session_date"] for r in rows})
    print(f"aligned {len(rows)} headlines from {in_path} -> {out_path}")
    print(
        f"timing: pre_open={counts['pre_open']} intraday={counts['intraday']} "
        f"post_close={counts['post_close']}"
    )
    print(f"sessions spanned: {', '.join(sessions)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to align")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="aligned CSV to write")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path))


if __name__ == "__main__":
    main()
