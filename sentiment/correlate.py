"""Day 5 CLI: contemporaneous vs lagged sentiment/return correlation, plus an
event study on high-magnitude-sentiment headlines.

    python -m sentiment.correlate
    python -m sentiment.correlate --live      # re-fetch prices instead of fixtures
    python -m sentiment.correlate --event-threshold 0.3

Scope: only headlines that ``sentiment.tickers.resolve`` maps to exactly one
NSE ticker are used - a screener-list headline ("X among N stocks...") names
a company only as one example among several, and pairing it with X's return
alone would overstate what the headline is about. See ``sentiment/tickers.py``
and the README's Day 5 Findings for the full accounting of what was included,
excluded, and why.

For each resolved headline:

- ``contemporaneous_return``: open-to-close return of the trading session
  Day 4's ``align_headline`` assigns it (the first session whose open comes
  strictly after the headline was published - the first return the market
  had available to react with).
- ``lagged_return``: open-to-close return of the *next* trading session, or
  blank if that session has not happened yet (see README - this run's "now"
  is inside the most recent aligned session, so same-day headlines have no
  lagged return to report yet).

Both are correlated against VADER's ``compound`` score (Pearson r with a
95% CI). The event study splits headlines into "high-magnitude" (|compound|
above ``--event-threshold``) and the rest, and bootstraps a 95% CI for the
difference in mean contemporaneous return between the two groups.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sentiment.headline import Headline, read_csv
from sentiment.market_hours import align_headline
from sentiment.prices import PriceFetchError, bar_on, load_bars, next_session_bar
from sentiment.stats import bootstrap_mean_diff_ci, pearson_with_ci
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "correlation.csv"

# A headline in the "Share Price Highlights" liveblog template scores 0.296
# regardless of what actually happened to the stock (see README Findings) -
# above that, a headline is carrying sentiment content beyond the template.
DEFAULT_EVENT_THRESHOLD = 0.30

ROW_FIELDNAMES = [
    "title",
    "company",
    "ticker",
    "session_date",
    "compound",
    "contemporaneous_return",
    "lagged_return",
]


def build_rows(in_path: Path, live: bool = False) -> tuple[list[dict], list[tuple[str, str]]]:
    """Return (rows, unresolved) where ``unresolved`` is [(title, company), ...]
    for headlines whose company was recognised but whose ticker could not be
    fetched (see ``sentiment.tickers``)."""
    return rows_from_headlines(read_csv(in_path), live=live)


def rows_from_headlines(headlines: list[Headline], live: bool = False) -> tuple[list[dict], list[tuple[str, str]]]:
    """Same pairing ``build_rows`` does, but against already-loaded headlines
    rather than a CSV path. Lets Day 8's audit module rerun the alignment +
    pricing pipeline against a shuffled-timestamp copy of the headlines
    without writing a CSV to disk for every shuffle."""
    rows: list[dict] = []
    unresolved: list[tuple[str, str]] = []
    price_errors: list[tuple[str, str]] = []

    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        company, ticker = match
        if ticker is None:
            unresolved.append((h.title, company))
            continue

        alignment = align_headline(h.published_at)
        try:
            bars = load_bars(ticker, live=live)
        except PriceFetchError as exc:
            price_errors.append((ticker, str(exc)))
            continue

        session_bar = bar_on(bars, alignment.session_date)
        if session_bar is None:
            price_errors.append((ticker, f"no bar for session_date {alignment.session_date}"))
            continue
        lagged_bar = next_session_bar(bars, alignment.session_date)

        scored = score_headline(h)
        rows.append(
            {
                "title": h.title,
                "company": company,
                "ticker": ticker,
                "session_date": alignment.session_date.isoformat(),
                "compound": scored.compound,
                "contemporaneous_return": session_bar.session_return,
                "lagged_return": lagged_bar.session_return if lagged_bar else None,
            }
        )

    if price_errors:
        for ticker, msg in price_errors:
            print(f"warning: {ticker}: {msg}", file=sys.stderr)

    return rows, unresolved


def write_rows_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run(in_path: Path, out_path: Path, live: bool, event_threshold: float) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    rows, unresolved = build_rows(in_path, live=live)
    if not rows:
        print("no headlines resolved to a fetchable ticker; nothing to correlate", file=sys.stderr)
        return 1

    write_rows_csv(rows, out_path)

    contemporaneous = [r["contemporaneous_return"] for r in rows]
    compounds = [r["compound"] for r in rows]
    lagged_rows = [r for r in rows if r["lagged_return"] is not None]

    print(f"{len(rows)} headlines resolved to a ticker with price data ({out_path})")
    if unresolved:
        names = ", ".join(f"{c} ({t[:40]}...)" for t, c in unresolved)
        print(f"{len(unresolved)} headline(s) named a company with no resolvable ticker: {names}")

    if len(rows) >= 2:
        contemp_stat = pearson_with_ci(compounds, contemporaneous)
        print(
            f"contemporaneous: r={contemp_stat.r:+.3f}  95% CI [{contemp_stat.ci_low:+.3f}, "
            f"{contemp_stat.ci_high:+.3f}]  n={contemp_stat.n}"
        )
    else:
        print(f"contemporaneous: not enough resolved headlines for a correlation (n={len(rows)})")

    if len(lagged_rows) >= 2:
        lagged_stat = pearson_with_ci(
            [r["compound"] for r in lagged_rows], [r["lagged_return"] for r in lagged_rows]
        )
        print(
            f"lagged:          r={lagged_stat.r:+.3f}  95% CI [{lagged_stat.ci_low:+.3f}, "
            f"{lagged_stat.ci_high:+.3f}]  n={lagged_stat.n}"
        )
    else:
        print(f"lagged:          not enough sessions with a next trading day yet (n={len(lagged_rows)})")

    high = [r["contemporaneous_return"] for r in rows if abs(r["compound"]) > event_threshold]
    low = [r["contemporaneous_return"] for r in rows if abs(r["compound"]) <= event_threshold]
    if high and low:
        event = bootstrap_mean_diff_ci(high, low)
        print(
            f"event study (|compound| > {event_threshold}): "
            f"mean return high={sum(high) / len(high):+.4f} (n={event.n_a})  "
            f"low={sum(low) / len(low):+.4f} (n={event.n_b})  "
            f"diff={event.diff:+.4f}  95% CI [{event.ci_low:+.4f}, {event.ci_high:+.4f}]"
        )
    else:
        print(f"event study: threshold {event_threshold} leaves one group empty (high={len(high)}, low={len(low)})")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="per-headline CSV to write")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=DEFAULT_EVENT_THRESHOLD,
        help="|compound| above this is 'high-magnitude' for the event study",
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.live, args.event_threshold))


if __name__ == "__main__":
    main()
