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
  had available to react with). When that weekday was an NSE holiday (no
  bar in the ticker's fixture), the ticker's next bar is used instead (P135).
- ``lagged_return``: open-to-close return of the *next* trading session, or
  blank if that session has not happened yet (see README - this run's "now"
  is inside the most recent aligned session, so same-day headlines have no
  lagged return to report yet).
- ``gap_return`` (P159): previous bar's close to the session bar's open, only
  for a headline published after that close (15:30 IST) and before 09:00 IST
  on the session date, when the pre-open call auction that sets the open
  starts; blank otherwise. Open-to-close starts after the open, so for news
  released outside market hours it misses the first price reaction. Printed
  as an exploratory line, not the pre-registered test.

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

from sentiment.headline import read_csv
from datetime import datetime, time

from sentiment.market_hours import IST, MARKET_CLOSE, align_headline
from sentiment.prices import Bar, PriceFetchError, load_bars, next_session_bar, session_bar
from sentiment.stats import bootstrap_mean_diff_ci, pearson_with_ci
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline
from sentiment import DISCLAIMER

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
    "source",
    "session_date",
    "compound",
    "contemporaneous_return",
    "lagged_return",
    "gap_return",
]

# NSE's pre-open call auction (09:00-09:08 IST) sets the open price, so a
# headline from 09:00 on may already be in it: no leak-free gap for it.
PRE_OPEN_AUCTION = time(9, 0)


def gap_return(published_at: datetime, bars: list[Bar], bar: Bar) -> float | None:
    """Previous close to ``bar``'s open, or None when that gap is not
    entirely after the headline (P159).

    The previous close is the last bar before ``bar`` in the ticker's own
    fixture, so a holiday-rolled session gaps from the last real close. The
    headline must come strictly after that close and before 09:00 IST on
    ``bar``'s date."""
    prev = None
    for b in bars:
        if b.date >= bar.date:
            break
        prev = b
    if prev is None:
        return None
    local = published_at.astimezone(IST)
    after_close = datetime.combine(prev.date, MARKET_CLOSE, tzinfo=IST)
    before_auction = datetime.combine(bar.date, PRE_OPEN_AUCTION, tzinfo=IST)
    if not (after_close < local < before_auction):
        return None
    return bar.open / prev.close - 1.0


def build_rows_from_headlines(
    headlines: list, live: bool = False, score_fn=score_headline
) -> tuple[list[dict], list[tuple[str, str]], list[tuple[str, str]]]:
    """Same as ``build_rows`` but starting from an already-loaded list of
    ``Headline`` objects rather than a CSV path - what ``build_rows`` itself
    uses, and what Day 8's leakage audit needs to rerun this pipeline against
    headlines whose timestamps it has shuffled in memory, without writing a
    scratch CSV just to read it back.

    Returns ``(rows, unresolved, price_errors)``: ``unresolved`` is headlines
    whose company is known but has no resolvable ticker; ``price_errors`` is
    headlines resolved to a ticker that still could not be paired with a
    price bar (D26, hub ``CRITIQUES.md`` - previously only logged to stderr,
    never counted by a caller).

    ``score_fn`` defaults to VADER (this function's original scorer) but
    accepts any ``Headline -> <has .compound>`` callable - ``sentiment.prereg``
    (Day 84, R18) passes FinBERT's ``score_headline`` instead, so the
    pre-registered statistic is never silently scored with VADER by reusing
    this same resolution/pairing logic."""
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

        bar = session_bar(bars, alignment.session_date)
        if bar is None:
            price_errors.append((ticker, f"no bar for session_date {alignment.session_date}"))
            continue
        lagged_bar = next_session_bar(bars, bar.date)

        scored = score_fn(h)
        rows.append(
            {
                "title": h.title,
                "company": company,
                "ticker": ticker,
                "source": h.source,
                "session_date": bar.date.isoformat(),
                "compound": scored.compound,
                "contemporaneous_return": bar.session_return,
                "lagged_return": lagged_bar.session_return if lagged_bar else None,
                "gap_return": gap_return(h.published_at, bars, bar),
            }
        )

    if price_errors:
        for ticker, msg in price_errors:
            print(f"warning: {ticker}: {msg}", file=sys.stderr)

    return rows, unresolved, price_errors


def build_rows(
    in_path: Path, live: bool = False, score_fn=score_headline
) -> tuple[list[dict], list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (rows, unresolved, price_errors) - see
    ``build_rows_from_headlines`` for what each list holds."""
    headlines = read_csv(in_path)
    return build_rows_from_headlines(headlines, live=live, score_fn=score_fn)


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

    rows, unresolved, price_errors = build_rows(in_path, live=live)
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
    if price_errors:
        print(
            f"{len(price_errors)} headline(s) resolved to a ticker but had no price bar for their "
            f"session and were dropped (see warnings above)"
        )

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

    gap_rows = [r for r in rows if r["gap_return"] is not None]
    if len(gap_rows) >= 2:
        gap_stat = pearson_with_ci([r["compound"] for r in gap_rows], [r["gap_return"] for r in gap_rows])
        print(
            f"overnight gap:   r={gap_stat.r:+.3f}  95% CI [{gap_stat.ci_low:+.3f}, "
            f"{gap_stat.ci_high:+.3f}]  n={gap_stat.n}  (previous close to open, headlines after the "
            f"close and before 09:00 IST only; exploratory, not the pre-registered test)"
        )
    else:
        print(f"overnight gap:   not enough out-of-hours headlines for a correlation (n={len(gap_rows)})")

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
    print(DISCLAIMER)
    sys.exit(run(args.in_path, args.out_path, args.live, args.event_threshold))


if __name__ == "__main__":
    main()
