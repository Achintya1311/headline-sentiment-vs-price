"""Refresh fixtures/prices/*.json for every ticker sentiment.tickers knows (D26, hub CRITIQUES.md).

    python -m sentiment.fetch_prices

Meant to run from ``.github/workflows/fetch-prices.yml`` on a schedule, the
same pattern ``collect-headlines.yml`` already uses in this repo and
``fetch-market-price.yml``/``fetch-largecaps.yml`` use elsewhere in this
portfolio. Not a new fetch path: it is ``sentiment.prices.load_bars(...,
live=True)``, the same ``--live`` path ``sentiment.correlate --live`` already
exercises per-headline, just run for every known ticker up front instead of
only the ones a resolved headline happens to name. ``load_bars``'s own merge
(``merge_bars``) keeps whatever a ticker's fixture already covers, so running
this on a schedule grows coverage over time rather than only ever holding the
one trailing month Yahoo's chart endpoint returns per call.
"""

from __future__ import annotations

import sys

from sentiment.prices import PriceFetchError, load_bars
from sentiment.tickers import known_tickers


def run() -> int:
    tickers = known_tickers()
    failures: list[tuple[str, str]] = []
    for ticker in tickers:
        try:
            bars = load_bars(ticker, live=True)
        except PriceFetchError as exc:
            failures.append((ticker, str(exc)))
            print(f"warning: {ticker}: {exc}", file=sys.stderr)
            continue
        span = f"{bars[0].date} to {bars[-1].date}" if bars else "no bars"
        print(f"{ticker}: {len(bars)} bars ({span})")

    print(f"{len(tickers) - len(failures)}/{len(tickers)} tickers refreshed")
    if failures:
        print(f"{len(failures)} ticker(s) failed to fetch (see warnings above)", file=sys.stderr)
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
