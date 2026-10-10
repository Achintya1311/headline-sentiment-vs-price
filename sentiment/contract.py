"""Integration day: the v0.5 sentiment contract this repo publishes to the spine.

    python -m sentiment.contract --ticker INFY.NS
    python -m sentiment.contract --ticker INFY.NS --contract outputs/sentiment_contract.json
    python -m sentiment.contract --ticker INFY.NS --window-days 14

For one ticker, averages VADER ``compound`` over the headlines
``sentiment.tickers.resolve`` maps to that ticker within a trailing window
(default 7 days) of the sample's own latest headline - this fixture has no
live "today" to anchor to, so "trailing 7 days" means the 7 days ending at
whatever the most recent committed headline's timestamp is, the same
fixture-relative-"now" choice Day 5/6 already made for lagged returns.

Writes the block as a file via ``--contract``, never as a Python import -
same convention every other satellite in this portfolio follows (see
reverse-dcf-engine's ``reverse_dcf.solve --contract`` for the pattern this
copies).

**Model deviation, recorded here and in README Limitations:** the contract
shape NEXT_STEPS.md committed to before this module existed shows
``"model": "finbert"`` as its example value. FinBERT cannot actually run in
this sandbox (torch is not installed here - a gap every checkpoint since
Day 3 has documented), so every contract this CLI writes honestly reports
``"model": "vader"`` instead. Shipping "finbert" because the fixture
*looks* like it would use it, when it demonstrably can't run here, would be
exactly the kind of invented number this portfolio exists to avoid.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from sentiment.correlate import DEFAULT_IN
from sentiment.headline import Headline, read_csv
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_WINDOW_DAYS = 7
MODEL = "vader"
COVERAGE = "partial"  # single RSS source (Economic Times only) - see README Findings/Day 1


@dataclass(frozen=True)
class SentimentContract:
    score_7d: float | None
    headline_count_7d: int
    model: str
    coverage: str

    def to_dict(self) -> dict:
        return {
            "sentiment": {
                "score_7d": self.score_7d,
                "headline_count_7d": self.headline_count_7d,
                "model": self.model,
                "coverage": self.coverage,
            }
        }


def headlines_for_ticker(headlines: list[Headline], ticker: str) -> list[Headline]:
    """Headlines ``sentiment.tickers.resolve`` maps to exactly this ticker."""
    matches = []
    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        _, resolved_ticker = match
        if resolved_ticker == ticker:
            matches.append(h)
    return matches


def build_contract(
    headlines: list[Headline], ticker: str, window_days: int = DEFAULT_WINDOW_DAYS
) -> SentimentContract:
    """Build the trailing-window contract for ``ticker``.

    The window anchors to the latest ``published_at`` in ``headlines``
    itself (there being no live "today" in a fixture-only sandbox), not to
    the wall clock - a run against a wider or more recent scrape would
    anchor later automatically.
    """
    ticker_headlines = headlines_for_ticker(headlines, ticker)
    if not ticker_headlines:
        return SentimentContract(score_7d=None, headline_count_7d=0, model=MODEL, coverage=COVERAGE)

    now = max(h.published_at for h in ticker_headlines)
    window_start = now - timedelta(days=window_days)
    windowed = [h for h in ticker_headlines if window_start <= h.published_at <= now]
    if not windowed:
        return SentimentContract(score_7d=None, headline_count_7d=0, model=MODEL, coverage=COVERAGE)

    scores = [score_headline(h).compound for h in windowed]
    return SentimentContract(
        score_7d=sum(scores) / len(scores),
        headline_count_7d=len(windowed),
        model=MODEL,
        coverage=COVERAGE,
    )


def run(in_path: Path, ticker: str, window_days: int, contract_path: Path | None) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    contract = build_contract(headlines, ticker, window_days=window_days)

    print(f"sentiment contract for {ticker} (trailing {window_days}d of this sample):")
    print(f"  score_7d          = {contract.score_7d:+.4f}" if contract.score_7d is not None else "  score_7d          = null (no headlines in window)")
    print(f"  headline_count_7d = {contract.headline_count_7d}")
    print(f"  model             = {contract.model}")
    print(f"  coverage          = {contract.coverage}")

    if contract_path:
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(contract.to_dict(), indent=2) + "\n")
        print(f"\nwrote sentiment contract to {contract_path}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--ticker", required=True, help="Yahoo Finance ticker to build the contract for, e.g. INFY.NS")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="trailing window size in days")
    parser.add_argument(
        "--contract",
        metavar="PATH",
        dest="contract_path",
        type=Path,
        default=None,
        help="write the v0.5 sentiment contract block (see to_dict()) as JSON to this path, "
        "for the spine to read as a file -- never as a Python import",
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.ticker, args.window_days, args.contract_path))


if __name__ == "__main__":
    main()
