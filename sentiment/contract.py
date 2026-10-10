"""Day 8 integration: the v0.5 ``sentiment`` contract this repo publishes to
Stock Stalker.

    python -m sentiment.contract --ticker FORTIS.NS
    python -m sentiment.contract --ticker RELIANCE.NS --contract outputs/sentiment_contract.json

NEXT_STEPS.md fixed this contract's shape before any of this module existed::

    {"sentiment": {"score_7d": 0.31, "headline_count_7d": 12, "model": "finbert", "coverage": "partial"}}

Two honest deviations from that plan, named here rather than hidden:

- ``model`` ships ``"vader"``, not ``"finbert"``. FinBERT
  (``sentiment/finbert.py``, Day 3) needs torch, which this sandbox could
  not install - a documented, pre-existing gap every day's README since
  Day 3 has recorded. There is no working FinBERT score to report, so
  claiming one here would be fabricated.
- ``score_7d`` is the mean VADER compound over whatever headlines for
  ``--ticker`` exist in this repo's own corpus within a trailing window of
  their latest publish time, not a rolling window measured against
  wall-clock "today". Every headline in
  ``fixtures/headlines/headlines_raw.csv`` was scraped on the same single
  calendar day (Mon 28 Sep 2026), so a genuine 7-*calendar*-day trailing
  window is never actually exercised by this fixture - it always equals
  "every headline this corpus has for that ticker".

This module never imports anything about Stock Stalker's screen, and
nothing here fails because the ticker it is pointed at has zero matching
headlines - ``coverage`` reports ``"none"`` rather than raising, the same
shape reverse-dcf-engine's and implied-vol-surface's own CLIs follow for a
ticker outside what they actually researched.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from sentiment.headline import Headline, read_csv
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
TRAILING_WINDOW_DAYS = 7
MODEL = "vader"


@dataclass(frozen=True)
class SentimentContractResult:
    ticker: str
    score_7d: float | None
    headline_count_7d: int
    model: str
    coverage: str  # "full" (every resolved headline falls in the window), "partial", or "none"

    def to_contract(self) -> dict:
        """The ``sentiment`` block this repo publishes to the spine (v0.5)."""
        return {
            "sentiment": {
                "score_7d": self.score_7d,
                "headline_count_7d": self.headline_count_7d,
                "model": self.model,
                "coverage": self.coverage,
            }
        }


def _headlines_for_ticker(headlines: list[Headline], ticker: str) -> list[Headline]:
    """Headlines whose ``sentiment.tickers.resolve`` match is exactly ``ticker``."""
    target = ticker.upper()
    matched = []
    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        _, mapped_ticker = match
        if mapped_ticker is not None and mapped_ticker.upper() == target:
            matched.append(h)
    return matched


def compute_sentiment_contract(in_path: Path, ticker: str) -> SentimentContractResult:
    """Build the trailing sentiment contract for ``ticker`` from the headlines at ``in_path``."""
    headlines = read_csv(in_path)
    resolved = _headlines_for_ticker(headlines, ticker)

    if not resolved:
        return SentimentContractResult(
            ticker=ticker, score_7d=None, headline_count_7d=0, model=MODEL, coverage="none"
        )

    latest = max(h.published_at for h in resolved)
    window_start = latest - timedelta(days=TRAILING_WINDOW_DAYS)
    windowed = [h for h in resolved if h.published_at >= window_start]
    scored = [score_headline(h) for h in windowed]
    mean_compound = sum(s.compound for s in scored) / len(scored)
    coverage = "full" if len(windowed) == len(resolved) else "partial"
    return SentimentContractResult(
        ticker=ticker,
        score_7d=mean_compound,
        headline_count_7d=len(windowed),
        model=MODEL,
        coverage=coverage,
    )


def run(in_path: Path, ticker: str, contract_path: Path | None) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    result = compute_sentiment_contract(in_path, ticker)
    score_repr = f"{result.score_7d:+.4f}" if result.score_7d is not None else "None"
    print(
        f"{result.ticker}: score_7d={score_repr}  headline_count_7d={result.headline_count_7d}  "
        f"model={result.model}  coverage={result.coverage}"
    )

    if contract_path is not None:
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(result.to_contract(), indent=2) + "\n")
        print(f"wrote sentiment contract to {contract_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--ticker", required=True, help="the ticker to report a trailing sentiment score for, e.g. FORTIS.NS"
    )
    parser.add_argument(
        "--contract",
        dest="contract_path",
        metavar="PATH",
        type=Path,
        default=None,
        help="write the v0.5 sentiment contract block (see to_contract()) as JSON to this path, "
        "for the spine to read as a file -- never as a Python import",
    )
    args = parser.parse_args(argv)
    return run(args.in_path, args.ticker, args.contract_path)


if __name__ == "__main__":
    sys.exit(main())
