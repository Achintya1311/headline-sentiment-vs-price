"""Integration day (v0.5): compute the sentiment contract Stock Stalker's
spine reads for one ticker.

    python -m sentiment.contract --ticker INFY.NS
    python -m sentiment.contract --ticker INFY.NS --model finbert
    python -m sentiment.contract --ticker INFY.NS --window-days 7 --out path.json

Writes ``{"sentiment": {"score_7d", "headline_count_7d", "model",
"coverage"}}`` (the shape NEXT_STEPS.md's "Contract to the spine" commits
to) as a file, never a Python import - the same pattern every other
satellite's integration day already used (reverse-dcf-engine's ``--contract``
flag, implied-vol-surface's, etc.).

``score_7d`` averages the chosen scorer's ``compound`` over headlines that
``sentiment.tickers.resolve`` maps to ``--ticker`` and whose Day-4 leak-free
``session_date`` falls in the trailing ``--window-days`` window ending at
``--as-of`` (default: the latest session_date in the fixture, since this
repo's own "now" is inside its most recent aligned session - see README).

``model`` defaults to ``vader``, not the ``finbert`` NEXT_STEPS.md's
illustrative contract names: FinBERT needs ``torch``, which Day 3's README
Limitations already records as not installable in this sandbox. The
contract's ``model`` field always names whichever scorer actually ran - a
contract claiming "finbert" while silently falling back to VADER would be
exactly the kind of unverified number this portfolio avoids. Pass
``--model finbert`` where torch is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from sentiment.correlate import DEFAULT_IN
from sentiment.headline import Headline, read_csv
from sentiment.market_hours import align_headline
from sentiment.tickers import resolve

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "sentiment_contract.json"
DEFAULT_WINDOW_DAYS = 7

# Below this many headlines in the window, coverage is "partial"; a count of
# zero is "none" rather than averaging an empty list.
FULL_COVERAGE_HEADLINES = 5


def _vader_compound(h: Headline) -> float:
    from sentiment.vader_score import score_headline

    return score_headline(h).compound


def _finbert_compound(h: Headline) -> float:
    from sentiment.finbert_score import score_headline

    return score_headline(h).compound


SCORERS: dict[str, Callable[[Headline], float]] = {
    "vader": _vader_compound,
    "finbert": _finbert_compound,
}


def resolved_ticker(h: Headline) -> str | None:
    """The ticker ``sentiment.tickers.resolve`` maps this headline's title to,
    or ``None`` if it names no single company or that company has no known
    ticker (see ``sentiment.tickers``)."""
    match = resolve(h.title)
    if match is None:
        return None
    return match[1]


def latest_session_date(headlines: list[Headline]) -> date | None:
    sessions = [align_headline(h.published_at).session_date for h in headlines]
    return max(sessions) if sessions else None


def headlines_in_window(
    headlines: list[Headline], ticker: str, as_of: date, window_days: int
) -> list[Headline]:
    """Headlines resolving to ``ticker`` whose leak-free session_date falls in
    the trailing ``window_days`` ending at ``as_of`` (inclusive both ends)."""
    target = ticker.upper()
    window_start = as_of - timedelta(days=window_days - 1)
    matched = []
    for h in headlines:
        resolved = resolved_ticker(h)
        if resolved is None or resolved.upper() != target:
            continue
        session_date = align_headline(h.published_at).session_date
        if window_start <= session_date <= as_of:
            matched.append(h)
    return matched


def coverage_for(headline_count: int) -> str:
    if headline_count == 0:
        return "none"
    if headline_count < FULL_COVERAGE_HEADLINES:
        return "partial"
    return "full"


@dataclass(frozen=True)
class SentimentContract:
    ticker: str
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


def build_contract(
    headlines: list[Headline],
    ticker: str,
    model: str,
    as_of: date | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> SentimentContract:
    if model not in SCORERS:
        raise ValueError(f"unknown model {model!r}; choose one of {sorted(SCORERS)}")
    resolved_as_of = as_of if as_of is not None else latest_session_date(headlines)
    if resolved_as_of is None:
        return SentimentContract(ticker=ticker, score_7d=None, headline_count_7d=0, model=model, coverage="none")

    matched = headlines_in_window(headlines, ticker, resolved_as_of, window_days)
    score_fn = SCORERS[model]
    score_7d = sum(score_fn(h) for h in matched) / len(matched) if matched else None
    return SentimentContract(
        ticker=ticker,
        score_7d=score_7d,
        headline_count_7d=len(matched),
        model=model,
        coverage=coverage_for(len(matched)),
    )


def run(
    in_path: Path, ticker: str, model: str, as_of: date | None, window_days: int, out_path: Path
) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        contract = build_contract(headlines, ticker, model, as_of, window_days)
    except ModuleNotFoundError as exc:
        print(f"cannot score with --model {model}: {exc} (see README Limitations)", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(contract.to_dict(), indent=2) + "\n")

    print(f"sentiment contract for {ticker} ({model}) -> {out_path}")
    print(
        f"score_7d={contract.score_7d if contract.score_7d is None else f'{contract.score_7d:+.4f}'}  "
        f"headline_count_7d={contract.headline_count_7d}  coverage={contract.coverage}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--ticker", required=True, help="ticker to compute the trailing-window sentiment contract for")
    parser.add_argument("--model", choices=sorted(SCORERS), default="vader", help="scorer to aggregate (default: vader)")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None, help="window end date, YYYY-MM-DD (default: latest session in the fixture)")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS, help="trailing window size in calendar days")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="contract JSON to write")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.ticker, args.model, args.as_of, args.window_days, args.out))


if __name__ == "__main__":
    main()
