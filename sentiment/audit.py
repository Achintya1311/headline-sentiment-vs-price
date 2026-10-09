"""Day 8 CLI: the ml-pipeline-audit / leakage pass.

``NEXT_STEPS.md``'s "Done when" criterion: shuffle the headline timestamps
and the signal must disappear. If a shuffled-timestamp control still
predicts returns, the pipeline is leaking and the result is an artifact.

Mechanically: reassign each headline's ``published_at`` to a different
headline's original timestamp (a random permutation of the same 50
timestamps across the same 50 titles), re-run Day 5's alignment and
contemporaneous-return pairing on the shuffled timestamps, and recompute
the Pearson correlation against VADER ``compound``. Doing this many times
builds a null distribution; the real (true-timestamp) estimate is compared
against it.

On this repo's own fixture, this permutation test is a weaker check than
it sounds: Day 5/6/7 already found the real contemporaneous correlation is
null (r=-0.185, 95% CI contains 0). A test that can only run on data with
no real signal in the first place cannot prove it would catch a genuine
leak - "shuffling an already-null estimate stays null" is also exactly
what a broken, non-leaking test would report. ``tests/test_audit.py``
closes that gap with a synthetic, ground-truth scenario where sentiment
*does* predict contemporaneous return by construction, and shows a fixed
(non-random) derangement of the timestamps collapses that correlation -
proof this audit has the power to fail, not just the opportunity to pass.

    python -m sentiment.audit
    python -m sentiment.audit --shuffles 1000 --seed 1
    python -m sentiment.audit --live
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.headline import Headline, read_csv
from sentiment.market_hours import align_headline
from sentiment.prices import Bar, PriceFetchError, bar_on, load_bars
from sentiment.stats import pearson_with_ci
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_SHUFFLES = 500
DEFAULT_SEED = 0


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` permuted across them.

    Titles, links and (downstream) tickers and compound scores stay with
    their original headline - only which timestamp each headline carries is
    reassigned. This preserves the exact marginal distribution of both the
    50 titles and the 50 timestamps while destroying any real correspondence
    between a headline's content and the trading session it lands in.
    """
    shuffled_ts = [h.published_at for h in headlines]
    rng.shuffle(shuffled_ts)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_ts)]


def tickers_for(headlines: list[Headline]) -> set[str]:
    """Every ticker Day 5's ``tickers.resolve`` would pair at least one
    headline's title with. Timestamps never affect this set - it only
    depends on title text - so it can be computed once and reused for every
    shuffled trial instead of re-resolving (and re-fetching prices for)
    the same tickers on every permutation.
    """
    out: set[str] = set()
    for h in headlines:
        match = resolve(h.title)
        if match and match[1]:
            out.add(match[1])
    return out


def load_all_bars(tickers: set[str], live: bool = False) -> dict[str, list[Bar]]:
    bars_by_ticker: dict[str, list[Bar]] = {}
    for ticker in tickers:
        try:
            bars_by_ticker[ticker] = load_bars(ticker, live=live)
        except PriceFetchError as exc:
            print(f"warning: {ticker}: {exc}", file=sys.stderr)
    return bars_by_ticker


def contemporaneous_r(
    headlines: list[Headline], bars_by_ticker: dict[str, list[Bar]]
) -> tuple[float | None, int]:
    """Pearson r (and n) between VADER ``compound`` and each headline's own
    ``published_at``-aligned contemporaneous return, given each ticker's
    bars. ``None`` if fewer than 2 headlines resolve to both a ticker with
    bars and a bar on their aligned session date."""
    compounds: list[float] = []
    returns: list[float] = []
    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        _, ticker = match
        if ticker is None or ticker not in bars_by_ticker:
            continue
        alignment = align_headline(h.published_at)
        session_bar = bar_on(bars_by_ticker[ticker], alignment.session_date)
        if session_bar is None:
            continue
        compounds.append(score_headline(h).compound)
        returns.append(session_bar.session_return)

    if len(compounds) < 2:
        return None, len(compounds)
    return pearson_with_ci(compounds, returns).r, len(compounds)


@dataclass(frozen=True)
class AuditResult:
    real_r: float | None
    real_n: int
    shuffled_rs: list[float]
    p_value: float | None  # two-sided empirical: P(|shuffled r| >= |real r|)


def run_audit(
    in_path: Path, n_shuffles: int = DEFAULT_SHUFFLES, seed: int = DEFAULT_SEED, live: bool = False
) -> AuditResult:
    headlines = read_csv(in_path)
    bars_by_ticker = load_all_bars(tickers_for(headlines), live=live)

    real_r, real_n = contemporaneous_r(headlines, bars_by_ticker)

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        r, _ = contemporaneous_r(shuffled, bars_by_ticker)
        if r is not None:
            shuffled_rs.append(r)

    p_value = None
    if real_r is not None and shuffled_rs:
        p_value = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r)) / len(shuffled_rs)

    return AuditResult(real_r=real_r, real_n=real_n, shuffled_rs=shuffled_rs, p_value=p_value)


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    result = run_audit(in_path, n_shuffles=n_shuffles, seed=seed, live=live)

    if result.real_r is None:
        print(f"real (true-timestamp) correlation: not enough resolved headlines (n={result.real_n})")
        return 1
    print(f"real (true-timestamp) correlation: r={result.real_r:+.3f}  n={result.real_n}")

    if not result.shuffled_rs:
        print("no shuffled trial produced a computable correlation; cannot build a null distribution")
        return 1

    abs_shuffled = sorted(abs(r) for r in result.shuffled_rs)
    mean_abs = sum(abs_shuffled) / len(abs_shuffled)
    p95 = abs_shuffled[int(0.95 * len(abs_shuffled)) - 1]
    print(
        f"shuffled-timestamp null ({len(result.shuffled_rs)} trials): "
        f"mean |r|={mean_abs:.3f}  95th pct |r|={p95:.3f}"
    )
    print(f"two-sided empirical p-value: {result.p_value:.3f}")

    if result.p_value is not None and result.p_value < 0.05:
        print(
            "FAIL: the real estimate is a statistically significant outlier against the "
            "shuffled null - a shuffled-timestamp control should not still predict returns. "
            "Something in alignment or pairing is leaking."
        )
        return 1

    print(
        "PASS (weakly): the real estimate is not distinguishable from shuffled noise. On this "
        "fixture the real correlation was already null (see README Day 5-7 Findings), so this "
        "run cannot by itself prove the audit would catch a genuine leak - see "
        "tests/test_audit.py's synthetic check for that."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--shuffles", type=int, default=DEFAULT_SHUFFLES, help="number of timestamp-shuffle trials"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for reproducible shuffles")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
