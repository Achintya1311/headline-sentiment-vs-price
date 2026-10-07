"""Day 8 CLI: the ml-pipeline-audit leakage/shuffle control.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 5000 --seed 1
    python -m sentiment.audit --live

NEXT_STEPS.md's "Done when" section sets the bar for this whole repo: shuffle
the headline timestamps and the signal must disappear. If a shuffled-
timestamp control still predicts returns, the pipeline is leaking - some
alignment or resolution bug is correlating headlines with returns regardless
of when the headline was actually published, and Day 5-7's results would be
an artifact of that bug, not a measurement of anything real.

This CLI is the permutation test that claim rests on. Each resolved headline
(same resolution as ``sentiment.correlate``: exactly one named company,
mapped to a fetchable ticker) keeps its own VADER ``compound`` score, but its
``published_at`` is reassigned to another resolved headline's timestamp -
this breaks the link between what a headline says and which trading session
it lands on, while leaving the marginal distribution of both compounds and
timestamps untouched. Repeating that thousands of times with different
random permutations gives a null distribution for the contemporaneous
Pearson r Day 5 reported; the real, unshuffled r is compared against it.

A real r that sits comfortably inside the shuffled null distribution (large
two-sided p-value) is consistent with there being no timestamp-dependent
signal in this pipeline, leaking or otherwise - exactly what Day 5/6's
already-near-zero correlations would predict. A real r far out in the tail
of the shuffled distribution would be the opposite: evidence that something
in the resolution/alignment/pricing path produces a signal independent of
real timing, i.e. a leak. See the README's Day 8 Findings and Limitations
for why, on this fixture, this test cannot do more than fail to find a leak
- it cannot prove one is absent.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sentiment.correlate import DEFAULT_IN
from sentiment.headline import read_csv
from sentiment.market_hours import align_headline
from sentiment.prices import Bar, PriceFetchError, bar_on, load_bars
from sentiment.stats import pearson_r
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_N_SHUFFLES = 2000
DEFAULT_SEED = 0
MIN_POINTS = 2


@dataclass(frozen=True)
class Resolved:
    """One headline that resolved to a ticker with price data, with its own
    compound score and bars already attached - the audit only ever reassigns
    ``published_at`` across this fixed list, so ticker resolution, scoring,
    and price loading each happen once, not once per shuffle trial."""

    title: str
    ticker: str
    compound: float
    published_at: datetime
    bars: list[Bar]


@dataclass(frozen=True)
class ShuffleResult:
    real_r: float
    real_n: int
    null_rs: list[float]
    p_value: float


def resolve_headlines(in_path: Path, live: bool = False) -> list[Resolved]:
    headlines = read_csv(in_path)
    resolved: list[Resolved] = []
    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        _, ticker = match
        if ticker is None:
            continue
        try:
            bars = load_bars(ticker, live=live)
        except PriceFetchError:
            continue
        resolved.append(
            Resolved(
                title=h.title,
                ticker=ticker,
                compound=score_headline(h).compound,
                published_at=h.published_at,
                bars=bars,
            )
        )
    return resolved


def contemporaneous_return(headline: Resolved, published_at: datetime) -> float | None:
    """The open-to-close return of ``headline``'s own ticker, on the session
    ``published_at`` (not necessarily ``headline.published_at`` - this is how
    a shuffle trial re-pairs a headline's price series with someone else's
    timestamp) aligns to. ``None`` if that session has no bar in the fixture."""
    alignment = align_headline(published_at)
    bar = bar_on(headline.bars, alignment.session_date)
    return bar.session_return if bar else None


def correlation_for_timestamps(resolved: list[Resolved], timestamps: list[datetime]) -> tuple[float, int] | None:
    """Pearson r between each headline's compound and the contemporaneous
    return produced by pairing it with the matching entry of ``timestamps``
    (same order, same length as ``resolved``). ``None`` if fewer than
    ``MIN_POINTS`` headlines land on a session with a bar."""
    xs: list[float] = []
    ys: list[float] = []
    for headline, ts in zip(resolved, timestamps):
        ret = contemporaneous_return(headline, ts)
        if ret is not None:
            xs.append(headline.compound)
            ys.append(ret)
    if len(xs) < MIN_POINTS:
        return None
    return pearson_r(xs, ys), len(xs)


def run_shuffle_test(resolved: list[Resolved], n_shuffles: int, seed: int) -> ShuffleResult:
    real = correlation_for_timestamps(resolved, [r.published_at for r in resolved])
    if real is None:
        raise ValueError(f"only {len(resolved)} resolved headline(s) - not enough for a correlation")
    real_r, real_n = real

    rng = random.Random(seed)
    timestamps = [r.published_at for r in resolved]
    null_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = timestamps[:]
        rng.shuffle(shuffled)
        result = correlation_for_timestamps(resolved, shuffled)
        if result is not None:
            null_rs.append(result[0])

    as_or_more_extreme = sum(1 for r in null_rs if abs(r) >= abs(real_r))
    p_value = as_or_more_extreme / len(null_rs) if null_rs else float("nan")

    return ShuffleResult(real_r=real_r, real_n=real_n, null_rs=null_rs, p_value=p_value)


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    resolved = resolve_headlines(in_path, live=live)
    try:
        result = run_shuffle_test(resolved, n_shuffles, seed)
    except ValueError as exc:
        print(f"cannot run the shuffle audit: {exc}", file=sys.stderr)
        return 1

    null_rs = result.null_rs
    mean_null = sum(null_rs) / len(null_rs) if null_rs else float("nan")
    sorted_null = sorted(null_rs)
    lo = sorted_null[int(0.025 * len(sorted_null))] if sorted_null else float("nan")
    hi = sorted_null[int(0.975 * len(sorted_null)) - 1] if sorted_null else float("nan")

    print(f"real (unshuffled) contemporaneous: r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled-timestamp null ({len(null_rs)}/{n_shuffles} usable trials): "
        f"mean r={mean_null:+.3f}  95% range [{lo:+.3f}, {hi:+.3f}]"
    )
    print(f"two-sided p-value (fraction of shuffles with |r| >= |real r|): {result.p_value:.3f}")

    if result.p_value < 0.05:
        print(
            "real r sits in the tail of its own shuffled-timestamp null - the signal did NOT "
            "disappear under shuffling. That is the leak signature NEXT_STEPS.md warns about: "
            "investigate before trusting any correlation this pipeline reports."
        )
    else:
        print(
            "real r is unremarkable against its shuffled-timestamp null - consistent with no "
            "timestamp-dependent signal, leaking or otherwise. This does not prove the pipeline "
            "is leak-free in general, and it does not mean sentiment predicts returns; it means "
            "this audit found nothing to contradict Day 5/6's own near-zero result."
        )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help="number of random timestamp permutations to build the null distribution from",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
