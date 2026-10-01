"""Day 8 CLI: the shuffled-timestamp leakage audit the README's "Correctness
gate" section promises ("shuffled-timestamp control: randomise headline
times and the signal must disappear... this is the test that decides
whether the repo is finished").

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 2000 --seed 0
    python -m sentiment.audit --live

Scope: Day 5's **contemporaneous** correlation only (VADER ``compound`` vs
the aligned session's open-to-close return) - the headline number in the
README's Day 5 Findings. Lagged return and the event study would need the
same treatment before they could be called audited; that is named as a gap
below and in the README, not silently assumed to be covered by this.

What gets shuffled, and why this is the right thing to shuffle: each
resolved headline keeps its own title, ticker, and VADER ``compound`` score
(all content-derived, independent of time) but is paired with a different
headline's ``published_at`` timestamp before ``market_hours.align_headline``
runs. Everything downstream - which session a headline is credited with,
which bar that implies - is untouched. If the real correlation reflects
headlines genuinely reacting to (or anticipating) a price move, scrambling
*which trading session* each score gets credited with should destroy it;
a pipeline bug that let some signal through regardless of correct alignment
would not be destroyed by this, which is exactly what this test is for.

This is a permutation test, not one shuffle: ``--n-shuffles`` independent
permutations build a null distribution of |r|, and the fraction of that
distribution at least as extreme as the real |r| is a permutation p-value.
A p-value that is not small says the real result is not distinguishable
from what a random alignment would produce by chance - which, given Day 5's
own r=-0.185 with a 95% CI of [-0.555, +0.246], is the expected, honest
outcome here: there was very little signal for a leak to inflate in the
first place. See README Findings/Limitations for what that means and does
not mean.
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
from sentiment.stats import pearson_r, pearson_with_ci
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_N_SHUFFLES = 2000
DEFAULT_SEED = 0
MIN_PAIRS = 4  # pearson_with_ci's Fisher z-transform needs n >= 4 for a CI


@dataclass(frozen=True)
class Resolved:
    """One headline that resolved to a ticker with price data: everything
    content-derived (title, ticker, compound) kept apart from the one thing
    a shuffle is allowed to swap out (``published_at``)."""

    title: str
    ticker: str
    compound: float
    published_at: datetime
    bars: list[Bar]


def load_resolved(in_path: Path, live: bool = False) -> list[Resolved]:
    """The same resolution Day 5's ``correlate.build_rows`` does (ticker
    lookup, price fixture load, VADER score), kept separate from it because
    the audit needs the per-headline ``bars`` series kept around - Day 5's
    rows only keep the two return numbers already computed for one fixed
    timestamp."""
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


def contemporaneous_pairs(
    resolved: list[Resolved], timestamps: list[datetime]
) -> tuple[list[float], list[float]]:
    """Pair each ``resolved[i]``'s own ticker/compound with ``timestamps[i]``
    (the identity assignment for the real run, a permutation for a shuffle)
    and return the ``(compounds, contemporaneous_returns)`` for whichever
    pairs land on a session date that headline's own ticker fixture has a
    bar for. A pair can legitimately drop out here (no bar for that
    session) the same way a real headline can in ``correlate.build_rows``;
    see the module docstring and README Limitations for why that did not
    happen on this fixture's actual timestamp range."""
    if len(resolved) != len(timestamps):
        raise ValueError("resolved and timestamps must be the same length")
    compounds: list[float] = []
    returns: list[float] = []
    for r, ts in zip(resolved, timestamps):
        alignment = align_headline(ts)
        bar = bar_on(r.bars, alignment.session_date)
        if bar is None:
            continue
        compounds.append(r.compound)
        returns.append(bar.session_return)
    return compounds, returns


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    real_ci_low: float
    real_ci_high: float
    shuffled_rs: list[float]
    n_shuffles_used: int
    n_shuffles_degenerate: int
    p_value: float


def run_permutation_test(
    resolved: list[Resolved], n_shuffles: int = DEFAULT_N_SHUFFLES, seed: int = DEFAULT_SEED
) -> AuditResult:
    """Build the null distribution of |r| from ``n_shuffles`` independent
    permutations of the real timestamps across ``resolved``, and compare the
    real (identity-assignment) |r| against it.

    Raises ``ValueError`` if the real assignment does not even clear
    ``MIN_PAIRS`` - there is nothing to audit without that."""
    real_timestamps = [r.published_at for r in resolved]
    real_compounds, real_returns = contemporaneous_pairs(resolved, real_timestamps)
    if len(real_compounds) < MIN_PAIRS:
        raise ValueError(
            f"only {len(real_compounds)} resolved headline(s) have a contemporaneous bar - "
            f"need at least {MIN_PAIRS} to audit a correlation"
        )
    real_stat = pearson_with_ci(real_compounds, real_returns)

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    degenerate = 0
    for _ in range(n_shuffles):
        shuffled_timestamps = real_timestamps[:]
        rng.shuffle(shuffled_timestamps)
        compounds, returns = contemporaneous_pairs(resolved, shuffled_timestamps)
        if len(compounds) < MIN_PAIRS:
            degenerate += 1
            continue
        shuffled_rs.append(pearson_r(compounds, returns))

    if not shuffled_rs:
        raise ValueError("every shuffle was degenerate (fewer than MIN_PAIRS pairs) - cannot build a null distribution")

    p_value = sum(1 for sr in shuffled_rs if abs(sr) >= abs(real_stat.r)) / len(shuffled_rs)

    return AuditResult(
        real_r=real_stat.r,
        real_n=real_stat.n,
        real_ci_low=real_stat.ci_low,
        real_ci_high=real_stat.ci_high,
        shuffled_rs=shuffled_rs,
        n_shuffles_used=len(shuffled_rs),
        n_shuffles_degenerate=degenerate,
        p_value=p_value,
    )


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    resolved = load_resolved(in_path, live=live)
    if not resolved:
        print("no headlines resolved to a ticker with price data; nothing to audit", file=sys.stderr)
        return 1

    try:
        result = run_permutation_test(resolved, n_shuffles=n_shuffles, seed=seed)
    except ValueError as exc:
        print(f"cannot audit: {exc}", file=sys.stderr)
        return 1

    mean_shuffled = sum(result.shuffled_rs) / len(result.shuffled_rs)
    print(f"{len(resolved)} headlines resolved; real (unshuffled) contemporaneous correlation:")
    print(f"  r={result.real_r:+.3f}  95% CI [{result.real_ci_low:+.3f}, {result.real_ci_high:+.3f}]  n={result.real_n}")
    print(
        f"{result.n_shuffles_used} shuffled-timestamp permutations "
        f"({result.n_shuffles_degenerate} degenerate, dropped): "
        f"mean r={mean_shuffled:+.3f}  range [{min(result.shuffled_rs):+.3f}, {max(result.shuffled_rs):+.3f}]"
    )
    print(f"permutation p-value (|shuffled r| >= |real r|): {result.p_value:.3f}")
    if result.p_value < 0.05:
        print(
            "p < 0.05: the real correlation is more extreme than 95% of shuffled-timestamp "
            "controls - this is the one case the correctness gate is watching for. Investigate "
            "before trusting the real result; see README Findings for what it would mean."
        )
    else:
        print(
            "p >= 0.05: the real correlation is not distinguishable from what randomising "
            "which session each headline is credited with produces by chance. Consistent with "
            "Day 5's own near-zero, wide-CI result - there was little signal here for a leak to "
            "inflate, so this shuffle test mostly confirms there is nothing to disappear, not "
            "that a real effect vanished under scrutiny. See README Limitations."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp permutations to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
