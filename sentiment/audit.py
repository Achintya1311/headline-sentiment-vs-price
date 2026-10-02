"""Day 8 CLI: the leakage audit - shuffle headline timestamps, confirm the
sentiment/return signal disappears.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --permutations 2000 --seed 1

NEXT_STEPS.md's "Done when" names the exact test: *"shuffle the headline
timestamps and the signal must disappear. If a shuffled-timestamp control
still predicts returns, the pipeline is leaking and the result is an
artifact."* Day 5's CI-contains-zero finding (r=-0.185, 95% CI [-0.555,
+0.246], n=23) is a weak version of that check - it only says this one run
isn't significant, not that the pipeline would fail to show a spurious
correlation if the alignment code were broken in some way a single run
can't reveal.

This module runs the actual control. For each of the 23 headlines Day 5
resolves to a ticker (``sentiment.tickers.resolve``), the only thing a
headline's ``published_at`` controls downstream is which trading session's
``sentiment.market_hours.align_headline`` assigns it to, and therefore
which ``sentiment.prices`` bar's return gets paired with its (content-only)
VADER compound score. Shuffling ``published_at`` across the resolved
headlines - keeping each headline's own title/ticker/compound fixed -
re-runs that exact alignment -> session_date -> return pipeline with a
timestamp that no longer belongs to that headline's content, breaking the
one honest mechanism that could connect sentiment to return. Repeating the
shuffle many times (seeded, for a reproducible CI assertion) builds a null
distribution of correlation coefficients; the real, correctly-aligned r is
then checked against it with a two-sided permutation p-value. If the real
r is not an outlier against that null, the result is statistically
indistinguishable from what a broken timestamp link alone can produce - the
signal "disappeared" in the sense NEXT_STEPS.md means.

See the README's Day 8 Findings and "why this might be spurious" section
for what this does and does not prove on the committed fixture, and
``tests/test_audit.py`` for the synthetic positive control that proves the
shuffle can actually detect a real, timestamp-mediated signal when the
fixture happens to contain one.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

from sentiment.headline import read_csv
from sentiment.market_hours import align_headline
from sentiment.prices import Bar, PriceFetchError, bar_on, load_bars
from sentiment.stats import PearsonResult, pearson_with_ci
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_PERMUTATIONS = 500
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.05
MIN_ROWS = 4  # pearson_with_ci's Fisher z-transform needs n >= 4, see stats.py


@dataclass(frozen=True)
class ResolvedHeadline:
    """A headline ``sentiment.tickers.resolve`` maps to exactly one
    fetchable ticker, kept with its own compound score and its own
    ``published_at`` - the two fields a shuffle run recombines across
    different headlines."""

    title: str
    ticker: str
    compound: float
    published_at: datetime


@dataclass(frozen=True)
class AuditResult:
    observed: PearsonResult
    null_results: list[PearsonResult]
    p_value: float

    @property
    def null_rs(self) -> list[float]:
        return [r.r for r in self.null_results]

    @property
    def null_false_positive_rate(self) -> float:
        """Share of shuffled runs whose own 95% CI excludes zero - the rate
        a broken timestamp link alone produces a result that *looks*
        significant by chance. Nominally ~5% at a 95% CI; see README
        Findings for why this is itself a noisy estimate at n=23."""
        if not self.null_results:
            return 0.0
        false_positives = sum(1 for r in self.null_results if r.ci_low > 0 or r.ci_high < 0)
        return false_positives / len(self.null_results)

    @property
    def leaking(self) -> bool:
        return self.p_value < DEFAULT_ALPHA


def build_resolved(in_path: Path) -> list[ResolvedHeadline]:
    """Every headline that resolves to a single, fetchable ticker (Day 5's
    rule - see ``sentiment.tickers``), scored with VADER. No price lookup
    here: a resolved headline's ticker may still have no fixture, which is
    handled per-timestamp in ``contemporaneous_return`` so the same
    resolved list can be reused across every shuffle."""
    headlines = read_csv(in_path)
    resolved: list[ResolvedHeadline] = []
    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        _, ticker = match
        if ticker is None:
            continue
        scored = score_headline(h)
        resolved.append(
            ResolvedHeadline(title=h.title, ticker=ticker, compound=scored.compound, published_at=h.published_at)
        )
    return resolved


def contemporaneous_return(
    ticker: str, published_at: datetime, bars_cache: dict[str, list[Bar]], live: bool
) -> float | None:
    """The open-to-close return of the session ``published_at`` leak-free
    aligns to for ``ticker``, or ``None`` if that ticker has no price
    fixture or no bar on that session. ``bars_cache`` is shared across an
    entire permutation run so each ticker's fixture is loaded once, not once
    per shuffle."""
    if ticker not in bars_cache:
        try:
            bars_cache[ticker] = load_bars(ticker, live=live)
        except PriceFetchError:
            bars_cache[ticker] = []
    bars = bars_cache[ticker]
    if not bars:
        return None
    alignment = align_headline(published_at)
    bar = bar_on(bars, alignment.session_date)
    return bar.session_return if bar is not None else None


def correlate_with_timestamps(
    resolved: list[ResolvedHeadline],
    timestamps: list[datetime],
    bars_cache: dict[str, list[Bar]],
    live: bool,
) -> PearsonResult | None:
    """Pair each ``resolved[i]``'s compound with the return whichever
    timestamp ``timestamps[i]`` aligns to for ``resolved[i]``'s ticker, and
    correlate. Passing ``resolved``'s own ``published_at`` list back
    unshuffled reproduces the real, leak-free result; any permutation of it
    is the shuffled-timestamp control."""
    compounds: list[float] = []
    returns: list[float] = []
    for rh, ts in zip(resolved, timestamps):
        ret = contemporaneous_return(rh.ticker, ts, bars_cache, live)
        if ret is None:
            continue
        compounds.append(rh.compound)
        returns.append(ret)
    if len(compounds) < MIN_ROWS:
        return None
    return pearson_with_ci(compounds, returns)


def permutation_null(
    resolved: list[ResolvedHeadline],
    bars_cache: dict[str, list[Bar]],
    live: bool,
    n_permutations: int,
    seed: int,
) -> list[PearsonResult]:
    """``n_permutations`` shuffled-timestamp correlation results,
    deterministic for a given seed so a CI assertion sees the same null
    distribution on every run."""
    rng = random.Random(seed)
    own_timestamps = [rh.published_at for rh in resolved]
    null_results: list[PearsonResult] = []
    for _ in range(n_permutations):
        shuffled = own_timestamps[:]
        rng.shuffle(shuffled)
        result = correlate_with_timestamps(resolved, shuffled, bars_cache, live)
        if result is not None:
            null_results.append(result)
    return null_results


def permutation_p_value(observed_r: float, null_rs: list[float]) -> float:
    """Two-sided permutation p-value with the standard +1/+1 correction
    (Davison & Hinkley) so a finite permutation count never reports an exact
    zero: the fraction of shuffled runs at least as extreme as the real
    result, counting the real result itself as one more sample from the
    null it is being compared against."""
    as_extreme = sum(1 for r in null_rs if abs(r) >= abs(observed_r))
    return (as_extreme + 1) / (len(null_rs) + 1)


def run_audit(
    in_path: Path,
    live: bool = False,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
) -> AuditResult | None:
    resolved = build_resolved(in_path)
    if len(resolved) < MIN_ROWS:
        return None

    bars_cache: dict[str, list[Bar]] = {}
    observed = correlate_with_timestamps(resolved, [rh.published_at for rh in resolved], bars_cache, live)
    if observed is None:
        return None

    null_results = permutation_null(resolved, bars_cache, live, n_permutations, seed)
    if not null_results:
        return None

    p_value = permutation_p_value(observed.r, [r.r for r in null_results])
    return AuditResult(observed=observed, null_results=null_results, p_value=p_value)


def run(in_path: Path, live: bool, n_permutations: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    result = run_audit(in_path, live=live, n_permutations=n_permutations, seed=seed)
    if result is None:
        print("not enough resolved headlines with price data to audit", file=sys.stderr)
        return 1

    obs = result.observed
    null_mean = mean(result.null_rs)
    null_sd = pstdev(result.null_rs)
    print(f"real alignment:  r={obs.r:+.3f}  95% CI [{obs.ci_low:+.3f}, {obs.ci_high:+.3f}]  n={obs.n}")
    print(
        f"shuffled null ({n_permutations} shuffles, seed={seed}): "
        f"mean r={null_mean:+.3f}  sd={null_sd:.3f}  "
        f"own-CI-excludes-zero rate={result.null_false_positive_rate:.1%}"
    )
    print(f"two-sided permutation p-value for the real result: {result.p_value:.3f}")

    if result.leaking:
        print(
            f"FAIL: real r is more extreme than {1 - DEFAULT_ALPHA:.0%} of shuffled-timestamp "
            "runs - the signal did not disappear when the true timestamp link was broken."
        )
        return 1

    print(
        f"PASS: real r is not distinguishable from the shuffled-timestamp null "
        f"(p={result.p_value:.3f} >= {DEFAULT_ALPHA}) - no evidence the result survives "
        "once the true timestamp link is broken."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="number of shuffled-timestamp control runs to build the null distribution from",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.permutations, args.seed))


if __name__ == "__main__":
    main()
