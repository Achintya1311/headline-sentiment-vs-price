"""Day 8 CLI: the leakage audit NEXT_STEPS.md's "Done when" section requires
before any correlation in this repo is believed.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 1000 --seed 1
    python -m sentiment.audit --live

The check: take the real headlines, randomly reassign their ``published_at``
timestamps across each other (same titles, same tickers - only *when* each
headline supposedly landed is scrambled), rebuild the contemporaneous
correlation the same way ``sentiment.correlate`` does, and repeat many times
to get a null distribution of what a timestamp-scrambled control looks like.

This exists to catch one specific failure mode: a hidden leak that makes the
pipeline's apparent predictive power insensitive to whether the timing is
actually honoured - for example, something upstream of alignment that quietly
ties a headline to a return regardless of ``session_date``. If a real
correlation's strength survives scrambling the one thing it is supposed to
depend on, that correlation was never really coming from correct, leak-free
timing in the first place.

Two different things can be true about this repo's own committed fixture,
and both get reported honestly rather than picking the flattering one:

1. Day 5-7 already found no detectable contemporaneous signal on this
   fixture (r=-0.185, 95% CI comfortably containing zero). There is no real
   signal here for shuffling to destroy, so running this audit against the
   real fixture mostly just confirms shuffling doesn't manufacture one out
   of nothing - it cannot demonstrate the audit's detection power, because
   there is nothing for it to detect.
2. ``tests/test_audit.py`` builds a synthetic case with a real,
   timing-dependent signal injected on purpose (sentiment deliberately
   correlated with the return of the session each headline is engineered to
   land on) and shows the real, leak-free pipeline recovers it strongly
   while shuffling collapses it - the actual proof that this audit has
   teeth, which the real null fixture alone cannot provide.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import PearsonResult, pearson_with_ci

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0

# The same "high-magnitude" cut sentiment.correlate's event study already
# uses for |compound| - reused here as the bar a real correlation's
# magnitude has to clear before shuffling it can mean anything.
SIGNAL_THRESHOLD = 0.30


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Same headlines, same titles and tickers - ``published_at`` values
    randomly reassigned across rows. This severs the one thing Day 4's
    alignment depends on (when a headline landed relative to market hours)
    while leaving which company each headline names untouched, so any
    correlation that genuinely depends on correct timing should collapse."""
    shuffled_timestamps = [h.published_at for h in headlines]
    rng.shuffle(shuffled_timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_timestamps)]


def contemporaneous_r(headlines: list[Headline], live: bool = False) -> PearsonResult | None:
    """The same statistic ``sentiment.correlate`` reports as "contemporaneous",
    rebuilt from an in-memory headline list. ``None`` if fewer than 2
    headlines resolve to a ticker with a bar for their aligned session - a
    shuffle can land a headline on a date its ticker's fixture has no bar
    for, shrinking the usable rows below what a real run would see."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_with_ci(compounds, returns)


@dataclass(frozen=True)
class AuditResult:
    real: PearsonResult
    shuffled_rs: list[float]

    @property
    def mean_abs_shuffled(self) -> float:
        return sum(abs(r) for r in self.shuffled_rs) / len(self.shuffled_rs)

    @property
    def p_value(self) -> float:
        """Fraction of timestamp-shuffled draws at least as extreme as the
        real statistic - the standard permutation-test p-value. Reported as
        a diagnostic, not the leak gate itself: a small p-value here just
        means the real result is statistically unusual for pure chance, in
        either direction (see ``signal_survived_shuffle`` for the actual
        leak check)."""
        extreme = sum(1 for r in self.shuffled_rs if abs(r) >= abs(self.real.r))
        return extreme / len(self.shuffled_rs)

    @property
    def had_signal_to_lose(self) -> bool:
        """Whether the real, correctly-aligned statistic even clears the
        high-magnitude bar. Below it there is no real signal for shuffling
        to disprove - true of this repo's own committed fixture, see the
        README's Day 8 Findings."""
        return abs(self.real.r) >= SIGNAL_THRESHOLD

    @property
    def signal_survived_shuffle(self) -> bool:
        """The actual leak signature: a real signal exists *and* the
        timestamp-shuffled controls reproduce comparable magnitude on
        average. In a leak-free pipeline, destroying the real timing should
        destroy a genuinely timing-dependent signal - if scrambling
        published_at barely moves the statistic, whatever is driving the
        correlation does not actually depend on correct alignment, which is
        exactly what "the pipeline is leaking" means here."""
        return self.had_signal_to_lose and self.mean_abs_shuffled >= abs(self.real.r) / 2


def run_audit(
    headlines: list[Headline], n_shuffles: int, seed: int, live: bool = False
) -> AuditResult | None:
    real = contemporaneous_r(headlines, live=live)
    if real is None:
        return None

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        stat = contemporaneous_r(shuffle_published_at(headlines, rng), live=live)
        if stat is not None:
            shuffled_rs.append(stat.r)

    if not shuffled_rs:
        raise RuntimeError(
            "every timestamp shuffle left fewer than 2 resolvable rows; "
            "cannot build a null distribution from this fixture"
        )
    return AuditResult(real=real, shuffled_rs=shuffled_rs)


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = run_audit(headlines, n_shuffles=n_shuffles, seed=seed, live=live)
    if result is None:
        print("fewer than 2 headlines resolved to a ticker with price data; nothing to audit", file=sys.stderr)
        return 1

    n = len(result.shuffled_rs)
    ordered = sorted(result.shuffled_rs)
    lo = ordered[int(0.025 * n)]
    hi = ordered[min(int(0.975 * n), n - 1)]

    print(f"real (correctly-timed) contemporaneous r = {result.real.r:+.3f}  n={result.real.n}")
    print(
        f"{n} timestamp-shuffled controls: mean |r| = {result.mean_abs_shuffled:.3f}, "
        f"95% range [{lo:+.3f}, {hi:+.3f}]"
    )
    print(f"permutation p-value (|shuffled r| >= |real r|) = {result.p_value:.3f}")

    if not result.had_signal_to_lose:
        print(
            f"no real signal to test: |real r|={abs(result.real.r):.3f} does not clear the "
            f"{SIGNAL_THRESHOLD} high-magnitude bar. Day 5-7 already found this fixture has no "
            "detectable contemporaneous signal - there is nothing here for shuffling to disprove. "
            "See tests/test_audit.py's synthetic case for a worked example with a real signal to lose."
        )
        return 0
    if result.signal_survived_shuffle:
        print(
            "FAIL: a real signal exists and timestamp-shuffled controls reproduce comparable "
            "magnitude on average - the signal does not depend on correct timing, which is the "
            "leak this audit exists to catch. Investigate before trusting this correlation."
        )
        return 1
    print("PASS: the real signal collapses under timestamp shuffling, as a leak-free result should.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to audit")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp-shuffle draws"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
