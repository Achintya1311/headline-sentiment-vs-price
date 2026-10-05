"""Day 8 CLI: timestamp-shuffle leakage control.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 1000 --tolerance 0.1

NEXT_STEPS.md's "done when": shuffle headline timestamps and the signal must
disappear. If a shuffled-timestamp control still predicts returns, the
pipeline is leaking and the result is an artifact.

Read literally, that is a claim about the *shuffled* run on its own, not
about how it compares to the real one: once ``published_at`` is randomly
reassigned, Day 4's alignment step has no genuine temporal information left
to work with, so the resulting contemporaneous-return correlation should
be small across many independent reshuffles. The real run's own r is
reported alongside purely as context (it's the number Day 5-7 already
discuss), never as the pass condition - a pipeline with no real signal
(which, per Day 5-7's findings, is what this one has) would trivially
"pass" a real-vs-shuffled comparison without the control ever having
exercised anything.

**The gate is an effect-size threshold on the shuffled mean, not a
significance test.** An earlier version of this module tested H0:
mean(shuffled r) == 0 with a one-sample z-test, which has a sharp, known
flaw for exactly this use: its power grows with ``n_shuffles``, so turning
the shuffle count up makes the test *more* likely to reject a mean that
never changes in size - it would flag a practically-irrelevant bias purely
because it ran enough reshuffles to shrink the standard error, which is not
what "the signal must disappear" is asking. ``--tolerance`` instead asks
whether |mean(shuffled r)| is *small in absolute terms* (default 0.1, the
same order of magnitude as the weak correlations Day 5 itself reports),
independent of how many shuffles were run to measure it. The z-test's CI
and p-value are still printed for context, never for the decision.

**What running this against the real fixture actually found**, and why it
is not read as a leak - see README Findings/Limitations for the full
write-up: with only 23 distinct tickers and exactly one headline per
ticker, and the fixture spanning only two trading sessions (Mon 28 / Tue 29
Sep 2026), a timestamp shuffle never moves a headline to a different
ticker's return - it only ever toggles, for that one fixed (ticker,
compound) pair, which of that *same* ticker's two adjacent-day returns gets
used. Those two cross-sectional correlations are not noise independent of
each other: corr(compound, ret_28)=+0.336 and corr(compound, ret_29)=-0.212
across the 23 tickers - opposite signs, on this one small sample. A
timestamp shuffle blends those two (weighted ~42%/58%, the real pre-open
vs. not split in this fixture) rather than averaging them away to zero, so
the shuffled mean lands at a small but nonzero -0.01 to -0.02 depending on
seed - comfortably inside ``--tolerance``, but not exactly zero, and not
for a reason a longer run of shuffles would fix. A genuinely independent
control would need headlines spread across enough *different* trading days
that a shuffle could land a ticker on a return uncorrelated with its real
one - which only a wider scrape (more days, not more shuffles) can give
this fixture.

This is designed to run unattended in CI with no network and no live
fetch: ``published_at`` is *permuted among* the 50 real headlines, never
resampled to invented times, so every shuffle lands inside the dates the
committed price fixtures cover; and every rerun (real and shuffled) reads
price fixtures only (``live=False``), never re-fetching.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import stdev

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 500
DEFAULT_TOLERANCE = 0.1
MIN_ROWS = 4  # pearson_with_ci's own floor for a meaningful r; see stats.py


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Permute ``published_at`` among ``headlines``; every other field
    (title, source, link - everything ticker resolution and sentiment
    scoring read) stays attached to its original headline. A true
    permutation of the real timestamps, not resampled random times, so a
    shuffle can never invent a session_date the committed price fixtures
    don't cover."""
    shuffled_times = [h.published_at for h in headlines]
    rng.shuffle(shuffled_times)
    return [replace(h, published_at=t) for h, t in zip(headlines, shuffled_times)]


def contemporaneous_r(headlines: list[Headline]) -> tuple[float, int]:
    """Contemporaneous Pearson r between VADER compound and same-session
    return, rebuilding Day 4/5's pipeline from this exact headline list.
    Always reads price fixtures (``live=False``) - shuffled reruns must stay
    offline and deterministic to run unattended in CI."""
    rows, _ = build_rows_from_headlines(headlines, live=False)
    if len(rows) < MIN_ROWS:
        return float("nan"), len(rows)
    r = pearson_r([row["compound"] for row in rows], [row["contemporaneous_return"] for row in rows])
    return r, len(rows)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    shuffled_rs: list[float]
    shuffled_mean: float
    shuffled_ci_low: float
    shuffled_ci_high: float
    p_value: float
    passed: bool


def run_audit(in_path: Path, n_shuffles: int, seed: int, tolerance: float) -> AuditResult:
    headlines = read_csv(in_path)
    real_r, real_n = contemporaneous_r(headlines)
    if real_n < MIN_ROWS:
        raise ValueError(f"only {real_n} resolved headline(s) in {in_path}; not enough to audit")

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        r, n = contemporaneous_r(shuffle_timestamps(headlines, rng))
        if n >= MIN_ROWS:
            shuffled_rs.append(r)

    if len(shuffled_rs) < 2:
        raise ValueError(
            f"only {len(shuffled_rs)} usable shuffle(s) (<4 resolved rows on the rest); "
            "cannot build a null distribution to compare against"
        )

    n = len(shuffled_rs)
    mean = sum(shuffled_rs) / n
    se = stdev(shuffled_rs) / math.sqrt(n)
    if se == 0:
        p_value = 1.0 if mean == 0 else 0.0
        ci_low = ci_high = mean
    else:
        z = mean / se
        p_value = 2 * (1 - _normal_cdf(abs(z)))
        margin = 1.959963985 * se  # stats.Z_95
        ci_low, ci_high = mean - margin, mean + margin

    return AuditResult(
        real_r=real_r,
        real_n=real_n,
        shuffled_rs=shuffled_rs,
        shuffled_mean=mean,
        shuffled_ci_low=ci_low,
        shuffled_ci_high=ci_high,
        p_value=p_value,
        passed=abs(mean) < tolerance,
    )


def run(in_path: Path, n_shuffles: int, seed: int, tolerance: float) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    try:
        result = run_audit(in_path, n_shuffles, seed, tolerance)
    except ValueError as exc:
        print(f"cannot audit: {exc}", file=sys.stderr)
        return 1

    dropped = n_shuffles - len(result.shuffled_rs)
    percentile = sorted(result.shuffled_rs)
    lo_p = percentile[int(0.025 * len(percentile))]
    hi_p = percentile[min(int(0.975 * len(percentile)), len(percentile) - 1)]

    print(f"real contemporaneous r={result.real_r:+.3f} (n={result.real_n}) - context only, not the pass condition")
    print(
        f"{len(result.shuffled_rs)}/{n_shuffles} timestamp-shuffled reruns usable "
        f"({dropped} dropped for <{MIN_ROWS} resolved rows)"
    )
    print(
        f"shuffled null: mean r={result.shuffled_mean:+.4f}  95% CI [{result.shuffled_ci_low:+.4f}, "
        f"{result.shuffled_ci_high:+.4f}]  (individual-shuffle 95% range [{lo_p:+.3f}, {hi_p:+.3f}])  "
        f"z-test p={result.p_value:.3f} (diagnostic only, see module docstring)"
    )

    if result.passed:
        print(
            f"PASS: |mean(shuffled r)|={abs(result.shuffled_mean):.4f} < tolerance {tolerance} - "
            "scrambling the timestamps leaves no practically meaningful correlation."
        )
        return 0

    print(
        f"FAIL: |mean(shuffled r)|={abs(result.shuffled_mean):.4f} >= tolerance {tolerance} - "
        "scrambling the timestamps should have left no practically meaningful correlation, but didn't. "
        "Check for a dependency between sentiment and return that doesn't require correct timestamps "
        "before trusting any real-data result."
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to audit")
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help="number of timestamp permutations to build the null distribution from",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, for a reproducible null distribution")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="|mean(shuffled r)| must stay below this to pass - an effect-size bar, not a p-value",
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.tolerance))


if __name__ == "__main__":
    main()
