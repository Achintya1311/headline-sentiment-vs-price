"""Day 8 CLI: ml-pipeline-audit pass - a shuffle-control leakage test, plus a
synthetic positive control proving the test would catch a real signal if one
existed.

    python -m sentiment.audit
    python -m sentiment.audit --n-perm 5000

NEXT_STEPS.md's "Done when" names the gate directly: shuffle the headline
timestamps and the signal must disappear. Shuffling ``published_at`` only
matters through what it changes downstream - which trading session (and so
which return) a headline ends up paired with - so this module shuffles that
pairing directly: permute the per-headline (compound, return) correspondence
``build_rows``/``usable_rows`` already computed, recompute the same
statistic, and repeat many times to build a null distribution.

That is a deliberate substitution for literally permuting raw timestamps,
not a shortcut taken for convenience: every headline in the committed
fixture was scraped within roughly a 36-hour window, so a real timestamp
shuffle could only ever relabel a headline into one of two realised session
dates (Monday 28 Sep or Tuesday 29 Sep). That is a much weaker scramble than
breaking the full correspondence across all 23 resolved companies' returns,
and would under-test exactly the thing this gate exists to catch. Permuting
the (sentiment, return) pairing directly is the stronger, more general
version of the same test: it destroys any real correspondence while leaving
both marginal distributions - the actual compound scores, the actual
realised returns - untouched.

Two checks, because Day 5/6 already found a null result on the real data -
a shuffle test that can only ever say "still null" on data with nothing in
it would be a test with no power, not a passed gate:

1. Synthetic positive control: inject a genuine, known linear relationship
   into synthetic (compound, return) pairs sized like the real sample, and
   confirm the harness (a) detects it before shuffling and (b) the shuffle
   collapses it back toward zero. This is the proof the test has teeth -
   without it, "the real data shows no signal under shuffling" would be
   indistinguishable from "this test couldn't detect a signal if it tried".
2. Real-fixture audit: run the same shuffle test against the real resolved
   headlines - Day 5's contemporaneous correlation and Day 6's next-day
   regression. If either real statistic turns out to be a significant
   outlier against its own shuffled null, something downstream is leaking
   and the "no signal" finding in Findings is not trustworthy as written.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows
from sentiment.regress import DEFAULT_TRAIN_FRAC, fit_and_evaluate, time_split, usable_rows
from sentiment.stats import pearson_r

DEFAULT_N_PERM = 2000
SIGNIFICANCE = 0.05

# Sized and shaped after the real contemporaneous sample (see README Day 5
# Findings: n=23) so the control's power is representative of what this
# pipeline could actually detect on data like its own, not an easier,
# larger synthetic case that would pass for the wrong reason.
SYNTHETIC_N = 23
SYNTHETIC_SLOPE = 0.08
SYNTHETIC_NOISE_STD = 0.03
SYNTHETIC_SEED = 20260930


@dataclass(frozen=True)
class ShuffleTestResult:
    observed: float
    n_perm: int
    p_value: float
    null_mean: float
    null_std: float

    @property
    def significant(self) -> bool:
        return self.p_value < SIGNIFICANCE


def shuffle(values: list[float], seed: int) -> list[float]:
    """A random permutation of ``values`` - same multiset, different order."""
    xs = list(values)
    random.Random(seed).shuffle(xs)
    return xs


def correlation_shuffle_test(
    xs: list[float], ys: list[float], n_perm: int = DEFAULT_N_PERM, seed: int = 0
) -> ShuffleTestResult:
    """Permutation test for Pearson r: shuffle ``ys`` against ``xs`` ``n_perm``
    times to build a null distribution of r with the real correspondence
    destroyed but both marginals intact, then report where the observed
    (unshuffled) r falls in that null. Two-sided - a leak could show up as
    an unexpectedly strong correlation of either sign."""
    observed = pearson_r(xs, ys)
    null_rs = [pearson_r(xs, shuffle(ys, seed=seed * 1_000_003 + i + 1)) for i in range(n_perm)]
    n_as_extreme = sum(1 for r in null_rs if abs(r) >= abs(observed))
    p_value = (n_as_extreme + 1) / (n_perm + 1)
    mean = sum(null_rs) / len(null_rs)
    variance = sum((r - mean) ** 2 for r in null_rs) / len(null_rs)
    return ShuffleTestResult(
        observed=observed, n_perm=n_perm, p_value=p_value, null_mean=mean, null_std=math.sqrt(variance)
    )


def regression_shuffle_test(
    rows: list[dict],
    train_frac: float = DEFAULT_TRAIN_FRAC,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
) -> ShuffleTestResult | None:
    """Same idea as ``correlation_shuffle_test``, applied to Day 6's
    out-of-sample R^2: shuffle the compound column against lagged_return
    (the chronological row order the time split depends on is left alone -
    only which score goes with which row is permuted) and refit, ``n_perm``
    times. One-sided - a leak would show up as an inflated R^2, not a
    suppressed one.

    Returns ``None`` when the compound column has zero variance (every row
    shares the same score): shuffling a constant changes nothing, so the
    test would otherwise report a meaningless p-value of 1.0 instead of
    admitting it has nothing to test. The committed fixture hits exactly
    this case - see README Day 6/8 Findings.
    """
    compounds = [r["compound"] for r in rows]
    if len(set(compounds)) <= 1:
        return None

    train, test = time_split(rows, train_frac)
    observed = fit_and_evaluate(train, test).test_r2

    null_r2s = []
    for i in range(n_perm):
        shuffled_compounds = shuffle(compounds, seed=seed * 1_000_003 + i + 1)
        shuffled_rows = [{**r, "compound": c} for r, c in zip(rows, shuffled_compounds)]
        s_train, s_test = time_split(shuffled_rows, train_frac)
        null_r2s.append(fit_and_evaluate(s_train, s_test).test_r2)

    n_as_extreme = sum(1 for r2 in null_r2s if r2 >= observed)
    p_value = (n_as_extreme + 1) / (n_perm + 1)
    mean = sum(null_r2s) / len(null_r2s)
    variance = sum((r2 - mean) ** 2 for r2 in null_r2s) / len(null_r2s)
    return ShuffleTestResult(
        observed=observed, n_perm=n_perm, p_value=p_value, null_mean=mean, null_std=math.sqrt(variance)
    )


def synthetic_signal_rows(n: int, slope: float, noise_std: float, seed: int) -> list[dict]:
    """``n`` synthetic (compound, lagged_return) pairs with a real, known
    linear relationship plus Gaussian noise - a positive control. Real
    headline sentiment never looks like this; the point is purely to prove
    the shuffle test has the power to catch a genuine correspondence if the
    real pipeline ever had one, which a null real fixture alone cannot
    demonstrate."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0)
        y = slope * x + rng.gauss(0.0, noise_std)
        rows.append({"compound": x, "lagged_return": y})
    return rows


def run(in_path: Path, n_perm: int, seed: int, live: bool) -> int:
    print("=== Synthetic positive control (proves the shuffle test has power) ===")
    synthetic_rows = synthetic_signal_rows(SYNTHETIC_N, SYNTHETIC_SLOPE, SYNTHETIC_NOISE_STD, SYNTHETIC_SEED)
    synth_xs = [r["compound"] for r in synthetic_rows]
    synth_ys = [r["lagged_return"] for r in synthetic_rows]

    synth_corr = correlation_shuffle_test(synth_xs, synth_ys, n_perm=n_perm, seed=seed)
    print(
        f"correlation: observed r={synth_corr.observed:+.3f}  "
        f"null mean={synth_corr.null_mean:+.3f} std={synth_corr.null_std:.3f}  "
        f"p={synth_corr.p_value:.4f}  n={SYNTHETIC_N}"
    )

    synth_reg = regression_shuffle_test(synthetic_rows, n_perm=n_perm, seed=seed)
    assert synth_reg is not None  # synthetic compounds are continuous by construction
    print(
        f"regression:  observed test R^2={synth_reg.observed:+.3f}  "
        f"null mean={synth_reg.null_mean:+.3f} std={synth_reg.null_std:.3f}  "
        f"p={synth_reg.p_value:.4f}"
    )

    power_confirmed = synth_corr.significant and synth_reg.significant
    if power_confirmed:
        print("the injected signal is detected pre-shuffle and destroyed by shuffling - the audit has power.")
    else:
        print("WARNING: the synthetic positive control was not caught - the audit may have no power.")

    print()
    print("=== Real-fixture audit ===")
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    rows, _ = build_rows(in_path, live=live)
    if len(rows) < 4:
        print(f"too few resolved headlines for a shuffle test (n={len(rows)})", file=sys.stderr)
        return 1

    real_xs = [r["compound"] for r in rows]
    real_ys = [r["contemporaneous_return"] for r in rows]
    real_corr = correlation_shuffle_test(real_xs, real_ys, n_perm=n_perm, seed=seed)
    print(
        f"contemporaneous: observed r={real_corr.observed:+.3f}  "
        f"null mean={real_corr.null_mean:+.3f} std={real_corr.null_std:.3f}  "
        f"p={real_corr.p_value:.4f}  n={len(rows)}"
    )
    if real_corr.significant:
        print("SIGNIFICANT vs the shuffled null - the 'no signal' finding may not hold; investigate for leakage.")
    else:
        print("not significant vs the shuffled null - consistent with Day 5's finding of no correlation.")

    lagged_rows = usable_rows(rows)
    real_reg = regression_shuffle_test(lagged_rows, n_perm=n_perm, seed=seed)
    if real_reg is None:
        print(
            f"regression: skipped - all {len(lagged_rows)} usable (next-day-return) headlines share "
            "the identical compound score (see README Day 6/8 Findings); nothing to shuffle."
        )
        real_reg_bad = False
    else:
        print(
            f"regression: observed test R^2={real_reg.observed:+.3f}  "
            f"null mean={real_reg.null_mean:+.3f} std={real_reg.null_std:.3f}  p={real_reg.p_value:.4f}"
        )
        if real_reg.significant:
            print("SIGNIFICANT vs the shuffled null - investigate for leakage.")
        else:
            print("not significant vs the shuffled null - consistent with Day 6's finding of no predictive power.")
        real_reg_bad = real_reg.significant

    print()
    if not power_confirmed:
        print("FAIL: synthetic positive control did not demonstrate power.")
        return 1
    if real_corr.significant or real_reg_bad:
        print("FAIL: a real statistic was significant against its shuffled null - possible leakage.")
        return 1
    print("PASS: the audit has power, and the real pipeline shows no signal that shuffling needed to destroy.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM, help="number of shuffles per test")
    parser.add_argument("--seed", type=int, default=0, help="base seed for the shuffle permutations")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_perm, args.seed, args.live))


if __name__ == "__main__":
    main()
