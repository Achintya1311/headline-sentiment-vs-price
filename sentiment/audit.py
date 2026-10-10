"""Day 8 CLI: the leakage/audit pass NEXT_STEPS.md's "Done when" names as
the real gate for this repo.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --n-shuffles 500 --seed 1

NEXT_STEPS.md states the gate plainly: "shuffle the headline timestamps and
the signal must disappear. If a shuffled-timestamp control still predicts
returns, the pipeline is leaking." That is a statement about the *shuffled*
runs' own behaviour, not a comparison against the real run's correlation -
so that is exactly what this audits.

Day 4 already guarantees, by construction, that no aligned session's return
predates the headline it is attributed to (``Alignment.leak_free()``). This
is a different, harder-to-get-right-by-staring-at-the-code failure mode:
could the pipeline manufacture an apparently "significant" correlation
(a 95% CI that excludes zero, Day 5's own bar for calling something a
signal) even when the headline/timestamp correspondence is nonsense?

The control: take the resolved headlines, randomly permute which headline
is assigned which *other* headline's real ``published_at`` (same ticker,
same compound score, same multiset of real timestamps - only which
headline owns which timestamp is scrambled), rebuild the contemporaneous
pairing through the real pipeline, and recompute Pearson r with its CI.
Repeat ``--n-shuffles`` times and count how often that CI excludes zero.
Pure chance alone puts that rate at roughly 5% (the same alpha Day 5's CI
is built at); this audit fails if it comes back well above that, which
would mean the pipeline finds "signal" in a pairing that is, by
construction, meaningless.

See ``tests/test_audit.py`` for a constructed case where this genuinely
fires (degenerate per-ticker price data that looks "predictive" under any
pairing at all, shuffled or not) - proof the gate can catch something, not
just always pass because the committed fixture's real result already
happens to be null (see README Day 8 Findings for what it finds there).
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

DEFAULT_N_SHUFFLES = 200
DEFAULT_SEED = 0

# Per-shuffle "significant" means its own 95% CI excludes zero, so pure
# chance alone should flag roughly 5% of shuffles. FAIL_THRESHOLD is 3x that
# nominal rate - generous on purpose, since with a few hundred shuffles and
# a couple dozen headlines per shuffle, the estimated rate itself is noisy;
# this is meant to catch a pipeline that is structurally broken, not to
# relitigate ordinary sampling variation.
NOMINAL_FALSE_POSITIVE_RATE = 0.05
FAIL_THRESHOLD = 3 * NOMINAL_FALSE_POSITIVE_RATE


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list of ``Headline``s with ``published_at`` permuted
    across the list. Same headlines, same companies, same tickers, same
    exact multiset of real timestamps (so the pre_open/intraday/post_close
    mix is unchanged) - only which headline owns which timestamp is
    scrambled, breaking the one correspondence this audit exists to stress:
    a headline's own content paired with its own real publish time.
    """
    shuffled_timestamps = [h.published_at for h in headlines]
    rng.shuffle(shuffled_timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_timestamps)]


def pairing_pearson(headlines: list[Headline], live: bool = False) -> PearsonResult | None:
    """Pearson r and 95% CI between VADER compound and contemporaneous
    return for ``headlines`` as given - ``None`` if fewer than 2 rows
    resolve to a ticker with price data (not enough to correlate)."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None
    return pearson_with_ci([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def is_significant(result: PearsonResult) -> bool:
    """A 95% CI that excludes zero - the same bar Day 5's Findings use."""
    return result.ci_low > 0 or result.ci_high < 0


@dataclass(frozen=True)
class AuditResult:
    real: PearsonResult
    shuffled_significant_rate: float
    n_valid_shuffles: int
    passed: bool


def run_audit(
    headlines: list[Headline],
    live: bool = False,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
) -> AuditResult | None:
    """Returns ``None`` if the real pairing does not have enough resolved
    rows to compute a correlation at all - nothing to audit yet."""
    real = pairing_pearson(headlines, live=live)
    if real is None:
        return None

    rng = random.Random(seed)
    n_significant = 0
    n_valid = 0
    for _ in range(n_shuffles):
        shuffled = shuffle_published_at(headlines, rng)
        result = pairing_pearson(shuffled, live=live)
        if result is None:
            continue
        n_valid += 1
        if is_significant(result):
            n_significant += 1

    if n_valid == 0:
        return None

    rate = n_significant / n_valid
    return AuditResult(
        real=real,
        shuffled_significant_rate=rate,
        n_valid_shuffles=n_valid,
        passed=rate <= FAIL_THRESHOLD,
    )


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = run_audit(headlines, live=live, n_shuffles=n_shuffles, seed=seed)
    if result is None:
        print("fewer than 2 resolved headlines; nothing to audit", file=sys.stderr)
        return 1

    real = result.real
    print(
        f"real contemporaneous: r={real.r:+.3f}  95% CI [{real.ci_low:+.3f}, {real.ci_high:+.3f}]  n={real.n}"
        + ("  (significant)" if is_significant(real) else "  (not significant)")
    )
    print(
        f"shuffled-timestamp control: {result.shuffled_significant_rate:.1%} of "
        f"{result.n_valid_shuffles} shuffles came back 'significant' by the same CI test "
        f"(chance alone predicts ~{NOMINAL_FALSE_POSITIVE_RATE:.0%}, fail threshold {FAIL_THRESHOLD:.0%})"
    )

    if not result.passed:
        print(
            "FAIL: a meaningless, scrambled headline/timestamp pairing still comes back "
            "'significant' far more often than chance - the pipeline is finding structure "
            "that does not depend on real timing. Do not trust the real-pairing result above."
        )
        return 1

    print(
        "PASS: scrambling the headline/timestamp pairing does not manufacture spurious "
        "significance beyond the chance rate - whatever the real-pairing result says, it is "
        "not an artifact of this kind of leak."
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
        help="number of timestamp-shuffled pairings to estimate the chance-significance rate from",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
