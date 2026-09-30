"""Day 8 CLI: ml-pipeline-audit - the leakage control NEXT_STEPS.md's "Done
when" section names as the thing that decides whether this repo is finished.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 2000
    python -m sentiment.audit --live

The check: shuffle which ``published_at`` timestamp goes with which headline
(the headline *text*, and therefore its VADER score, never moves - only the
time it is said to have been said moves), rebuild Day 4's leak-free alignment
and Day 5's contemporaneous-return pairing on top of that shuffled timing,
and recompute the same Pearson r Day 5 reports. Repeat many times to build a
null distribution of "what r looks like when sentiment and timing are
deliberately decoupled", then report where the real, unshuffled r sits in
that distribution (a two-sided permutation-test p-value).

This is a control, not a re-run of Day 5: it does not change the finding
(Day 5/6/7 already found no signal), it tests whether the *pipeline* would
have shown one if the timestamps carried no real information. See
``tests/test_audit.py`` for the half of this check that a permutation test
against real, already-null data cannot prove by itself - that the shuffle
actually destroys a real, synthetic signal when one is deliberately planted.
Without that positive-control test, a shuffle audit that always reports
"no signal" would pass even if it were silently broken (e.g. shuffling
nothing at all).
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, rows_for_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 500


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` values permuted
    across headlines - the same multiset of timestamps, reassigned to
    different headline text. Everything downstream of a timestamp (Day 4's
    session alignment, Day 5's contemporaneous/lagged return) is recomputed
    from this shuffled time, so a headline's sentiment is now paired with a
    session it was not actually published ahead of."""
    times = [h.published_at for h in headlines]
    rng.shuffle(times)
    return [replace(h, published_at=t) for h, t in zip(headlines, times)]


def contemporaneous_r(rows: list[dict]) -> float | None:
    """Pearson r between VADER compound and contemporaneous return, or None
    if there are fewer than 2 resolved rows to correlate (can happen on a
    shuffle that reassigns every sentiment-bearing headline to a trading day
    with no price fixture)."""
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float
    real_n: int
    shuffled_rs: list[float]
    n_shuffles_requested: int
    n_shuffles_usable: int
    p_value: float

    @property
    def shuffled_mean(self) -> float:
        return sum(self.shuffled_rs) / len(self.shuffled_rs)

    @property
    def shuffled_abs_max(self) -> float:
        return max(abs(r) for r in self.shuffled_rs)


def run_shuffle_audit(
    headlines: list[Headline],
    live: bool = False,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = 0,
) -> ShuffleAuditResult:
    """Build the real (unshuffled) contemporaneous r, then ``n_shuffles``
    shuffled-timestamp controls, and return a permutation-test comparison.

    ``p_value`` is the two-sided permutation-test fraction of shuffled
    controls whose |r| is at least as large as the real |r| - the standard
    reading is "how often would a pipeline with no genuine timing
    information produce a result this extreme by chance." A *small*
    p-value here would be the actual leakage alarm: it would mean shuffling
    away the real timestamps barely moves the statistic, i.e. the pipeline's
    apparent result does not actually depend on timing being real.
    """
    real_rows, _ = rows_for_headlines(headlines, live=live)
    real_r = contemporaneous_r(real_rows)
    if real_r is None:
        raise ValueError(f"only {len(real_rows)} resolved headline(s) - not enough to correlate")

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled_headlines = shuffle_published_at(headlines, rng)
        shuffled_rows, _ = rows_for_headlines(shuffled_headlines, live=live)
        r = contemporaneous_r(shuffled_rows)
        if r is not None:
            shuffled_rs.append(r)

    if not shuffled_rs:
        raise ValueError("every shuffle produced fewer than 2 resolvable rows; cannot build a null distribution")

    at_least_as_extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
    p_value = at_least_as_extreme / len(shuffled_rs)

    return ShuffleAuditResult(
        real_r=real_r,
        real_n=len(real_rows),
        shuffled_rs=shuffled_rs,
        n_shuffles_requested=n_shuffles,
        n_shuffles_usable=len(shuffled_rs),
        p_value=p_value,
    )


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_shuffle_audit(headlines, live=live, n_shuffles=n_shuffles, seed=seed)
    except ValueError as exc:
        print(f"cannot run shuffle audit: {exc}", file=sys.stderr)
        return 1

    print(f"real (unshuffled) contemporaneous r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled-timestamp control: {result.n_shuffles_usable}/{result.n_shuffles_requested} usable shuffles, "
        f"mean r={result.shuffled_mean:+.3f}  max|r|={result.shuffled_abs_max:.3f}"
    )
    print(
        f"permutation p-value (P[|shuffled r| >= |real r|]) = {result.p_value:.3f}"
    )
    if result.p_value < 0.05:
        print(
            "p < 0.05: the real result is more extreme than shuffled-timestamp controls almost "
            "ever produce - if this were paired with a real (non-null) finding elsewhere in this "
            "repo, that finding would look genuine, not an artifact of the alignment step."
        )
    else:
        print(
            "p >= 0.05: shuffling away the real timestamps produces results just as extreme as "
            "the real one about as often as chance predicts. Consistent with Day 5/6/7's own "
            "finding of no signal - there is nothing here for the audit to catch, and the audit "
            "cannot by itself prove the pipeline would catch a real leak (see tests/test_audit.py "
            "for the synthetic-signal check that proves that separately)."
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
        help="number of shuffled-timestamp controls to build the null distribution from",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, for a reproducible report")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
