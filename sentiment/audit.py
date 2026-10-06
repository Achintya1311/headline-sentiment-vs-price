"""Day 8 CLI: leakage audit via headline-timestamp shuffling.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 1

NEXT_STEPS.md's "done when" is a specific, falsifiable test: shuffle the
headline timestamps and the signal must disappear. If a shuffled-timestamp
control still "predicts" returns as often as the real pipeline does, Day 4's
timestamp-to-session alignment is not actually where any correlation is
coming from - something else in the pipeline is leaking the right answer
regardless of when a headline says it was published, and the result is an
artifact.

Mechanics: ``shuffle_timestamps`` keeps every headline's title, source and
score exactly as scraped, and only permutes which ``published_at`` value
each one is stamped with. That is the one input Day 4's alignment and Day 5's
correlation actually depend on, so re-running the same, unmodified pipeline
against many such shuffles and checking whether the correlation and its
statistical significance survive is a direct test of whether the pipeline
needs correct timestamps to produce its result, or ignores them.

This audit always reads prices from the committed fixtures (``live=False``
for every shuffle trial, regardless of the CLI's own ``--live`` flag) -
running hundreds of live fetches per invocation would be both slow and a
pointless real-network cost for a question that fixtures answer exactly as
well. ``--live`` here only controls the one, real (unshuffled) baseline run.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import PearsonResult, pearson_with_ci, permutation_p_value

DEFAULT_N_SHUFFLES = 200
DEFAULT_SEED = 0

# How often a shuffled-timestamp control is allowed to turn up a "significant"
# (95% CI excluding zero) correlation purely by chance before this audit calls
# it a leak rather than noise. A well-behaved 95% CI should do this on about
# 5% of trials; this is set generously above that (double, rounded up) so a
# couple of unlucky draws in a few hundred shuffles don't fail the audit on
# their own, while a control that is "significant" on a large fraction of
# shuffles - the actual leak pattern this test exists to catch - still fails.
MAX_FALSE_POSITIVE_RATE = 0.12

BuildRowsFn = Callable[[list[Headline], bool], tuple[list[dict], list]]


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list of headlines with ``published_at`` reassigned by a
    random permutation across the same headlines. Every other field (title,
    source, link, scraped_at - and therefore the VADER score, which is a pure
    function of title) stays exactly as scraped. This is the minimal change
    that breaks Day 4's alignment without touching anything else."""
    shuffled_ts = [h.published_at for h in headlines]
    rng.shuffle(shuffled_ts)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_ts)]


def correlation_for(
    headlines: list[Headline], live: bool, build_rows_fn: BuildRowsFn
) -> PearsonResult | None:
    """Contemporaneous-return correlation for one set of (possibly shuffled)
    headlines, re-running the real build_rows pipeline. ``None`` if too few
    headlines resolved to a ticker with price data to correlate at all."""
    rows, _ = build_rows_fn(headlines, live)
    if len(rows) < 2:
        return None
    return pearson_with_ci([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


@dataclass(frozen=True)
class ShuffleAuditResult:
    actual: PearsonResult | None
    shuffled: list[PearsonResult]
    n_shuffles_requested: int
    p_value: float | None
    false_positive_rate: float | None

    @property
    def leak_detected(self) -> bool:
        """True if a shuffled-timestamp control "predicts returns" far more
        often than chance would - the literal failure condition NEXT_STEPS.md
        names. Undefined (treated as a pass, nothing to flag) if there were
        too few usable shuffles to estimate a rate from."""
        return self.false_positive_rate is not None and self.false_positive_rate > MAX_FALSE_POSITIVE_RATE

    @property
    def passed(self) -> bool:
        return not self.leak_detected


def _is_significant(result: PearsonResult) -> bool:
    """95% CI excludes zero. ``pearson_with_ci`` returns (-1, 1) for n < 4
    (not enough points for the Fisher z-transform's variance to be defined),
    which never excludes zero - exactly right here, since "significant" is
    not a meaningful label for a correlation computed from 2-3 points."""
    return result.ci_low > 0 or result.ci_high < 0


def shuffle_audit(
    headlines: list[Headline],
    live: bool = False,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
    build_rows_fn: BuildRowsFn = build_rows_from_headlines,
) -> ShuffleAuditResult:
    actual = correlation_for(headlines, live=live, build_rows_fn=build_rows_fn)

    rng = random.Random(seed)
    shuffled: list[PearsonResult] = []
    for _ in range(n_shuffles):
        trial = shuffle_timestamps(headlines, rng)
        result = correlation_for(trial, live=False, build_rows_fn=build_rows_fn)
        if result is not None:
            shuffled.append(result)

    p_value = permutation_p_value(actual.r, [r.r for r in shuffled]) if actual and shuffled else None
    false_positive_rate = (
        sum(1 for r in shuffled if _is_significant(r)) / len(shuffled) if shuffled else None
    )

    return ShuffleAuditResult(
        actual=actual,
        shuffled=shuffled,
        n_shuffles_requested=n_shuffles,
        p_value=p_value,
        false_positive_rate=false_positive_rate,
    )


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = shuffle_audit(headlines, live=live, n_shuffles=n_shuffles, seed=seed)

    if result.actual is None:
        print("no headlines resolved to a ticker with price data; nothing to audit", file=sys.stderr)
        return 1
    if not result.shuffled:
        print(
            "every shuffle produced fewer than 2 resolvable rows; cannot estimate a null "
            "distribution from this fixture",
            file=sys.stderr,
        )
        return 1

    a = result.actual
    print(f"actual (unshuffled):  r={a.r:+.3f}  95% CI [{a.ci_low:+.3f}, {a.ci_high:+.3f}]  n={a.n}")
    print(
        f"shuffled timestamps:  {len(result.shuffled)}/{result.n_shuffles_requested} trials usable, "
        f"mean |r|={sum(abs(r.r) for r in result.shuffled) / len(result.shuffled):.3f}, "
        f"{sum(1 for r in result.shuffled if _is_significant(r))} significant at 95% "
        f"({result.false_positive_rate:.1%})"
    )
    print(f"permutation p-value (actual vs shuffled-null): {result.p_value:.3f}")

    if result.passed:
        print(
            f"PASS: shuffled-timestamp controls turn up a 'significant' correlation "
            f"{result.false_positive_rate:.1%} of the time (<= {MAX_FALSE_POSITIVE_RATE:.0%} allowed) - "
            "no evidence the pipeline's correlation depends on anything but the real, "
            "leak-free alignment. The signal disappears under shuffling because there was "
            "never more than noise for shuffling to disappear."
        )
    else:
        print(
            f"FAIL: shuffled-timestamp controls are 'significant' {result.false_positive_rate:.1%} of "
            f"the time (> {MAX_FALSE_POSITIVE_RATE:.0%} allowed) - a randomised, wrong alignment is "
            "producing a correlation almost as often as the real one, which means the real "
            "correlation is not actually coming from the timestamp-based alignment. Treat any "
            "upstream result from this pipeline as an artifact until this is root-caused."
        )
    return 0 if result.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices for the real, unshuffled baseline only")
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp permutations")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
