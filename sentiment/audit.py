"""Day 8 CLI: the ml-pipeline-audit pass NEXT_STEPS.md's "Done when" section
promises - a shuffled-timestamp leakage control - plus a demonstration of
why it matters.

    python -m sentiment.audit
    python -m sentiment.audit --permutations 500 --seed 0

What this tests: Day 4's alignment is leak-free by construction (
``Alignment.leak_free()`` is asserted on every row - see
``sentiment/market_hours.py``), but construction being leak-free is not
the same as the measured correlation being free of leakage - a bug
anywhere else (ticker resolution, scoring, the correlation itself) could
still let a look-ahead or look-back relationship through undetected. The
control: reassign which headline got which publish time, recompute the
same contemporaneous-return correlation on the shuffled data, and check
where the real result falls in the resulting null distribution. If the
real pipeline has no timestamp-driven signal to begin with, its statistic
should look like an ordinary draw from that null, not an outlier - this is
a two-sided permutation test, not a single before/after comparison.

A second run demonstrates the control actually has power to catch a real
leak: the same correlation recomputed using the naive, non-leak-free
alignment Day 4 replaced (a headline's own calendar date, no pre-open /
intraday / post-close distinction) shows what shuffling is supposed to
catch. See the README's Day 8 Findings for what this fixture actually
shows either way.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.market_hours import IST, align_headline
from sentiment.stats import pearson_with_ci

DEFAULT_PERMUTATIONS = 500
DEFAULT_SEED = 0
ALPHA = 0.05


def leak_free_session_date(published_at: datetime) -> date:
    """Day 4's real alignment - what the pipeline actually uses."""
    return align_headline(published_at).session_date


def naive_same_day_session_date(published_at: datetime) -> date:
    """The pre-Day-4 bug, kept here only as a comparison case: a headline's
    own calendar date in IST, with no pre-open/intraday/post-close
    distinction. An intraday or post-close headline paired this way is
    credited with (or blamed for) price action that happened before it
    existed - look-back leakage, not prediction."""
    return published_at.astimezone(IST).date()


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Reassign ``published_at`` values across ``headlines``, keeping every
    other field (title, scraped_at, ...) attached to its original headline.
    Breaks whatever relationship exists between a headline's content and
    its publish time - exactly the relationship any timestamp-driven
    leakage would depend on."""
    shuffled_times = [h.published_at for h in headlines]
    rng.shuffle(shuffled_times)
    return [replace(h, published_at=t) for h, t in zip(headlines, shuffled_times)]


@dataclass(frozen=True)
class PermutationResult:
    real_r: float
    real_n: int
    null_rs: list[float]
    p_value: float

    @property
    def null_mean(self) -> float:
        return sum(self.null_rs) / len(self.null_rs)


def _contemporaneous_r(headlines: list[Headline], session_date_fn, live: bool) -> tuple[float, int] | None:
    rows, _ = build_rows_from_headlines(headlines, live=live, align_fn=session_date_fn, verbose=False)
    if len(rows) < 2:
        return None
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_with_ci(compounds, returns).r, len(rows)


def permutation_test(
    headlines: list[Headline],
    session_date_fn,
    n_perm: int,
    seed: int,
    live: bool = False,
) -> PermutationResult:
    """Two-sided permutation test: is the real contemporaneous Pearson r an
    outlier against the distribution of the same statistic under repeated
    random timestamp reassignment?

    ``p_value`` uses the standard add-one correction (``(b+1)/(n+1)``,
    Davison & Hinkley) - the real draw is itself one valid realisation under
    the null of "timestamps carry no information", so the denominator
    counts it too; a p-value of exactly 0 from a finite sample would
    otherwise overstate the evidence.
    """
    real = _contemporaneous_r(headlines, session_date_fn, live)
    if real is None:
        raise ValueError("fewer than 2 resolved headlines; nothing to test")
    real_r, real_n = real

    rng = random.Random(seed)
    null_rs: list[float] = []
    for _ in range(n_perm):
        shuffled = shuffle_timestamps(headlines, rng)
        result = _contemporaneous_r(shuffled, session_date_fn, live)
        if result is not None:
            null_rs.append(result[0])

    if not null_rs:
        raise ValueError("no permutation produced 2+ resolved headlines; cannot build a null distribution")

    extreme = sum(1 for r in null_rs if abs(r) >= abs(real_r))
    p_value = (extreme + 1) / (len(null_rs) + 1)

    return PermutationResult(real_r=real_r, real_n=real_n, null_rs=null_rs, p_value=p_value)


def run(in_path: Path, n_perm: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)

    print(f"shuffled-timestamp leakage control ({n_perm} permutations, seed={seed})")
    print()
    print("real pipeline (Day 4's leak-free alignment):")
    try:
        leak_free = permutation_test(headlines, leak_free_session_date, n_perm, seed, live)
    except ValueError as exc:
        print(f"  cannot run: {exc}", file=sys.stderr)
        return 1

    print(f"  real r={leak_free.real_r:+.3f} (n={leak_free.real_n})")
    print(
        f"  shuffled-timestamp null: mean r={leak_free.null_mean:+.3f} "
        f"over {len(leak_free.null_rs)} valid permutations"
    )
    print(f"  two-sided permutation p={leak_free.p_value:.3f}")
    if leak_free.p_value < ALPHA:
        passed = False
        print(
            f"  FAIL: real |r| is a significant outlier against the shuffled-timestamp null "
            f"(p<{ALPHA}) - the pipeline may be leaking. See the README's Day 8 Findings."
        )
    else:
        passed = True
        print(
            f"  PASS: real r is consistent with the shuffled-timestamp null (p>={ALPHA}) - "
            "shuffling timestamps does not make the result more or less extreme, which is what "
            "'no detectable timestamp-driven signal' looks like."
        )

    print()
    print("for comparison only - naive same-calendar-day alignment (the pre-Day-4 bug, not used by the real pipeline):")
    try:
        naive = permutation_test(headlines, naive_same_day_session_date, n_perm, seed, live)
    except ValueError as exc:
        print(f"  cannot run: {exc}", file=sys.stderr)
    else:
        print(f"  real r={naive.real_r:+.3f} (n={naive.real_n})")
        print(
            f"  shuffled-timestamp null: mean r={naive.null_mean:+.3f} "
            f"over {len(naive.null_rs)} valid permutations"
        )
        print(f"  two-sided permutation p={naive.p_value:.3f}")
        if max(naive.null_rs) - min(naive.null_rs) < 1e-9:
            print(
                "  NOTE: every permutation produced the identical statistic. This fixture's 50 "
                "headlines were all scraped on one calendar day, and this naive alignment keeps "
                "only the date component of published_at, discarding time-of-day - shuffling "
                "times within a single date is a no-op for it. The control cannot demonstrate "
                "catching this particular bug on this fixture; see tests/test_audit.py for a "
                "synthetic multi-day case where the same code does catch it, and the README's "
                "Day 8 Findings for why this is recorded as a real limitation, not patched over."
            )
        else:
            print("  exists only to show the control has power to catch a leak if one existed.")

    return 0 if passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--permutations", type=int, default=DEFAULT_PERMUTATIONS, help="number of timestamp shuffles to draw"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic null distribution")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.permutations, args.seed, args.live))


if __name__ == "__main__":
    main()
