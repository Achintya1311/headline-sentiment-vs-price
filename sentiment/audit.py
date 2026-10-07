"""Day 8 CLI: the leakage/audit pass NEXT_STEPS.md's "Done when" section
requires - shuffle headline timestamps and confirm whatever correlation Day
5 found is not an artifact of the pipeline, not of a real (or accidentally
leaked) relationship between sentiment and price.

    python -m sentiment.audit
    python -m sentiment.audit --n-perm 5000 --alpha 0.05

How it works: take the same resolved (compound, contemporaneous_return)
pairs ``sentiment.correlate`` builds, and compute their real Pearson r.
Then, many times, randomly permute *which headline carries which
published_at* - everything else (title, company, ticker, compound) stays
with the original headline - and rebuild the rows from scratch, so a
shuffled run can also land on a different trading session, a different
price bar, or even drop out entirely if no bar exists for the session a
shuffled timestamp now resolves to. Recompute r each time. That null
distribution answers the actual question: could the real alignment's
correlation just as easily have come from a random timestamp assignment?
If so, Day 4's timing logic isn't earning its keep - the correlation
doesn't depend on getting the timing right, which is exactly what a leak
(or a coincidence dressed up as a result) would look like.

This is deliberately not the same as shuffling the final (x, y) pairs
directly: shuffling *timestamps* goes through the real alignment and price
lookup code on every trial, so a bug in that code (not just in the
correlation step) would also show up as an inflated real-vs-null gap.

See the README's "Why this might be spurious" section for what this test
can and cannot tell you on this fixture, and for the other threats to the
earlier days' results it does not cover (most of them are not about
leakage at all).
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r, permutation_p_value

DEFAULT_N_PERM = 2000
DEFAULT_ALPHA = 0.05


def shuffle_headline_times(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` randomly permuted
    across them - the same multiset of real timestamps, reassigned to
    different headlines, so title/company/ticker/compound (none of which
    depend on the timestamp) are untouched and only the timing-dependent
    alignment can change."""
    times = [h.published_at for h in headlines]
    shuffled_times = times[:]
    rng.shuffle(shuffled_times)
    return [replace(h, published_at=t) for h, t in zip(headlines, shuffled_times)]


@dataclass(frozen=True)
class PermutationResult:
    real_r: float
    real_n: int
    null_rs: list[float]
    p_value: float


def permutation_test(
    headlines: list[Headline],
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
    live: bool = False,
) -> PermutationResult | None:
    """Run the shuffle control. Returns ``None`` if the real, correctly
    aligned data does not even resolve to the 2+ points a correlation
    needs - there is then nothing for a shuffle control to audit."""
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(real_rows) < 2:
        return None
    real_r = pearson_r(
        [r["compound"] for r in real_rows], [r["contemporaneous_return"] for r in real_rows]
    )

    rng = random.Random(seed)
    null_rs: list[float] = []
    for _ in range(n_perm):
        shuffled = shuffle_headline_times(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=live)
        if len(rows) < 2:
            # A shuffled timestamp can land a headline on a session with no
            # price bar at all (see sentiment.prices.bar_on) - that trial's
            # row drops out rather than being forced into the comparison,
            # same as the real pipeline does for the unshuffled data.
            continue
        null_rs.append(
            pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])
        )

    if not null_rs:
        return None
    return PermutationResult(
        real_r=real_r, real_n=len(real_rows), null_rs=null_rs, p_value=permutation_p_value(real_r, null_rs)
    )


def run(in_path: Path, n_perm: int, seed: int, alpha: float, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = permutation_test(headlines, n_perm=n_perm, seed=seed, live=live)
    if result is None:
        print("fewer than 2 resolved headlines in both the real and every shuffled run; nothing to audit")
        return 1

    mean_abs_null = sum(abs(r) for r in result.null_rs) / len(result.null_rs)
    print(f"real contemporaneous r = {result.real_r:+.3f}  (n={result.real_n})")
    print(
        f"shuffled-timestamp null: {len(result.null_rs)} trials, mean |r| = {mean_abs_null:.3f}, "
        f"range [{min(result.null_rs):+.3f}, {max(result.null_rs):+.3f}]"
    )
    print(f"empirical two-sided p-value = {result.p_value:.4f} (alpha={alpha})")

    if result.p_value <= alpha:
        print(
            "FAIL: the real alignment's correlation is a significant outlier against the "
            "shuffled-timestamp null - the pipeline may be leaking look-ahead, or this result "
            "is an artifact of something other than genuine timing-dependent signal."
        )
        return 1

    print(
        "PASS: the real alignment's correlation is not distinguishable from what random "
        "timestamp assignment already produces on this fixture - no evidence the pipeline "
        "manufactures significance out of broken timing. See README 'Why this might be "
        "spurious' for what this does not rule out."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM, help="number of shuffled trials")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, for a reproducible null distribution")
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="significance level for the leakage gate"
    )
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_perm, args.seed, args.alpha, args.live))


if __name__ == "__main__":
    main()
