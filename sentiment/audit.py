"""Day 8 CLI: the leakage/audit pass - shuffle headline timestamps and check
whatever contemporaneous-return correlation exists disappears.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --n-shuffles 500 --seed 1

This is the "Correctness gate" the README and NEXT_STEPS.md name as the test
that decides whether the repo is finished: Day 4's market-hours alignment is
the only thing standing between "the market reacted to this headline" and
"this return happened before the headline existed." If that alignment is
doing real work, a headline's own publish time matters - swap it for a
different headline's publish time and whatever correlation existed should
not survive, because the headline is now being scored against a session it
never actually preceded.

``shuffle_timestamps`` permutes ``published_at`` across headlines (each
headline keeps its own title and therefore its own ticker and VADER score;
only the timestamp used to align it to a trading session changes). This is
a permutation test, not a single before/after comparison: the real
(unshuffled) Pearson r is compared against the distribution of r's from
``--n-shuffles`` independent shuffles, each rebuilt through the same
alignment -> price -> correlation pipeline
(``sentiment.correlate.build_rows_from_headlines``), and the reported
p-value is the two-sided fraction of shuffles at least as extreme as the
real statistic.

See the README's Day 8 Findings and "Why this might be spurious" section
for what this fixture's small, saturated sample means this test can and
cannot show - in particular, Day 5/6 already found the real statistic is
indistinguishable from zero before any shuffling, so this test mostly
confirms there was no signal to leak in the first place, not that a real
signal survived a strong control.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 300
DEFAULT_SEED = 0
MIN_USABLE = 2


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    n_shuffles_requested: int
    shuffle_rs: list[float]
    shuffle_ns: list[int]
    p_value: float


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list with ``published_at`` values randomly reassigned
    across headlines. Each headline keeps its own source, title and link -
    and therefore its own ticker resolution and VADER score - but is aligned
    to a trading session using a *different* headline's publish time.

    This preserves the marginal distribution of publish times exactly (the
    same set of timestamps, the same overall pre-open/intraday/post-close
    mix) while breaking any real link between what a headline says and when
    it was actually published, which is the one thing Day 4's alignment
    logic depends on to be meaningful.
    """
    times = [h.published_at for h in headlines]
    shuffled_times = times[:]
    rng.shuffle(shuffled_times)
    return [replace(h, published_at=t) for h, t in zip(headlines, shuffled_times)]


def correlation_for(headlines: list[Headline], live: bool) -> tuple[float, int]:
    """Rebuild rows through the real pipeline for this exact set of
    headlines and return (contemporaneous Pearson r, n usable). n can be
    below ``MIN_USABLE`` if, for this particular set of timestamps, too few
    headlines resolve to a ticker with a price bar on the session they land
    on - a shuffle can easily move a headline onto a date this fixture's
    one-month price snapshot has no bar for (see README Findings)."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < MIN_USABLE:
        return 0.0, len(rows)
    compounds = [r["compound"] for r in rows]
    contemporaneous = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, contemporaneous), len(rows)


def run_audit(headlines: list[Headline], live: bool, n_shuffles: int, seed: int) -> AuditResult:
    real_r, real_n = correlation_for(headlines, live)

    rng = random.Random(seed)
    shuffle_rs: list[float] = []
    shuffle_ns: list[int] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        r, n = correlation_for(shuffled, live)
        shuffle_ns.append(n)
        if n >= MIN_USABLE:
            shuffle_rs.append(r)

    if shuffle_rs:
        as_extreme = sum(1 for r in shuffle_rs if abs(r) >= abs(real_r))
        p_value = as_extreme / len(shuffle_rs)
    else:
        p_value = float("nan")

    return AuditResult(
        real_r=real_r,
        real_n=real_n,
        n_shuffles_requested=n_shuffles,
        shuffle_rs=shuffle_rs,
        shuffle_ns=shuffle_ns,
        p_value=p_value,
    )


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = run_audit(headlines, live, n_shuffles, seed)

    print(f"real (unshuffled) contemporaneous r={result.real_r:+.3f}  n={result.real_n}")
    usable = len(result.shuffle_rs)
    print(f"{result.n_shuffles_requested} timestamp-shuffled control runs, {usable} usable (n>={MIN_USABLE} resolved)")

    if usable == 0:
        print("no shuffle produced enough usable rows; cannot form a null distribution on this fixture")
        return 0

    abs_rs = [abs(r) for r in result.shuffle_rs]
    mean_abs = sum(abs_rs) / len(abs_rs)
    print(f"shuffled |r|: mean={mean_abs:.3f}  min={min(abs_rs):.3f}  max={max(abs_rs):.3f}")
    print(f"permutation p-value (two-sided, |r_shuffle| >= |r_real|): {result.p_value:.3f}")

    if result.p_value > 0.05:
        print(
            "p > 0.05: the real statistic is not distinguishable from the shuffled-timestamp null "
            "distribution. On this fixture that is consistent with Day 5/6's own null finding "
            "(contemporaneous r was already inside a CI containing zero) rather than independent "
            "proof the alignment logic is leak-free - see README Findings."
        )
    else:
        print(
            "p <= 0.05: the real statistic sits in the tail of the shuffled null distribution. "
            "On a fixture this small and this saturated, that would be a surprising result worth "
            "re-checking by hand, not one to report as a discovery."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp-shuffle control runs"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for reproducible shuffles")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
