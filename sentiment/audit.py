"""Day 8 CLI: ml-pipeline-audit - the shuffled-timestamp leakage control.

    python -m sentiment.audit
    python -m sentiment.audit --trials 1000 --seed 1
    python -m sentiment.audit --live

Per NEXT_STEPS.md's "Done when": a null correlation (Days 5-7's honest
result) does not by itself prove this pipeline is leak-free - a pipeline
that silently used look-ahead information could also land on a near-zero
number on this one small sample by coincidence. The actual control is to
shuffle headline *timestamps* across the headline set, rebuild Day 5's
whole alignment-to-correlation pipeline on the shuffled timestamps, and
recompute the statistic. If a scrambled-timestamp control still "predicts"
returns about as well as the real pairing does, something downstream of
``published_at`` is not actually depending on it the way Day 4 assumes,
and the result is an artifact.

Only ``published_at`` is shuffled. Title, link, source and the VADER score
derived from the title are untouched - the failure mode this guards
against is in *when* a headline is treated as having happened, not in
what it says. Shuffling is a permutation test: the real correlation is
compared against the distribution of correlations produced by many random
re-pairings of (headline content, timestamp), and a p-value says how
unusual the real pairing's statistic is relative to that chance
distribution.

This cannot by itself tell a genuine short-horizon reaction apart from a
leak - both would survive the shuffle. What it does prove, mechanically,
is whether the pipeline's output actually depends on ``published_at`` at
all; ``tests/test_audit.py`` includes a synthetic positive control that
checks this audit has the power to catch a real timestamp-driven
association, not just rubber-stamp a pipeline that ignores timestamps
altogether.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_TRIALS = 500
SIGNIFICANCE = 0.05


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list with the same ``published_at`` values randomly
    reassigned across headlines - everything else (title, link, source, and
    therefore the VADER score derived from the title) stays exactly where
    it was. This breaks the one thing Day 4 exists to get right: which
    timestamp a headline's return is honestly attributed through."""
    shuffled_times = [h.published_at for h in headlines]
    rng.shuffle(shuffled_times)
    return [replace(h, published_at=t) for h, t in zip(headlines, shuffled_times)]


def contemporaneous_r(headlines: list[Headline], live: bool = False) -> tuple[float | None, int]:
    """Rebuild Day 5's pipeline on ``headlines`` and return (r, n) for the
    compound/contemporaneous-return correlation, or (None, n) if fewer than
    2 headlines resolved to a ticker with price data."""
    rows, _unresolved = build_rows(None, live=live, headlines=headlines)
    if len(rows) < 2:
        return None, len(rows)
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, returns), len(rows)


@dataclass(frozen=True)
class ShuffleAudit:
    real_r: float | None
    real_n: int
    trials: int
    shuffled_rs: list[float]
    p_value: float | None


def run_shuffle_audit(
    headlines: list[Headline],
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    live: bool = False,
) -> ShuffleAudit:
    """Permutation test: compare the real timestamp pairing's correlation
    against ``trials`` random re-pairings of the same headlines' content
    and timestamps. ``p_value`` is the two-sided fraction of shuffled
    trials at least as extreme as the real statistic (Davison & Hinkley's
    +1 correction, so it is never reported as exactly zero)."""
    real_r, real_n = contemporaneous_r(headlines, live=live)
    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(trials):
        shuffled = shuffle_published_at(headlines, rng)
        r, _n = contemporaneous_r(shuffled, live=live)
        if r is not None:
            shuffled_rs.append(r)

    p_value = None
    if real_r is not None and shuffled_rs:
        extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
        p_value = (extreme + 1) / (len(shuffled_rs) + 1)

    return ShuffleAudit(real_r=real_r, real_n=real_n, trials=trials, shuffled_rs=shuffled_rs, p_value=p_value)


def run(in_path: Path, trials: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    audit = run_shuffle_audit(headlines, trials=trials, seed=seed, live=live)

    if audit.real_r is None:
        print(f"real data: not enough resolved headlines for a correlation (n={audit.real_n})")
        return 0

    print(f"real:     r={audit.real_r:+.3f}  n={audit.real_n}")
    if audit.shuffled_rs:
        mean_abs_shuffled = sum(abs(r) for r in audit.shuffled_rs) / len(audit.shuffled_rs)
        print(
            f"shuffled: mean|r|={mean_abs_shuffled:.3f}  over {len(audit.shuffled_rs)}/{audit.trials} "
            "usable trials (timestamps permuted, content untouched)"
        )
    if audit.p_value is not None:
        print(f"permutation p-value: {audit.p_value:.3f}")

    if audit.p_value is None or audit.p_value > SIGNIFICANCE:
        print(
            "PASS: the real correlation is not distinguishable from a random-timestamp shuffle - "
            "no detectable timestamp-driven signal to begin with, consistent with Days 5-7's own null result."
        )
        return 0

    print(
        "REVIEW: the real correlation sits in the extreme tail of the shuffled-timestamp null "
        f"(p={audit.p_value:.3f} <= {SIGNIFICANCE}). This does not by itself prove a leak - a genuine "
        "short-horizon reaction would also survive the shuffle - but it means the result depends on "
        "real chronological pairing and needs the alignment logic checked by hand before it is trusted.",
        file=sys.stderr,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to audit")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="number of timestamp shuffles")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, for a reproducible CI run")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.trials, args.seed, args.live))


if __name__ == "__main__":
    main()
