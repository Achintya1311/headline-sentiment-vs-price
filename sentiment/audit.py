"""Day 8 CLI: the shuffled-timestamp leakage audit.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --n-shuffles 2000 --seed 0

NEXT_STEPS.md's "Done when" is explicit: shuffle the headline timestamps
and the signal must disappear. If a shuffled-timestamp control still
predicts returns as well as the real alignment does, the pipeline is
leaking - some other channel (ticker identity, duplicate session returns,
an ordering bug) is doing the "predicting", not the timestamp-gated
alignment Day 4 built.

What gets shuffled: the ``published_at`` values of the resolved headlines
are permuted *among the headlines themselves* (same multiset of real
timestamps, reassigned to different titles/compounds/tickers). Everything
else - titles, VADER scores, ticker resolution, the price fixtures - is
untouched. Re-running Day 4's ``align_headline`` and Day 5's
``build_rows_from_headlines`` on the shuffled timestamps produces a new,
fake session_date/contemporaneous_return pairing for each trial. Repeating
this many times builds a null distribution of the contemporaneous Pearson
r a *correctly-timed-by-chance* pairing would produce; the real run's r is
then located inside that distribution.

This is a permutation test, not a vibe check: ``p_value`` is the fraction
of shuffles whose |r| is at least as large as the real |r|. A large
p-value means the real correlation is unremarkable next to random pairings
- consistent with "the signal disappears under shuffling" and with Day
5/6's own finding that this fixture's r is not distinguishable from zero.
A small p-value would mean something in the real run is *more* correlated
than randomly-timed chance allows, which - given nothing in this pipeline
should let timestamp order matter beyond session assignment - would point
at a leak worth chasing down, not a result worth reporting.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 1000
DEFAULT_SEED = 0


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list of ``Headline``s with the same set of
    ``published_at`` timestamps, permuted across headlines. Titles, links,
    and every other field stay with their original headline - only which
    timestamp each headline carries changes, which is what breaks the
    timing Day 4's alignment depends on."""
    shuffled_times = [h.published_at for h in headlines]
    rng.shuffle(shuffled_times)
    return [replace(h, published_at=t) for h, t in zip(headlines, shuffled_times)]


def contemporaneous_r(headlines: list[Headline], live: bool = False) -> float | None:
    """The contemporaneous Pearson r Day 5 would report for this exact set
    of headlines (which may have shuffled timestamps). ``None`` if fewer
    than 2 headlines resolve to a ticker with price data - too few for a
    correlation, and excluded from the null distribution rather than
    counted as a 0.0 that would understate how dispersed it really is."""
    from sentiment.correlate import build_rows_from_headlines

    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def run_shuffle_trials(
    headlines: list[Headline], n_shuffles: int, seed: int, live: bool = False
) -> list[float]:
    """The null distribution: one contemporaneous r per shuffle trial."""
    rng = random.Random(seed)
    rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_published_at(headlines, rng)
        r = contemporaneous_r(shuffled, live=live)
        if r is not None:
            rs.append(r)
    return rs


def permutation_p_value(real_r: float, null_rs: list[float]) -> float:
    """Two-sided permutation p-value: the fraction of the null distribution
    at least as extreme as the real statistic. ``1.0`` (not an error) if
    the null distribution is empty - no trials to compare against means no
    evidence either way, which should read as "cannot evaluate", not as
    "passed"."""
    if not null_rs:
        return 1.0
    at_least_as_extreme = sum(1 for r in null_rs if abs(r) >= abs(real_r))
    return at_least_as_extreme / len(null_rs)


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    real_r = contemporaneous_r(headlines, live=live)
    if real_r is None:
        print("not enough resolved headlines for a real contemporaneous correlation", file=sys.stderr)
        return 1

    null_rs = run_shuffle_trials(headlines, n_shuffles=n_shuffles, seed=seed, live=live)
    if not null_rs:
        print("no shuffle trial produced a usable correlation; cannot run the audit", file=sys.stderr)
        return 1

    p = permutation_p_value(real_r, null_rs)
    null_mean = sum(null_rs) / len(null_rs)
    null_abs_sorted = sorted(abs(r) for r in null_rs)
    null_abs_p95 = null_abs_sorted[int(0.95 * (len(null_abs_sorted) - 1))]

    print(f"real contemporaneous r = {real_r:+.3f}")
    print(
        f"shuffled-timestamp null: n={len(null_rs)} trials, mean r={null_mean:+.3f}, "
        f"95th pct |r|={null_abs_p95:.3f}"
    )
    print(f"permutation p-value (two-sided) = {p:.3f}")

    if abs(real_r) <= null_abs_p95:
        print(
            "PASS: the real correlation sits inside the shuffled-timestamp null's own typical "
            "spread - it disappears under shuffling because it was never distinguishable from "
            "randomly-timed pairing to begin with. No evidence of a leak; also no evidence of a "
            "real signal (see Day 5's own finding and this day's README section)."
        )
        return 0

    print(
        "FAIL: the real correlation is more extreme than the shuffled-timestamp null typically "
        "produces. Either there is a genuine timing-dependent signal here (unlikely given this "
        "fixture's near-zero variance in compound scores - see Day 5/6 Findings) or something in "
        "the pipeline lets the result depend on more than the headline's actual timestamp. "
        "Investigate before trusting this number."
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp-shuffle trials to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
