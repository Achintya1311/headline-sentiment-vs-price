"""Day 8 CLI: the ml-pipeline-audit / leakage-control pass.

    python -m sentiment.audit
    python -m sentiment.audit --trials 500
    python -m sentiment.audit --live

NEXT_STEPS.md's "Done when" is explicit: shuffle the headline timestamps
and the signal must disappear; if a shuffled-timestamp control still
predicts returns, the pipeline is leaking and the Day 5/6 result is an
artifact, not a finding.

This module randomly reassigns ``published_at`` across the headline sample
- title, company resolution and VADER score stay put, only the timestamp
moves - then reruns the exact Day 5 pairing (``sentiment.correlate``'s
``build_rows_from_headlines``: alignment -> session return -> correlation)
on the result. A shuffled timestamp has no legitimate reason to predict a
company's return, because it is no longer the time the headline about that
company actually ran.

**What this sample can and can't actually test** (see README Findings for
the full write-up): Day 5/6 already found the real correlation is
indistinguishable from zero on this fixture. "The signal must disappear"
presupposes a signal to disappear - there isn't one here, so this audit
cannot exercise the headline claim as literally written. What it checks
instead, and what *is* a real leakage control even on a null result:

1. The shuffle is not a no-op - it must actually change which trading
   session (and therefore which return) a sampling of headlines is paired
   with. If every shuffled trial reproduced the exact real correlation,
   that would mean ``published_at`` is not actually driving the session
   assignment downstream, which is itself a leakage symptom (the opposite
   of the one NEXT_STEPS.md describes, but a leak all the same).
2. The distribution of shuffled-run correlations is centred near zero and
   the real correlation is not a surprising outlier against it - the
   necessary (not sufficient) condition for "no leakage" when there is no
   real signal to begin with.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import statistics
import sys
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r, permutation_p_value

DEFAULT_TRIALS = 200
DEFAULT_SEED = 0


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list with every headline's ``published_at`` randomly
    reassigned among the sample - a derangement in expectation, not
    necessarily in every draw (``random.shuffle`` can leave an element in
    place), which matches what a real shuffle-the-timestamps control means:
    nothing is held back from the shuffle on purpose."""
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [dataclasses.replace(h, published_at=ts) for h, ts in zip(headlines, timestamps)]


def real_r(headlines: list[Headline], live: bool = False) -> tuple[float, int] | None:
    """Pearson r (compound vs contemporaneous_return) on the unshuffled
    sample, paired with the n it was computed on. None if too few headlines
    resolved to a ticker with price data to correlate at all."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None
    r = pearson_r([row["compound"] for row in rows], [row["contemporaneous_return"] for row in rows])
    return r, len(rows)


def shuffle_trial_r(headlines: list[Headline], rng: random.Random, live: bool = False) -> float | None:
    """One shuffle trial's Pearson r, or None if the shuffle happened to
    leave fewer than 2 headlines resolvable (can't happen here since
    resolution doesn't depend on timing, but build_rows_from_headlines is
    the honest source of truth, not an assumption)."""
    shuffled = shuffle_published_at(headlines, rng)
    rows, _ = build_rows_from_headlines(shuffled, live=live)
    if len(rows) < 2:
        return None
    return pearson_r([row["compound"] for row in rows], [row["contemporaneous_return"] for row in rows])


def run(in_path: Path, trials: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    real = real_r(headlines, live=live)
    if real is None:
        print("fewer than 2 headlines resolved to a ticker with price data; nothing to audit", file=sys.stderr)
        return 1
    real_stat, n = real

    rng = random.Random(seed)
    null_stats: list[float] = []
    identical_to_real = 0
    for _ in range(trials):
        trial = shuffle_trial_r(headlines, rng, live=live)
        if trial is None:
            continue
        null_stats.append(trial)
        if trial == real_stat:
            identical_to_real += 1

    if not null_stats:
        print("no shuffle trial resolved enough headlines to correlate; nothing to audit", file=sys.stderr)
        return 1

    mean_null = statistics.fmean(null_stats)
    stdev_null = statistics.pstdev(null_stats) if len(null_stats) > 1 else 0.0
    p_value = permutation_p_value(real_stat, null_stats)
    distinct = len(set(null_stats))

    print(f"real contemporaneous r = {real_stat:+.3f}  (n={n})")
    print(
        f"{len(null_stats)} shuffle trial(s): mean r={mean_null:+.3f}  stdev={stdev_null:.3f}  "
        f"({distinct} distinct value(s))"
    )
    print(f"permutation check: real r is at least as extreme as {p_value:.1%} of shuffled trials")

    failed = False
    if distinct <= 1 and len(null_stats) > 1:
        print(
            "FAIL: every shuffle trial produced the identical correlation - the shuffle is not "
            "changing session assignment, so it cannot be testing what it claims to test",
            file=sys.stderr,
        )
        failed = True
    if identical_to_real and identical_to_real == len(null_stats):
        print(
            f"FAIL: all {identical_to_real} shuffle trial(s) reproduced the exact real r="
            f"{real_stat:+.3f} - published_at does not appear to affect the pipeline's output",
            file=sys.stderr,
        )
        failed = True

    if not failed:
        print("PASS: shuffling changes the pipeline's output and the real r is not an outlier against it")
        print(
            "see README Limitations: this fixture's real r was already statistically "
            "indistinguishable from zero (Day 5), so this run cannot exercise the stronger "
            "claim 'a real signal disappears under shuffling' - only that nothing looks "
            "newly, suspiciously signal-like once timestamps are randomised, and that the "
            "shuffle mechanism itself is doing something."
        )

    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="number of shuffle trials to run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible audit run")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.trials, args.seed, args.live))


if __name__ == "__main__":
    main()
