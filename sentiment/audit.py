"""Day 8 CLI: the leakage/audit pass - shuffle headline timestamps and check
whether the sentiment/return "signal" survives.

    python -m sentiment.audit
    python -m sentiment.audit --trials 500 --seed 1

This is the "Done when" control from ``projects/headline-sentiment-vs-price/
NEXT_STEPS.md``: Day 5's contemporaneous correlation between VADER's
``compound`` and a headline's aligned-session return is only trustworthy if
it depends on headlines actually being paired with their own real
publication time. If the correlation would look just as "significant" no
matter which headline got which timestamp, the pipeline isn't measuring a
timing-dependent reaction - it's measuring an artifact that would survive
even a broken alignment step.

The audit is a permutation test, repeated ``--trials`` times:

1. Randomly reassign ``published_at`` across the headline set (title,
   company, link, source - everything else - stays with its own headline,
   so only the headline/timestamp pairing changes, never which ticker a
   headline resolves to or how VADER scores it).
2. Re-run Day 4's alignment and Day 5's contemporaneous-correlation pipeline
   on the shuffled set.
3. Check whether that shuffled run's 95% CI (Fisher z, same as
   ``sentiment.correlate``) excludes zero - i.e. whether it still "predicts
   returns" by the same standard the real run is judged by.

If shuffled runs come back "significant" far more often than a true null
would (roughly 5% of the time, at a 95% CI), the signal is not actually
tied to real timestamps - NEXT_STEPS.md calls that a leak, and this CLI
exits 1. See ``tests/test_audit.py`` for the control case this is built to
catch: a synthetic fixture with a real, engineered sentiment/return
relationship, where correct alignment recovers it and shuffling reliably
destroys it - proof the test has teeth, not just that it passes vacuously.

On the real committed fixture, this is expected to pass, and not because
the audit is toothless: Day 5/6's Findings already established there is no
real correlation on this data to begin with (see README). A pass here does
not create evidence of a clean result where Day 5/6 found none - it only
confirms that *whatever* the pipeline reports, real or null, does not
depend on timestamps being randomly reassignable without consequence.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import PearsonResult, pearson_with_ci

DEFAULT_TRIALS = 200
DEFAULT_SEED = 0

# How much more often than a true null's ~5% false-positive rate a shuffled
# run is allowed to look "significant" before this audit calls it a leak
# rather than noise. Set well above 5% so this project's small samples (a
# 95% CI on n=23 is already fairly wide) don't trip the gate on ordinary
# sampling variation.
LEAK_THRESHOLD = 0.20


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return new ``Headline``s with ``published_at`` randomly permuted
    across the list. Every other field stays with its own headline, so only
    the headline/timestamp pairing changes - not which ticker a headline
    resolves to (that comes from the title) or how it scores."""
    shuffled_timestamps = [h.published_at for h in headlines]
    rng.shuffle(shuffled_timestamps)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=new_ts,
            published_raw=h.published_raw,
            scraped_at=h.scraped_at,
        )
        for h, new_ts in zip(headlines, shuffled_timestamps)
    ]


def contemporaneous_stat(headlines: list[Headline], live: bool = False) -> PearsonResult | None:
    """Day 5's contemporaneous Pearson r (with 95% CI) for this headline set,
    or ``None`` if fewer than 2 headlines resolved to a ticker with price
    data - not enough to correlate at all."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None
    return pearson_with_ci([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def is_significant(stat: PearsonResult) -> bool:
    """True if ``stat``'s 95% CI excludes zero."""
    return not (stat.ci_low <= 0.0 <= stat.ci_high)


@dataclass(frozen=True)
class AuditResult:
    real: PearsonResult
    real_significant: bool
    trials_run: int
    shuffled_significant_count: int
    shuffled_significant_fraction: float
    leak_suspected: bool


def run_audit(
    headlines: list[Headline],
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> AuditResult:
    real = contemporaneous_stat(headlines, live=live)
    if real is None:
        raise ValueError("fewer than 2 resolved headlines - nothing to audit")

    rng = random.Random(seed)
    trials_run = 0
    shuffled_significant = 0
    for _ in range(trials):
        shuffled = shuffle_timestamps(headlines, rng)
        stat = contemporaneous_stat(shuffled, live=live)
        if stat is None:
            continue
        trials_run += 1
        if is_significant(stat):
            shuffled_significant += 1

    if trials_run == 0:
        raise ValueError("every shuffle resolved fewer than 2 headlines - cannot build a null distribution")

    fraction = shuffled_significant / trials_run
    return AuditResult(
        real=real,
        real_significant=is_significant(real),
        trials_run=trials_run,
        shuffled_significant_count=shuffled_significant,
        shuffled_significant_fraction=fraction,
        leak_suspected=fraction > LEAK_THRESHOLD,
    )


def run(in_path: Path, trials: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_audit(headlines, trials=trials, seed=seed, live=live)
    except ValueError as exc:
        print(f"cannot audit: {exc}", file=sys.stderr)
        return 1

    print(
        f"real contemporaneous r = {result.real.r:+.3f}  95% CI [{result.real.ci_low:+.3f}, "
        f"{result.real.ci_high:+.3f}]  n={result.real.n}"
        f"  ({'significant' if result.real_significant else 'not significant'})"
    )
    print(
        f"shuffled-timestamp control: {result.shuffled_significant_count}/{result.trials_run} "
        f"trials ({result.shuffled_significant_fraction:.1%}) still came back significant"
    )

    if result.leak_suspected:
        print(
            f"FAIL: shuffled timestamps still look significant {result.shuffled_significant_fraction:.1%} "
            f"of the time (> {LEAK_THRESHOLD:.0%}) - the correlation does not depend on headlines "
            "keeping their real publication time. That is a leak, not a finding: something upstream "
            "of the correlation (alignment, the ticker/price lookup, or the correlation itself) is "
            "not actually using the timestamp the way Day 4 assumes."
        )
        return 1

    print(
        f"PASS: shuffled timestamps come back significant only {result.shuffled_significant_fraction:.1%} "
        f"of the time (<= {LEAK_THRESHOLD:.0%}), consistent with a true null under random reassignment. "
        "On this fixture that is expected either way: Day 5/6 already found no real correlation to "
        "begin with (see README Findings) - this audit confirms that null does not depend on the "
        "timestamps being real, rather than manufacturing a pass."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="number of random timestamp shuffles")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic audit")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.trials, args.seed, args.live))


if __name__ == "__main__":
    main()
