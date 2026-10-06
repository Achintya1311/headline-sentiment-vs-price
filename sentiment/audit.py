"""Day 8 CLI: the leakage/audit pass. Shuffle headline timestamps, confirm
whatever "signal" Days 5-7 found disappears - the "done when" gate the
README's Correctness gate section and NEXT_STEPS.md both name as the test
that decides whether this repo is finished.

    python -m sentiment.audit
    python -m sentiment.audit --iterations 500
    python -m sentiment.audit --event-threshold 0.3

What this actually does: take the real, scraped headlines and compute Day
5's two statistics (the contemporaneous VADER-vs-return correlation, and
the event-study mean-return difference between high- and low-magnitude
headlines) from the real, correct timestamp -> trading-session alignment.
Then, many times, take the *same* headlines, randomly permute which
headline got which ``published_at`` (titles, scores and tickers stay put -
only the clock is shuffled), re-run the real alignment and price-join
pipeline (``sentiment.correlate.build_rows_from_headlines``) on the
shuffled timestamps, and recompute both statistics. That rebuilds the
distribution those statistics would take under the null hypothesis that
timing carries no information - the headline still says what it says, but
it is now (falsely) paired with whichever session the shuffled clock landed
it on.

Reading the p-value: a **small** p-value means the real, correctly-aligned
statistic is an outlier against the shuffled distribution - breaking the
true timestamp -> session pairing changed the number a lot, so whatever
relationship exists depends on the real timing, not an artifact. A
**large** p-value means shuffling barely moves the statistic - the pipeline
would report much the same number no matter which headline got paired with
which session, which is exactly the leak NEXT_STEPS.md warns about *if* the
real, unshuffled statistic were also large: "a shuffled-timestamp control
[that] still predicts returns" means the apparent predictive power has
nothing to do with correct alignment. A large p-value is only reassuring
when paired with an unremarkable observed value - which is this project's
actual case: Day 5 already found the real contemporaneous correlation's own
95% CI spans zero, so there is no large "signal" here for a leak to be
hiding behind in the first place (see README Findings/Limitations for why:
VADER's scores are heavily saturated at one value, the samples are tiny,
and the "market" is a proxy, not a real index).

This is not the same permutation every time: with the fixture's real
headline count (23 resolved headlines, drawn from an ~50-headline,
one-week RSS snapshot), a shuffled timestamp usually still lands within the
same one or two trading sessions the real timestamps span, and a handful of
shuffles may occasionally roll a headline's aligned session past the end of
that ticker's committed one-month price fixture, dropping it from that
iteration's rows. Iterations are only skipped (not counted as zero) when
fewer than 2 rows resolve, since a correlation needs at least 2 points.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_EVENT_THRESHOLD, DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import PermutationResult, pearson_r, permutation_p_value

DEFAULT_ITERATIONS = 500
DEFAULT_SEED = 0


@dataclass(frozen=True)
class AuditResult:
    n_real_rows: int
    contemporaneous: PermutationResult | None
    event_diff: PermutationResult | None
    n_skipped: int
    n_iterations: int


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list with the same headlines but ``published_at`` values
    randomly redistributed among them - everything else (title, source,
    link, scraped_at) stays with its original headline."""
    timestamps = [h.published_at for h in headlines]
    shuffled = timestamps[:]
    rng.shuffle(shuffled)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled)]


def _contemporaneous_r(rows: list[dict]) -> float | None:
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def _event_diff(rows: list[dict], event_threshold: float) -> float | None:
    high = [r["contemporaneous_return"] for r in rows if abs(r["compound"]) > event_threshold]
    low = [r["contemporaneous_return"] for r in rows if abs(r["compound"]) <= event_threshold]
    if not high or not low:
        return None
    return sum(high) / len(high) - sum(low) / len(low)


def run_audit(
    headlines: list[Headline],
    live: bool,
    event_threshold: float,
    n_iterations: int,
    seed: int,
) -> AuditResult:
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    observed_r = _contemporaneous_r(real_rows)
    observed_diff = _event_diff(real_rows, event_threshold)

    rng = random.Random(seed)
    null_r: list[float] = []
    null_diff: list[float] = []
    n_skipped = 0

    for _ in range(n_iterations):
        shuffled = shuffle_published_at(headlines, rng)
        shuffled_rows, _ = build_rows_from_headlines(shuffled, live=False)
        r = _contemporaneous_r(shuffled_rows)
        diff = _event_diff(shuffled_rows, event_threshold)
        if r is None and diff is None:
            n_skipped += 1
            continue
        if r is not None:
            null_r.append(r)
        if diff is not None:
            null_diff.append(diff)

    contemporaneous = permutation_p_value(observed_r, null_r) if observed_r is not None and null_r else None
    event_diff = permutation_p_value(observed_diff, null_diff) if observed_diff is not None and null_diff else None

    return AuditResult(
        n_real_rows=len(real_rows),
        contemporaneous=contemporaneous,
        event_diff=event_diff,
        n_skipped=n_skipped,
        n_iterations=n_iterations,
    )


def _verdict(label: str, result: PermutationResult | None, alpha: float = 0.05) -> str:
    if result is None:
        return f"{label}: not enough resolved rows (real or shuffled) to run the control"
    n_null = len(result.null_samples)
    verdict = (
        "stands out from the shuffled null - depends on the real timestamps, not an artifact of "
        "which headline gets paired with which session"
        if result.p_value < alpha
        else "indistinguishable from the shuffled null - shuffling the timestamps barely moves it, so its "
        "size (whatever it is) is not evidence of a timing-dependent relationship"
    )
    return (
        f"{label}: observed={result.observed:+.4f}  shuffled null: mean={result.null_mean:+.4f} "
        f"std={result.null_std:.4f} (n={n_null})  p={result.p_value:.3f}  -> {verdict}"
    )


def run(in_path: Path, live: bool, event_threshold: float, n_iterations: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = run_audit(headlines, live=live, event_threshold=event_threshold, n_iterations=n_iterations, seed=seed)

    print(f"{result.n_real_rows} headlines resolved to a ticker with price data under the real timestamps")
    print(f"{n_iterations} shuffled-timestamp trials requested, {result.n_skipped} skipped (fewer than 2 usable rows)")
    print(_verdict("contemporaneous r", result.contemporaneous))
    print(_verdict("event-study diff ", result.event_diff))

    if result.contemporaneous is not None and result.contemporaneous.p_value >= 0.05:
        print(
            "\nNot a strong result: large p-values here are reassuring only because the real, "
            "unshuffled contemporaneous r is itself small (Day 5's own 95% CI already spans zero) - "
            "there is no sizeable 'signal' for a leak to be hiding behind. See README Limitations for "
            "what would need to change (a wider date range, a non-saturated sentiment score, a real "
            "benchmark index) before this control could distinguish a real effect from a leaking one "
            "rather than just confirming there is nothing here to leak."
        )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=DEFAULT_EVENT_THRESHOLD,
        help="|compound| above this is 'high-magnitude' for the event-study statistic",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="number of shuffled-timestamp trials to run",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible shuffle")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.event_threshold, args.iterations, args.seed))


if __name__ == "__main__":
    main()
