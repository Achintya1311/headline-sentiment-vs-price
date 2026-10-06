"""Day 8 CLI: the leakage/audit pass the README's "Correctness gate" and
NEXT_STEPS.md both name - shuffle headline timestamps and confirm whatever
signal the pipeline reports does not survive.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --n-shuffles 500 --event-threshold 0.3

Day 4's ``Alignment.leak_free()`` already asserts, on every row, every run,
that a headline's aligned session cannot start before the headline existed.
That proves the *alignment rule* cannot look ahead. It says nothing about
whether the *statistics* built on top of a correct alignment (Day 5's
correlation, Day 5's event-study group difference) are themselves real, or
could have come out just as easily from the wrong headline-to-return
pairing. This module tests that second thing: permute which headline's
sentiment score lands on which trading session by shuffling ``published_at``
across the fixture's own headlines (same timestamps, scrambled assignment),
rerun Day 4's real alignment and Day 5's real statistics on each shuffle,
and compare the correctly-aligned, observed statistic to that null
distribution.

A statistic that only looks real because of a lucky pairing will sit near
the middle of its own shuffled-null distribution - indistinguishable from
noise. One that is genuinely driven by the correct headline/return
correspondence should stand out from it. See the README's Day 8 Findings
for what this fixture actually shows: Day 5/6 already found both
correlations statistically indistinguishable from zero, which bounds what
this control can demonstrate - see "Why this might still be spurious"
there.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from sentiment.correlate import DEFAULT_EVENT_THRESHOLD, DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0

Statistic = Callable[[list[dict]], "tuple[float, int] | None"]


@dataclass(frozen=True)
class ShuffleResult:
    name: str
    observed: float
    n_observed: int
    null_mean: float
    null_std: float
    n_null: int
    p_value: float

    def leaks(self, alpha: float = 0.05) -> bool:
        """True if the observed statistic is an outlier against its own
        shuffled-null distribution - i.e. something a wrong pairing could not
        plausibly have produced by chance. False does not mean "no leak"; it
        means this control found nothing to distinguish the real pairing
        from a random one, which is also what it reports when there was
        never much of a signal to begin with (see module docstring)."""
        return self.p_value < alpha


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Permute ``published_at`` timestamps across ``headlines``, leaving every
    other field (title, source, link) fixed. Reuses Day 4's real
    ``align_headline`` on the scrambled timestamps - this shuffles which
    headline gets which timing, not the alignment rule itself."""
    shuffled_times = [h.published_at for h in headlines]
    rng.shuffle(shuffled_times)
    return [replace(h, published_at=t) for h, t in zip(headlines, shuffled_times)]


def contemporaneous_r(rows: list[dict]) -> tuple[float, int] | None:
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows]), len(rows)


def lagged_r(rows: list[dict]) -> tuple[float, int] | None:
    lagged = [r for r in rows if r["lagged_return"] is not None]
    if len(lagged) < 2:
        return None
    r_value = pearson_r([r["compound"] for r in lagged], [r["lagged_return"] for r in lagged])
    return r_value, len(lagged)


def make_event_diff(threshold: float) -> Statistic:
    def event_diff(rows: list[dict]) -> tuple[float, int] | None:
        high = [r["contemporaneous_return"] for r in rows if abs(r["compound"]) > threshold]
        low = [r["contemporaneous_return"] for r in rows if abs(r["compound"]) <= threshold]
        if not high or not low:
            return None
        return sum(high) / len(high) - sum(low) / len(low), len(high) + len(low)

    return event_diff


def run_shuffle_test(
    headlines: list[Headline],
    name: str,
    statistic: Statistic,
    n_shuffles: int,
    seed: int,
) -> ShuffleResult | None:
    observed_rows, _ = build_rows_from_headlines(headlines, live=False)
    observed_stat = statistic(observed_rows)
    if observed_stat is None:
        return None
    observed, n_observed = observed_stat

    rng = random.Random(seed)
    null_values: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_published_at(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=False)
        value = statistic(rows)
        if value is not None:
            null_values.append(value[0])

    if not null_values:
        return None

    n = len(null_values)
    mean = sum(null_values) / n
    variance = sum((v - mean) ** 2 for v in null_values) / n
    std = variance**0.5
    extreme = sum(1 for v in null_values if abs(v) >= abs(observed))
    p_value = extreme / n

    return ShuffleResult(
        name=name,
        observed=observed,
        n_observed=n_observed,
        null_mean=mean,
        null_std=std,
        n_null=n,
        p_value=p_value,
    )


def run(in_path: Path, n_shuffles: int, seed: int, event_threshold: float) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)

    checks: list[tuple[str, Statistic]] = [
        ("contemporaneous r", contemporaneous_r),
        ("lagged r", lagged_r),
        (f"event diff (|compound| > {event_threshold})", make_event_diff(event_threshold)),
    ]

    any_leak = False
    ran_any = False
    for name, statistic in checks:
        result = run_shuffle_test(headlines, name, statistic, n_shuffles, seed)
        if result is None:
            print(f"{name}: not enough data to run the shuffle control")
            continue
        ran_any = True
        any_leak = any_leak or result.leaks()
        verdict = "OUTLIER vs shuffled null - investigate for leakage" if result.leaks() else "within shuffled null - no leak evidence"
        print(
            f"{name}: observed={result.observed:+.4f} (n={result.n_observed})  "
            f"shuffled null mean={result.null_mean:+.4f} std={result.null_std:.4f} (n_shuffles={result.n_null})  "
            f"p={result.p_value:.3f}  -> {verdict}"
        )

    if not ran_any:
        print("no statistic had enough data to audit", file=sys.stderr)
        return 1

    if any_leak:
        print("\nFAIL: at least one statistic is an outlier against its shuffled-timestamp null.")
        return 1

    print("\nPASS: every statistic checked is within its own shuffled-timestamp null distribution.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp shuffles to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic null")
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=DEFAULT_EVENT_THRESHOLD,
        help="|compound| above this is 'high-magnitude' for the event-diff check",
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.event_threshold))


if __name__ == "__main__":
    main()
