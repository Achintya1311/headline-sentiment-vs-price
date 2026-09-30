"""Day 8 CLI: shuffle-timestamp leakage audit.

    python -m sentiment.audit
    python -m sentiment.audit --iterations 500 --seed 7
    python -m sentiment.audit --live

NEXT_STEPS.md's "done when" bar for this whole repo: shuffle the headline
timestamps and the signal must disappear. If a shuffled-timestamp control
still predicts returns as well as the real timestamps do, the pipeline is
leaking future price information into the sentiment/return pairing, and
every earlier day's correlation and regression result would be an artifact
of that leak, not a finding.

This is a permutation test, not a unit test: it treats the whole chain from
a headline's raw ``published_at`` through ``sentiment.market_hours.
align_headline`` and ``sentiment.prices.bar_on`` to a Pearson r as one
black-box function of the timestamp pool, and asks whether the real,
correctly-timestamped run is distinguishable from many reruns of that same
function with the timestamps randomly reassigned among the same headlines -
sentiment content, company, and ticker held fixed, only WHO published WHEN
changes. See README's "Why this might be spurious" for what this control
can, and cannot, rule out on this repo's own already-null fixture.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_ITERATIONS = 500
DEFAULT_SEED = 0
SIGNIFICANCE = 0.05
# |r| below this is not a "signal" in the first place - nothing to audit for
# leakage, whatever the permutation p-value says. Matches the event study's
# own |compound| > 0.3 "high-magnitude" cutoff in spirit (sentiment.correlate.
# DEFAULT_EVENT_THRESHOLD), applied here to r instead of compound.
NOTABLE_EFFECT = 0.30

RowsBuilder = Callable[[list[Headline]], list[dict]]
Statistic = Callable[[list[dict]], "float | None"]


@dataclass(frozen=True)
class AuditResult:
    real_stat: float
    n_shuffles_used: int
    shuffled_stats: list[float]
    p_value: float

    @property
    def passes(self) -> bool:
        """The audit only has something to fail: a *notable* real effect
        that a shuffled-timestamp control reproduces just as easily.

        p_value here is P(|shuffled stat| >= |real stat|) under random
        timestamp reassignment. Low p_value means shuffling usually produces
        something weaker than the real result - the real pairing needed the
        genuine timestamps to work, so the "signal" is not an artifact of a
        mechanism that ignores timing. High p_value means shuffled runs
        reproduce the real result about as often as not - if the real
        result was small to begin with that is expected and uninteresting,
        but if it was notable, that is exactly NEXT_STEPS.md's failure mode:
        "a shuffled-timestamp control still predicts returns" means the
        pipeline is leaking (or the correlation survives on something other
        than genuine per-headline timing, e.g. a ticker-level confound -
        see README's "Why this might be spurious").
        """
        if abs(self.real_stat) < NOTABLE_EFFECT:
            return True
        return self.p_value < SIGNIFICANCE


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` reassigned among
    the same pool by an in-place Fisher-Yates shuffle (``rng.shuffle``).

    Everything else - title, source, link, and the sentiment content itself
    - stays exactly as scraped; only which timestamp attaches to which
    headline changes. A headline keeping its own original timestamp is a
    possible outcome, matching an honest random permutation rather than a
    derangement that would bias the control.
    """
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, timestamps)]


def contemporaneous_r(rows: list[dict]) -> float | None:
    """The same statistic ``sentiment.correlate`` reports as "contemporaneous"
    - Pearson r between VADER compound and the aligned session's open-to-close
    return. ``None`` (not raised) when there are too few rows to define r, so
    a shuffle that happens to resolve fewer tickers just drops out of the
    control distribution instead of crashing the whole audit."""
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def permutation_test(
    headlines: list[Headline],
    build_rows: RowsBuilder,
    statistic: Statistic = contemporaneous_r,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> AuditResult:
    """Shuffle-timestamp permutation test against ``statistic`` (default:
    the contemporaneous Pearson r Day 5/`sentiment.correlate` reports).

    ``build_rows`` is injected rather than hardcoded to ``sentiment.
    correlate.build_rows_from_headlines`` so a test can substitute a
    deliberately leaking implementation and confirm this function actually
    flags it - see tests/test_audit.py's positive control, which is the
    only thing standing between this being a real audit and pure theatre.
    """
    real_stat = statistic(build_rows(headlines))
    if real_stat is None:
        raise ValueError("not enough resolved headlines to compute a statistic")

    rng = random.Random(seed)
    shuffled_stats: list[float] = []
    for _ in range(iterations):
        shuffled_headlines = shuffle_timestamps(headlines, rng)
        stat = statistic(build_rows(shuffled_headlines))
        if stat is not None:
            shuffled_stats.append(stat)

    if not shuffled_stats:
        raise ValueError("every shuffled-timestamp run produced no usable rows")

    at_least_as_extreme = sum(1 for s in shuffled_stats if abs(s) >= abs(real_stat))
    p_value = at_least_as_extreme / len(shuffled_stats)

    return AuditResult(
        real_stat=real_stat,
        n_shuffles_used=len(shuffled_stats),
        shuffled_stats=shuffled_stats,
        p_value=p_value,
    )


def run(in_path: Path, iterations: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)

    def build(hs: list[Headline]) -> list[dict]:
        rows, _ = build_rows_from_headlines(hs, live=live)
        return rows

    try:
        result = permutation_test(headlines, build, iterations=iterations, seed=seed)
    except ValueError as exc:
        print(f"cannot audit: {exc}", file=sys.stderr)
        return 1

    mean_abs_shuffled = sum(abs(s) for s in result.shuffled_stats) / len(result.shuffled_stats)
    print(f"real (leak-free) alignment: contemporaneous r={result.real_stat:+.3f}")
    print(
        f"{result.n_shuffles_used} shuffled-timestamp control runs: "
        f"mean |r|={mean_abs_shuffled:.3f}  p(|shuffled r| >= |real r|)={result.p_value:.3f}"
    )
    if abs(result.real_stat) < NOTABLE_EFFECT:
        print(
            f"PASS: |r|={abs(result.real_stat):.3f} is below the {NOTABLE_EFFECT:.2f} 'notable "
            "effect' bar - nothing here for a leak to be hiding behind, consistent with "
            "Day 5/6/7's own null finding on this fixture."
        )
    elif result.passes:
        print(
            f"PASS: real r is notable AND stands out from the shuffled control (p={result.p_value:.3f} "
            f"< {SIGNIFICANCE:.2f}) - shuffling timestamps mostly weakens it, so the effect needs "
            "genuine per-headline timing to appear, not just a leaking or timestamp-blind pipeline."
        )
    else:
        print(
            f"FAIL: real r is notable but a shuffled-timestamp control reproduces it just as "
            f"easily (p={result.p_value:.3f} >= {SIGNIFICANCE:.2f}) - the pipeline may be leaking, "
            "or the correlation survives on something other than genuine timing (e.g. a "
            "ticker-level confound). See README 'Why this might be spurious' before trusting "
            "this number."
        )
    return 0 if result.passes else 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS, help="number of shuffled-timestamp control runs"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible control")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.iterations, args.seed, args.live))


if __name__ == "__main__":
    main()
