"""Day 8 CLI: the leakage/audit pass NEXT_STEPS.md's "Done when" section
names as the test that decides whether this repo's results are believed.

    python -m sentiment.audit
    python -m sentiment.audit --trials 500
    python -m sentiment.audit --live

Shuffle headline publish timestamps and re-run Day 5's contemporaneous
correlation many times. Shuffling severs the one link ``sentiment/
market_hours.py`` depends on - a headline's real publish time versus the
market's - while leaving everything else (which ticker a headline resolves
to, its VADER compound, which price bars exist) untouched. If the resulting
correlation routinely comes back "significant" anyway, the pipeline is not
actually using real timing to produce its result - it is leaking price
information through some other path, and the real (unshuffled) number is an
artifact, not a finding.

This is a narrower claim than "the result is real": it only tests leakage
through the *timestamp-alignment* path this repo's Day 4 module owns. A
score that was itself computed from future price data (never done here, but
see the README's "why this might be spurious" section) would sail through
this control untouched, because it was never routed through
``align_headline`` in the first place. See the README for what this control
does and does not rule out.
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

# Under a true null (no real relationship, no leak), a 95% CI excludes zero
# by pure chance about 5% of the time. Flag the shuffled control as showing
# a leak only once it clears this far past that chance rate - small enough
# to catch a real leak (which drives the rate close to 100%), large enough
# that one or two unlucky trials out of a few hundred do not cry wolf.
SIGNIFICANT_FRACTION_ALARM = 0.25


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` (same order, same identity per slot)
    with ``published_at`` values randomly permuted across the list.

    Same multiset of timestamps, reassigned to different headlines - this
    severs any real link between a given headline's own content and the
    time it carries, without changing the timestamp distribution Day 4's
    alignment logic sees, or anything else about the headline (title,
    ticker resolution, VADER score).
    """
    shuffled_times = [h.published_at for h in headlines]
    rng.shuffle(shuffled_times)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=t,
            published_raw=h.published_raw,
            scraped_at=h.scraped_at,
        )
        for h, t in zip(headlines, shuffled_times)
    ]


def is_significant(stat: PearsonResult) -> bool:
    """True if ``stat``'s 95% CI excludes zero - the standard, if generous,
    "distinguishable from no correlation" bar."""
    return stat.ci_low > 0 or stat.ci_high < 0


def trial_correlation(headlines: list[Headline], live: bool = False) -> PearsonResult | None:
    """Contemporaneous compound/return correlation for ``headlines``, or
    ``None`` if fewer than 2 headlines resolve to a ticker with price data."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None
    return pearson_with_ci([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


@dataclass(frozen=True)
class AuditResult:
    real: PearsonResult | None
    shuffled: list[PearsonResult]
    n_significant: int
    fraction_significant: float

    @property
    def leak_suspected(self) -> bool:
        return self.fraction_significant > SIGNIFICANT_FRACTION_ALARM


def run_shuffle_trials(
    headlines: list[Headline], trials: int, seed: int, live: bool = False
) -> AuditResult:
    real = trial_correlation(headlines, live=live)

    rng = random.Random(seed)
    shuffled: list[PearsonResult] = []
    for _ in range(trials):
        control = trial_correlation(shuffle_timestamps(headlines, rng), live=live)
        if control is not None:
            shuffled.append(control)

    n_significant = sum(1 for stat in shuffled if is_significant(stat))
    fraction = n_significant / len(shuffled) if shuffled else 0.0
    return AuditResult(real=real, shuffled=shuffled, n_significant=n_significant, fraction_significant=fraction)


def run(in_path: Path, live: bool, trials: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = run_shuffle_trials(headlines, trials=trials, seed=seed, live=live)

    if result.real is None:
        print("real pipeline: fewer than 2 headlines resolved to a ticker - nothing to audit", file=sys.stderr)
        return 1

    print(
        f"real (unshuffled):   r={result.real.r:+.3f}  95% CI [{result.real.ci_low:+.3f}, "
        f"{result.real.ci_high:+.3f}]  n={result.real.n}  "
        f"{'significant' if is_significant(result.real) else 'not significant'}"
    )
    print(
        f"shuffled-timestamp control: {result.n_significant}/{len(result.shuffled)} trials "
        f"({result.fraction_significant:.1%}) came back significant at 95% "
        f"(~5% expected by chance alone under no leak)"
    )

    if result.leak_suspected:
        print(
            "LEAK SUSPECTED: shuffled-timestamp trials are significant far more often than chance "
            "would predict - the correlation does not depend on real publish timing. Do not trust "
            "the unshuffled number until this is root-caused.",
            file=sys.stderr,
        )
        return 1

    print(
        "no leak detected: shuffling headline timestamps makes the correlation disappear at "
        "roughly the rate pure chance would, so whatever the real number shows, it is not an "
        "artifact of the timestamp-alignment pipeline."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="number of shuffled-timestamp trials")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible CI run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.trials, args.seed))


if __name__ == "__main__":
    main()
