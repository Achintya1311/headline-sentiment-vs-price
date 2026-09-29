"""Day 8 CLI: the leakage/audit pass - a shuffled-timestamp permutation control.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --trials 1000 --seed 1

NEXT_STEPS.md's "Done when" bar for this repo: shuffle the headline
timestamps and the signal must disappear. Concretely: take the headlines
actually scraped, randomly reassign *which* headline got *which* published
timestamp (every timestamp that was scraped is still used exactly once, but
no longer paired with its original headline), and rerun Day 5's contemporaneous
correlation on the result. A headline's title (and therefore its VADER
``compound`` score and the ticker ``sentiment.tickers.resolve`` maps it to)
never changes - only its timestamp does, which is what Day 4's alignment uses
to pick which trading session's return gets attached to it. If a real
sentiment/return relationship exists, decoupling headline content from real
publish time should destroy it. If a shuffled run still "predicts" returns
about as well as the real one, the pipeline is leaking - something other than
genuine timing is doing the work.

One shuffle proves nothing (it could disappear or survive by chance on a
small sample); this runs many shuffles and reports a proper permutation
p-value: the fraction of shuffled trials whose |correlation| is at least as
large as the real, unshuffled run's. See README's "Why this might be
spurious" section for what a p-value computed against this repo's own
already-null result can and cannot show, and ``tests/test_audit.py`` for a
synthetic case with a real injected relationship, proving this control
actually has the power to catch a leak rather than always reporting "no
leak" regardless of the data.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_TRIALS = 300
DEFAULT_SEED = 0


def shuffle_timestamps(headlines: list[Headline], seed: int) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` permuted among them.

    Every headline keeps its own title, link and source (and therefore its
    own sentiment score and resolved ticker); every timestamp that was
    actually scraped is still used exactly once. What breaks is the pairing
    between the two: a headline's content is decoupled from when it was said
    to have been published, so whichever trading session Day 4's alignment
    now attaches it to is no longer a session the market could plausibly have
    reacted to it in.
    """
    rng = random.Random(seed)
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=ts,
            published_raw=h.published_raw,
            scraped_at=h.scraped_at,
        )
        for h, ts in zip(headlines, timestamps)
    ]


def contemporaneous_r(headlines: list[Headline], live: bool = False) -> tuple[float, int]:
    """Pearson r between VADER compound and contemporaneous return across
    every headline that resolves to a ticker with price data, plus the n it
    was computed from. Needs n >= 2; returns (0.0, n) below that rather than
    raising, the same graceful-degradation ``sentiment.correlate.run`` uses."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return 0.0, len(rows)
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, returns), len(rows)


@dataclass(frozen=True)
class AuditResult:
    observed_r: float
    observed_n: int
    trial_rs: list[float]
    p_value: float
    mean_abs_shuffled_r: float


def run_audit(headlines: list[Headline], trials: int, seed: int, live: bool = False) -> AuditResult:
    """Run the shuffled-timestamp permutation test.

    ``p_value`` is the fraction of shuffled trials whose |r| is at least as
    large as the real, unshuffled |r| - the standard permutation-test
    definition of "could the real result plausibly have arisen from a
    timestamp-content pairing this random?". A trial that ends up with fewer
    than 2 resolved headlines (possible in principle if a shuffle scatters
    every headline onto a session date with no price bar) is dropped rather
    than counted as r=0, so it can't quietly inflate or deflate the p-value.
    """
    observed_r, observed_n = contemporaneous_r(headlines, live=live)

    trial_rs: list[float] = []
    for i in range(trials):
        shuffled = shuffle_timestamps(headlines, seed=seed + i)
        r, n = contemporaneous_r(shuffled, live=live)
        if n >= 2:
            trial_rs.append(r)

    if not trial_rs:
        raise ValueError("no shuffle trial produced enough resolved headlines to correlate")

    at_least_as_extreme = sum(1 for r in trial_rs if abs(r) >= abs(observed_r))
    p_value = at_least_as_extreme / len(trial_rs)
    mean_abs_shuffled_r = sum(abs(r) for r in trial_rs) / len(trial_rs)

    return AuditResult(
        observed_r=observed_r,
        observed_n=observed_n,
        trial_rs=trial_rs,
        p_value=p_value,
        mean_abs_shuffled_r=mean_abs_shuffled_r,
    )


def run(in_path: Path, trials: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    if not headlines:
        print(f"{in_path} has no headlines to audit", file=sys.stderr)
        return 1

    try:
        result = run_audit(headlines, trials=trials, seed=seed, live=live)
    except ValueError as exc:
        print(f"cannot run audit: {exc}", file=sys.stderr)
        return 1

    print(f"observed contemporaneous r={result.observed_r:+.3f} (n={result.observed_n}, real timestamps)")
    print(f"{len(result.trial_rs)} shuffled-timestamp trials: mean |r|={result.mean_abs_shuffled_r:.3f}")
    print(
        f"permutation p-value: {result.p_value:.3f} "
        "(fraction of shuffled trials with |r| >= observed |r|)"
    )
    if result.p_value < 0.05:
        print(
            "p < 0.05: the real, correctly-timed pairing produces a correlation almost no "
            "random timestamp shuffle reproduces - if this had been a real finding, that "
            "would be evidence against a pure look-ahead leak (not proof of a real market "
            "effect on its own)."
        )
    else:
        print(
            "p >= 0.05: shuffled controls routinely produce a correlation at least this "
            "large by chance alone. This does NOT mean the pipeline leaks - the real result "
            "was already close to zero (see README Findings), so there is barely any signal "
            "for a shuffle to make 'disappear'. This run is a null-vs-null comparison, not a "
            "leak-detection pass with teeth; see README 'Why this might be spurious' and "
            "tests/test_audit.py's synthetic case for the version of this test that actually "
            "has the power to catch a real leak."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS, help="number of shuffled-timestamp permutation trials"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base seed for the shuffle trials")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.trials, args.seed, args.live))


if __name__ == "__main__":
    main()
