"""Day 8 CLI: ml-pipeline-audit - shuffle headline timestamps and confirm the
sentiment/return signal disappears.

    python -m sentiment.leakage_audit
    python -m sentiment.leakage_audit --trials 1000 --seed 1

The gate this project is built on (NEXT_STEPS.md "Done when", the README's
Correctness gate, and the hub's knowaboutit.md): randomly reassign the
*timestamps* of the scraped headlines among themselves - same titles, same
tickers, same VADER scores, just attached to the wrong moment in time - and
rerun Day 4's alignment through Day 6's out-of-sample regression. Because
``align_headline`` reads only ``published_at``, a shuffle breaks exactly one
thing: which trading session a headline's content gets credited with
reacting to. If the pipeline's apparent predictive power survives that, it
was never conditioning on *when* the news happened, and the real run's
result would not be trustworthy.

Two signals already reported by Day 5/6 are tracked per run:

- ``lagged_r``: Pearson r between VADER compound and the *next* session's
  open-to-close return (Day 5).
- ``test_r2``: out-of-sample R^2 of the chronological train/test regression
  (Day 6).

Shuffling timestamps also reshuffles *which* headlines end up with a
next-day return to regress against at all (see this module's own README
section) - a trial where the resulting split is degenerate (too few usable
rows) is skipped and counted, not silently dropped.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.regress import DEFAULT_TRAIN_FRAC, fit_and_evaluate, time_split, usable_rows
from sentiment.stats import pearson_with_ci

DEFAULT_TRIALS = 500
DEFAULT_SEED = 0


@dataclass(frozen=True)
class RunMetrics:
    n_resolved: int
    n_lagged: int
    lagged_r: float | None
    n_train: int | None
    n_test: int | None
    test_r2: float | None


@dataclass(frozen=True)
class AuditResult:
    real: RunMetrics
    null: list[RunMetrics]
    skipped_r2: int


def shuffle_timestamps(headlines: list[Headline], seed: int) -> list[Headline]:
    """Headlines with the same multiset of ``published_at`` values randomly
    reassigned across them. Title, source, link and scraped_at stay with
    their original headline - only *when* each headline is said to have
    been published moves, which is exactly the one thing the leakage
    control is supposed to scramble."""
    rng = random.Random(seed)
    shuffled_ts = [h.published_at for h in headlines]
    rng.shuffle(shuffled_ts)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=ts,
            published_raw=h.published_raw,
            scraped_at=h.scraped_at,
        )
        for h, ts in zip(headlines, shuffled_ts)
    ]


def compute_metrics(rows: list[dict], train_frac: float) -> RunMetrics:
    """The same two headline numbers Day 5/6 already report, computed from
    whatever ``rows`` a (real or shuffled) run produced. Either metric comes
    back ``None`` when the run did not produce enough usable rows to compute
    it - never a crash, and never a fabricated 0.0 standing in for "could not
    be computed"."""
    lagged_rows = [r for r in rows if r["lagged_return"] is not None]
    lagged_r = None
    if len(lagged_rows) >= 2:
        lagged_r = pearson_with_ci(
            [r["compound"] for r in lagged_rows], [r["lagged_return"] for r in lagged_rows]
        ).r

    n_train = n_test = None
    test_r2 = None
    usable = usable_rows(rows)
    if usable:
        try:
            train, test = time_split(usable, train_frac)
        except ValueError:
            pass
        else:
            result = fit_and_evaluate(train, test)
            n_train, n_test, test_r2 = result.n_train, result.n_test, result.test_r2

    return RunMetrics(
        n_resolved=len(rows),
        n_lagged=len(lagged_rows),
        lagged_r=lagged_r,
        n_train=n_train,
        n_test=n_test,
        test_r2=test_r2,
    )


def run_audit(in_path: Path, trials: int, seed: int, train_frac: float, live: bool = False) -> AuditResult:
    headlines = read_csv(in_path)
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real = compute_metrics(real_rows, train_frac)

    null: list[RunMetrics] = []
    skipped_r2 = 0
    for trial in range(trials):
        shuffled = shuffle_timestamps(headlines, seed=seed + trial)
        rows, _ = build_rows_from_headlines(shuffled, live=live)
        metrics = compute_metrics(rows, train_frac)
        null.append(metrics)
        if metrics.test_r2 is None:
            skipped_r2 += 1

    return AuditResult(real=real, null=null, skipped_r2=skipped_r2)


def percentile_rank(value: float, population: list[float]) -> float:
    """Fraction of ``population`` at or above ``value`` - how far into the
    upper tail of its own shuffled null distribution the real result sits."""
    if not population:
        return float("nan")
    return sum(1 for p in population if p >= value) / len(population)


def run(in_path: Path, trials: int, seed: int, train_frac: float, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    result = run_audit(in_path, trials, seed, train_frac, live=live)
    real, null = result.real, result.null

    print(f"real run: {real.n_resolved} resolved headlines, {real.n_lagged} with a next-day return")
    print(
        f"real lagged r:  {real.lagged_r:+.3f}"
        if real.lagged_r is not None
        else "real lagged r:  n/a (fewer than 2 usable rows)"
    )
    if real.test_r2 is not None:
        print(f"real test R^2:  {real.test_r2:+.3f}  (train n={real.n_train}, test n={real.n_test})")
    else:
        print("real test R^2:  n/a (not enough usable rows for a time split)")

    lagged_null = [m.lagged_r for m in null if m.lagged_r is not None]
    r2_null = [m.test_r2 for m in null if m.test_r2 is not None]

    print(f"\n{trials} shuffled-timestamp trials (seed={seed}):")
    print(f"  lagged r:  computable in {len(lagged_null)}/{trials}")
    if lagged_null:
        print(
            f"    null mean={sum(lagged_null) / len(lagged_null):+.3f}  "
            f"min={min(lagged_null):+.3f}  max={max(lagged_null):+.3f}"
        )
        if real.lagged_r is not None:
            frac = percentile_rank(real.lagged_r, lagged_null)
            print(f"    real r is at or above {frac:.0%} of the shuffled null distribution")

    print(f"  test R^2:  computable in {len(r2_null)}/{trials} ({result.skipped_r2} skipped - degenerate split)")
    if r2_null:
        print(
            f"    null mean={sum(r2_null) / len(r2_null):+.3f}  "
            f"min={min(r2_null):+.3f}  max={max(r2_null):+.3f}"
        )
        if real.test_r2 is not None:
            frac = percentile_rank(real.test_r2, r2_null)
            print(f"    real R^2 is at or above {frac:.0%} of the shuffled null distribution")

    print(
        "\nleakage check: the real run must not be a positive outlier against its own "
        "shuffled-timestamp null - see README Day 8 Findings for how to read this fixture's result."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="number of shuffled-timestamp trials")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base RNG seed (trial i uses seed+i)")
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC, help="same meaning as sentiment.regress")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.trials, args.seed, args.train_frac, args.live))


if __name__ == "__main__":
    main()
