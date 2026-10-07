"""Day 8 CLI: the shuffled-timestamp leakage audit, plus a synthetic check
that the audit actually has the power to catch a real leak.

    python -m sentiment.audit
    python -m sentiment.audit --trials 500 --seed 1

This is the "Done when" gate NEXT_STEPS.md and the README's correctness
gate both name: "shuffle headline timestamps and the signal must disappear.
If a shuffled-timestamp control still predicts returns, the pipeline is
leaking." Two checks run, not one:

1. **Real-data placebo test.** Permute which headline got which
   ``published_at`` among this repo's own committed timestamps (same
   multiset of times, reassigned to different headline content), realign
   every row from scratch through the real ``sentiment.align`` logic, and
   recompute the same statistics Day 5/6 report (contemporaneous Pearson r,
   out-of-sample R^2). Repeated over many trials this gives a null
   distribution; the real (unshuffled) result is checked against it.

2. **Synthetic power check.** The real fixture's own finding (README Day
   5/6) is already a null result - 17 of 23 headlines tied at the same
   VADER score, out-of-sample R^2 = 0.000 exactly because the regressor has
   zero variance. A placebo test that always reports "no leakage" on data
   that already had no detectable signal before shuffling proves nothing -
   a test that cannot fail is not a test. A synthetic dataset with a real,
   noise-free sentiment -> next-day-return relationship wired in by
   construction (same clean-signal idea ``tests/test_regress.py`` already
   uses) is run through the exact same ``align_headline`` + shuffle
   machinery, to show shuffling destroys a genuine signal when one exists.
   That is what gives check 1's clean result on real data any evidentiary
   weight - see the README's "why this might be spurious" section for what
   this still cannot rule out.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.market_hours import IST, align_headline, is_trading_day
from sentiment.regress import DEFAULT_TRAIN_FRAC, fit_and_evaluate, time_split, usable_rows
from sentiment.stats import pearson_with_ci

DEFAULT_TRIALS = 500
DEFAULT_SEED = 0

# How strong the synthetic, noise-free signal must look before shuffling,
# and how weak it must look after, for the audit's own power check to pass.
# Thresholds are deliberately loose - this is a sanity check on the test
# apparatus, not a precision measurement.
SYNTHETIC_UNSHUFFLED_MIN_R2 = 0.8
SYNTHETIC_SHUFFLED_MAX_MEAN_R2 = 0.3

# Two-sided placebo p-value floor: if the real result ranks inside the top
# 5% most extreme of its own shuffled null distribution, that is flagged as
# a possible leak rather than waved through.
REAL_DATA_P_VALUE_FLOOR = 0.05


@dataclass(frozen=True)
class PipelineStat:
    """The two statistics Day 5/6 already report, recomputed from one set
    of rows so the audit can compare real vs. shuffled on the same footing."""

    contemporaneous_r: float | None
    oos_r2: float | None


def stat_from_rows(rows: list[dict], train_frac: float) -> PipelineStat:
    contemp_r: float | None = None
    if len(rows) >= 2:
        contemp_r = pearson_with_ci(
            [r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows]
        ).r

    oos_r2: float | None = None
    lagged = usable_rows(rows)
    if lagged:
        try:
            train, test = time_split(lagged, train_frac)
            oos_r2 = fit_and_evaluate(train, test).test_r2
        except ValueError:
            oos_r2 = None

    return PipelineStat(contemporaneous_r=contemp_r, oos_r2=oos_r2)


def shuffle_published_at(headlines: list[Headline], seed: int) -> list[Headline]:
    """Reassign this exact set of ``published_at`` values across the same
    headlines, breaking the link between what a headline says and when it
    was really published while leaving the set of real timestamps (and so
    the set of real trading sessions in play) unchanged.

    Headlines are re-sorted by their new ``published_at`` afterwards,
    matching ``sentiment.headline.write_csv``'s own oldest-first ordering -
    the real pipeline always sees headlines in that order, and Day 6's
    chronological train/test split depends on it.
    """
    rng = random.Random(seed)
    times = [h.published_at for h in headlines]
    rng.shuffle(times)
    shuffled = [replace(h, published_at=t, published_raw=t.isoformat()) for h, t in zip(headlines, times)]
    return sorted(shuffled, key=lambda h: h.published_at)


def run_real_data_trials(
    headlines: list[Headline], trials: int, seed: int, train_frac: float
) -> list[PipelineStat]:
    stats = []
    for i in range(trials):
        shuffled = shuffle_published_at(headlines, seed + i)
        rows, _ = build_rows_from_headlines(shuffled, live=False)
        stats.append(stat_from_rows(rows, train_frac))
    return stats


def placebo_p_value(real_value: float | None, null_values: list[float | None]) -> float | None:
    """Two-sided fraction of ``null_values`` at least as extreme as
    ``real_value``. ``None`` is excluded from the comparison (a trial that
    could not compute the statistic, e.g. not enough usable rows, says
    nothing about leakage either way)."""
    if real_value is None:
        return None
    population = [v for v in null_values if v is not None]
    if not population:
        return None
    count = sum(1 for v in population if abs(v) >= abs(real_value))
    return count / len(population)


# --- Synthetic power check ---------------------------------------------

_SYNTHETIC_BASE_DATE = date(2026, 9, 1)


def _trading_dates(n: int, start: date) -> list[date]:
    dates = []
    d = start
    while len(dates) < n:
        if is_trading_day(d):
            dates.append(d)
        d += timedelta(days=1)
    return dates


@dataclass(frozen=True)
class SyntheticWorld:
    """A fake universe of ``n_dates`` trading sessions, each with its own
    'true' next-day return, and ``headlines_per_date`` headlines per date
    whose sentiment score is an exact (noise-free) linear function of that
    date's true return - built in by construction, the same clean-signal
    idea ``tests/test_regress.py``'s own synthetic test uses, so a
    correlation-destroying operation (timestamp shuffling) has a real
    effect to destroy."""

    headlines: list[Headline]
    compound_by_link: dict[str, float]
    true_return_by_date: dict[date, float]


def make_synthetic_world(n_dates: int, headlines_per_date: int, true_slope: float) -> SyntheticWorld:
    dates = _trading_dates(n_dates, _SYNTHETIC_BASE_DATE)
    # Spread of returns around zero, not all the same sign, so a shuffle
    # that pairs a headline with the wrong date's return is a real miss,
    # not an accidental near-match.
    true_return_by_date = {d: 0.01 * (i - (n_dates - 1) / 2) for i, d in enumerate(dates)}

    headlines: list[Headline] = []
    compound_by_link: dict[str, float] = {}
    for d in dates:
        for j in range(headlines_per_date):
            link = f"synthetic-{d.isoformat()}-{j}"
            published_at = datetime.combine(d, time(8, 0 + j, 0), tzinfo=IST)  # pre-open
            headlines.append(
                Headline(
                    source="synthetic",
                    title=f"synthetic headline for {d.isoformat()} #{j}",
                    link=link,
                    published_at=published_at,
                    published_raw=published_at.isoformat(),
                    scraped_at=published_at,
                )
            )
            compound_by_link[link] = true_slope * true_return_by_date[d]

    return SyntheticWorld(
        headlines=headlines, compound_by_link=compound_by_link, true_return_by_date=true_return_by_date
    )


def synthetic_rows(world: SyntheticWorld, headlines: list[Headline]) -> list[dict]:
    """Realign ``headlines`` (a real or shuffled copy of ``world.headlines``)
    through the real ``align_headline`` logic and attach each row's
    ``lagged_return`` from the date it lands on - exactly what
    ``build_rows_from_headlines`` does to real headlines, at the statistics
    layer only, with no ticker or price-fixture plumbing needed."""
    rows = []
    for h in headlines:
        alignment = align_headline(h.published_at)
        lagged_return = world.true_return_by_date.get(alignment.session_date)
        if lagged_return is None:
            continue  # shuffled past the edge of the synthetic world's date range
        rows.append(
            {
                "compound": world.compound_by_link[h.link],
                "contemporaneous_return": lagged_return,
                "lagged_return": lagged_return,
            }
        )
    return rows


@dataclass(frozen=True)
class SyntheticPowerCheck:
    unshuffled_r2: float | None
    shuffled_r2s: list[float | None]
    shuffled_mean_r2: float | None

    @property
    def passed(self) -> bool:
        if self.unshuffled_r2 is None or self.unshuffled_r2 < SYNTHETIC_UNSHUFFLED_MIN_R2:
            return False
        if self.shuffled_mean_r2 is None:
            return False
        return self.shuffled_mean_r2 < SYNTHETIC_SHUFFLED_MAX_MEAN_R2


def run_synthetic_power_check(
    n_dates: int, headlines_per_date: int, true_slope: float, trials: int, seed: int, train_frac: float
) -> SyntheticPowerCheck:
    world = make_synthetic_world(n_dates, headlines_per_date, true_slope)

    unshuffled_rows = synthetic_rows(world, world.headlines)
    unshuffled_stat = stat_from_rows(unshuffled_rows, train_frac)

    shuffled_r2s: list[float | None] = []
    for i in range(trials):
        shuffled = shuffle_published_at(world.headlines, seed + i)
        rows = synthetic_rows(world, shuffled)
        shuffled_r2s.append(stat_from_rows(rows, train_frac).oos_r2)

    non_null = [v for v in shuffled_r2s if v is not None]
    shuffled_mean = sum(non_null) / len(non_null) if non_null else None

    return SyntheticPowerCheck(
        unshuffled_r2=unshuffled_stat.oos_r2, shuffled_r2s=shuffled_r2s, shuffled_mean_r2=shuffled_mean
    )


# --- CLI -----------------------------------------------------------------


def run(in_path: Path, trials: int, seed: int, train_frac: float) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    ok = True

    print("=== synthetic power check: can this control catch a real signal? ===")
    power = run_synthetic_power_check(
        n_dates=6, headlines_per_date=4, true_slope=20.0, trials=trials, seed=seed, train_frac=train_frac
    )
    print(f"unshuffled synthetic out-of-sample R^2 = {power.unshuffled_r2}")
    print(f"shuffled synthetic out-of-sample R^2, mean over {trials} trials = {power.shuffled_mean_r2}")
    if power.passed:
        print("PASS: the control detects the built-in signal unshuffled, and destroys it when shuffled.")
    else:
        print(
            "FAIL: the control either can't see a real signal unshuffled, or doesn't destroy one when "
            "shuffled - the real-data check below cannot be trusted until this passes."
        )
        ok = False

    print()
    print("=== real-data placebo test: does the real fixture's result stand out from chance? ===")
    headlines = read_csv(in_path)
    rows, _ = build_rows_from_headlines(headlines, live=False)
    real_stat = stat_from_rows(rows, train_frac)
    null_stats = run_real_data_trials(headlines, trials, seed, train_frac)

    contemp_p = placebo_p_value(real_stat.contemporaneous_r, [s.contemporaneous_r for s in null_stats])
    oos_p = placebo_p_value(real_stat.oos_r2, [s.oos_r2 for s in null_stats])

    print(f"real contemporaneous r = {real_stat.contemporaneous_r}  (placebo p-value: {contemp_p})")
    print(f"real out-of-sample R^2 = {real_stat.oos_r2}  (placebo p-value: {oos_p})")

    for name, p in (("contemporaneous r", contemp_p), ("out-of-sample R^2", oos_p)):
        if p is None:
            print(f"{name}: not enough usable rows to judge - neither pass nor fail.")
            continue
        if p < REAL_DATA_P_VALUE_FLOOR:
            print(
                f"FAIL: real {name} ranks in the most extreme {p:.1%} of its own shuffled-timestamp "
                "null distribution - that is not what a leak-free pipeline should produce."
            )
            ok = False
        else:
            print(f"PASS: real {name} is unremarkable against its shuffled-timestamp null distribution.")

    print()
    if ok:
        print("AUDIT PASSED: no evidence the pipeline leaks future information through timestamps.")
    else:
        print("AUDIT FAILED: see above.")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to audit")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="shuffled-timestamp trials to run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base RNG seed (trial i uses seed + i)")
    parser.add_argument(
        "--train-frac",
        type=float,
        default=DEFAULT_TRAIN_FRAC,
        help="train/test split fraction, matching sentiment.regress's own default",
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.trials, args.seed, args.train_frac))


if __name__ == "__main__":
    main()
