"""Day 8 CLI: the shuffled-timestamp leakage control (``ml-pipeline-audit``).

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 1000 --seed 1
    python -m sentiment.audit --live

NEXT_STEPS.md's "Done when" section names this as the test that decides
whether the whole repo is finished: shuffle headline timestamps and the
signal must disappear. If a shuffled-timestamp control still predicts
returns, the pipeline is leaking future price information through some path
other than the leak-free construction Day 4-6 built, and every earlier
day's result is an artifact, not a finding.

This runs two checks, not one, because they answer different questions:

1. **Leakage check (real data).** ``published_at`` is permuted across the
   real 50-headline fixture (same multiset of timestamps, reassigned to
   different headlines - so each headline keeps its own VADER compound
   score, which depends only on title text, but is attributed to a
   different trading session) and the *actual* ``sentiment.correlate``
   pipeline - ``align_headline``, ticker resolution, price lookup, Pearson
   r - is re-run from scratch on the shuffled input, many times. The check
   is on the *shuffled* r's directly, not on how they compare to the real
   one: per NEXT_STEPS.md, "if a shuffled-timestamp control still predicts
   returns, the pipeline is leaking" - so this asserts the shuffled r's
   stay small (near zero), regardless of what the real r happened to be.
   Comparing the real r to the shuffled distribution instead (a standard
   permutation-test p-value) would miss the most dangerous bug: a pipeline
   that ignores ``published_at`` entirely would make every shuffle a no-op,
   so the real r would trivially sit inside its own unchanged "null" and
   look fine. Testing the shuffled r's magnitude directly catches that.

2. **Power check (synthetic signal).** Day 5-7 already found the real
   compound/return correlation is statistically indistinguishable from zero
   (contemporaneous r=-0.185, 95% CI comfortably containing zero - see
   README Findings). Running check 1 on real data alone would therefore
   "pass" whether or not the shuffle actually does anything: there is
   nothing there for it to destroy. Before trusting check 1, this builds a
   synthetic set of (ticker, session_date, return) rows from the real,
   committed price fixtures with a sentiment score deliberately engineered
   to be a strong, known function of that same row's own return - a real
   signal, by construction - and confirms the *same* shuffle mechanic
   collapses it. If it didn't, check 1's "no signal after shuffling" would
   be meaningless noise, not evidence of a leak-free pipeline.

See README Limitations for what this audit does not establish.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import sentiment.prices as prices
from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_with_ci

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0
# How strongly the synthetic power check ties compound to return before
# clipping to VADER's [-1, 1] compound range. Real daily open-to-close
# returns in these fixtures run a few percent, so a gain of 10 pushes most
# of them well off the clip boundary while still leaving some clipped -
# real-looking, not a toy r=1.000.
SYNTHETIC_SIGNAL_GAIN = 10.0
# Thresholds the CLI's pass/fail verdict is judged against. Not statistical
# dogma - picked to be strict enough that a genuinely leak-free pipeline and
# a genuinely working shuffle pass comfortably, and loose enough that this
# audit isn't itself flaky from one run to the next.
LEAK_MAX_MEAN_ABS_R = 0.25
POWER_MIN_R_BEFORE = 0.5
POWER_MAX_MEAN_ABS_R_AFTER = 0.25


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list with ``published_at`` permuted across ``headlines``.

    Same multiset of timestamps as the input - nothing about *when*
    headlines exist changes in aggregate - but each headline's own title
    (and therefore its VADER compound score, and its ticker resolution,
    both of which depend only on title text) is now attributed to a
    different point in time, and therefore a different trading session and
    a different return. That is exactly what "shuffle the headline
    timestamps" means operationally.
    """
    times = [h.published_at for h in headlines]
    rng.shuffle(times)
    return [replace(h, published_at=t) for h, t in zip(headlines, times)]


def correlation_r(headlines: list[Headline], live: bool = False) -> float | None:
    """Pearson r between VADER compound and contemporaneous return over
    whatever headlines resolve to a ticker with price data, or ``None`` if
    fewer than 2 do (too few to correlate)."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None
    compounds = [r["compound"] for r in rows]
    contemporaneous = [r["contemporaneous_return"] for r in rows]
    return pearson_with_ci(compounds, contemporaneous).r


@dataclass(frozen=True)
class LeakageCheckResult:
    real_r: float | None
    n_resolved: int
    shuffled_rs: list[float]

    @property
    def mean_abs_shuffled_r(self) -> float | None:
        if not self.shuffled_rs:
            return None
        return sum(abs(r) for r in self.shuffled_rs) / len(self.shuffled_rs)

    @property
    def passed(self) -> bool:
        """True if the shuffled-timestamp control shows no meaningful
        predictive power - the signal "disappeared" per NEXT_STEPS.md,
        whatever the real (unshuffled) r happened to be. ``None`` (no
        shuffled r's resolved at all) is reported, not silently passed."""
        mean_after = self.mean_abs_shuffled_r
        return mean_after is not None and mean_after <= LEAK_MAX_MEAN_ABS_R


def leakage_check(
    headlines: list[Headline], n_shuffles: int, seed: int, live: bool = False
) -> LeakageCheckResult:
    rows, _ = build_rows_from_headlines(headlines, live=live)
    n_resolved = len(rows)
    real_r = correlation_r(headlines, live=live)

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        r = correlation_r(shuffled, live=live)
        if r is not None:
            shuffled_rs.append(r)

    return LeakageCheckResult(real_r=real_r, n_resolved=n_resolved, shuffled_rs=shuffled_rs)


@dataclass(frozen=True)
class SyntheticRow:
    ticker: str
    session_date: str
    compound: float
    contemporaneous_return: float


def build_synthetic_rows(signal_gain: float = SYNTHETIC_SIGNAL_GAIN) -> list[SyntheticRow]:
    """Synthetic (ticker, session_date, compound, return) rows with compound
    a known, strong linear function of that row's own return - signal by
    construction - built from the real price fixtures already committed
    under ``fixtures/prices/`` (so the returns themselves are genuine daily
    moves, not invented numbers; only the sentiment/return pairing is
    synthetic). Exists only to prove the shuffle mechanic below has the
    power to detect a real signal when one exists; see module docstring.
    """
    tickers = sorted(p.stem for p in prices.FIXTURE_DIR.glob("*.json"))
    rows: list[SyntheticRow] = []
    for ticker in tickers:
        for bar in prices.load_bars(ticker):
            compound = max(-1.0, min(1.0, signal_gain * bar.session_return))
            rows.append(
                SyntheticRow(
                    ticker=ticker,
                    session_date=bar.date.isoformat(),
                    compound=compound,
                    contemporaneous_return=bar.session_return,
                )
            )
    return rows


def synthetic_r(rows: list[SyntheticRow]) -> float | None:
    if len(rows) < 2:
        return None
    return pearson_with_ci(
        [r.compound for r in rows], [r.contemporaneous_return for r in rows]
    ).r


def shuffle_synthetic_pairing(rows: list[SyntheticRow], rng: random.Random) -> list[SyntheticRow]:
    """The row-level analogue of ``shuffle_timestamps``: reassign which
    (ticker, session_date, return) tuple each compound score is attributed
    to, breaking the sentiment/return correspondence while leaving both
    marginal distributions - the set of compound scores, the set of returns
    - exactly as they were."""
    pairings = [(r.ticker, r.session_date, r.contemporaneous_return) for r in rows]
    rng.shuffle(pairings)
    return [
        SyntheticRow(ticker=ticker, session_date=session_date, compound=row.compound, contemporaneous_return=ret)
        for row, (ticker, session_date, ret) in zip(rows, pairings)
    ]


@dataclass(frozen=True)
class PowerCheckResult:
    r_before: float | None
    n_rows: int
    shuffled_rs: list[float]

    @property
    def mean_abs_r_after(self) -> float | None:
        if not self.shuffled_rs:
            return None
        return sum(abs(r) for r in self.shuffled_rs) / len(self.shuffled_rs)

    @property
    def passed(self) -> bool:
        """True if the undisturbed synthetic signal is clearly detectable
        and shuffling clearly collapses it - proof the leakage check's
        shuffle mechanic has teeth, not just a test that always "passes"
        because there was never anything to find."""
        mean_after = self.mean_abs_r_after
        return (
            self.r_before is not None
            and mean_after is not None
            and abs(self.r_before) >= POWER_MIN_R_BEFORE
            and mean_after <= POWER_MAX_MEAN_ABS_R_AFTER
        )


def power_check(n_shuffles: int, seed: int, signal_gain: float = SYNTHETIC_SIGNAL_GAIN) -> PowerCheckResult:
    rows = build_synthetic_rows(signal_gain=signal_gain)
    r_before = synthetic_r(rows)

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_synthetic_pairing(rows, rng)
        r = synthetic_r(shuffled)
        if r is not None:
            shuffled_rs.append(r)

    return PowerCheckResult(r_before=r_before, n_rows=len(rows), shuffled_rs=shuffled_rs)


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)

    power = power_check(n_shuffles, seed)
    print(f"power check (synthetic signal, n={power.n_rows} rows, gain={SYNTHETIC_SIGNAL_GAIN}):")
    if power.r_before is None:
        print("  not enough synthetic rows to correlate - no price fixtures found")
    else:
        mean_after = power.mean_abs_r_after
        print(f"  r before shuffle = {power.r_before:+.3f}")
        print(
            f"  mean |r| after {len(power.shuffled_rs)} shuffles = "
            f"{mean_after:+.3f}" if mean_after is not None else "  no shuffled rows resolved"
        )
        verdict = "PASS" if power.passed else "FAIL"
        print(
            f"  [{verdict}] signal detectable before shuffle (|r|>={POWER_MIN_R_BEFORE}) "
            f"and collapses after (mean|r|<={POWER_MAX_MEAN_ABS_R_AFTER})"
        )

    leakage = leakage_check(headlines, n_shuffles, seed, live=live)
    print(f"\nleakage check (real data, {leakage.n_resolved} headlines resolved to a ticker):")
    if leakage.real_r is None:
        print("  not enough resolved headlines to correlate - cannot run this check")
    else:
        print(f"  real r (unshuffled) = {leakage.real_r:+.3f}  (context only - see README Findings)")
        mean_after = leakage.mean_abs_shuffled_r
        if mean_after is not None:
            print(f"  mean |r| across {len(leakage.shuffled_rs)} timestamp shuffles = {mean_after:+.3f}")
            verdict = "PASS" if leakage.passed else "FAIL"
            print(
                f"  [{verdict}] shuffled-timestamp control shows no predictive power "
                f"(mean|r|<={LEAK_MAX_MEAN_ABS_R} required)"
            )
        else:
            print("  no shuffled r's resolved - cannot run this check")

    overall_pass = power.passed and leakage.passed
    print(f"\noverall: {'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        print(
            "see README 'Why this might be spurious' and Limitations for what a FAIL here would "
            "and would not mean.",
            file=sys.stderr,
        )
    return 0 if overall_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of random timestamp shuffles to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible report")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
