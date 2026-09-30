"""Day 8 CLI: the correctness gate - shuffle headline timestamps and check
whether any sentiment/return correlation depends on them at all.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500
    python -m sentiment.audit --in path/to.csv

NEXT_STEPS.md's "Done when" section names this test directly: "shuffle the
headline timestamps and the signal must disappear. If a shuffled-timestamp
control still predicts returns, the pipeline is leaking and the result is
an artifact."

What "shuffle" means here: keep every headline's title/company/ticker fixed
and randomly reassign *which headline got which publish time* - a
permutation of the timestamp column, not fresh random ones. A permutation
keeps the exact real distribution of timestamps (the same trading-day /
pre-open / intraday / post-close mix Day 4 found), so a shuffled run's
alignment difficulty is not artificially easier or harder than the real
run's; only the correspondence between a headline's content and its own
true publish time is destroyed.

Rerunning ``sentiment.correlate``'s own pipeline (``build_rows_from_
headlines``) against many such shuffles gives a null distribution for "what
does this correlation look like when the headline-to-session pairing is
random, but everything else about the data is real." Two different things
can go wrong, and this audit checks for the specific one the leakage test
names:

- If the pipeline is honest, only the *true* publish time can correctly
  place a headline in the session it actually predates. Shuffle that away
  and the contemporaneous-return pairing becomes effectively random per
  ticker, so the shuffled correlations should scatter around zero - their
  own mean's 95% CI should contain zero. This is the leak check this module
  runs: if that CI does *not* contain zero, shuffling timestamps did not
  wash out the correlation, which means something other than the headline's
  true publish time is driving it - the definition of leakage.
- Separately, this also reports where the real (unshuffled) correlation
  sits against the shuffled null (a permutation-test p-value), for context.
  A real result that is a clear outlier against an otherwise zero-centred
  null is evidence of a real, timing-dependent relationship - interesting,
  but not itself evidence of leakage, and not what this gate exists to
  catch. See the module docstring's limitations note and the README's "why
  this might be spurious" section for what this audit does *not* rule out.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import Z_95, pearson_r

DEFAULT_N_SHUFFLES = 200


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list of the same headlines with ``published_at`` (and
    its accompanying ``published_raw`` text) permuted across them - every
    headline keeps its own title/company/ticker, but is paired with a
    *different* headline's publish time. A permutation, not fresh random
    timestamps, so the real mix of trading-day/pre-open/intraday/post-close
    cases is preserved exactly; only which headline each timestamp belongs
    to changes.
    """
    times = [h.published_at for h in headlines]
    raw = [h.published_raw for h in headlines]
    order = list(range(len(headlines)))
    rng.shuffle(order)
    return [replace(h, published_at=times[i], published_raw=raw[i]) for h, i in zip(headlines, order)]


def contemporaneous_r(rows: list[dict]) -> float | None:
    """Pearson r between VADER compound and contemporaneous return, or
    ``None`` if fewer than 2 rows resolved (not enough to correlate)."""
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    n_real_rows: int
    shuffled_rs: list[float]
    shuffled_mean: float
    shuffled_ci_low: float
    shuffled_ci_high: float
    permutation_p: float

    @property
    def leak_suspected(self) -> bool:
        """The whole point of the gate: if shuffling timestamps did not pull
        the correlation back toward zero - the shuffled mean's own 95% CI
        excludes zero - the correlation does not depend on headlines being
        paired with their true publish time, which is what "leaking" means
        here."""
        return not (self.shuffled_ci_low <= 0.0 <= self.shuffled_ci_high)


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Mean and a 95% CI for that mean (normal approximation over the
    sample of shuffle runs itself, not the Fisher z-transform Day 5's
    ``pearson_with_ci`` uses for a single r - here the "sample" is many
    independent shuffled correlations, not many headlines)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean, mean
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(variance / n)
    return mean, mean - Z_95 * se, mean + Z_95 * se


def permutation_p_value(real_r: float, shuffled_rs: list[float]) -> float:
    """Two-sided permutation p-value: the fraction of shuffle runs at least
    as extreme as the real result, +1/+1 smoothed so it is never exactly
    zero (a real result more extreme than every single shuffle does not
    mean the true p-value is 0, only that it is smaller than 1/(n+1))."""
    if not shuffled_rs:
        raise ValueError("no shuffled runs produced a correlation to compare against")
    extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
    return (extreme + 1) / (len(shuffled_rs) + 1)


def run_audit(headlines: list[Headline], n_shuffles: int, seed: int, live: bool = False) -> AuditResult:
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real_r = contemporaneous_r(real_rows)
    if real_r is None:
        raise ValueError(f"fewer than 2 headlines resolved to a ticker (n={len(real_rows)}); nothing to audit")

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled_rows, _ = build_rows_from_headlines(shuffle_timestamps(headlines, rng), live=live)
        r = contemporaneous_r(shuffled_rows)
        if r is not None:
            shuffled_rs.append(r)

    if not shuffled_rs:
        raise ValueError("no shuffled run produced enough resolved headlines to correlate")

    mean, lo, hi = _mean_ci(shuffled_rs)
    p = permutation_p_value(real_r, shuffled_rs)
    return AuditResult(
        real_r=real_r,
        n_real_rows=len(real_rows),
        shuffled_rs=shuffled_rs,
        shuffled_mean=mean,
        shuffled_ci_low=lo,
        shuffled_ci_high=hi,
        permutation_p=p,
    )


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_audit(headlines, n_shuffles, seed, live=live)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"real contemporaneous r = {result.real_r:+.3f}  (n={result.n_real_rows})")
    print(
        f"shuffled-timestamp null: mean r = {result.shuffled_mean:+.3f}  "
        f"95% CI [{result.shuffled_ci_low:+.3f}, {result.shuffled_ci_high:+.3f}]  "
        f"over {len(result.shuffled_rs)} shuffles"
    )
    print(f"permutation p-value (context only, |shuffled r| >= |real r|): {result.permutation_p:.3f}")

    if result.leak_suspected:
        print(
            "FAIL: the shuffled-timestamp null does not straddle zero - correlation survives even "
            "when headlines are paired with the wrong publish time. That means it does not depend "
            "on the true timestamp, which is what leakage looks like here. Do not trust this result.",
            file=sys.stderr,
        )
        return 1

    print("PASS: shuffling timestamps pulls the correlation back to a null centred on zero, as it should.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp permutations to run"
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, for a reproducible null distribution")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
