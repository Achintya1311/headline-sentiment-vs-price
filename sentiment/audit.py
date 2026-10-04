"""Day 8 CLI: the leakage/audit pass NEXT_STEPS.md's "Done when" commits to.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 0
    python -m sentiment.audit --live

The control: take every headline exactly as scraped and scored (same title,
same VADER ``compound``, same resolved ticker - ``sentiment.tickers.resolve``
never looks at a timestamp), but *shuffle which headline got which
``published_at``* before handing the batch to Day 4/5's own alignment and
correlation code unchanged. Shuffling breaks the one thing a timestamp
controls - which trading session a headline's return is measured against -
while leaving every other column exactly as real. If the resulting
contemporaneous correlation is routinely just as strong (or stronger) than
the one computed on real timestamps, the pipeline cannot be trusting the
timestamp for anything load-bearing: either it is leaking (some other path
lets the "right" answer through regardless of alignment) or the correlation
was never using the timestamp's information in the first place.

This is a permutation test, not a single before/after comparison: one shuffle
is as easy to get lucky or unlucky with as the n=23 real sample already is
(see README Day 5 Findings), so ``--n-shuffles`` independent reshuffles build
a null distribution and the real result is read against it, not against one
draw. Everything is seeded, so a run is reproducible and fit for CI.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import build_rows
from sentiment.headline import Headline, read_csv, write_csv
from sentiment.stats import PearsonResult, pearson_with_ci

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0

# Nominal false-positive rate of a 95% CI "excludes zero" call. A shuffled
# control that trips this far more often than chance would is the red flag
# this module exists to catch - see the module docstring and README Day 8.
SIGNIFICANCE_RATE_ALARM = 3.0  # shuffled significant-rate > 3x the nominal 5% trips the audit


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    real_significant: bool  # real 95% CI excludes zero
    null_r: list[float]
    n_shuffles_used: int
    n_shuffles_requested: int
    p_value: float  # fraction of shuffles with |r| >= |real_r|
    null_mean: float
    null_std: float
    significant_rate: float  # fraction of shuffles whose own 95% CI excludes zero

    @property
    def leak_suspected(self) -> bool:
        """See README Day 8 Findings for why this is the one check this
        audit can honestly make on a sample this small and this saturated:
        it cannot prove "no leakage" in general, only that shuffling the one
        thing (timestamp -> session alignment) that could leak does not
        manufacture significance far more often than a correctly-calibrated
        95% CI would by chance alone."""
        return self.significant_rate > SIGNIFICANCE_RATE_ALARM * 0.05


def shuffle_timestamps(headlines: list[Headline], seed: int) -> list[Headline]:
    """Return a new list with the same headlines (same title, same
    ``scraped_at``) but ``published_at`` values permuted across the batch -
    a bijection of real timestamps onto different headlines, not fresh
    synthetic ones, so the marginal distribution of publish times is
    unchanged and only the *pairing* is randomised."""
    import random

    rng = random.Random(seed)
    times = [h.published_at for h in headlines]
    shuffled_times = times[:]
    rng.shuffle(shuffled_times)
    return [replace(h, published_at=t) for h, t in zip(headlines, shuffled_times)]


def correlation_for(headlines: list[Headline], live: bool = False) -> PearsonResult | None:
    """Run the real Day 5 pipeline (``sentiment.correlate.build_rows``,
    untouched) against an in-memory batch of headlines by round-tripping
    through a temp CSV - the same file-shaped input the CLI reads, so this
    audit exercises the exact code path a human running the pipeline would,
    not a reimplementation of it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shuffled.csv"
        write_csv(headlines, path)
        rows, _ = build_rows(path, live=live)
    if len(rows) < 2:
        return None
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_with_ci(compounds, returns)


def run_audit(
    in_path: Path,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> AuditResult:
    headlines = read_csv(in_path)
    real_stat = correlation_for(headlines, live=live)
    if real_stat is None:
        raise ValueError("not enough resolved headlines in the real data to audit")

    null_r: list[float] = []
    significant_count = 0
    for i in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, seed=seed + i)
        stat = correlation_for(shuffled, live=live)
        if stat is None:
            continue
        null_r.append(stat.r)
        if stat.ci_low > 0 or stat.ci_high < 0:
            significant_count += 1

    if not null_r:
        raise ValueError("no shuffle produced a resolvable correlation - cannot audit")

    p_value = sum(1 for r in null_r if abs(r) >= abs(real_stat.r)) / len(null_r)

    return AuditResult(
        real_r=real_stat.r,
        real_n=real_stat.n,
        real_significant=real_stat.ci_low > 0 or real_stat.ci_high < 0,
        null_r=null_r,
        n_shuffles_used=len(null_r),
        n_shuffles_requested=n_shuffles,
        p_value=p_value,
        null_mean=statistics.fmean(null_r),
        null_std=statistics.pstdev(null_r),
        significant_rate=significant_count / len(null_r),
    )


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    try:
        result = run_audit(in_path, n_shuffles=n_shuffles, seed=seed, live=live)
    except ValueError as exc:
        print(f"audit could not run: {exc}", file=sys.stderr)
        return 1

    print(f"real (true timestamps):      r={result.real_r:+.3f}  n={result.real_n}  "
          f"significant={result.real_significant}")
    print(f"shuffled-timestamp control:   {result.n_shuffles_used}/{result.n_shuffles_requested} usable shuffles, "
          f"mean r={result.null_mean:+.3f}  std={result.null_std:.3f}")
    print(f"p(|shuffled r| >= |real r|) = {result.p_value:.3f}")
    print(f"shuffled 95%-CI 'significant' rate = {result.significant_rate:.1%} "
          f"(nominal false-positive rate under a correctly calibrated CI is ~5%)")

    if result.leak_suspected:
        print(
            "FAIL: shuffled-timestamp controls look 'significant' far more often than chance - "
            "the pipeline may not actually depend on correct timestamp alignment. See README Day 8.",
            file=sys.stderr,
        )
        return 1

    print("PASS: shuffling timestamps does not manufacture spurious significance beyond chance rate.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of random timestamp permutations")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base RNG seed (shuffle i uses seed+i)")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
