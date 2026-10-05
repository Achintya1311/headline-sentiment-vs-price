"""Day 8 CLI: the leakage/audit pass NEXT_STEPS.md's "Done when" names as
the test that decides whether this repo's results can be trusted.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 200
    python -m sentiment.audit --live

"Shuffle the headline timestamps and the signal must disappear. If a
shuffled-timestamp control still predicts returns, the pipeline is leaking
and the result is an artifact."

A single shuffle permutes ``published_at`` among the same set of headlines -
same titles, same compound scores, same scraped content, but each headline
now carries a timestamp that belonged to a different one. That severs the
link between what a headline said and the trading session
``sentiment.market_hours.align_headline`` assigns it to, without touching
anything else the pipeline reads (the ticker a headline resolves to comes
from its title, not its timestamp, so shuffling never changes *which*
company a headline is tested against - only *which session's* return it is
paired with).

The check compares the real-alignment correlation's magnitude against the
mean magnitude measured across many independent shuffles:

- If shuffling collapses the correlation towards zero, whatever the real
  alignment measured depended on headlines landing on their own, correctly
  computed session - consistent with a leak-free pipeline.
- If shuffling barely moves the correlation, the measured relationship does
  not actually depend on correct timestamp alignment at all, which is
  exactly the artifact this check exists to catch (see ``tests/test_audit.py``
  for a constructed example of both cases).

Read alongside ``tests/test_audit.py``: Day 5 already found no significant
correlation under correct alignment on this repo's own fixture (the 95% CI
contains zero), so there is little real signal here for a shuffle to
visibly destroy - the audit below necessarily passes on this fixture almost
by default. The synthetic positive and negative controls in the test file
plant a timestamp-dependent correlation and a timestamp-blind "leak" on
purpose, to prove this check would actually catch a real leak if this
fixture ever had one. See README Findings and Limitations for why that
distinction matters and is not glossed over.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0

# Below this, the real-alignment correlation itself is too small to call a
# "signal" in the first place - there is nothing meaningful for a shuffle to
# make disappear, and reporting a PASS/FAIL retention ratio against noise
# would be comparing noise to noise.
MIN_REAL_SIGNAL = 0.10

# A shuffle that still carries at least half of the real correlation's
# magnitude has not made the signal "disappear" by any reasonable reading
# of that word.
RETENTION_THRESHOLD = 0.50


@dataclasses.dataclass(frozen=True)
class AuditResult:
    r_real: float
    n_real: int
    r_shuffled: list[float]

    @property
    def mean_abs_shuffled(self) -> float:
        return sum(abs(r) for r in self.r_shuffled) / len(self.r_shuffled)

    @property
    def has_real_signal(self) -> bool:
        return abs(self.r_real) >= MIN_REAL_SIGNAL

    @property
    def retained_fraction(self) -> float | None:
        """Fraction of the real correlation's magnitude that survives
        timestamp shuffling, or ``None`` if there was no real signal to
        begin with (see ``has_real_signal``)."""
        if not self.has_real_signal:
            return None
        return self.mean_abs_shuffled / abs(self.r_real)

    @property
    def leak_suspected(self) -> bool:
        """True if timestamp shuffling did not make the real correlation
        disappear - the failure mode NEXT_STEPS.md's "Done when" names."""
        retained = self.retained_fraction
        return retained is not None and retained >= RETENTION_THRESHOLD


def shuffle_timestamps(headlines: list[Headline], seed: int) -> list[Headline]:
    """Return headlines with ``published_at`` permuted across the list -
    same titles, same scraped content, but each headline now carries a
    timestamp that belonged to a different one. A permutation (not
    independent re-randomisation) keeps the exact same multiset of
    timestamps in play, so any difference in the resulting correlation
    comes from which headline got which timestamp, not from a different
    spread of timestamps altogether."""
    rng = random.Random(seed)
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [dataclasses.replace(h, published_at=ts) for h, ts in zip(headlines, timestamps)]


def run_audit(
    headlines: list[Headline],
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> AuditResult:
    """Run the real pipeline once, then ``n_shuffles`` timestamp-shuffled
    copies of it, and report how much of the real correlation's magnitude
    survives. Raises ``ValueError`` if the real pass, or every shuffle,
    leaves too few resolved rows to correlate (n < 2) - a pipeline
    precondition failure, not an audit result."""
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(real_rows) < 2:
        raise ValueError(f"not enough resolved headlines to correlate (n={len(real_rows)})")
    r_real = pearson_r(
        [r["compound"] for r in real_rows],
        [r["contemporaneous_return"] for r in real_rows],
    )

    seed_rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffle_seed = seed_rng.randrange(2**31)
        shuffled_headlines = shuffle_timestamps(headlines, shuffle_seed)
        rows, _ = build_rows_from_headlines(shuffled_headlines, live=live)
        if len(rows) < 2:
            continue
        shuffled_rs.append(
            pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])
        )

    if not shuffled_rs:
        raise ValueError("no timestamp shuffle produced enough resolved rows to correlate")

    return AuditResult(r_real=r_real, n_real=len(real_rows), r_shuffled=shuffled_rs)


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_audit(headlines, n_shuffles=n_shuffles, seed=seed, live=live)
    except ValueError as exc:
        print(f"audit could not run: {exc}", file=sys.stderr)
        return 1

    print(f"real alignment:      r={result.r_real:+.3f}  n={result.n_real}")
    print(
        f"shuffled timestamps: mean|r|={result.mean_abs_shuffled:.3f} over {len(result.r_shuffled)} shuffles"
    )

    if not result.has_real_signal:
        print(
            f"no real-alignment signal to audit: |r|={abs(result.r_real):.3f} is below the "
            f"{MIN_REAL_SIGNAL:.2f} floor this check treats as a signal in the first place - "
            "there is nothing here for a shuffle to meaningfully destroy."
        )
        return 0

    retained = result.retained_fraction
    print(f"retained after shuffling: {retained:.1%} of the real correlation's magnitude")
    if result.leak_suspected:
        print(
            "FAIL: shuffling timestamps did not make the signal disappear - "
            "that is the leak NEXT_STEPS.md warns about. Do not trust the real-alignment result."
        )
        return 1
    print("PASS: shuffling timestamps collapses the real correlation, as a leak-free pipeline should.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp permutations to test against"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for the sequence of per-shuffle seeds")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
