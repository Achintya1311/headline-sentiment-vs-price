"""Day 8 CLI: pipeline leakage audit - shuffle headline timestamps and
confirm any sentiment/return relationship disappears.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500
    python -m sentiment.audit --seed 1

This is NEXT_STEPS.md's "Done when" gate, made executable: shuffle the
headline timestamps and the signal must disappear. If a shuffled-timestamp
run still reproduces the real run's correlation, something other than
correct time alignment is driving it - the pipeline is leaking, and every
earlier day's finding built on it would be unproven, not demonstrated.

What gets shuffled: each headline keeps its own title (and therefore its
own ticker - ``sentiment.tickers.resolve`` reads headline text, not time)
but is re-timestamped with another headline's real ``published_at``. Day 4's
alignment then assigns it a different trading session for the *same*
ticker, so the shuffle re-pairs each headline's sentiment score with a
different day's return without touching ticker resolution, scoring, or the
price fixtures - the only thing that changes between the real run and a
shuffled run is which session's return a headline's compound is tested
against. That isolates exactly what the alignment module is supposed to
get right.

Repeating the shuffle many times (``--n-shuffles``) builds a null
distribution for the contemporaneous Pearson r a random time-pairing would
produce; the real run's r is compared against it as a permutation test.

Why this audit is honest but weak on the committed fixture: Day 5 already
found the real contemporaneous r is small and not significant on its own
(r=-0.185, 95% CI comfortably containing zero - see README). "The signal
disappears under shuffling" is nearly a tautology when there is barely a
signal to begin with; a pipeline that produced noise for every input,
including a genuinely correctly-aligned one, would pass this gate for
free. ``tests/test_audit.py`` carries the part that actually tests
something: a synthetic fixture with a strong, deliberately planted
relationship between compound and contemporaneous return, checked both
before shuffling (confirms this audit can detect a real signal) and after
(confirms shuffling destroys a real signal, not just a weak one). Without
both halves, this CLI's clean result on the real fixture would not be
evidence of anything.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0
SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float
    real_n: int
    shuffled_rs: list[float]

    @property
    def n_shuffles(self) -> int:
        return len(self.shuffled_rs)

    @property
    def shuffled_mean(self) -> float:
        return sum(self.shuffled_rs) / len(self.shuffled_rs)

    @property
    def p_value(self) -> float:
        """Two-sided permutation p-value: the fraction of timestamp-shuffled
        runs whose |r| is at least as extreme as the real run's. Low means
        the real correlation would be unlikely to arise from a random
        time-pairing alone - i.e. correct alignment is doing real work."""
        extreme = sum(1 for r in self.shuffled_rs if abs(r) >= abs(self.real_r))
        return extreme / len(self.shuffled_rs)


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Permute ``published_at`` across ``headlines``. Each returned headline
    keeps its own title/source/link - and so its own ticker - but is
    re-timestamped with another headline's real publish time, picked by a
    random permutation of the whole list."""
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, timestamps)]


def contemporaneous_r(rows: list[dict]) -> float | None:
    """Pearson r between compound and contemporaneous_return, or None if
    there are fewer than 2 rows to correlate (degrades the same way
    ``sentiment.correlate.run`` does, rather than raising)."""
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def run_shuffle_audit(
    headlines: list[Headline],
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    live: bool = False,
    seed: int = DEFAULT_SEED,
) -> ShuffleAuditResult:
    """Compute the real contemporaneous r, then rebuild it ``n_shuffles``
    times with timestamps permuted, re-running the real alignment/price
    pipeline each time. Trials that resolve to fewer than 2 priced rows
    (a shuffled session date can fall outside a ticker's fixture window)
    are skipped rather than padded with a fabricated value."""
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real_r = contemporaneous_r(real_rows)
    if real_r is None:
        raise ValueError(f"not enough resolved headlines to audit (n={len(real_rows)})")

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled_headlines = shuffle_timestamps(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled_headlines, live=live)
        r = contemporaneous_r(rows)
        if r is not None:
            shuffled_rs.append(r)

    if not shuffled_rs:
        raise ValueError("every shuffled trial produced fewer than 2 priced rows; cannot build a null distribution")

    return ShuffleAuditResult(real_r=real_r, real_n=len(real_rows), shuffled_rs=shuffled_rs)


def run(in_path: Path, n_shuffles: int, live: bool, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_shuffle_audit(headlines, n_shuffles=n_shuffles, live=live, seed=seed)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"real contemporaneous r={result.real_r:+.3f}  (n={result.real_n})")
    print(
        f"{result.n_shuffles} timestamp-shuffled run(s): mean r={result.shuffled_mean:+.3f}  "
        f"min={min(result.shuffled_rs):+.3f}  max={max(result.shuffled_rs):+.3f}"
    )
    print(f"permutation p-value (two-sided, |shuffled r| >= |real r|): {result.p_value:.3f}")

    if result.p_value < SIGNIFICANCE_LEVEL and abs(result.real_r) > abs(result.shuffled_mean):
        print(
            "real r sits outside the shuffled-timestamp null distribution: consistent with a "
            "genuine timing-dependent relationship that correct alignment is required to see."
        )
    elif abs(result.real_r) > 0.3 and result.p_value >= SIGNIFICANCE_LEVEL:
        print(
            "WARNING: the real run shows a sizeable correlation, but timestamp-shuffled runs "
            "reproduce it about as often as chance would - this is the leak pattern NEXT_STEPS.md "
            "warns about. Do not trust this result until the leak is found."
        )
    else:
        print(
            "real r is unremarkable next to random timestamp shuffles: consistent with Day 5's "
            "null finding on this fixture, not evidence either way about whether correct "
            "alignment matters - there is no real signal here for a leak to fake. See "
            "README Limitations for why this audit has little power on the committed fixture, "
            "and tests/test_audit.py for a synthetic case where a real signal is planted and "
            "this same check is shown to catch it."
        )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp-shuffle trials to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible shuffle sequence")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.live, args.seed))


if __name__ == "__main__":
    main()
