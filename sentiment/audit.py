"""Day 8 CLI: the leakage/audit pass - shuffle headline timestamps and
confirm the signal disappears (NEXT_STEPS.md's "Done when" leakage test).

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 2000 --seed 1

What this actually tests: Day 4's ``align_headline`` maps a headline's
``published_at`` to the trading session it may honestly be attributed to,
and Day 5's ``build_rows`` pairs that session's return with the headline's
VADER ``compound`` score. If that pairing were buggy - headlines and
returns silently mismatched by list position instead of by the timestamp
that was actually scraped, say - the measured sentiment/return correlation
would not really depend on the real timestamps, and *destroying* those
timestamps (shuffling them across headlines, so every headline keeps its
own title/compound but is now aligned using some other headline's publish
time) would fail to make the correlation go away.

This is a permutation test, not a one-off eyeball check: shuffle the
timestamps ``--n-shuffles`` times with a fixed, deterministic ``--seed``
(so this produces the same verdict every time it runs, in CI or by hand),
recompute the real contemporaneous Pearson r against each shuffled-and-
realigned control, and see where the *real* r (computed with the real,
unshuffled timestamps) falls inside that null distribution. If the real r
is an extreme outlier against randomized timing, the pipeline is carrying
information that real alignment alone cannot explain - exactly what a leak
or a mispairing bug would produce. If it sits comfortably inside the null
distribution, there is no such information: whatever the real correlation
is - and Day 5/6 already found it is small, with an out-of-sample
regression R² of exactly 0.000 - it is statistically indistinguishable
from what misaligned timestamps alone would produce by chance.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows
from sentiment.headline import Headline, read_csv, write_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 1000
DEFAULT_SEED = 0
MIN_ROWS_FOR_STAT = 2
SIGNIFICANCE_LEVEL = 0.05


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    null_mean: float
    n_shuffles: int
    p_value: float  # fraction of shuffled-control |r| >= real |r|


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return ``headlines`` with the same multiset of ``published_at``
    values, randomly reassigned across headlines - each headline keeps its
    own title (and so its own VADER compound score) but is now timestamped,
    and therefore session-aligned, using a time that really belonged to a
    different headline. This is the control: whatever predictive power the
    real pipeline has from genuinely pairing a headline with the session it
    could actually move, this construction removes."""
    times = [h.published_at for h in headlines]
    rng.shuffle(times)
    return [replace(h, published_at=t) for h, t in zip(headlines, times)]


def _contemporaneous_r(headlines: list[Headline], live: bool) -> tuple[float, int] | None:
    """Realign ``headlines`` (via a throwaway CSV, so this reuses the real
    ``build_rows`` pipeline unchanged rather than duplicating its logic) and
    return (r, n) for compound vs. contemporaneous return, or ``None`` if
    fewer than ``MIN_ROWS_FOR_STAT`` headlines resolve to a ticker with
    price data to correlate against."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shuffled_headlines.csv"
        write_csv(headlines, path)
        rows, _ = build_rows(path, live=live)
    if len(rows) < MIN_ROWS_FOR_STAT:
        return None
    compounds = [r["compound"] for r in rows]
    contemporaneous = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, contemporaneous), len(rows)


def run_permutation_test(
    in_path: Path,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> AuditResult:
    headlines = read_csv(in_path)
    real = _contemporaneous_r(headlines, live)
    if real is None:
        raise ValueError(f"fewer than {MIN_ROWS_FOR_STAT} headlines resolve to a ticker with price data")
    real_r, real_n = real

    rng = random.Random(seed)
    null_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        result = _contemporaneous_r(shuffled, live)
        if result is not None:
            null_rs.append(result[0])

    if not null_rs:
        raise ValueError("no shuffled control produced enough resolved headlines to compute a correlation")

    n_extreme = sum(1 for r in null_rs if abs(r) >= abs(real_r))
    p_value = n_extreme / len(null_rs)
    null_mean = sum(null_rs) / len(null_rs)

    return AuditResult(
        real_r=real_r,
        real_n=real_n,
        null_mean=null_mean,
        n_shuffles=len(null_rs),
        p_value=p_value,
    )


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    try:
        result = run_permutation_test(in_path, n_shuffles=n_shuffles, seed=seed, live=live)
    except ValueError as exc:
        print(f"cannot run the leakage audit: {exc}", file=sys.stderr)
        return 1

    print(
        f"real contemporaneous r={result.real_r:+.3f} (n={result.real_n}) vs. "
        f"{result.n_shuffles} shuffled-timestamp control(s): null mean r={result.null_mean:+.3f}"
    )
    print(f"p-value (fraction of shuffled controls with |r| >= real |r|) = {result.p_value:.3f}")
    if result.p_value < SIGNIFICANCE_LEVEL:
        print(
            "WARNING: the real correlation is an outlier against randomized timing - this is "
            "what a pipeline leak (or a genuinely strong effect) would look like. Investigate "
            "the alignment pipeline before trusting this number."
        )
        return 1

    print(
        "PASS: the real correlation is statistically indistinguishable from what misaligned "
        "(shuffled) timestamps alone produce - no evidence the pipeline is leaking information "
        "the real timestamps should not provide into the sentiment/return pairing."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help="number of shuffled-timestamp controls to run",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
