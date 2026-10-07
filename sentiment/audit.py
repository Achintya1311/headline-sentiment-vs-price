"""Day 8 CLI: ml-pipeline-audit leakage control.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 0
    python -m sentiment.audit --live

NEXT_STEPS.md's "done when" bar for this project: shuffle every headline's
``published_at`` timestamp across the corpus, keep everything else (title,
source, the VADER score it implies) fixed, and recompute Day 5's
contemporaneous correlation. Shuffling destroys the one thing Day 4's
alignment pipeline is supposed to get right - *which* trading session a
headline's return should be paired against - while leaving sentiment
content and price data untouched. If a shuffled-timestamp run still
"predicts" returns as well as the real timestamps do, something downstream
of alignment is leaking (a ticker-level price regularity standing in for
the headline's own timing, say) and Day 5/6/7's numbers should not be
trusted.

Caveat read against this repo's own Day 5/6 Findings, not glossed over: the
real-timestamp contemporaneous correlation is already indistinguishable
from zero (r=-0.185, 95% CI comfortably containing 0, n=23). A test framed
as "does the signal disappear under shuffling" has nothing to make
disappear here - there is no signal in the real data for a leak to be
hiding inside. The honest version of this test is a permutation p-value:
where does the real statistic fall inside the distribution of statistics
the same pipeline produces under shuffled timestamps? See the module
docstring in ``tests/test_audit.py`` for a synthetic positive control that
proves this machinery actually has the power to catch a real leak when one
exists - run only against a fabricated signal, never against this repo's
real fixture, precisely because the real fixture has no signal to inject
a leak into in the first place.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_with_ci

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0
MIN_ROWS = 4  # pearson_with_ci's Fisher-z CI needs n >= 4; see sentiment/stats.py


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float
    real_n: int
    n_shuffles: int
    n_usable_shuffles: int
    shuffled_rs: list[float]
    p_value: float  # two-sided: fraction of shuffled |r| >= real |r|


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Reassign ``published_at`` across ``headlines`` at random - same
    multiset of timestamps, different headline each one lands on - leaving
    every other field (title, source, link, scraped_at) untouched. This
    breaks exactly the link Day 4's alignment depends on without touching
    sentiment content or price data, which is what makes it a control for
    leakage in the alignment step specifically rather than a generic
    shuffle of the whole dataset.
    """
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, timestamps)]


def _contemporaneous_r(headlines: list[Headline], live: bool) -> float | None:
    rows, _ = rows_from_headlines(headlines, live=live, quiet=True)
    if len(rows) < MIN_ROWS:
        return None
    return pearson_with_ci(
        [r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows]
    ).r


def run_shuffle_audit(
    headlines: list[Headline],
    live: bool = False,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
) -> ShuffleAuditResult:
    real_rows, _ = rows_from_headlines(headlines, live=live, quiet=True)
    if len(real_rows) < MIN_ROWS:
        raise ValueError(
            f"only {len(real_rows)} resolved headline(s) - need >= {MIN_ROWS} for a correlation"
        )
    real_r = pearson_with_ci(
        [r["compound"] for r in real_rows], [r["contemporaneous_return"] for r in real_rows]
    ).r

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        r = _contemporaneous_r(shuffled, live)
        if r is not None:
            shuffled_rs.append(r)

    if not shuffled_rs:
        raise ValueError("no shuffle produced enough resolved rows to correlate")

    extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
    p_value = extreme / len(shuffled_rs)

    return ShuffleAuditResult(
        real_r=real_r,
        real_n=len(real_rows),
        n_shuffles=n_shuffles,
        n_usable_shuffles=len(shuffled_rs),
        shuffled_rs=shuffled_rs,
        p_value=p_value,
    )


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_shuffle_audit(headlines, live=live, n_shuffles=n_shuffles, seed=seed)
    except ValueError as exc:
        print(f"cannot run leakage audit: {exc}", file=sys.stderr)
        return 1

    mean_abs_shuffled = sum(abs(r) for r in result.shuffled_rs) / len(result.shuffled_rs)
    print(f"real timestamps:     r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled timestamps: mean |r|={mean_abs_shuffled:.3f} over "
        f"{result.n_usable_shuffles}/{result.n_shuffles} usable shuffles"
    )
    print(
        f"two-sided permutation p-value: {result.p_value:.3f} "
        "(fraction of shuffles with |r| >= real |r|)"
    )
    if result.p_value < 0.05:
        print(
            "real |r| sits outside 95% of the shuffled-null distribution - the real timestamp "
            "pairing finds more structure than a random one. Worth a second look before trusting "
            "it, not a pass: see README Findings for what could produce this without a real "
            "sentiment signal."
        )
    else:
        print(
            "real |r| is unremarkable against the shuffled-null distribution - consistent with "
            "Day 5/6's finding that there is no real correlation here to leak in the first "
            "place. See README Limitations: on this fixture, this test cannot tell "
            "'leakage-free' apart from 'signal-free' - both look identical from here."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp permutations to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
