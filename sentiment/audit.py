"""Day 8 CLI: shuffle-timestamp leakage control, run in CI not once by hand.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 1

NEXT_STEPS.md's "done when" bar for this repo: shuffle every headline's
``published_at`` and the sentiment/return signal must disappear. A
shuffled-timestamp control that still predicts returns means the pipeline is
leaking - something downstream of alignment is using information the
headline could not honestly have had yet.

This runs Day 5's ``build_rows_from_headlines`` pipeline once on the real,
published timestamps, then ``--n-shuffles`` times on copies of the same
headlines with ``published_at`` permuted across them. Title - and therefore
company, ticker, and VADER ``compound`` - stays put; only *when* each
headline is said to have been published changes. Permuting timestamps
changes each headline's ``session_date``, which changes which bar its
ticker's contemporaneous/lagged return is pulled from - that is the one
place a look-ahead bug could hide, so it is the one thing this control
disturbs.

The real statistic is compared against the shuffled null distribution with a
permutation p-value: the fraction of shuffles whose |r| is at least as large
as the real |r|. A high p-value here is necessary but not sufficient
evidence of no leakage on this fixture - see the caveat ``run`` prints and
the README's Day 8 Findings for why.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` permuted across
    them. Every timestamp that existed still exists exactly once; only which
    headline it is attached to changes - title, company, ticker, and
    ``compound`` are untouched, so any difference in the resulting signal
    traces to timestamp-dependent alignment and nothing else."""
    shuffled_timestamps = [h.published_at for h in headlines]
    rng.shuffle(shuffled_timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_timestamps)]


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float | None
    real_n: int
    shuffled_rs: list[float]
    skipped_shuffles: int
    p_value: float | None


def run_shuffle_audit(headlines: list[Headline], live: bool, n_shuffles: int, seed: int) -> ShuffleAuditResult:
    """Compute the real contemporaneous r, then the same statistic across
    ``n_shuffles`` timestamp permutations, keyed by a single ``seed`` so the
    CLI and its tests see the same shuffles every run."""
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real_r: float | None = None
    if len(real_rows) >= 2:
        real_r = pearson_r(
            [r["compound"] for r in real_rows], [r["contemporaneous_return"] for r in real_rows]
        )

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    skipped = 0
    for _ in range(n_shuffles):
        rows, _ = build_rows_from_headlines(shuffle_timestamps(headlines, rng), live=live)
        if len(rows) < 2:
            # a shuffled session_date can land on a day a ticker's fixture
            # has no bar for, so a shuffle can resolve fewer rows than the
            # real run - skip it rather than fabricating a correlation from
            # too few points.
            skipped += 1
            continue
        shuffled_rs.append(
            pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])
        )

    p_value = None
    if real_r is not None and shuffled_rs:
        p_value = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r)) / len(shuffled_rs)

    return ShuffleAuditResult(
        real_r=real_r,
        real_n=len(real_rows),
        shuffled_rs=shuffled_rs,
        skipped_shuffles=skipped,
        p_value=p_value,
    )


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = run_shuffle_audit(headlines, live, n_shuffles, seed)

    if result.real_r is None:
        print("fewer than 2 resolved headlines on the real timestamps; nothing to audit", file=sys.stderr)
        return 1
    if not result.shuffled_rs:
        print("every shuffle left fewer than 2 resolved rows; cannot build a null distribution", file=sys.stderr)
        return 1

    mean_shuffled = statistics.mean(result.shuffled_rs)
    stdev_shuffled = statistics.pstdev(result.shuffled_rs) if len(result.shuffled_rs) > 1 else 0.0

    print(f"real (unshuffled) contemporaneous r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled null ({len(result.shuffled_rs)} of {n_shuffles} shuffles usable, "
        f"{result.skipped_shuffles} skipped for <2 resolved rows): "
        f"mean r={mean_shuffled:+.3f}  sd={stdev_shuffled:.3f}"
    )
    print(f"permutation p-value: {result.p_value:.3f}  (fraction of shuffles with |r| >= real |r|)")

    if result.p_value >= 0.05:
        print(
            "real |r| sits inside the shuffled null - shuffling timestamps did not need to do "
            "anything, because the real statistic shows no signal to begin with. This passes the "
            "letter of the leakage test but not its purpose: the test is only powered to catch "
            "leakage when there is a real correlation to destroy, and Day 5/6 already found there "
            "is not one on this fixture. See README Day 8 Findings and tests/test_audit.py's "
            "synthetic case for evidence the control has power when a correlation does exist."
        )
    else:
        print(
            "real |r| sits in the tail of the shuffled null - the correlation depends on the real "
            "timestamp alignment surviving. That is what a genuine (non-leaking) result should look "
            "like, but it is also what a leak would look like - this control cannot tell the two "
            "apart on its own. Inspect the alignment logic before trusting either reading."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp permutations to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic report")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
