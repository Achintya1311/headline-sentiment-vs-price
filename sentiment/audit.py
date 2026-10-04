"""Day 8 CLI: the ml-pipeline-audit leakage control.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 2000 --seed 1

This repo's own "done when" rule (NEXT_STEPS.md, README): shuffle the
headline timestamps and the signal must disappear. If a shuffled-timestamp
control still predicts returns as well as the real timestamps do, the
pipeline is leaking something a timestamp is not supposed to carry, and
whatever correlation Day 5-7 reported is an artifact, not a finding.

This reruns the *real* pipeline for every shuffle - Day 4's ``align_headline``
and Day 5's price pairing (``sentiment.correlate.rows_from_headlines``) - not
a generic permutation test bolted onto the final numbers. Only
``published_at`` is permuted across headlines; title, source, link and
hence each headline's resolved company/ticker are untouched, so a shuffle
changes *which trading session* a headline is scored against without
changing what it says or which company it is about. That is the specific
thing Day 4 exists to get right, so it is the specific thing this audit
has to be able to break.

What this test does **not** catch: a confound tied to company identity
rather than timing (e.g. a stock that is simply trending the same direction
on both candidate sessions) survives a timestamp shuffle unchanged, because
the ticker a headline pairs with never moves. See the README's Day 8
Limitations for why that matters and what would catch it instead.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "audit_shuffle_null.csv"
DEFAULT_N_SHUFFLES = 1000
DEFAULT_SEED = 0
MIN_ROWS_FOR_STAT = 2


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` permuted across
    the list - the same multiset of real publish times, randomly reassigned
    to different headlines. Title, source, link and ``scraped_at`` are left
    alone, so ticker resolution is unaffected; only which trading session
    each headline's alignment lands on can change.
    """
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, timestamps)]


def contemporaneous_r(headlines: list[Headline], live: bool = False) -> tuple[float | None, int]:
    """Pearson r between VADER compound and contemporaneous return, rebuilt
    from scratch against ``headlines`` (real or shuffled). ``None`` if fewer
    than ``MIN_ROWS_FOR_STAT`` headlines resolve to a priced ticker."""
    rows, _ = rows_from_headlines(headlines, live=live)
    if len(rows) < MIN_ROWS_FOR_STAT:
        return None, len(rows)
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, returns), len(rows)


def shuffle_null_distribution(
    headlines: list[Headline], n_shuffles: int, seed: int, live: bool = False
) -> list[float]:
    """Rerun the pipeline ``n_shuffles`` times against independently shuffled
    timestamps, returning the resulting Pearson r for every shuffle that had
    enough resolved rows to compute one. Deterministic for a fixed seed."""
    rng = random.Random(seed)
    null_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        r, _ = contemporaneous_r(shuffled, live=live)
        if r is not None:
            null_rs.append(r)
    return null_rs


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list, ``pct`` in [0, 1]."""
    if not sorted_values:
        raise ValueError("need at least one value")
    idx = min(int(pct * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[idx]


def empirical_two_sided_p(observed: float, null_rs: list[float]) -> float:
    """Fraction of the null distribution at least as extreme (in absolute
    value) as ``observed`` - an empirical two-sided permutation p-value."""
    if not null_rs:
        raise ValueError("empty null distribution")
    extreme = sum(1 for r in null_rs if abs(r) >= abs(observed))
    return extreme / len(null_rs)


def write_null_csv(null_rs: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shuffle_index", "r"])
        for i, r in enumerate(null_rs):
            writer.writerow([i, r])


def run(in_path: Path, out_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    observed_r, n = contemporaneous_r(headlines, live=live)
    if observed_r is None:
        print(f"not enough resolved headlines for a correlation (n={n}); nothing to audit", file=sys.stderr)
        return 1

    null_rs = shuffle_null_distribution(headlines, n_shuffles, seed, live=live)
    if not null_rs:
        print(
            "no shuffle produced enough resolved rows for a correlation; cannot build a null distribution",
            file=sys.stderr,
        )
        return 1
    write_null_csv(null_rs, out_path)

    sorted_null = sorted(null_rs)
    lo = percentile(sorted_null, 0.025)
    hi = percentile(sorted_null, 0.975)
    mean_null = sum(null_rs) / len(null_rs)
    p = empirical_two_sided_p(observed_r, null_rs)

    print(f"observed contemporaneous r = {observed_r:+.3f} (n={n}, real timestamps)")
    print(
        f"shuffled-timestamp null (n_shuffles={len(null_rs)}, seed={seed}): "
        f"mean={mean_null:+.3f}  95% band [{lo:+.3f}, {hi:+.3f}]  -> {out_path}"
    )
    print(f"two-sided empirical p = {p:.3f} (fraction of shuffles at least as extreme as observed)")

    if lo <= observed_r <= hi:
        print(
            "PASS: the observed r sits inside the shuffled-timestamp null band - indistinguishable "
            "from a pipeline with no timestamp-dependent signal to lose. Consistent with Day 5-7's own "
            "null finding on this fixture; it is not independent proof the pipeline can never leak (see "
            "README Limitations for what this specific control does not catch)."
        )
    else:
        print(
            "FAIL: the observed r sits outside the shuffled-timestamp null band - real timestamps predict "
            "returns better than permuted ones. Something is leaking look-ahead through the alignment or "
            "pairing step; do not trust Day 5-7's correlation numbers until this is root-caused."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="per-shuffle r values CSV to write"
    )
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp shuffles to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
