"""Day 8 CLI: ml-pipeline-audit - shuffle headline timestamps and check
whether the sentiment/return relationship survives.

    python -m sentiment.audit
    python -m sentiment.audit --n-perm 2000 --seed 1
    python -m sentiment.audit --live

This is the test the README's "Correctness gate" and NEXT_STEPS.md's "Done
when" both commit to: shuffle headline *timestamps* only (title, source,
link, and therefore sentiment score stay attached to their original row)
across the same fixture, rerun Day 4's alignment and Day 5's
contemporaneous/lagged pairing on each shuffle, and compare the real
Pearson r against the resulting null distribution. A genuine, non-leaking
result should look unremarkable next to that null distribution: the news
content didn't move, only which trading session it got credited to, so if
the real pairing isn't exploiting a timestamp-alignment bug, randomising
the pairing shouldn't matter much either. A real r that stands far outside
the null distribution is the signature of exactly that kind of leak.

See README Limitations for why this fixture's own real data cannot show
the "a real signal gets destroyed" half of that story: Day 5/6 already
found r indistinguishable from zero before any shuffling - there's no
signal here for a leak to have inflated. tests/test_audit.py carries a
synthetic case with a real, planted sentiment/return relationship to prove
the mechanism actually detects one when it exists, rather than this CLI's
null result on real data being the only case it was ever run against.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "audit_permutation.csv"
DEFAULT_N_PERM = 999
DEFAULT_SEED = 0
MIN_ROWS_FOR_R = 2


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Reassign the *set* of ``published_at`` values across ``headlines`` at
    random - the same multiset of real timestamps this fixture actually has,
    just landing on a different headline each time. Title, source, link and
    ``scraped_at`` stay with their original row, so only the
    timestamp-to-content pairing the alignment pipeline depends on is
    broken; the sentiment score (which depends only on the title) is
    untouched."""
    shuffled_timestamps = [h.published_at for h in headlines]
    rng.shuffle(shuffled_timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_timestamps)]


def contemporaneous_r(rows: list[dict]) -> float | None:
    if len(rows) < MIN_ROWS_FOR_R:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def lagged_r(rows: list[dict]) -> float | None:
    usable = [r for r in rows if r["lagged_return"] is not None]
    if len(usable) < MIN_ROWS_FOR_R:
        return None
    return pearson_r([r["compound"] for r in usable], [r["lagged_return"] for r in usable])


def permutation_pvalue(real_r: float | None, null_rs: list[float]) -> float | None:
    """Two-sided permutation p-value: how often a random timestamp shuffle
    produces a correlation at least as extreme as the real one. The +1 in
    numerator and denominator is the standard finite-permutation correction
    (Davison & Hinkley) - it keeps this from ever reporting p=0 just because
    none of a finite number of shuffles happened to tie or beat the real
    value."""
    if real_r is None or not null_rs:
        return None
    extreme = sum(1 for r in null_rs if abs(r) >= abs(real_r))
    return (extreme + 1) / (len(null_rs) + 1)


def percentile_rank(real_r: float | None, null_rs: list[float]) -> float | None:
    """Where the real r falls within the sorted null distribution, 0-100."""
    if real_r is None or not null_rs:
        return None
    below = sum(1 for r in null_rs if r <= real_r)
    return 100 * below / len(null_rs)


@dataclass(frozen=True)
class AuditResult:
    real_contemp_r: float | None
    real_contemp_n: int
    real_lagged_r: float | None
    real_lagged_n: int
    null_contemp_rs: list[float]
    null_lagged_rs: list[float]
    n_perm: int


def run_audit(headlines: list[Headline], n_perm: int, seed: int, live: bool) -> AuditResult:
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real_contemp = contemporaneous_r(real_rows)
    real_lagged = lagged_r(real_rows)
    real_lagged_n = len([r for r in real_rows if r["lagged_return"] is not None])

    rng = random.Random(seed)
    null_contemp: list[float] = []
    null_lagged: list[float] = []
    for _ in range(n_perm):
        shuffled = shuffle_timestamps(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=live, quiet=True)
        c = contemporaneous_r(rows)
        if c is not None:
            null_contemp.append(c)
        lag = lagged_r(rows)
        if lag is not None:
            null_lagged.append(lag)

    return AuditResult(
        real_contemp_r=real_contemp,
        real_contemp_n=len(real_rows),
        real_lagged_r=real_lagged,
        real_lagged_n=real_lagged_n,
        null_contemp_rs=null_contemp,
        null_lagged_rs=null_lagged,
        n_perm=n_perm,
    )


def write_null_distribution_csv(result: AuditResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["permutation", "contemporaneous_r", "lagged_r"])
        max_len = max(len(result.null_contemp_rs), len(result.null_lagged_rs))
        for i in range(max_len):
            c = result.null_contemp_rs[i] if i < len(result.null_contemp_rs) else ""
            lag = result.null_lagged_rs[i] if i < len(result.null_lagged_rs) else ""
            writer.writerow([i, c, lag])


def describe(name: str, real_r: float | None, real_n: int, null_rs: list[float], n_perm: int) -> str:
    if real_r is None:
        return f"{name}: not enough resolved headlines for a real correlation (n={real_n})"
    p = permutation_pvalue(real_r, null_rs)
    pct = percentile_rank(real_r, null_rs)
    lines = [
        f"{name}: real r={real_r:+.3f} (n={real_n})  "
        f"null distribution from {len(null_rs)}/{n_perm} usable shuffles: "
        f"p={p:.3f}  percentile={pct:.0f}"
    ]
    if p is not None and p > 0.05:
        lines.append(
            f"  -> real {name} r is not distinguishable from a random timestamp shuffle (p>0.05): "
            "no signal here for a leak to have inflated."
        )
    else:
        lines.append(
            f"  -> real {name} r sits outside the random-shuffle null distribution (p<=0.05): "
            "investigate before trusting this number - a genuine, non-leaking result should not "
            "depend this much on which exact session a headline's timestamp happens to land it on."
        )
    return "\n".join(lines)


def run(in_path: Path, out_path: Path, live: bool, n_perm: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    if not headlines:
        print(f"{in_path} has no headlines to audit", file=sys.stderr)
        return 1

    result = run_audit(headlines, n_perm=n_perm, seed=seed, live=live)
    write_null_distribution_csv(result, out_path)

    print(f"shuffled headline timestamps {n_perm} time(s) (seed={seed}), rerunning Day 4/5's pipeline each time")
    print(f"null distribution written to {out_path}")
    print(describe("contemporaneous", result.real_contemp_r, result.real_contemp_n, result.null_contemp_rs, n_perm))
    print(describe("lagged", result.real_lagged_r, result.real_lagged_n, result.null_lagged_rs, n_perm))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to audit")
    parser.add_argument(
        "--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="null-distribution CSV to write"
    )
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM, help="number of timestamp shuffles to run")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution"
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.live, args.n_perm, args.seed))


if __name__ == "__main__":
    main()
