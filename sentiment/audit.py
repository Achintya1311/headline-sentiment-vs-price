"""Day 8 CLI: the shuffled-timestamp leakage audit - the correctness gate
NEXT_STEPS.md's "Done when" names: shuffle every headline's ``published_at``
among the others and confirm whatever correlation Day 5 reports disappears.

    python -m sentiment.audit
    python -m sentiment.audit --n-perm 1000 --seed 7

For each of ``--n-perm`` reshuffles, every headline keeps its own title (and
therefore the ticker ``sentiment.tickers.resolve`` maps it to) but is given a
randomly reassigned ``published_at`` drawn from the same multiset of
timestamps the real headlines actually carry. Day 4's alignment and Day 5's
``build_rows``/Pearson-r machinery then rerun, completely unmodified, against
that scrambled input. This produces an empirical null distribution for the
contemporaneous r a pipeline with no real timestamp dependence would show by
chance; the real run's r is compared against it with a two-sided permutation
p-value.

A small p-value means the real r is unusually extreme next to its own
shuffled-timestamp control - the correlation collapses when the true
timing is destroyed, consistent with it depending on genuinely correct
alignment rather than on a leak. A p-value that is *not* small means
shuffling did not meaningfully weaken the correlation, which has two very
different possible explanations this run cannot tell apart on its own:
either something downstream of the timestamp is leaking information the
shuffled timestamp could not have carried, or there was too little real
correlation here in the first place for the control to have anything to
destroy. See the README's Day 8 Findings and Limitations for which one this
fixture is in, and ``tests/test_audit.py`` for both failure modes
reproduced directly against a synthetic fixture built to exercise each one.
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

DEFAULT_N_PERM = 500
DEFAULT_SEED = 0
MIN_ROWS_FOR_R = 2


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return headlines carrying the same multiset of ``published_at`` values,
    randomly re-paired to a (possibly different) headline. Everything else -
    title, source, link - stays exactly as scraped, so only the alignment
    step sees different input; which ticker a headline resolves to is
    untouched."""
    shuffled_ts = [h.published_at for h in headlines]
    rng.shuffle(shuffled_ts)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_ts)]


def contemporaneous_r(rows: list[dict]) -> float | None:
    """Pearson r between ``compound`` and ``contemporaneous_return`` across
    ``rows``, or ``None`` if there are too few rows to correlate."""
    if len(rows) < MIN_ROWS_FOR_R:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    n_rows: int
    null_rs: list[float]
    p_value: float  # two-sided: fraction of shuffled |r| >= |real r|

    @property
    def null_mean(self) -> float:
        return statistics.fmean(self.null_rs)

    @property
    def null_stdev(self) -> float:
        return statistics.pstdev(self.null_rs) if len(self.null_rs) > 1 else 0.0


def run_audit(headlines: list[Headline], n_perm: int, seed: int, live: bool = False) -> AuditResult:
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real_r = contemporaneous_r(real_rows)
    if real_r is None:
        raise ValueError(f"only {len(real_rows)} resolved headline(s) - not enough for a correlation")

    rng = random.Random(seed)
    null_rs: list[float] = []
    for _ in range(n_perm):
        shuffled = shuffle_timestamps(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=live, verbose=False)
        r = contemporaneous_r(rows)
        if r is not None:
            null_rs.append(r)

    if not null_rs:
        raise ValueError("every shuffle produced too few rows for a correlation - cannot build a null distribution")

    as_extreme = sum(1 for r in null_rs if abs(r) >= abs(real_r))
    p_value = as_extreme / len(null_rs)

    return AuditResult(real_r=real_r, n_rows=len(real_rows), null_rs=null_rs, p_value=p_value)


def run(in_path: Path, n_perm: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    if not headlines:
        print(f"{in_path} has no headlines to audit", file=sys.stderr)
        return 1

    try:
        result = run_audit(headlines, n_perm=n_perm, seed=seed, live=live)
    except ValueError as exc:
        print(f"cannot run the audit: {exc}", file=sys.stderr)
        return 1

    print(f"real contemporaneous r={result.real_r:+.3f} (n={result.n_rows})")
    print(
        f"{len(result.null_rs)} shuffled-timestamp re-runs: null r mean={result.null_mean:+.3f} "
        f"stdev={result.null_stdev:.3f}"
    )
    print(f"two-sided permutation p-value (shuffled |r| >= real |r|): {result.p_value:.3f}")

    if result.p_value < 0.05:
        print(
            "the correlation collapses under shuffled timestamps - consistent with it depending "
            "on genuinely correct alignment, not a pipeline leak."
        )
    else:
        print(
            "shuffling headline timestamps did not meaningfully weaken the correlation. This run "
            "cannot tell whether that is because something downstream of the timestamp is leaking "
            "information, or because there was too little real correlation here for the control to "
            "have anything to destroy - see the README's Day 8 Findings and Limitations."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to audit")
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM, help="number of timestamp reshuffles to run")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution"
    )
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_perm, args.seed, args.live))


if __name__ == "__main__":
    main()
