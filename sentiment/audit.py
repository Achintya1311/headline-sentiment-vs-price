"""Day 8 CLI: the leakage/audit pass - shuffle headline timestamps and
confirm whatever correlation ``sentiment.correlate`` finds does not survive.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 1000 --seed 1
    python -m sentiment.audit --live

Day 5/6 already found a null result by eye (contemporaneous r=-0.185 with a
95% CI that comfortably contains zero). This module asks a different,
narrower question than "is there a signal": it is a leakage control on the
*pipeline*, not a significance test of sentiment itself. A headline's
``compound`` score and resolved ``ticker`` depend only on its title, never
its timestamp; its ``session_date`` (and therefore its contemporaneous
return) depends only on its timestamp via Day 4's ``align_headline``. So
randomly permuting *which headline got which publish timestamp*, holding
every title fixed, severs exactly the one link the correlation could
honestly exploit - the real timing of the news relative to the market.

If the real (unshuffled) correlation sits comfortably inside the
distribution of correlations produced by that random re-pairing, the result
is consistent with "no detectable timing-dependent signal", matching Day
5/6's own finding. If the real correlation is a clear outlier against that
null distribution, *that* would be the sign to distrust: either a genuine
signal (extraordinary, given n=23 and a saturated predictor - see the
README's "why this might be spurious" section) or, more likely given this
pipeline's history of off-by-one timestamp bugs elsewhere in the portfolio,
a leak in the alignment logic that an eyeball check of one run would not
catch.

The permutation is over the *pool of timestamps that were actually
published* - it reassigns them, never invents new ones - so a shuffle can
land a headline on a different real publish moment, but never onto a
timestamp that never happened.
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
MIN_ROWS_FOR_CORRELATION = 2


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    null_rs: list[float]
    n_shuffles_requested: int
    n_shuffles_skipped: int
    p_value: float

    @property
    def null_mean_abs_r(self) -> float:
        return sum(abs(r) for r in self.null_rs) / len(self.null_rs)

    @property
    def leaks(self) -> bool:
        """``True`` if the real correlation is a statistical outlier against
        the shuffled-timestamp null distribution (p < 0.05) - the signature
        the README's "Done when" section calls a failed leakage test: a
        shuffled-timestamp control that still predicts returns as well as,
        or better than, the real pairing."""
        return self.p_value < 0.05


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list of the same headlines with ``published_at`` values
    randomly permuted across them. Every timestamp in the returned list was
    a real publish time of *some* headline in the input - this reassigns the
    pool, it never fabricates a new one. Titles, companies and scores travel
    with their original ``Headline`` object and are untouched."""
    shuffled_ts = [h.published_at for h in headlines]
    rng.shuffle(shuffled_ts)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_ts)]


def _contemporaneous_r(rows: list[dict]) -> float | None:
    if len(rows) < MIN_ROWS_FOR_CORRELATION:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def run_leakage_audit(
    in_path: Path,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> AuditResult:
    headlines = read_csv(in_path)

    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real_r = _contemporaneous_r(real_rows)
    if real_r is None:
        raise ValueError(
            f"only {len(real_rows)} resolved headline(s) with price data - not enough to correlate"
        )

    rng = random.Random(seed)
    null_rs: list[float] = []
    skipped = 0
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=live, quiet=True)
        r = _contemporaneous_r(rows)
        if r is None:
            skipped += 1
            continue
        null_rs.append(r)

    if not null_rs:
        raise ValueError("every shuffle produced too few resolved rows to correlate - cannot build a null distribution")

    at_least_as_extreme = sum(1 for r in null_rs if abs(r) >= abs(real_r))
    p_value = at_least_as_extreme / len(null_rs)

    return AuditResult(
        real_r=real_r,
        real_n=len(real_rows),
        null_rs=null_rs,
        n_shuffles_requested=n_shuffles,
        n_shuffles_skipped=skipped,
        p_value=p_value,
    )


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    try:
        result = run_leakage_audit(in_path, n_shuffles=n_shuffles, seed=seed, live=live)
    except ValueError as exc:
        print(f"cannot run leakage audit: {exc}", file=sys.stderr)
        return 1

    print(f"real contemporaneous correlation: r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled-timestamp null distribution: {len(result.null_rs)} usable shuffle(s) "
        f"(requested {result.n_shuffles_requested}, {result.n_shuffles_skipped} skipped - too few rows resolved)"
    )
    print(f"null mean |r|={result.null_mean_abs_r:.3f}")
    print(
        f"p-value (fraction of shuffles with |r| >= real |r|): {result.p_value:.3f}"
    )
    if result.leaks:
        print(
            "FAIL: the real correlation is a statistical outlier against the shuffled-timestamp "
            "control (p < 0.05) - a random re-pairing of headlines to sessions should not predict "
            "returns this well. Investigate the alignment pipeline for a leak before trusting this result."
        )
    else:
        print(
            "PASS: the real correlation is not distinguishable from a random timestamp pairing "
            "(p >= 0.05) - consistent with Day 5/6's own null finding, not evidence against a "
            "leak in general, only against this specific one."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help="number of random timestamp permutations to build the null distribution from",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible CI run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
