"""Day 8 CLI: ml-pipeline-audit - the leakage control NEXT_STEPS.md's "Done
when" section requires before any of this repo's correlation or regression
result can be trusted.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500
    python -m sentiment.audit --live

Day 5 already found a null contemporaneous correlation on the committed
fixture (r=-0.185, 95% CI comfortably containing zero - see README). A null
r alone does not prove the pipeline is leak-free, though: a bug that paired
a headline's sentiment with the wrong trading session's return could just as
easily have produced a null r on this one fixture by coincidence, and a
leaking pipeline that happened to land on a real-looking signal here would
be reported as a finding, not caught as a bug.

This module runs the actual test NEXT_STEPS.md's "Done when" section
describes: randomly reassign headlines' ``published_at`` timestamps among
each other (same headlines, same compound scores, same tickers - only which
trading session Day 4's alignment lands each headline on changes), rebuild
the contemporaneous-return correlation under that broken timing, and repeat
many times. That gives a null distribution of what "r" looks like when a
headline's sentiment is paired with an essentially-random session of its
own ticker instead of the correct one. The real run's r is then compared
against that distribution with a permutation-test p-value, not eyeballed -
if the real r is not an outlier against the shuffled-timing null, there is
no evidence the pipeline is leaking; if it is, the alignment logic (or
something upstream of it) needs to be re-audited before the result is
trusted.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_N_SHUFFLES = 200
DEFAULT_SEED = 0


def shuffle_published_at(headlines: list[Headline], seed: int) -> list[Headline]:
    """Return a new list with the same ``published_at`` timestamps randomly
    reassigned across headlines - title, link, source and scraped_at stay
    with their original headline. This is the control: it destroys the true
    headline -> trading-session mapping while leaving the sentiment-bearing
    text, the ticker each headline names, and the set of timestamps used all
    exactly as they were. If a pipeline still finds a sentiment/return
    relationship after this shuffle, that relationship was never coming from
    the headline's real timing in the first place.
    """
    rng = random.Random(seed)
    shuffled_timestamps = [h.published_at for h in headlines]
    rng.shuffle(shuffled_timestamps)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=new_ts,
            published_raw=h.published_raw,
            scraped_at=h.scraped_at,
        )
        for h, new_ts in zip(headlines, shuffled_timestamps)
    ]


def contemporaneous_r(headlines: list[Headline], live: bool = False) -> float | None:
    """Pearson r between VADER compound and contemporaneous return over
    whatever headlines resolve to a priced ticker, or ``None`` if fewer than
    2 do (not enough to correlate)."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def permutation_p_value(real_r: float, shuffled_rs: list[float]) -> float:
    """Two-sided permutation-test p-value: how often a shuffled-timing run's
    |r| is at least as extreme as the real run's. Add-one smoothing (the
    standard finite-permutation correction) keeps this from reporting an
    impossible p=0.0 just because none of the sampled shuffles happened to
    match or exceed it."""
    if not shuffled_rs:
        raise ValueError("need at least one shuffled run to compare against")
    at_least_as_extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
    return (at_least_as_extreme + 1) / (len(shuffled_rs) + 1)


def run_audit(
    headlines: list[Headline], n_shuffles: int, seed: int, live: bool = False
) -> tuple[float | None, list[float], int]:
    """Return (real_r, shuffled_rs, dropped) - ``dropped`` counts shuffled
    runs that resolved to fewer than 2 priced headlines (the shuffle moved a
    headline onto a session_date with no bar for its ticker) and so
    contributed no r to the null distribution."""
    real_r = contemporaneous_r(headlines, live=live)
    shuffled_rs: list[float] = []
    dropped = 0
    for i in range(n_shuffles):
        shuffled = shuffle_published_at(headlines, seed=seed + i)
        r = contemporaneous_r(shuffled, live=live)
        if r is None:
            dropped += 1
            continue
        shuffled_rs.append(r)
    return real_r, shuffled_rs, dropped


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    real_r, shuffled_rs, dropped = run_audit(headlines, n_shuffles=n_shuffles, seed=seed, live=live)

    if real_r is None:
        print("fewer than 2 headlines resolved to a priced ticker; nothing to audit", file=sys.stderr)
        return 1
    if not shuffled_rs:
        print(
            "every shuffled-timing run resolved to fewer than 2 priced headlines; "
            "cannot build a null distribution",
            file=sys.stderr,
        )
        return 1

    mean_abs_shuffled = sum(abs(r) for r in shuffled_rs) / len(shuffled_rs)
    p = permutation_p_value(real_r, shuffled_rs)

    print(f"real contemporaneous r={real_r:+.3f} (true headline timing, n={len(headlines)} headlines read)")
    print(
        f"shuffled-timing null: {len(shuffled_rs)} usable run(s) of {n_shuffles} attempted "
        f"({dropped} dropped for <2 priced rows after shuffling)"
    )
    print(f"  mean |shuffled r|={mean_abs_shuffled:.3f}  range=[{min(shuffled_rs):+.3f}, {max(shuffled_rs):+.3f}]")
    print(f"permutation p-value (two-sided, real r vs shuffled-timing null): {p:.4f}")
    if p < 0.05:
        print(
            "WARNING: real r is an outlier against the shuffled-timing null - "
            "re-audit the alignment/pairing logic before trusting this result"
        )
    else:
        print("real r is consistent with the shuffled-timing null: no evidence this pipeline is leaking")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help="number of shuffled-timing control runs to build the null distribution from",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base RNG seed for the shuffles")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
