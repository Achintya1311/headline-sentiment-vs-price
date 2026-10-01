"""Day 8 CLI: the ml-pipeline-audit pass - shuffle headline timestamps and
confirm Day 5's contemporaneous sentiment/return correlation disappears.

    python -m sentiment.audit
    python -m sentiment.audit --n-permutations 500 --seed 0
    python -m sentiment.audit --live

Done when (NEXT_STEPS.md / README "Correctness gate"): shuffle the headline
timestamps and the signal must disappear. If a shuffled-timestamp control
still predicts returns, the pipeline is leaking and the result is an
artifact. This module is what turns that into a test that runs in CI
(``tests/test_audit.py``), not something eyeballed once by hand.

What "shuffle" means here, concretely: ``published_at`` is permuted across
the *same* set of headlines - each headline keeps its own title, company and
VADER score, but is re-assigned another headline's publish timestamp. The
full pipeline (``sentiment.market_hours.align_headline`` -> ``session_date``
-> ``sentiment.prices.bar_on``) is then re-run from scratch on the shuffled
timestamps, exactly like ``sentiment.correlate.build_rows`` does for real
ones. A leak hiding inside alignment or the bar lookup - not just inside the
final (compound, return) pairs - has to survive this, because the whole
chain re-runs on every permutation, not a paper reshuffle of two parallel
lists computed once.

The statistic audited is Day 5's contemporaneous Pearson r (the headline
number ``sentiment.correlate`` reports). A permutation null distribution is
built by recomputing r many times under reshuffled timestamps; the
two-sided p-value is the fraction of those null r's at least as extreme as
the real, unshuffled |r|.

On its own, "the real result isn't extreme" is a weak pass on a fixture that
is already null (see README Day 5/6 Findings) - a pipeline that silently
always returned r=0 would pass it too. ``tests/test_audit.py`` also runs
this same machinery against a synthetic fixture with a manufactured,
genuinely timestamp-dependent relationship and checks the opposite: that
case must *not* pass, which is what proves the test can actually catch a
real (or leaked) effect rather than rubber-stamping whatever it's given.
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

DEFAULT_N_PERMUTATIONS = 500
DEFAULT_SEED = 0

# Safety cap on how many shuffles to attempt before giving up on reaching
# n_permutations valid ones - guards against an unlucky fixture where almost
# every reshuffled timestamp lands on a session with no committed bar.
MAX_ATTEMPTS_FACTOR = 20


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Reassign ``published_at`` across ``headlines``, keeping everything else
    (title, company, the sentiment it will score to) exactly as scraped.

    ``published_raw`` is rewritten to match the reassigned timestamp - the
    original raw feed string no longer describes a real publish time once
    the timestamp has moved to a different headline, and keeping the stale
    string around would be misleading, not faithful.
    """
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [replace(h, published_at=ts, published_raw=ts.isoformat()) for h, ts in zip(headlines, timestamps)]


def contemporaneous_r(headlines: list[Headline], live: bool = False) -> tuple[float | None, int]:
    """Re-run the full build-rows pipeline on ``headlines`` and return the
    Pearson r of VADER compound vs contemporaneous return, and n. ``r`` is
    ``None`` when fewer than 2 rows resolve - too little to correlate, which
    is a different thing from a defined r of 0."""
    rows, _ = build_rows_from_headlines(headlines, live=live, warn=False)
    if len(rows) < 2:
        return None, len(rows)
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, returns), len(rows)


@dataclass(frozen=True)
class PermutationResult:
    real_r: float
    real_n: int
    null_rs: list[float]
    p_value: float

    @property
    def n_permutations(self) -> int:
        return len(self.null_rs)


def permutation_test(
    headlines: list[Headline],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> PermutationResult:
    """Build a null distribution of the contemporaneous r by reshuffling
    headline timestamps ``n_permutations`` times and recomputing r from
    scratch on each shuffle, then compare the real, unshuffled r against it.

    A reshuffled timestamp can land a headline's session on a trading day
    its ticker has no committed bar for, leaving fewer than 2 resolved rows
    for that shuffle - those draws carry no information either way and are
    skipped rather than counted as r=0, which would bias the null toward
    "no effect" for the wrong reason.
    """
    real_r, real_n = contemporaneous_r(headlines, live=live)
    if real_r is None:
        raise ValueError(f"only {real_n} resolved headline(s) with a price - not enough to correlate")

    rng = random.Random(seed)
    null_rs: list[float] = []
    attempts = 0
    max_attempts = max(n_permutations * MAX_ATTEMPTS_FACTOR, 1)
    while len(null_rs) < n_permutations and attempts < max_attempts:
        attempts += 1
        shuffled = shuffle_timestamps(headlines, rng)
        r, _ = contemporaneous_r(shuffled, live=live)
        if r is None:
            continue
        null_rs.append(r)

    if not null_rs:
        raise ValueError(
            f"no shuffled-timestamp permutation produced 2+ resolvable rows in {attempts} attempts - "
            "cannot build a null distribution"
        )

    extreme = sum(1 for r in null_rs if abs(r) >= abs(real_r))
    p_value = extreme / len(null_rs)

    return PermutationResult(real_r=real_r, real_n=real_n, null_rs=null_rs, p_value=p_value)


def run(in_path: Path, n_permutations: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = permutation_test(headlines, n_permutations=n_permutations, seed=seed, live=live)
    except ValueError as exc:
        print(f"cannot audit: {exc}", file=sys.stderr)
        return 1

    mean_abs_null = sum(abs(r) for r in result.null_rs) / result.n_permutations
    print(f"real contemporaneous r={result.real_r:+.3f} (n={result.real_n})")
    print(
        f"shuffled-timestamp null: {result.n_permutations} permutations "
        f"({attempts_note(n_permutations, result.n_permutations)}), mean |r|={mean_abs_null:.3f}"
    )
    print(f"two-sided p-value (fraction of |null r| >= |real r|): {result.p_value:.3f}")
    if result.p_value < 0.05:
        print(
            "FAILS the Day 8 gate: the real result is more extreme than 95% of its own "
            "shuffled-timestamp null distribution. That means either a genuine "
            "timestamp-dependent relationship or a leak - this test cannot tell those apart "
            "on its own, only that the correlation does not disappear under shuffling. "
            "Do not trust the correlation until this is explained."
        )
    else:
        print(
            "PASSES the Day 8 gate: the real result is unremarkable against its own "
            "shuffled-timestamp null distribution - consistent with Day 5/6's finding of no "
            "detectable sentiment signal on this fixture, not with a leak."
        )
    return 0


def attempts_note(requested: int, achieved: int) -> str:
    if achieved == requested:
        return "all requested"
    return f"{achieved} of {requested} requested - the rest left too few resolvable rows to count"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help="number of shuffled-timestamp permutations to build the null distribution from",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution"
    )
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_permutations, args.seed, args.live))


if __name__ == "__main__":
    main()
