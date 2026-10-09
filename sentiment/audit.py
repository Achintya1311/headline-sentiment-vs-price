"""Day 8 CLI: the shuffled-timestamp correctness gate.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 1
    python -m sentiment.audit --live

The repo's own "Correctness gate" (see README): randomise which headline
each real ``published_at`` timestamp belongs to, rerun Day 4's alignment and
Day 5's contemporaneous correlation on that shuffled pairing, and compare
the real pipeline's |r| against the distribution |r| takes under many
independent shuffles. Shuffling a headline's timestamp changes which
trading session (and so which day's return) it is paired with, without
touching its title, ticker, or VADER score - exactly the headline-content
vs. market-reaction link Day 4's leak-free alignment exists to get right.
If that link is real, destroying it should make the correlation collapse;
if the real correlation already looks like a typical draw from the shuffled
distribution, there was nothing in the pipeline to destroy in the first
place.

This is a permutation test, not a pass/fail switch: it reports a two-sided
p-value (the fraction of shuffles whose |r| is at least as extreme as the
real pipeline's), because "no evidence of signal" and "evidence of no
signal" are different claims - see README Limitations. The real fixture is
expected to score high (not significant): Day 5/6 already found this
sample's correlation statistically indistinguishable from zero before this
audit existed. ``tests/test_audit.py`` carries a synthetic self-test that
injects a real, deliberate dependency between score and return and confirms
this same machinery flags *that* as significant - proof the gate has teeth,
not just a rubber stamp on an already-null result.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from random import Random

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "audit_shuffle.csv"
DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0


def shuffle_timestamps(headlines: list[Headline], rng: Random) -> list[Headline]:
    """Return a new list of the same headlines with the *set* of
    ``published_at`` values randomly reassigned across them.

    Every timestamp that comes out of this was a real timestamp of some
    headline going in - only which headline's title/ticker/score it is now
    attached to has changed. That is what "shuffle headline timestamps"
    means here: break the pairing between what a headline says and when the
    market could have reacted to it, while leaving everything else (which
    company it names, what it says, how VADER scores it) untouched.
    """
    shuffled_at = [h.published_at for h in headlines]
    rng.shuffle(shuffled_at)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_at)]


def _contemporaneous_r(headlines: list[Headline], live: bool) -> tuple[float, int]:
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return 0.0, len(rows)
    xs = [r["compound"] for r in rows]
    ys = [r["contemporaneous_return"] for r in rows]
    return pearson_r(xs, ys), len(rows)


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float
    real_n: int
    shuffled_rs: list[float]
    p_value: float


def run_audit(headlines: list[Headline], live: bool, n_shuffles: int, seed: int) -> ShuffleAuditResult:
    """Run the real pipeline once, then ``n_shuffles`` independent timestamp
    shuffles, and return the real contemporaneous r alongside the null
    distribution of r the shuffles produce.

    A shuffle that happens to leave too few resolved/priced headlines to
    correlate (fewer than 2) contributes no r to the null distribution
    rather than a fabricated 0.0 - padding the null with zeros would bias
    the permutation p-value toward "significant" by making the null look
    tighter around zero than it really is.
    """
    real_r, real_n = _contemporaneous_r(headlines, live)

    rng = Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        r, n = _contemporaneous_r(shuffled, live)
        if n >= 2:
            shuffled_rs.append(r)

    as_extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
    p_value = (as_extreme + 1) / (len(shuffled_rs) + 1) if shuffled_rs else 1.0

    return ShuffleAuditResult(real_r=real_r, real_n=real_n, shuffled_rs=shuffled_rs, p_value=p_value)


def write_shuffled_csv(result: ShuffleAuditResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shuffle_index", "r"])
        for i, r in enumerate(result.shuffled_rs):
            writer.writerow([i, r])


def run(in_path: Path, out_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    if len(headlines) < 2:
        print(f"{in_path} has too few headlines to audit", file=sys.stderr)
        return 1

    result = run_audit(headlines, live, n_shuffles, seed)
    write_shuffled_csv(result, out_path)

    if result.shuffled_rs:
        mean_shuffled = sum(result.shuffled_rs) / len(result.shuffled_rs)
        var_shuffled = sum((r - mean_shuffled) ** 2 for r in result.shuffled_rs) / len(result.shuffled_rs)
        sd_shuffled = var_shuffled**0.5
    else:
        mean_shuffled = 0.0
        sd_shuffled = 0.0

    print(f"real contemporaneous r={result.real_r:+.3f} (n={result.real_n})")
    print(
        f"shuffled-timestamp null: {len(result.shuffled_rs)}/{n_shuffles} usable shuffle(s), "
        f"mean r={mean_shuffled:+.3f} sd={sd_shuffled:.3f} -> {out_path}"
    )
    print(f"two-sided permutation p-value: {result.p_value:.3f}")
    if result.p_value < 0.05:
        print(
            "p < 0.05: the real pairing's |r| is more extreme than at least 95% of "
            "shuffled-timestamp pairings - this fixture's correlation does not look like noise."
        )
    else:
        print(
            "p >= 0.05: the real pairing's |r| is indistinguishable from a random "
            "headline-to-session pairing on this fixture - consistent with Day 5/6's own "
            "finding of no detectable correlation here. Not proof that no relationship "
            "exists, only that this audit found nothing to tell apart from noise."
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to audit")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="shuffled-r CSV to write")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of independent timestamp shuffles"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible CLI run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
