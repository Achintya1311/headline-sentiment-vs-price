"""Day 8 CLI: the shuffle-timestamp leakage audit.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 2000 --seed 0 --event-threshold 0.3

This is the test the README's "Correctness gate" and NEXT_STEPS.md's "Done
when" section both name as the one that decides whether the repo is
finished: shuffle headline timestamps and confirm the signal disappears. If
a shuffled-timestamp control still predicts returns as well as the real,
leak-free alignment does, the pipeline is leaking look-ahead information
somewhere upstream of this check, and Day 5-7's results cannot be trusted.

What "shuffle" means here: permute the fixture's headlines' ``published_at``
values among each other - title, compound score, and ticker resolution stay
with the original headline, only which timestamp it carries changes - then
re-run Day 4's alignment and Day 5's contemporaneous correlation exactly as
``sentiment.correlate`` does. Day 1's scrape landed every headline in this
fixture on one calendar day (Monday 28 Sep 2026), so a shuffled timestamp
can only move a headline between the two sessions that day's time-of-day
split actually produces (``sentiment/market_hours.py``): Monday itself
(pre-open) or Tuesday (intraday/post-close), never to an arbitrary date.

Repeating the shuffle many times and recomputing r each time builds a null
distribution: the range of correlations this fixture's compound scores and
returns could produce from an alignment that carries no causal information
at all (a headline paired with a session it could not possibly have reacted
to, or reacted to, by construction - the shuffle itself is not leak-free).
The real, leak-free r should land as an unremarkable draw from that null
distribution - not a positive outlier - for the pipeline to be cleared of
look-ahead leakage. See the README's Day 8 Findings for what this fixture's
audit actually shows and why a null result here is weaker evidence than it
might look.
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r, permutation_p_value

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "audit_shuffle_null.csv"
DEFAULT_N_SHUFFLES = 1000
DEFAULT_SEED = 0


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` permuted across them.

    Everything else about each headline (source, title, link, scraped_at) is
    left exactly as scraped, so a shuffled run still resolves the same
    tickers and scores the same compound values as the real run - the only
    thing a shuffle can change is which trading session's return a headline
    ends up paired with.
    """
    timestamps = [h.published_at for h in headlines]
    shuffled_timestamps = timestamps[:]
    rng.shuffle(shuffled_timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_timestamps)]


def contemporaneous_r(rows: list[dict]) -> float | None:
    """Pearson r between compound and contemporaneous_return, or None if
    fewer than 2 rows resolved - not enough to correlate."""
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    null_rs: list[float]
    p_value: float

    @property
    def null_mean(self) -> float:
        return statistics.fmean(self.null_rs)

    @property
    def null_stdev(self) -> float:
        return statistics.pstdev(self.null_rs) if len(self.null_rs) > 1 else 0.0


def run_audit(
    headlines: list[Headline],
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> AuditResult:
    """Run the real (leak-free) alignment once, then ``n_shuffles`` randomly
    reassigned ones, and return both r's plus the permutation p-value.

    Price fixtures are read once per ticker and cached (``bars_cache``),
    not once per shuffle - the fixtures are the same for every trial since
    only the headline/session pairing changes, not the price data itself.
    """
    bars_cache: dict = {}
    real_rows, _ = build_rows_from_headlines(headlines, live=live, bars_cache=bars_cache)
    real_r = contemporaneous_r(real_rows)
    if real_r is None:
        raise ValueError(f"only {len(real_rows)} resolved headline(s) - not enough for a correlation")

    rng = random.Random(seed)
    null_rs: list[float] = []
    skipped = 0
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=live, bars_cache=bars_cache)
        r = contemporaneous_r(rows)
        if r is None:
            skipped += 1
            continue
        null_rs.append(r)

    if not null_rs:
        raise ValueError("no shuffle produced a resolvable correlation; cannot build a null distribution")
    if skipped:
        print(
            f"warning: {skipped}/{n_shuffles} shuffle(s) left fewer than 2 resolved headlines "
            "and were skipped (excluded from the null distribution, not counted as zero)",
            file=sys.stderr,
        )

    p_value = permutation_p_value(real_r, null_rs)
    return AuditResult(real_r=real_r, real_n=len(real_rows), null_rs=null_rs, p_value=p_value)


def write_null_csv(result: AuditResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["shuffle_contemporaneous_r"])
        for r in result.null_rs:
            writer.writerow([r])


def run(in_path: Path, out_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_audit(headlines, n_shuffles=n_shuffles, seed=seed, live=live)
    except ValueError as exc:
        print(f"cannot run the audit: {exc}", file=sys.stderr)
        return 1

    write_null_csv(result, out_path)

    print(f"real (leak-free) alignment: contemporaneous r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled-timestamp null ({len(result.null_rs)} trials): "
        f"mean={result.null_mean:+.3f}  stdev={result.null_stdev:.3f}  "
        f"min={min(result.null_rs):+.3f}  max={max(result.null_rs):+.3f}  ({out_path})"
    )
    print(f"two-sided permutation p-value: {result.p_value:.3f}")

    if result.p_value < 0.05:
        print(
            "FAIL: the real r is an outlier against the shuffled-timestamp null - "
            "the alignment pipeline may be leaking look-ahead information. Do not trust "
            "Day 5-7's results until this is root-caused."
        )
        return 1

    print(
        "PASS: the real r is an unremarkable draw from the shuffled-timestamp null - "
        "no evidence the alignment pipeline is leaking. See README Day 8 Findings for "
        "why this is weaker evidence than a plain pass/fail suggests on this fixture."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="null-distribution CSV to write")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of shuffled-timestamp trials to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
