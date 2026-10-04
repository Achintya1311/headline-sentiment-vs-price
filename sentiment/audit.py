"""Day 8 CLI: leakage audit - shuffle headline timestamps and confirm the
sentiment/return signal disappears.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 1

NEXT_STEPS.md's "done when": shuffle the headline timestamps and the signal
must disappear. If a shuffled-timestamp control still predicts returns, the
pipeline is leaking and the result is an artifact.

What "disappear" means here, precisely: Day 4's market-hours alignment
(``sentiment.market_hours.align_headline``) is the only thing that ties a
headline to a specific trading session's return. If that is the sole
channel connecting ``compound`` to ``contemporaneous_return``, then
randomising which headline got which ``published_at`` - while leaving every
headline's title (and so its ticker and compound score) untouched - should
destroy whatever session-to-return pairing existed, and the resulting
correlation should scatter around zero across many such shuffles. If it
instead stays consistently far from zero, something other than the real,
leak-free alignment is driving the correlation - a bug, not a finding.

This audit deliberately does NOT try to answer "is there a real signal" -
Day 5/6's own 95%-CI correlations already address that, honestly, with a
null result. It checks one specific failure mode only: does breaking the
timestamp -> session_date link also break the correlation. See the
README's Day 8 section for why that is a narrower claim than "this
pipeline has no leaks of any kind."
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r, percentile_ci

DEFAULT_N_SHUFFLES = 300
DEFAULT_SEED = 0

# Below this, treat the shuffled-null std as "zero" - floating-point noise
# from re-deriving the same float arithmetic in a different row order, not a
# real spread (e.g. a build function that in fact ignores the shuffle
# entirely still accumulates ~1e-16 of rounding error across repeats).
STD_ZERO_TOLERANCE = 1e-9

BuildFn = Callable[[list[Headline], bool], tuple[list[dict], list[tuple[str, str]]]]


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list with the same headlines (title, and so ticker and
    compound score, untouched) but with ``published_at`` values permuted
    across them - breaking any real pairing between a headline's content and
    when it was actually published, while keeping the same multiset of
    timestamps in play."""
    shuffled_times = [h.published_at for h in headlines]
    rng.shuffle(shuffled_times)
    return [replace(h, published_at=t) for h, t in zip(headlines, shuffled_times)]


def contemporaneous_r(rows: list[dict]) -> float | None:
    """Pearson r between ``compound`` and ``contemporaneous_return`` across
    ``rows``, or ``None`` if there are fewer than the 2 points ``pearson_r``
    needs."""
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float
    real_n: int
    shuffled_rs: list[float]

    @property
    def shuffled_mean(self) -> float:
        return sum(self.shuffled_rs) / len(self.shuffled_rs)

    @property
    def shuffled_std(self) -> float:
        mean = self.shuffled_mean
        return (sum((r - mean) ** 2 for r in self.shuffled_rs) / len(self.shuffled_rs)) ** 0.5

    def null_ci(self) -> tuple[float, float]:
        """95% percentile interval of the shuffled-timestamp null distribution."""
        return percentile_ci(self.shuffled_rs)

    def passes(self) -> bool:
        """The leak check passes when shuffling actually perturbs the
        pipeline's output (std > 0 - a build path that ignores the shuffled
        timestamp entirely would otherwise pass vacuously) and the resulting
        null distribution is consistent with zero: breaking the timestamp ->
        session alignment also breaks the correlation, as it should if the
        real correlation came only from that leak-free alignment and
        nothing else."""
        if self.shuffled_std <= STD_ZERO_TOLERANCE:
            return False
        lo, hi = self.null_ci()
        return lo <= 0.0 <= hi


def run_shuffle_audit(
    headlines: list[Headline],
    build_fn: BuildFn = build_rows_from_headlines,
    live: bool = False,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
) -> ShuffleAuditResult:
    real_rows, _ = build_fn(headlines, live)
    real_r = contemporaneous_r(real_rows)
    if real_r is None:
        raise ValueError("not enough resolved headlines to audit (need >= 2)")

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_published_at(headlines, rng)
        rows, _ = build_fn(shuffled, live)
        r = contemporaneous_r(rows)
        if r is not None:
            shuffled_rs.append(r)

    if not shuffled_rs:
        raise ValueError("no shuffle produced enough resolved headlines to audit")

    return ShuffleAuditResult(real_r=real_r, real_n=len(real_rows), shuffled_rs=shuffled_rs)


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_shuffle_audit(headlines, live=live, n_shuffles=n_shuffles, seed=seed)
    except ValueError as exc:
        print(f"cannot run leakage audit: {exc}", file=sys.stderr)
        return 1

    lo, hi = result.null_ci()
    print(f"real contemporaneous r = {result.real_r:+.3f}  (n={result.real_n})")
    print(
        f"{len(result.shuffled_rs)} shuffles of published_at -> "
        f"null r: mean={result.shuffled_mean:+.3f}  std={result.shuffled_std:.3f}  "
        f"95% range [{lo:+.3f}, {hi:+.3f}]"
    )

    if result.shuffled_std <= STD_ZERO_TOLERANCE:
        print(
            "INCONCLUSIVE: shuffling published_at did not change the pipeline's output at all - "
            "the build path isn't actually using the timestamp to assign returns, so this audit "
            "cannot tell leak-free from leaking here.",
            file=sys.stderr,
        )
        return 1

    if result.passes():
        print(
            "PASS: the shuffled-timestamp null is consistent with zero - breaking the real "
            "timestamp -> session alignment also breaks the correlation, as it should if the "
            "real correlation came only from leak-free alignment."
        )
        return 0

    print(
        "FAIL: the shuffled-timestamp null stays away from zero - something other than the "
        "real, leak-free alignment is driving the correlation. Treat the pipeline as leaking "
        "until this is root-caused.",
        file=sys.stderr,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp shuffles to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for the shuffles (deterministic)")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
