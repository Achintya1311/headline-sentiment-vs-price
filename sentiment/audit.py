"""Day 8 CLI: ml-pipeline audit - shuffle headline timestamps and confirm the
sentiment/return "signal" disappears.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 1000 --seed 1

NEXT_STEPS.md's "Done when": shuffle the headline timestamps and the signal
must disappear. If a shuffled-timestamp control still predicts returns, the
pipeline is leaking and the result is an artifact.

This is a calibration check, not a one-off eyeball comparison. Randomly
reassign the scraped headlines' ``published_at`` timestamps among each other
(same multiset of times, randomly re-paired with headline content, so each
headline keeps its own title/company/ticker but gets someone else's publish
time), re-run Day 4's alignment and Day 5's contemporaneous correlation on
the result, and repeat ``--n-shuffles`` times. Shuffling the timestamp is the
only thing that changes: which trading session a headline aligns to, and
therefore which return it is paired with. A headline's VADER compound score
never changes (it is a property of the title, not the time), so if the
pipeline is honest, a headline's real timing should be the only reason its
sentiment ever lines up with a particular session's return.

Day 5's own 95%-CI machinery already tells us, per run, whether a
correlation "looks significant" (CI excludes zero). Under genuinely random
timing there is no true relationship to find, so by construction a 95% CI
should wrongly exclude zero on about 5% of shuffles - pure sampling noise.
If the observed rate is far above that, something other than correct timing
is driving the correlation whenever it appears: a leak, not a signal that
disappears when timing is randomised.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import PearsonResult, pearson_with_ci

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0
MIN_ROWS_FOR_CI = 4

# Under a true null, a 95% CI should wrongly exclude zero ~5% of the time.
# Flag only once the rate is well above what sampling noise alone explains
# at the --n-shuffles sizes this CLI actually runs with.
NOMINAL_FALSE_POSITIVE_RATE = 0.05
FLAG_THRESHOLD = 0.15


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list of headlines with ``published_at`` values permuted
    across them - every other field (title, link, source) stays with its
    original row.

    This is deliberately the only thing that changes. A headline's title
    (and therefore its VADER score and which ticker ``sentiment.tickers``
    resolves it to) is untouched; only *when* it is said to have been
    published moves. That isolates Day 4's alignment logic as the one thing
    under test: does the correlation depend on headlines being paired with
    the session their own timing says they may honestly be tested against,
    or would any random pairing do just as well?
    """
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


def _contemporaneous_stat(headlines: list[Headline]) -> PearsonResult | None:
    rows, _ = build_rows_from_headlines(headlines, live=False)
    if len(rows) < MIN_ROWS_FOR_CI:
        return None
    return pearson_with_ci([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


@dataclass(frozen=True)
class AuditResult:
    real: PearsonResult
    null_rs: list[float]
    n_ci_excludes_zero: int

    @property
    def n_valid(self) -> int:
        return len(self.null_rs)

    @property
    def false_positive_rate(self) -> float:
        return self.n_ci_excludes_zero / self.n_valid


def run_shuffle_audit(headlines: list[Headline], n_shuffles: int, seed: int) -> AuditResult:
    """Run the shuffle-timestamp leakage control described in the module
    docstring. Raises ``ValueError`` if the real pipeline, or every shuffle,
    resolves too few headlines to even compute a correlation's CI - there is
    nothing to audit then."""
    real = _contemporaneous_stat(headlines)
    if real is None:
        raise ValueError(
            f"fewer than {MIN_ROWS_FOR_CI} resolved headlines - not enough to correlate or audit"
        )

    rng = random.Random(seed)
    null_rs: list[float] = []
    n_ci_excludes_zero = 0
    for _ in range(n_shuffles):
        stat = _contemporaneous_stat(shuffle_timestamps(headlines, rng))
        if stat is None:
            continue
        null_rs.append(stat.r)
        if stat.ci_low > 0 or stat.ci_high < 0:
            n_ci_excludes_zero += 1

    if not null_rs:
        raise ValueError("no shuffle produced a scoreable correlation - cannot build a null distribution")

    return AuditResult(real=real, null_rs=null_rs, n_ci_excludes_zero=n_ci_excludes_zero)


def run(in_path: Path, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_shuffle_audit(headlines, n_shuffles, seed)
    except ValueError as exc:
        print(f"cannot audit: {exc}", file=sys.stderr)
        return 1

    null_mean = sum(result.null_rs) / len(result.null_rs)

    print(
        f"real contemporaneous correlation: r={result.real.r:+.3f}  "
        f"95% CI [{result.real.ci_low:+.3f}, {result.real.ci_high:+.3f}]  n={result.real.n}"
    )
    print(f"shuffled-timestamp control ({result.n_valid} shuffles, seed={seed}): mean r={null_mean:+.3f}")
    print(
        f"false-positive rate (shuffles whose 95% CI excludes zero): "
        f"{result.false_positive_rate:.1%}  (nominal expectation under random timing: "
        f"{NOMINAL_FALSE_POSITIVE_RATE:.0%})"
    )

    if result.false_positive_rate > FLAG_THRESHOLD:
        print(
            f"FLAG: {result.false_positive_rate:.1%} of shuffles look 'significant' by their own "
            f"95% CI, well above the ~{NOMINAL_FALSE_POSITIVE_RATE:.0%} random timing should "
            "produce. The correlation does not depend on correct alignment - that is what a leak "
            "looks like. Do not trust the Day 5 result until this is root-caused."
        )
        return 1

    print(
        "PASS: shuffled timing produces a 'significant' correlation about as often as chance alone "
        "would (~5% of the time), not systematically. Nothing suggests the pipeline manufactures a "
        "correlation independent of real headline timing."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to audit")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp permutations to try"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
