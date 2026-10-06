"""Day 8 CLI: the leakage audit NEXT_STEPS.md's "Done when" clause commits
to - shuffle headline timestamps and confirm the correlation signal does not
survive it.

    python -m sentiment.audit
    python -m sentiment.audit --trials 500 --seed 1

This is a permutation test, not a re-run of Day 5's analysis: ``compound``,
``title`` and ``company`` stay fixed per headline, but ``published_at`` is
permuted across the whole headline set before re-running Day 4's alignment
and Day 5's ``build_rows`` pipeline. That breaks the one link every later
day's work depends on - which headline's content attaches to which real
timestamp - while leaving everything else (which ticker a headline resolves
to, what price fixture backs it, how many headlines exist) untouched.

The real Pearson r is then compared against the distribution of r's the
same pipeline produces across many independent shuffles. If the real r is
not an outlier against that null distribution (a large two-sided p-value),
the correlation is statistically indistinguishable from what pure chance on
a dataset with no usable timing information produces - the "signal", such
as it is, does not depend on the headlines actually being aligned to the
sessions they were really published before. See README Findings for why
that is exactly what Day 5/6 already found by a different route (saturated
scores, zero-variance regressor) and why this audit does not contradict
them - it formalizes the same conclusion as a test that keeps re-checking
itself on every future run, per NEXT_STEPS.md's "this test runs in CI, not
once by hand."

A real leak - a correlation driven by something other than genuine,
correctly-ordered timing - is not guaranteed to show up here. See README
Limitations: a correlation that depends on *which ticker* a headline names
rather than *when* it was published survives this shuffle intact, because
shuffling timestamps never touches ticker assignment. This audit catches
timing-dependent artifacts, not every possible leak.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_TRIALS = 300
DEFAULT_SEED = 0


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list of ``Headline``s with the same ``published_at``
    values, randomly reassigned across headlines - everything else (title,
    source, link, ``scraped_at``) stays with its original headline.

    This is a permutation, not a resample: the same multiset of real
    timestamps is used, just relabelled, so a shuffled run still spans the
    same sessions the real run does and cannot fail merely for landing
    outside the price fixtures' date range.
    """
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=ts,
            published_raw=h.published_raw,
            scraped_at=h.scraped_at,
        )
        for h, ts in zip(headlines, timestamps)
    ]


def contemporaneous_r(rows: list[dict]) -> float | None:
    """Pearson r between ``compound`` and ``contemporaneous_return`` across
    ``rows``, or ``None`` if fewer than 2 rows resolved - too few for
    ``pearson_r`` to be defined, and too few for a shuffle trial to count."""
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float
    real_n: int
    trials_run: int
    trials_skipped: int
    shuffled_rs: list[float]
    mean_shuffled_r: float
    p_value: float

    @property
    def looks_like_noise(self) -> bool:
        """True if the real correlation is not a two-sided outlier (p > 0.05)
        against the shuffled-timestamp null distribution."""
        return self.p_value > 0.05


def run_shuffle_audit(
    headlines: list[Headline], n_trials: int = DEFAULT_TRIALS, seed: int = DEFAULT_SEED, live: bool = False
) -> ShuffleAuditResult:
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real_r = contemporaneous_r(real_rows)
    if real_r is None:
        raise ValueError(f"only {len(real_rows)} headline(s) resolved to a ticker - need at least 2 to audit")

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    skipped = 0
    for _ in range(n_trials):
        shuffled = shuffle_timestamps(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=live)
        r = contemporaneous_r(rows)
        if r is None:
            skipped += 1
            continue
        shuffled_rs.append(r)

    if not shuffled_rs:
        raise ValueError("every shuffle trial resolved fewer than 2 headlines - cannot build a null distribution")

    extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
    p_value = extreme / len(shuffled_rs)

    return ShuffleAuditResult(
        real_r=real_r,
        real_n=len(real_rows),
        trials_run=len(shuffled_rs),
        trials_skipped=skipped,
        shuffled_rs=shuffled_rs,
        mean_shuffled_r=sum(shuffled_rs) / len(shuffled_rs),
        p_value=p_value,
    )


def run(in_path: Path, n_trials: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_shuffle_audit(headlines, n_trials=n_trials, seed=seed, live=live)
    except ValueError as exc:
        print(f"cannot audit: {exc}", file=sys.stderr)
        return 1

    print(f"real contemporaneous r={result.real_r:+.3f} (n={result.real_n})")
    print(
        f"shuffled-timestamp null: {result.trials_run} trial(s) "
        f"({result.trials_skipped} skipped for <2 resolved rows), "
        f"mean r={result.mean_shuffled_r:+.3f}"
    )
    print(f"two-sided p-value (real r vs shuffled null) = {result.p_value:.3f}")

    if result.looks_like_noise:
        print(
            "signal does not survive timestamp shuffle: the real correlation is not "
            "distinguishable from what randomly mislabelling timestamps already produces. "
            "Consistent with Day 5/6's own null finding - not evidence of leakage, and not "
            "evidence of a real signal either."
        )
    else:
        print(
            "WARNING: the real correlation is an outlier against the shuffled-timestamp null "
            "(p <= 0.05) - investigate before trusting this number. Note this audit only "
            "catches timing-dependent artifacts; see README Limitations."
        )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="number of timestamp-shuffle trials")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null distribution")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.trials, args.seed, args.live))


if __name__ == "__main__":
    main()
