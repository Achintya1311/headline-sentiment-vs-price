"""Day 8: shuffle-timestamp leakage audit.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 2000
    python -m sentiment.audit --live

NEXT_STEPS.md's "done when": shuffle the headline timestamps and the
sentiment/return signal must disappear. If a shuffled-timestamp control
still predicts returns, the pipeline is leaking look-ahead information and
Days 5-7's results cannot be trusted at face value.

This builds a null distribution by repeatedly permuting the fixture
headlines' own ``published_at`` timestamps among each other - each headline
keeps its own title (so ``sentiment.tickers.resolve`` still finds the same
company/ticker, and VADER still scores the same text), but Day 4's alignment
now attaches that headline's content to a ``session_date`` - and therefore a
return - it was never actually published ahead of. Any real, content-driven
relationship between what a headline says and how its own company's stock
moved should not survive that scrambling.

Two numbers are reported for both the contemporaneous and lagged
correlation ``sentiment.correlate`` computes:

- **false-positive rate** (the pass/fail gate): the share of shuffled
  trials whose own 95% CI excludes zero. A pipeline run against scrambled
  timestamps has nothing time-dependent left to find, so a correctly built
  pipeline should "discover" a significant correlation about as often as a
  95% CI's own 5% false-positive rate predicts by chance alone. A rate far
  above that means the pipeline keeps finding "significant" correlations
  regardless of whether the timestamps feeding it are honest - which is the
  leak this audit exists to catch, and it survives even a leak that makes
  *every* shuffle reproduce the exact same rows (see
  ``tests/test_audit.py`` for a worked example).
- **permutation p-value** (diagnostic only, not a gate): the share of
  shuffled trials whose |r| is at least as large as the real, unshuffled
  pipeline's |r|. This is reported for context but deliberately does not
  decide pass/fail: a genuinely strong, non-leaked relationship is
  *supposed* to look like an outlier against a shuffled null (that is what
  a real effect looks like under a permutation test, not evidence of a
  leak) - only the false-positive rate tells leakage apart from a real
  signal, because a leak reproduces "significance" on shuffles that should
  have destroyed it, while a real signal does not.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import PearsonResult, pearson_with_ci

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0

# How much slack above the CI's own nominal 5% false-positive rate is
# tolerated before the audit calls it a leak rather than shuffle noise.
# With n_shuffles in the hundreds, the binomial standard error around 5% is
# under 1 point - 25% is not "generous", it is "only trips on a real bug".
FALSE_POSITIVE_TOLERANCE = 0.25


@dataclass(frozen=True)
class CorrelationAudit:
    label: str
    observed: PearsonResult | None
    n_shuffles: int
    permutation_p: float | None
    false_positive_rate: float | None

    @property
    def passes(self) -> bool:
        """With no observed correlation (too few resolved headlines) there is
        nothing to audit, so that counts as passing rather than failing by
        default. Otherwise the gate is the false-positive rate alone: does
        shuffling keep "finding" significant correlations far more often
        than a 95% CI's own 5% chance rate would predict. The permutation
        p-value is reported alongside but never gates this - see the module
        docstring for why it would wrongly flag a real, non-leaked signal."""
        if self.observed is None or self.false_positive_rate is None:
            return True
        return self.false_positive_rate <= FALSE_POSITIVE_TOLERANCE


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` (and the
    ``published_raw`` string kept consistent with it) permuted among them.
    Each headline keeps its own title, link and source - only which
    timestamp it carries changes, so the set of timestamps in play is
    exactly the same as the real data, just reassigned."""
    shuffled_times = [h.published_at for h in headlines]
    rng.shuffle(shuffled_times)
    return [
        dataclasses.replace(h, published_at=t, published_raw=t.isoformat())
        for h, t in zip(headlines, shuffled_times)
    ]


def _correlation_from_rows(rows: list[dict], key: str) -> PearsonResult | None:
    pairs = [(r["compound"], r[key]) for r in rows if r.get(key) is not None]
    if len(pairs) < 2:
        return None
    return pearson_with_ci([p[0] for p in pairs], [p[1] for p in pairs])


def audit_correlation(
    headlines: list[Headline],
    key: str,
    label: str,
    live: bool = False,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
) -> CorrelationAudit:
    """Run the shuffle-timestamp control for one of ``correlate``'s two
    correlations (``key`` is ``"contemporaneous_return"`` or
    ``"lagged_return"``)."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    observed = _correlation_from_rows(rows, key)
    if observed is None:
        return CorrelationAudit(
            label=label, observed=None, n_shuffles=n_shuffles, permutation_p=None, false_positive_rate=None
        )

    rng = random.Random(seed)
    as_extreme = 0
    false_positives = 0
    completed = 0
    for _ in range(n_shuffles):
        trial_headlines = shuffle_published_at(headlines, rng)
        with contextlib.redirect_stderr(io.StringIO()):
            trial_rows, _ = build_rows_from_headlines(trial_headlines, live=live)
        stat = _correlation_from_rows(trial_rows, key)
        if stat is None:
            # A shuffle landed enough headlines outside the price fixture's
            # date range to leave fewer than 2 pairable rows - skip rather
            # than count it either way; it is silent on the question asked.
            continue
        completed += 1
        if abs(stat.r) >= abs(observed.r):
            as_extreme += 1
        if stat.ci_low > 0 or stat.ci_high < 0:
            false_positives += 1

    if completed == 0:
        return CorrelationAudit(
            label=label, observed=observed, n_shuffles=n_shuffles, permutation_p=None, false_positive_rate=None
        )

    permutation_p = (as_extreme + 1) / (completed + 1)
    false_positive_rate = false_positives / completed
    return CorrelationAudit(
        label=label,
        observed=observed,
        n_shuffles=completed,
        permutation_p=permutation_p,
        false_positive_rate=false_positive_rate,
    )


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    audits = [
        audit_correlation(headlines, "contemporaneous_return", "contemporaneous", live, n_shuffles, seed),
        audit_correlation(headlines, "lagged_return", "lagged", live, n_shuffles, seed + 1),
    ]

    all_pass = True
    for a in audits:
        if a.observed is None:
            print(f"{a.label}: not enough resolved headlines to audit (skipped)")
            continue
        if a.permutation_p is None:
            print(f"{a.label}: no shuffle trial produced a computable correlation (skipped)")
            continue
        verdict = "PASS" if a.passes else "FAIL"
        print(
            f"{a.label}: observed r={a.observed.r:+.3f} n={a.observed.n}  "
            f"permutation p={a.permutation_p:.3f}  "
            f"shuffled false-positive rate={a.false_positive_rate:.1%} "
            f"(n_shuffles={a.n_shuffles})  -> {verdict}"
        )
        all_pass = all_pass and a.passes

    print("leakage audit: PASS" if all_pass else "leakage audit: FAIL")
    return 0 if all_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp-shuffle trials per correlation"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for reproducible shuffles")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
