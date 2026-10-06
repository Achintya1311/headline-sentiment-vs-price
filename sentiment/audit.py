"""Day 8 CLI: ml-pipeline-audit - shuffle headline timestamps and confirm
the sentiment/return "signal" disappears.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 2000 --seed 1
    python -m sentiment.audit --live

See the repo's "Correctness gate" and NEXT_STEPS.md's "Done when": this is
the test that decides whether any of Day 5-7's numbers are believable. A
headline's sentiment is paired with a return through exactly one channel -
``sentiment.market_hours.align_headline(published_at)`` picks the
leak-free ``session_date``, and that date picks the price bar. If the
reported correlation exists for some *other* reason (a bug that ignores the
real alignment, a confound that happens to track headline order rather than
true timing, ...), scrambling which ``published_at`` goes with which
headline should not touch it. If it does survive scrambling, the pipeline
is leaking and the correlation is an artifact, not a finding.

Method: a permutation test. Ticker resolution and VADER scoring depend only
on a headline's *text*, never its timestamp, so both are computed once up
front. What gets shuffled is the mapping from (fixed) headline to (fixed)
published_at timestamp; each shuffle re-runs only the timestamp-dependent
half of the pipeline - alignment, session_date, and the resulting
contemporaneous return - exactly the half Day 4 built to prevent look-ahead.
Repeating that many times builds a null distribution of |r|; the real
(unshuffled) |r| is reported against it as a two-sided permutation p-value:
the fraction of shuffles whose |r| is at least as large as the real one.

A large p-value means the real correlation is not statistically
distinguishable from one you would get by randomly scrambling which
timestamp belongs to which headline - consistent with "no detectable
signal", the same conclusion Day 5-7 already reached with a parametric CI.
A small p-value means the real correlation is bigger than nearly every
shuffled control, i.e. it depends on the genuine timestamp-to-session
mapping rather than coincidence - on a fixture this small, a reason to look
by hand, not an automatic verdict either way.

What this test does *not* catch by itself: a bug where shuffling has no
real effect on the computation (e.g. alignment silently ignoring the
timestamp it was given) would make every "shuffled" run reproduce the real
result exactly, which looks like total agreement with the null, not an
outlier - a permutation p-value alone cannot tell that apart from "nothing
to find here" without also checking that the shuffle mechanism actually
changes the computed pairs (see ``tests/test_audit.py``'s direct check on
``contemporaneous_pairs``, and the README's "Why this might be spurious"
section for what else this test cannot rule out).
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sentiment.headline import Headline, read_csv
from sentiment.market_hours import align_headline
from sentiment.prices import PriceFetchError, bar_on, load_bars
from sentiment.stats import pearson_r
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_N_SHUFFLES = 1000
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class ResolvedHeadline:
    """The timestamp-independent half of one headline: which ticker it
    names and what it scores, fixed before any shuffling happens."""

    title: str
    ticker: str
    compound: float
    published_at: datetime


def resolve_and_score(headlines: list[Headline]) -> list[ResolvedHeadline]:
    """Ticker-resolve and VADER-score every headline once.

    Headlines that name no single company, or whose company has no
    fetchable ticker (``sentiment.tickers.resolve``), are dropped - the same
    scope Day 5's ``build_rows`` uses. Doing this once up front means a
    shuffle only has to touch the cheap, timestamp-dependent half of the
    pipeline below, not re-run VADER thousands of times on text that never
    changes.
    """
    resolved = []
    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        _company, ticker = match
        if ticker is None:
            continue
        scored = score_headline(h)
        resolved.append(
            ResolvedHeadline(title=h.title, ticker=ticker, compound=scored.compound, published_at=h.published_at)
        )
    return resolved


def shuffle_timestamps(timestamps: list[datetime], rng: random.Random) -> list[datetime]:
    """Return a permutation of ``timestamps`` - same multiset of values, new
    order. Pulled out as its own function so the permutation itself (not the
    pipeline wired around it) can be tested directly."""
    shuffled = list(timestamps)
    rng.shuffle(shuffled)
    return shuffled


def contemporaneous_pairs(
    resolved: list[ResolvedHeadline], published_ats: list[datetime], live: bool = False
) -> tuple[list[float], list[float]]:
    """Pair each ``resolved[i]``'s (fixed) compound with the contemporaneous
    return implied by ``published_ats[i]`` - not necessarily
    ``resolved[i].published_at``. Passing a shuffled list of timestamps here
    is exactly the leakage control: everything else about the pipeline
    (which ticker, what sentiment score) stays the same as the real run.
    """
    compounds: list[float] = []
    returns: list[float] = []
    for r, ts in zip(resolved, published_ats):
        alignment = align_headline(ts)
        try:
            bars = load_bars(r.ticker, live=live)
        except PriceFetchError:
            continue
        bar = bar_on(bars, alignment.session_date)
        if bar is None:
            continue
        compounds.append(r.compound)
        returns.append(bar.session_return)
    return compounds, returns


def correlation_stat(resolved: list[ResolvedHeadline], published_ats: list[datetime], live: bool = False) -> float | None:
    compounds, returns = contemporaneous_pairs(resolved, published_ats, live=live)
    if len(compounds) < 2:
        return None
    return pearson_r(compounds, returns)


@dataclass(frozen=True)
class PermutationAuditResult:
    observed_r: float
    observed_n: int
    null_abs_r: list[float]
    n_shuffles: int

    @property
    def p_value(self) -> float:
        """Two-sided permutation p-value: the fraction of shuffles whose
        |r| is at least as large as the observed |r|, with add-one
        smoothing (Davison & Hinkley) so a finite number of shuffles never
        reports an impossible p-value of exactly 0."""
        observed_abs = abs(self.observed_r)
        at_least_as_extreme = sum(1 for null_r in self.null_abs_r if null_r >= observed_abs)
        return (at_least_as_extreme + 1) / (len(self.null_abs_r) + 1)

    def leak_free(self, alpha: float = DEFAULT_ALPHA) -> bool:
        """True if the observed correlation is NOT a significant outlier
        against the shuffled-timestamp null - i.e. scrambling which
        timestamp belongs to which headline destroys whatever correlation
        existed, rather than leaving it intact."""
        return self.p_value > alpha


def run_permutation_audit(
    headlines: list[Headline],
    live: bool = False,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
) -> PermutationAuditResult:
    resolved = resolve_and_score(headlines)
    real_timestamps = [r.published_at for r in resolved]
    observed = correlation_stat(resolved, real_timestamps, live=live)
    if observed is None:
        raise ValueError("fewer than 2 headlines resolve to a ticker with price data; nothing to audit")

    rng = random.Random(seed)
    null_abs_r: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(real_timestamps, rng)
        stat = correlation_stat(resolved, shuffled, live=live)
        if stat is not None:
            null_abs_r.append(abs(stat))

    if not null_abs_r:
        raise ValueError("every shuffle produced fewer than 2 usable pairs; cannot build a null distribution")

    return PermutationAuditResult(
        observed_r=observed,
        observed_n=len(contemporaneous_pairs(resolved, real_timestamps, live=live)[0]),
        null_abs_r=null_abs_r,
        n_shuffles=n_shuffles,
    )


def run(in_path: Path, live: bool, n_shuffles: int, seed: int, alpha: float) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_permutation_audit(headlines, live=live, n_shuffles=n_shuffles, seed=seed)
    except ValueError as exc:
        print(f"cannot run the leakage audit: {exc}", file=sys.stderr)
        return 1

    valid = len(result.null_abs_r)
    print(f"observed contemporaneous r={result.observed_r:+.3f} (n={result.observed_n})")
    print(
        f"shuffled-timestamp null ({valid}/{result.n_shuffles} shuffles valid): "
        f"mean|r|={statistics.mean(result.null_abs_r):.3f}  max|r|={max(result.null_abs_r):.3f}"
    )
    print(f"two-sided permutation p-value: {result.p_value:.3f}")

    if result.leak_free(alpha):
        print(
            f"PASS (p > {alpha}): the observed correlation is not statistically distinguishable "
            "from what randomly shuffling the headline timestamps produces. On this fixture that is "
            "the same conclusion Day 5-7 already reached by other means - no detectable signal, not "
            "a leak propping up a result that isn't really there. See README 'Why this might be "
            "spurious' for what this test does and does not rule out."
        )
        return 0
    print(
        f"NOTABLE (p <= {alpha}): the observed correlation is larger than nearly every randomly "
        "shuffled control - i.e. it depends on the real published_at-to-session mapping, not "
        "coincidence. On a fixture this small that is a flag to check by hand (ticker resolution, "
        "alignment, the price fixture itself), not an automatic leak finding."
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp shuffles in the null"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible null")
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="significance threshold for the permutation p-value"
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed, args.alpha))


if __name__ == "__main__":
    main()
