"""Day 8 CLI: leakage audit - shuffle headline timestamps and confirm any
sentiment/return signal disappears.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 1000 --seed 1

This is the "Done when" gate NEXT_STEPS.md commits to before Day 5's
correlation numbers are trusted: neither ticker resolution (``sentiment.
tickers.resolve``) nor VADER scoring (``sentiment.vader_score.score_headline``)
looks at ``published_at`` - only Day 4's ``align_headline`` does, to pick
which trading session a headline is tested against. So under a genuinely
leak-free pipeline, a headline's sentiment score and *which session's return
it gets compared to* should be independent of one another once you scramble
the timestamps: whatever correlation the real, correctly-dated alignment
finds should not survive relabelling headlines with someone else's
publish time.

The check: resolve every headline to (ticker, compound) once - that part
never changes - then repeat "align with this set of timestamps, correlate
compound against contemporaneous return" under many random permutations of
the timestamps across headlines. Two things have to both hold for this to
pass:

1. The real (correctly-aligned) correlation is not a statistical outlier
   against the distribution of shuffled-timestamp correlations (a
   permutation-test p-value on |r|).
2. Shuffled timestamps don't turn up a "statistically significant" 95% CI
   (one that excludes zero) much more often than the ~5% a well-behaved
   permutation should produce by chance alone - a sanity check independent
   of the real result, since a pipeline that manufactures significance from
   random relabelling is broken regardless of what the real alignment finds.

On this fixture the real result is already a null one (see Day 5's README
Findings, r=-0.185 with a CI that comfortably contains zero) - so passing
here mostly confirms there was nothing to lose. That is exactly why
``tests/test_audit.py`` also runs this same procedure against a *synthetic*
leak (a case built so the "sentiment" score is wired directly to its own
correctly-aligned session's return) and checks it correctly fails: without
that positive control, a pass on the real data would be indistinguishable
from a test with no power at all. See the README's "Why this result might
still be spurious" section for what this audit does and does not rule out.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sentiment.correlate import DEFAULT_IN
from sentiment.headline import Headline, read_csv
from sentiment.market_hours import align_headline
from sentiment.prices import Bar, PriceFetchError, bar_on, load_bars
from sentiment.stats import pearson_r, pearson_with_ci
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0

# Nominal false-positive rate a well-behaved 95% CI should hit by chance
# alone under a true null - used both as the pass/fail threshold for the
# real-vs-shuffled comparison and as a ceiling on how often shuffling itself
# should look "significant".
ALPHA = 0.05


@dataclass(frozen=True)
class Resolved:
    """A headline reduced to what the correlation step actually needs, split
    from ``published_at`` on purpose: everything here is timestamp-independent,
    so shuffling timestamps across a list of ``Resolved`` never has to touch
    ticker resolution or scoring again."""

    title: str
    ticker: str
    compound: float
    published_at: datetime


def resolve_headlines(headlines: list[Headline]) -> list[Resolved]:
    """Resolve+score every headline, same scope as ``sentiment.correlate.
    build_rows`` (only headlines naming exactly one company with a fetchable
    ticker) - independent of ``published_at``, which is kept alongside only
    so the real (unshuffled) run has something to compare shuffles against."""
    resolved: list[Resolved] = []
    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        _, ticker = match
        if ticker is None:
            continue
        scored = score_headline(h)
        resolved.append(Resolved(title=h.title, ticker=ticker, compound=scored.compound, published_at=h.published_at))
    return resolved


def load_bar_cache(resolved: list[Resolved]) -> dict[str, list[Bar]]:
    """Load each distinct ticker's price bars once, offline (fixtures only -
    this audit is a correctness gate meant to run unattended in CI, not a
    place to add live network calls repeated across hundreds of shuffles)."""
    cache: dict[str, list[Bar]] = {}
    for r in resolved:
        if r.ticker in cache:
            continue
        try:
            cache[r.ticker] = load_bars(r.ticker, live=False)
        except PriceFetchError:
            cache[r.ticker] = []
    return cache


def contemporaneous_pairs(
    resolved: list[Resolved], timestamps: list[datetime], bar_cache: dict[str, list[Bar]]
) -> tuple[list[float], list[float]]:
    """Align each ``resolved[i]`` headline to ``timestamps[i]`` (which may be
    a shuffle of the real timestamps) and look up that session's return.
    Returns ``(compounds, returns)`` for whichever headlines land on a
    session this fixture's price history actually covers - a shuffle can
    (and does) push a headline onto a session with no bar, which just drops
    that one point rather than erroring."""
    compounds: list[float] = []
    returns: list[float] = []
    for r, ts in zip(resolved, timestamps):
        alignment = align_headline(ts)
        bar = bar_on(bar_cache.get(r.ticker, []), alignment.session_date)
        if bar is None:
            continue
        compounds.append(r.compound)
        returns.append(bar.session_return)
    return compounds, returns


def permutation_p_value(real_stat: float, shuffled_stats: list[float]) -> float:
    """Two-sided permutation p-value: the fraction of shuffles whose |stat| is
    at least as extreme as the real one, add-one smoothed (Davison & Hinkley)
    so a finite number of shuffles never reports an impossible p=0 - there is
    always some chance an unseen shuffle could have matched or exceeded it."""
    if not shuffled_stats:
        return 1.0
    extreme = sum(1 for s in shuffled_stats if abs(s) >= abs(real_stat))
    return (extreme + 1) / (len(shuffled_stats) + 1)


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float
    real_n: int
    shuffled_rs: list[float]
    p_value: float
    pct_shuffles_significant: float

    @property
    def n_shuffles(self) -> int:
        return len(self.shuffled_rs)

    def passes(self, alpha: float = ALPHA) -> bool:
        """PASS: the real, leak-free-aligned result is not distinguishable
        from the shuffled-timestamp null (p-value > alpha), AND shuffling
        alone isn't manufacturing 'significant' correlations far more than
        chance predicts. Both have to hold - see module docstring."""
        return self.p_value > alpha and self.pct_shuffles_significant <= 2 * alpha


def audit_resolved(
    resolved: list[Resolved],
    bar_cache: dict[str, list[Bar]],
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
) -> ShuffleAuditResult:
    """The permutation core, split from ``run_shuffle_audit`` so it can be
    driven directly from already-resolved (headline-free) data - both by the
    real CLI path and by ``tests/test_audit.py``'s synthetic positive control,
    which has no real ticker/headline text to resolve."""
    real_timestamps = [r.published_at for r in resolved]
    real_compounds, real_returns = contemporaneous_pairs(resolved, real_timestamps, bar_cache)
    if len(real_compounds) < 2:
        raise ValueError(f"not enough headlines with a priced session to audit (n={len(real_compounds)})")
    real_r = pearson_r(real_compounds, real_returns)

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    n_significant = 0
    n_evaluable = 0
    for _ in range(n_shuffles):
        shuffled_timestamps = real_timestamps[:]
        rng.shuffle(shuffled_timestamps)
        compounds, returns = contemporaneous_pairs(resolved, shuffled_timestamps, bar_cache)
        if len(compounds) < 2:
            continue
        shuffled_rs.append(pearson_r(compounds, returns))
        if len(compounds) >= 4:
            n_evaluable += 1
            ci = pearson_with_ci(compounds, returns)
            if ci.ci_low > 0 or ci.ci_high < 0:
                n_significant += 1

    p_value = permutation_p_value(real_r, shuffled_rs)
    pct_significant = n_significant / n_evaluable if n_evaluable else 0.0

    return ShuffleAuditResult(
        real_r=real_r,
        real_n=len(real_compounds),
        shuffled_rs=shuffled_rs,
        p_value=p_value,
        pct_shuffles_significant=pct_significant,
    )


def run_shuffle_audit(
    headlines: list[Headline], n_shuffles: int = DEFAULT_N_SHUFFLES, seed: int = DEFAULT_SEED
) -> ShuffleAuditResult:
    resolved = resolve_headlines(headlines)
    if len(resolved) < 2:
        raise ValueError(f"not enough resolved headlines to audit (n={len(resolved)})")
    bar_cache = load_bar_cache(resolved)
    return audit_resolved(resolved, bar_cache, n_shuffles=n_shuffles, seed=seed)


def run(in_path: Path, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_shuffle_audit(headlines, n_shuffles=n_shuffles, seed=seed)
    except ValueError as exc:
        print(f"cannot run leakage audit: {exc}", file=sys.stderr)
        return 1

    mean_shuffled = sum(result.shuffled_rs) / len(result.shuffled_rs)
    print(f"real (leak-free aligned):  r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled-timestamp null:   mean r={mean_shuffled:+.3f}  "
        f"range [{min(result.shuffled_rs):+.3f}, {max(result.shuffled_rs):+.3f}]  "
        f"over {result.n_shuffles} shuffles"
    )
    print(f"permutation p-value (real vs. shuffled null): {result.p_value:.3f}")
    print(
        f"shuffles reading 'significant' by chance (95% CI excludes 0): "
        f"{result.pct_shuffles_significant:.1%}  (nominal ~{ALPHA:.0%} expected)"
    )

    if result.passes():
        print(
            "PASS: the real result is not distinguishable from random timestamp "
            "noise, and shuffling does not manufacture spurious significance. "
            "No evidence this pipeline is leaking."
        )
        return 0

    print(
        "FAIL: a shuffled-timestamp control still predicts returns as well as "
        "(or better than) the real, correctly-dated alignment - treat "
        "sentiment.correlate's reported correlation as an artifact, not a "
        "signal, until this is root-caused.",
        file=sys.stderr,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp permutations to draw"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible CI run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
