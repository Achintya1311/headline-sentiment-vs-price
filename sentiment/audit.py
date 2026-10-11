"""Day 8 CLI: the timestamp-shuffle leakage audit (NEXT_STEPS.md "Done when").

The repo's own correctness gate: shuffle the headline timestamps that drive
Day 4's session-date alignment, rebuild Day 5's contemporaneous-return
correlation under each shuffle, and compare the real (true-timestamp) r
against that shuffle-generated null distribution. A headline's compound
score and resolved ticker do not depend on timing, so reassigning
``published_at`` among headlines - then re-running the real alignment and
price-lookup code, not just permuting two already-computed number lists -
is what would expose a look-ahead leak in *this* pipeline specifically: if
the real pairing produces a correlation no permutation can match, something
other than genuine chronological reaction is driving it.

    python -m sentiment.audit
    python -m sentiment.audit --n-perm 5000
    python -m sentiment.audit --live

On its own this control is necessary but uninformative if it always reports
"no leak" - a test that can never fail proves nothing. ``tests/test_audit.py``
separately proves it has power: a synthetic case engineered to correlate
only when the true timestamp is used, where the shuffle correctly destroys
the correlation and the audit flags the true pairing as a significant
outlier against the shuffled null.

This is a statistical control on top of ``Alignment.leak_free()``
(``sentiment/market_hours.py``), which already asserts the no-look-ahead
invariant directly on every row - this module does not replace that, it
checks whether anything downstream of a correct alignment still manages to
look leak-like.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sentiment.headline import Headline, read_csv
from sentiment.market_hours import align_headline
from sentiment.prices import Bar, PriceFetchError, bar_on, load_bars
from sentiment.stats import pearson_r, permutation_p_value
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_N_PERM = 2000
SIGNIFICANCE = 0.05


@dataclass(frozen=True)
class _Resolved:
    ticker: str
    compound: float
    bars: list[Bar]


def _resolvable(headlines: list[Headline], live: bool = False) -> tuple[list[_Resolved], list[datetime]]:
    """The part of the pipeline that does not depend on *which* timestamp is
    used: ticker resolution, VADER scoring, and the price fixture itself.
    Precomputing this once means the shuffle loop below only has to redo
    alignment and the bar lookup - the part a shuffled timestamp actually
    changes - on every permutation, not re-resolve or re-score headlines or
    re-read a fixture file thousands of times.

    Returns ``(resolved, true_times)``, parallel lists - ``true_times[i]``
    is the real ``published_at`` behind ``resolved[i]``.
    """
    resolved: list[_Resolved] = []
    true_times: list[datetime] = []
    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        _, ticker = match
        if ticker is None:
            continue
        try:
            bars = load_bars(ticker, live=live)
        except PriceFetchError:
            continue
        resolved.append(_Resolved(ticker=ticker, compound=score_headline(h).compound, bars=bars))
        true_times.append(h.published_at)
    return resolved, true_times


def _returns_for_times(resolved: list[_Resolved], times: list[datetime]) -> tuple[list[float], list[float]]:
    """Re-run just the alignment + bar lookup for ``times`` (parallel to
    ``resolved``, either headlines' real ``published_at`` or a shuffled
    permutation of them) and return the (compounds, contemporaneous_returns)
    pairs that still resolve to a bar in this fixture's date range."""
    compounds: list[float] = []
    returns: list[float] = []
    for r, published_at in zip(resolved, times):
        alignment = align_headline(published_at)
        session_bar = bar_on(r.bars, alignment.session_date)
        if session_bar is None:
            continue
        compounds.append(r.compound)
        returns.append(session_bar.session_return)
    return compounds, returns


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    n_real: int
    p_value: float
    n_perm: int
    perm_mean: float
    perm_std: float


def run_audit(headlines: list[Headline], live: bool = False, n_perm: int = DEFAULT_N_PERM, seed: int = 0) -> AuditResult:
    resolved, true_times = _resolvable(headlines, live=live)
    if len(resolved) < 2:
        raise ValueError("fewer than 2 headlines resolved to a priced ticker; nothing to audit")

    real_compounds, real_returns = _returns_for_times(resolved, true_times)
    if len(real_compounds) < 2:
        raise ValueError("fewer than 2 headlines have a bar for their true aligned session; nothing to audit")
    real_r = pearson_r(real_compounds, real_returns)

    rng = random.Random(seed)
    perm_rs: list[float] = []
    for _ in range(n_perm):
        shuffled_times = true_times[:]
        rng.shuffle(shuffled_times)
        compounds, returns = _returns_for_times(resolved, shuffled_times)
        if len(compounds) >= 2:
            perm_rs.append(pearson_r(compounds, returns))

    if not perm_rs:
        raise ValueError("no permutation produced 2+ resolved headlines; cannot build a null distribution")

    p_value = permutation_p_value(real_r, perm_rs)
    perm_mean = sum(perm_rs) / len(perm_rs)
    perm_var = sum((r - perm_mean) ** 2 for r in perm_rs) / len(perm_rs)
    return AuditResult(
        real_r=real_r,
        n_real=len(real_compounds),
        p_value=p_value,
        n_perm=len(perm_rs),
        perm_mean=perm_mean,
        perm_std=perm_var**0.5,
    )


def run(in_path: Path, live: bool, n_perm: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1
    headlines = read_csv(in_path)
    try:
        result = run_audit(headlines, live=live, n_perm=n_perm, seed=seed)
    except ValueError as exc:
        print(f"audit could not run: {exc}", file=sys.stderr)
        return 1

    print(f"real (true-timestamp) contemporaneous r = {result.real_r:+.3f}  (n={result.n_real})")
    print(
        f"shuffle null: {result.n_perm} timestamp permutations, "
        f"mean r={result.perm_mean:+.3f}  std={result.perm_std:.3f}"
    )
    print(f"two-sided permutation p-value = {result.p_value:.4f}")
    if result.p_value < SIGNIFICANCE:
        print(
            f"WARNING: the real pairing is more extreme than {1 - SIGNIFICANCE:.0%} of timestamp-"
            "shuffled re-pairings - that is what look-ahead leakage would look like. Investigate "
            "before trusting any correlation this pipeline reports."
        )
        return 1
    print(
        "no evidence of timestamp-dependent leakage: the real pairing is statistically "
        "indistinguishable from a randomly re-timed one at the 5% level. Consistent with Day "
        "5-7's own null findings, not a separate confirmation that sentiment predicts returns here."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM, help="number of timestamp permutations")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed, for a reproducible report")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_perm, args.seed))


if __name__ == "__main__":
    main()
