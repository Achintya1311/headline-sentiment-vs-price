"""Day 8 CLI: the leakage/audit pass NEXT_STEPS.md's "Done when" names as
decisive - shuffle headline timestamps and confirm whatever correlation
Day 5 found does not survive.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --n-shuffles 2000 --seed 7

Everything about a resolved headline except *which timestamp decides its
session_date* is held fixed: the same ticker, the same VADER ``compound``
score (text-derived, never time-derived), and the same price bars. Only
``published_at`` is permuted across the resolved headlines before Day 4's
real ``align_headline`` logic runs on it, so each trial re-pairs compound
scores with contemporaneous/lagged returns using a session_date no more
informed by a headline's actual timing than chance.

Repeating that ``--n-shuffles`` times builds a null distribution for the
same Pearson r Day 5 reported; the real (unshuffled) r is compared against
it with a two-sided permutation p-value. See README Findings for why a
pipeline that was already reporting a null result makes this a weaker test
than it sounds, and why passing it is not proof of no leak.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sentiment.headline import read_csv
from sentiment.market_hours import align_headline
from sentiment.prices import Bar, PriceFetchError, bar_on, load_bars, next_session_bar
from sentiment.stats import pearson_with_ci
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_N_SHUFFLES = 1000
DEFAULT_SEED = 0


@dataclass(frozen=True)
class ResolvedHeadline:
    """A headline with everything time-independent precomputed once: the
    shuffle loop below never re-resolves a ticker, re-scores text, or
    re-fetches prices - only the timestamp fed to ``align_headline`` varies
    between trials."""

    title: str
    ticker: str
    compound: float
    published_at: datetime
    bars: list[Bar]


def resolve_headlines(in_path: Path, live: bool = False) -> tuple[list[ResolvedHeadline], list[tuple[str, str]]]:
    headlines = read_csv(in_path)
    resolved: list[ResolvedHeadline] = []
    unresolved: list[tuple[str, str]] = []
    price_errors: list[tuple[str, str]] = []

    for h in headlines:
        match = resolve(h.title)
        if match is None:
            continue
        company, ticker = match
        if ticker is None:
            unresolved.append((h.title, company))
            continue
        try:
            bars = load_bars(ticker, live=live)
        except PriceFetchError as exc:
            price_errors.append((ticker, str(exc)))
            continue
        scored = score_headline(h)
        resolved.append(
            ResolvedHeadline(title=h.title, ticker=ticker, compound=scored.compound, published_at=h.published_at, bars=bars)
        )

    if price_errors:
        for ticker, msg in price_errors:
            print(f"warning: {ticker}: {msg}", file=sys.stderr)

    return resolved, unresolved


def pair_with_timestamps(resolved: list[ResolvedHeadline], timestamps: list[datetime]) -> list[dict]:
    """Align each ``resolved[i]`` against ``timestamps[i]`` - which may or
    may not be that headline's own ``published_at`` - and pair it with the
    session/next-session return. This is the one seam the shuffle test
    needs: pass each headline's real timestamp back for the "real" run, a
    permutation of all of them for a "shuffled" trial."""
    rows: list[dict] = []
    for rh, ts in zip(resolved, timestamps):
        alignment = align_headline(ts)
        session_bar = bar_on(rh.bars, alignment.session_date)
        if session_bar is None:
            continue
        lagged_bar = next_session_bar(rh.bars, alignment.session_date)
        rows.append(
            {
                "ticker": rh.ticker,
                "compound": rh.compound,
                "session_date": alignment.session_date,
                "contemporaneous_return": session_bar.session_return,
                "lagged_return": lagged_bar.session_return if lagged_bar else None,
            }
        )
    return rows


@dataclass(frozen=True)
class CorrelationStat:
    contemporaneous_r: float | None
    n_contemporaneous: int
    lagged_r: float | None
    n_lagged: int


def correlate_rows(rows: list[dict]) -> CorrelationStat:
    contemp_r = None
    if len(rows) >= 2:
        contemp_r = pearson_with_ci([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows]).r

    lagged_rows = [r for r in rows if r["lagged_return"] is not None]
    lagged_r = None
    if len(lagged_rows) >= 2:
        lagged_r = pearson_with_ci(
            [r["compound"] for r in lagged_rows], [r["lagged_return"] for r in lagged_rows]
        ).r

    return CorrelationStat(
        contemporaneous_r=contemp_r, n_contemporaneous=len(rows), lagged_r=lagged_r, n_lagged=len(lagged_rows)
    )


def permutation_p_value(real_r: float, shuffled_rs: list[float]) -> float:
    """Two-sided permutation p-value with the standard +1 continuity
    correction, so a real statistic no shuffle ever matched is reported as
    "rarer than 1 in n_shuffles+1", never as a false exact zero."""
    if not shuffled_rs:
        raise ValueError("need at least one shuffled trial")
    extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
    return (extreme + 1) / (len(shuffled_rs) + 1)


@dataclass(frozen=True)
class AuditResult:
    real: CorrelationStat
    shuffled_contemporaneous: list[float]
    shuffled_lagged: list[float]
    contemporaneous_p: float | None
    lagged_p: float | None


def run_audit(resolved: list[ResolvedHeadline], n_shuffles: int, seed: int) -> AuditResult:
    real_timestamps = [rh.published_at for rh in resolved]
    real_rows = pair_with_timestamps(resolved, real_timestamps)
    real_stat = correlate_rows(real_rows)

    rng = random.Random(seed)
    shuffled_contemporaneous: list[float] = []
    shuffled_lagged: list[float] = []
    for _ in range(n_shuffles):
        shuffled_timestamps = real_timestamps[:]
        rng.shuffle(shuffled_timestamps)
        shuffled_rows = pair_with_timestamps(resolved, shuffled_timestamps)
        stat = correlate_rows(shuffled_rows)
        if stat.contemporaneous_r is not None:
            shuffled_contemporaneous.append(stat.contemporaneous_r)
        if stat.lagged_r is not None:
            shuffled_lagged.append(stat.lagged_r)

    contemporaneous_p = (
        permutation_p_value(real_stat.contemporaneous_r, shuffled_contemporaneous)
        if real_stat.contemporaneous_r is not None and shuffled_contemporaneous
        else None
    )
    lagged_p = (
        permutation_p_value(real_stat.lagged_r, shuffled_lagged)
        if real_stat.lagged_r is not None and shuffled_lagged
        else None
    )

    return AuditResult(
        real=real_stat,
        shuffled_contemporaneous=shuffled_contemporaneous,
        shuffled_lagged=shuffled_lagged,
        contemporaneous_p=contemporaneous_p,
        lagged_p=lagged_p,
    )


def _format_metric(name: str, real_r: float | None, shuffled: list[float], p: float | None, n: int) -> str:
    if real_r is None or not shuffled or p is None:
        return f"{name}: not enough data for a permutation test (n={n})"
    lo, hi = min(shuffled), max(shuffled)
    mean_abs = sum(abs(r) for r in shuffled) / len(shuffled)
    return (
        f"{name}: real r={real_r:+.3f} (n={n})  "
        f"shuffled r range [{lo:+.3f}, {hi:+.3f}], mean|r|={mean_abs:.3f}  "
        f"permutation p={p:.3f}"
    )


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    resolved, unresolved = resolve_headlines(in_path, live=live)
    if len(resolved) < 2:
        print(f"not enough resolved headlines to audit (n={len(resolved)})", file=sys.stderr)
        return 1

    result = run_audit(resolved, n_shuffles=n_shuffles, seed=seed)

    print(f"{len(resolved)} headlines resolved to a ticker with price data (real timestamps)")
    if unresolved:
        names = ", ".join(f"{c} ({t[:40]}...)" for t, c in unresolved)
        print(f"{len(unresolved)} headline(s) named a company with no resolvable ticker: {names}")
    print(f"shuffles: {n_shuffles} (seed={seed})")

    print(
        _format_metric(
            "contemporaneous",
            result.real.contemporaneous_r,
            result.shuffled_contemporaneous,
            result.contemporaneous_p,
            result.real.n_contemporaneous,
        )
    )
    print(
        _format_metric(
            "lagged         ",
            result.real.lagged_r,
            result.shuffled_lagged,
            result.lagged_p,
            result.real.n_lagged,
        )
    )

    print(
        "verdict: a small p (real r is an outlier against the shuffled null) would mean the "
        "pipeline finds something that depends on genuine timing - not observed here. A large p "
        "means the real result cannot be told apart from shuffled noise, which is consistent with "
        "'no detectable signal' but is not, by itself, proof the pipeline is leak-free - see README "
        "Limitations for why."
    )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp-shuffle trials")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for the shuffles (deterministic by default)")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
