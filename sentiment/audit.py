"""Day 8 CLI: ml-pipeline-audit - shuffle headline timestamps and check
whether the sentiment/return correlation depends on genuine time alignment.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --n-shuffles 5000 --seed 1

## What this tests

Day 4's ``market_hours.align_headline`` exists to guarantee that a headline
is only ever paired with a trading session whose open happened *after* the
headline was published - no look-ahead. Day 5 then correlated VADER's
``compound`` score against that session's return and found essentially
nothing (r=-0.185, 95% CI comfortably straddling zero, n=23).

This module is the check called for in NEXT_STEPS.md's "Done when": take
every headline that Day 5's pipeline resolves to a ticker, and instead of
using its real ``published_at``, randomly reassign timestamps *among* the
resolved headlines (keeping each headline's own title/ticker/compound
score fixed). Re-running ``market_hours.align_headline`` on a shuffled
timestamp sends the headline to an arbitrary, almost certainly wrong,
trading session - the pairing a genuinely leak-free pipeline should not be
able to exploit.

Repeating that shuffle many times builds an empirical null distribution of
the correlation statistic under "the timestamp carries no information".
The real (correctly-aligned) correlation is then compared against that
null via a two-sided permutation p-value. The leakage test **passes** when
the real correlation is *not* a standout relative to that null - i.e.
wrong timestamps predict about as well (or as badly) as the real one. If
wrong timestamps could reliably score notably *higher* than the null
typically does while the real correlation cannot, or if the real
correlation stands out as unusually extreme relative to shuffled controls
in a way this project's Day 5-7 findings give no reason to expect, that is
the signature of a bug leaking information through some channel other than
correct time alignment (e.g. a caching bug that returns the same bar
regardless of the date requested), and needs investigating before trusting
any number this pipeline produces.

## Why "passes" here does not mean "found a real effect"

Given Day 5's own real correlation is already statistically
indistinguishable from zero, the honest expectation is that this audit
finds the real value unremarkable against the shuffled null too - not
because the audit is powerful enough to rule out leakage, but because
there was barely a signal to leak in the first place. See the README's
Day 8 section for why a PASS here is a weak, not a strong, form of
reassurance on this sample size.
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
from sentiment.prices import Bar, PriceFetchError, bar_on, load_bars
from sentiment.stats import pearson_r
from sentiment.tickers import resolve
from sentiment.vader_score import score_headline

DEFAULT_IN = Path(__file__).resolve().parent.parent / "fixtures" / "headlines" / "headlines_raw.csv"
DEFAULT_N_SHUFFLES = 2000
DEFAULT_SEED = 0
ALPHA = 0.05


@dataclass(frozen=True)
class ResolvedHeadline:
    title: str
    ticker: str
    compound: float
    published_at: datetime


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    n_real: int
    null_rs: list[float]
    p_value: float

    @property
    def passes(self) -> bool:
        """True if the real correlation is not a statistical outlier against
        the shuffled-timestamp null - see the module docstring for why this
        is a weak, not a strong, form of reassurance here."""
        return self.p_value > ALPHA

    @property
    def null_mean(self) -> float:
        return sum(self.null_rs) / len(self.null_rs)

    @property
    def null_percentile(self) -> float:
        """Where |real_r| falls in the sorted |null_rs|, as a fraction in [0, 1]."""
        target = abs(self.real_r)
        below = sum(1 for r in self.null_rs if abs(r) <= target)
        return below / len(self.null_rs)


def resolve_headlines(in_path: Path) -> list[ResolvedHeadline]:
    """Score and resolve every headline once, up front - shuffling only ever
    touches ``published_at`` after this point, never the title, ticker, or
    compound score."""
    resolved: list[ResolvedHeadline] = []
    for h in read_csv(in_path):
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


def _bars_by_ticker(resolved: list[ResolvedHeadline], live: bool) -> dict[str, list[Bar] | None]:
    """Load each distinct ticker's bars once. ``None`` marks a ticker whose
    bars could not be loaded at all, so every shuffle can skip it cheaply."""
    cache: dict[str, list[Bar] | None] = {}
    for r in resolved:
        if r.ticker in cache:
            continue
        try:
            cache[r.ticker] = load_bars(r.ticker, live=live)
        except PriceFetchError:
            cache[r.ticker] = None
    return cache


def correlation_for_timestamps(
    resolved: list[ResolvedHeadline],
    timestamps: list[datetime],
    bars_by_ticker: dict[str, list[Bar] | None],
) -> tuple[float, int] | None:
    """Pearson r between each headline's (fixed) compound score and the
    contemporaneous return ``timestamps[i]`` (not necessarily ``resolved[i]
    .published_at``) aligns it to. Returns ``None`` if fewer than 2 headlines
    end up with a resolvable return."""
    xs: list[float] = []
    ys: list[float] = []
    for r, ts in zip(resolved, timestamps):
        bars = bars_by_ticker.get(r.ticker)
        if bars is None:
            continue
        session_date = align_headline(ts).session_date
        bar = bar_on(bars, session_date)
        if bar is None:
            continue
        xs.append(r.compound)
        ys.append(bar.session_return)
    if len(xs) < 2:
        return None
    return pearson_r(xs, ys), len(xs)


def run_audit(
    resolved: list[ResolvedHeadline],
    live: bool = False,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
) -> AuditResult:
    if len(resolved) < 2:
        raise ValueError(f"need at least 2 resolved headlines to audit, got {len(resolved)}")

    bars_by_ticker = _bars_by_ticker(resolved, live=live)

    real = correlation_for_timestamps(resolved, [r.published_at for r in resolved], bars_by_ticker)
    if real is None:
        raise ValueError("fewer than 2 headlines have a resolvable contemporaneous return; cannot audit")
    real_r, n_real = real

    rng = random.Random(seed)
    timestamps = [r.published_at for r in resolved]
    null_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = timestamps[:]
        rng.shuffle(shuffled)
        result = correlation_for_timestamps(resolved, shuffled, bars_by_ticker)
        if result is not None:
            null_rs.append(result[0])

    if not null_rs:
        raise ValueError("no shuffle produced 2+ resolvable returns; cannot build a null distribution")

    at_least_as_extreme = sum(1 for r in null_rs if abs(r) >= abs(real_r))
    p_value = (at_least_as_extreme + 1) / (len(null_rs) + 1)

    return AuditResult(real_r=real_r, n_real=n_real, null_rs=null_rs, p_value=p_value)


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    resolved = resolve_headlines(in_path)
    if len(resolved) < 2:
        print(f"only {len(resolved)} headline(s) resolved to a ticker; nothing to audit", file=sys.stderr)
        return 1

    try:
        result = run_audit(resolved, live=live, n_shuffles=n_shuffles, seed=seed)
    except ValueError as exc:
        print(f"cannot audit: {exc}", file=sys.stderr)
        return 1

    print(f"real (leak-free) alignment: r={result.real_r:+.3f}  n={result.n_real}")
    print(
        f"shuffled-timestamp null ({len(result.null_rs)} shuffles): "
        f"mean r={result.null_mean:+.3f}  |real r| sits at the {result.null_percentile:.0%} "
        f"percentile of |null r|"
    )
    print(f"two-sided permutation p-value: {result.p_value:.4f}")
    if result.passes:
        print(
            "PASS: the real correlation is not a statistical outlier against shuffled-timestamp "
            "controls - wrong timestamps predict about as well as the real one, i.e. there is no "
            "detectable signal for a timing leak to have inflated. See the README's Day 8 section "
            "for why this is a weak, not a strong, form of reassurance on n={}.".format(result.n_real)
        )
    else:
        print(
            "FAIL: the real correlation is a statistical outlier against shuffled-timestamp "
            "controls in a way Day 5-7's own findings give no reason to expect - investigate for "
            "a leak before trusting this number."
        )
    return 0 if result.passes else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp-shuffle permutations to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic CI run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed))


if __name__ == "__main__":
    main()
