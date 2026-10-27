"""Day 84 (hub CRITIQUES.md R18): the pre-registered follow-up test's data
gate and statistic.

    python -m sentiment.prereg
    python -m sentiment.prereg --live

Days 5-8 found no result on 23-50 headlines; the README's "Pre-registered
follow-up test" section fixed one further test in advance, before any of
this data existed, and said it would not be run - let alone reported as an
``r`` number - below a stated data bar. This module is that one test, kept
out of ``sentiment.correlate`` so the already-reported Day 5 VADER/raw-return
numbers are never confused with it.

The gate (see README): at least 1,000 headlines resolved to a priced ticker
(after excluding ones that name their own price move and, since R28, the
liveblog template titles), over at least 30
distinct sessions, from at least 3 distinct sources. ``docs/POWER.md`` is
why the gate moved there from the original, underpowered 100/10/no-source-
count bar.

Checking the gate never needs FinBERT: a headline's resolution, its price
pairing and whether its title names a price move are all independent of
which scorer eventually reads it. FinBERT - the heavier, network-on-first-
use dependency (see README Limitations) - is only invoked in
``run_prereg_test`` once the gate has already been confirmed met, which it
is not as of this day's run.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.finbert_score import score_texts
from sentiment.headline import read_csv
from sentiment.market import equal_weighted_daily_returns, load_universe_bars
from sentiment.overlay import abnormal_return_0_1
from sentiment.prices import load_bars
from sentiment.robustness import names_price_move, session_demeaned_r
from sentiment.stats import PearsonResult, pearson_with_ci

PREREG_MIN_HEADLINES = 1000
PREREG_MIN_SESSIONS = 30
PREREG_MIN_SOURCES = 3

# R28 (amended 27 Oct 2026, before the statistic was ever computed): the
# Economic Times "<Company> Share Price Highlights: <Company> Stock Price
# History" liveblog titles carry no news, and FinBERT reads every one as
# neutral (README Day 3). They were 29 of the 33 headlines passing the gate
# on 27 Oct, so leaving them in would let a met gate correlate a constant.
_LIVEBLOG_TEMPLATE = re.compile(r"Share Price Highlights:.*Stock Price History", re.IGNORECASE)


def is_liveblog_template(title: str) -> bool:
    return bool(_LIVEBLOG_TEMPLATE.search(title))


@dataclass(frozen=True)
class GateStatus:
    met: bool
    n_headlines: int
    n_sessions: int
    n_sources: int
    reasons: tuple[str, ...]


def check_gate(rows: list[dict]) -> GateStatus:
    """``rows`` must already be resolved, priced, price-naming-excluded and
    have a non-``None`` ``abnormal_return_0_1`` - i.e. exactly the usable
    set the pre-registered statistic would be computed on."""
    n_headlines = len(rows)
    n_sessions = len({r["session_date"] for r in rows})
    n_sources = len({r["source"] for r in rows})
    reasons = []
    if n_headlines < PREREG_MIN_HEADLINES:
        reasons.append(f"{n_headlines}/{PREREG_MIN_HEADLINES} headlines")
    if n_sessions < PREREG_MIN_SESSIONS:
        reasons.append(f"{n_sessions}/{PREREG_MIN_SESSIONS} sessions")
    if n_sources < PREREG_MIN_SOURCES:
        reasons.append(f"{n_sources}/{PREREG_MIN_SOURCES} sources")
    return GateStatus(met=not reasons, n_headlines=n_headlines, n_sessions=n_sessions, n_sources=n_sources, reasons=tuple(reasons))


def usable_rows(headlines: list, live: bool = False) -> list[dict]:
    """Headlines resolved to a priced ticker and session, minus ones that
    name their own price move or are liveblog template titles (R28), with
    each remaining row's ``[0, +1]``
    market-adjusted abnormal return attached - or dropped if that window
    isn't available yet. VADER's default scorer runs here only for its
    resolution/pairing side effect; its ``compound`` is discarded and
    overwritten with FinBERT's once (if) the gate is met, in
    ``run_prereg_test``."""
    return usable_rows_with_counts(headlines, live=live)[0]


def usable_rows_with_counts(headlines: list, live: bool = False) -> tuple[list[dict], int]:
    """:func:`usable_rows` plus how many resolved headlines were dropped as
    liveblog template titles (R28), so the gate line can say so."""
    rows, _, _ = build_rows_from_headlines(headlines, live=live)
    n_templates = sum(1 for r in rows if is_liveblog_template(r["title"]))
    candidates = [r for r in rows if not names_price_move(r["title"]) and not is_liveblog_template(r["title"])]
    if not candidates:
        return [], n_templates

    tickers = sorted({r["ticker"] for r in candidates})
    bars_by_ticker = {t: load_bars(t, live=live) for t in tickers}
    market_returns = equal_weighted_daily_returns(load_universe_bars())

    usable = []
    for r in candidates:
        bars = bars_by_ticker[r["ticker"]]
        event_date = date.fromisoformat(r["session_date"])
        abnormal = abnormal_return_0_1(bars, event_date, market_returns)
        if abnormal is None:
            continue
        row = dict(r)
        row["abnormal_return_0_1"] = abnormal
        usable.append(row)
    return usable, n_templates


@dataclass(frozen=True)
class PreregResult:
    gate: GateStatus
    stat: PearsonResult | None


def run_prereg_test(headlines: list, live: bool = False) -> PreregResult:
    """Check the gate; only score and correlate against FinBERT if it is met."""
    rows = usable_rows(headlines, live=live)
    gate = check_gate(rows)
    if not gate.met:
        return PreregResult(gate=gate, stat=None)

    finbert_scores = score_texts([r["title"] for r in rows])
    for r, s in zip(rows, finbert_scores):
        r["compound"] = s["positive"] - s["negative"]

    demeaned_compound = []
    demeaned_abnormal = []
    by_day: dict[str, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r["session_date"], []).append(r)
    means = {d: sum(x["abnormal_return_0_1"] for x in rs) / len(rs) for d, rs in by_day.items()}
    for r in rows:
        demeaned_compound.append(r["compound"])
        demeaned_abnormal.append(r["abnormal_return_0_1"] - means[r["session_date"]])

    stat = pearson_with_ci(demeaned_compound, demeaned_abnormal)
    return PreregResult(gate=gate, stat=stat)


def run(in_path: Path, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = run_prereg_test(headlines, live=live)
    gate = result.gate
    _, n_templates = usable_rows_with_counts(headlines, live=live)

    print(
        f"pre-registered follow-up test gate: {gate.n_headlines} headlines, "
        f"{gate.n_sessions} sessions, {gate.n_sources} source(s) "
        f"(need >={PREREG_MIN_HEADLINES}/{PREREG_MIN_SESSIONS}/{PREREG_MIN_SOURCES}); "
        f"{n_templates} liveblog template title(s) excluded (R28)"
    )
    if not gate.met:
        print(
            "underpowered, not null: gate not met (" + ", ".join(gate.reasons) + "). "
            "See docs/POWER.md and README's Pre-registered follow-up test section. "
            "r is not reported until the gate is met."
        )
        return 0

    stat = result.stat
    print(
        f"FinBERT compound vs [0,+1] abnormal return, session-demeaned: "
        f"r={stat.r:+.3f}  95% CI [{stat.ci_low:+.3f}, {stat.ci_high:+.3f}]  n={stat.n}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live))


if __name__ == "__main__":
    main()
