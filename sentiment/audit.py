"""Day 8 CLI: the shuffled-timestamp leakage audit this repo's README has
promised since Day 1 ("Correctness gate" / "Done when").

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500      # faster, noisier
    python -m sentiment.audit --alpha 0.10

What it does: take the same headlines Day 5's ``sentiment.correlate`` pairs
with returns, and run the *exact same* pairing pipeline many times with
every headline's ``published_at`` randomly reassigned from another headline
in the same set first (the multiset of real timestamps is unchanged - only
which headline gets which one is). Each shuffled headline keeps its own
title, company, ticker and VADER score; only Day 4's alignment - and
therefore the contemporaneous/lagged return it ends up paired with -
changes.

Why this is the right control: the contemporaneous/lagged correlation only
means something if it depends on headlines being paired with the session
their *own* real timestamp leak-free-aligns to. Breaking that pairing at
random should make the correlation behave like statistical noise. A
shuffled run that still comes back "significant" (its 95% CI excludes
zero) far more often than a true null would by chance - 5% of the time, at
the same 95% level ``sentiment.stats.pearson_with_ci`` already uses
elsewhere in this repo - means something other than correct timing is
driving the result: a leak in alignment, ticker resolution, or the universe
itself, not a real reaction to news.

This fixture's real contemporaneous/lagged correlations are already null
(see README Day 5 Findings: r=-0.185 and r=+0.000, both CIs containing
zero), so this audit mostly proves the gate is wired correctly rather than
catching a leak in the act - a "why this might be spurious" section in the
README names the ways a *real-looking* result from this same pipeline could
still be an artifact even if this control passed.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_with_ci

DEFAULT_N_SHUFFLES = 2000
DEFAULT_SEED = 0
DEFAULT_ALPHA = 0.05
# A correctly-wired shuffled control should trip its own alpha-level
# significance test about `alpha` of the time, by chance alone. Tolerate up
# to twice that before calling it a leak - a named, explicit judgment call
# (like the event-study threshold and train/test split elsewhere in this
# repo), not a hidden one.
TOLERANCE_MULTIPLIER = 2.0
MIN_N_FOR_CI = 4  # pearson_with_ci's own floor below which its CI is fixed at (-1, 1)


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Reassign the same multiset of ``published_at`` values across
    ``headlines`` at random. Title, link, company and sentiment stay with
    their original headline - only which trading session it gets aligned to
    (via the reassigned timestamp) changes."""
    shuffled_ts = [h.published_at for h in headlines]
    rng.shuffle(shuffled_ts)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=new_ts,
            published_raw=h.published_raw,
            scraped_at=h.scraped_at,
        )
        for h, new_ts in zip(headlines, shuffled_ts)
    ]


def _resolved_pairs(rows: list[dict], return_key: str) -> tuple[list[float], list[float]]:
    usable = [r for r in rows if r[return_key] is not None]
    return [r["compound"] for r in usable], [r[return_key] for r in usable]


@dataclass(frozen=True)
class StatAudit:
    name: str
    real_r: float | None
    real_n: int
    real_significant: bool
    n_shuffles_usable: int
    significant_rate: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.significant_rate <= self.tolerance


@dataclass(frozen=True)
class AuditReport:
    contemporaneous: StatAudit
    lagged: StatAudit

    @property
    def passed(self) -> bool:
        return self.contemporaneous.passed and self.lagged.passed


def audit_statistic(
    real_rows: list[dict],
    shuffled_rows_list: list[list[dict]],
    return_key: str,
    name: str,
    alpha: float,
    tolerance_multiplier: float,
) -> StatAudit:
    real_xs, real_ys = _resolved_pairs(real_rows, return_key)
    if len(real_xs) >= 2:
        real_stat = pearson_with_ci(real_xs, real_ys)
        real_r = real_stat.r
        real_significant = not (real_stat.ci_low <= 0.0 <= real_stat.ci_high)
    else:
        real_r = None
        real_significant = False

    n_usable = 0
    n_significant = 0
    for shuffled_rows in shuffled_rows_list:
        xs, ys = _resolved_pairs(shuffled_rows, return_key)
        if len(xs) < MIN_N_FOR_CI:
            # Below this, pearson_with_ci always reports (-1, 1) - "not
            # significant" by construction, not by evidence. Skip it rather
            # than padding the rate with guaranteed non-significant results.
            continue
        stat = pearson_with_ci(xs, ys)
        n_usable += 1
        if not (stat.ci_low <= 0.0 <= stat.ci_high):
            n_significant += 1

    significant_rate = n_significant / n_usable if n_usable else 0.0
    return StatAudit(
        name=name,
        real_r=real_r,
        real_n=len(real_xs),
        real_significant=real_significant,
        n_shuffles_usable=n_usable,
        significant_rate=significant_rate,
        tolerance=alpha * tolerance_multiplier,
    )


def run(
    in_path: Path,
    live: bool,
    n_shuffles: int,
    seed: int,
    alpha: float,
    tolerance_multiplier: float = TOLERANCE_MULTIPLIER,
) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    price_cache: dict = {}
    real_rows, _ = build_rows_from_headlines(headlines, live=live, price_cache=price_cache)
    if not real_rows:
        print("no headlines resolved to a fetchable ticker; nothing to audit", file=sys.stderr)
        return 1

    rng = random.Random(seed)
    shuffled_rows_list = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=False, price_cache=price_cache)
        shuffled_rows_list.append(rows)

    contemporaneous = audit_statistic(
        real_rows, shuffled_rows_list, "contemporaneous_return", "contemporaneous", alpha, tolerance_multiplier
    )
    lagged = audit_statistic(
        real_rows, shuffled_rows_list, "lagged_return", "lagged", alpha, tolerance_multiplier
    )
    report = AuditReport(contemporaneous=contemporaneous, lagged=lagged)

    print(f"{len(real_rows)} resolved headlines, {n_shuffles} timestamp shuffles (seed={seed})")
    for stat in (contemporaneous, lagged):
        real_r_str = f"{stat.real_r:+.3f}" if stat.real_r is not None else "n/a"
        print(
            f"{stat.name}: real r={real_r_str} (n={stat.real_n}, "
            f"{'significant' if stat.real_significant else 'not significant'} at {alpha:.0%}) | "
            f"shuffled-control significance rate={stat.significant_rate:.1%} "
            f"over {stat.n_shuffles_usable} usable shuffle(s) "
            f"(expected ~{alpha:.0%} by chance, tolerance {stat.tolerance:.0%}) -> "
            f"{'PASS' if stat.passed else 'FAIL'}"
        )

    if report.passed:
        print("leakage audit: PASS - a shuffled-timestamp control does not predict returns more than chance allows.")
    else:
        print(
            "leakage audit: FAIL - a shuffled-timestamp control is 'significant' far more often than chance allows; "
            "the real result does not depend on correct timestamps and should not be trusted.",
            file=sys.stderr,
        )
    return 0 if report.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of random timestamp shuffles to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic CI run")
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA, help="significance level for each shuffle's own CI test"
    )
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.n_shuffles, args.seed, args.alpha))


if __name__ == "__main__":
    main()
