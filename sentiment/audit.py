"""Day 8 CLI: the shuffled-timestamp leakage audit NEXT_STEPS.md's "Done
when" section names as the gate that decides whether this repo is finished.

    python -m sentiment.audit
    python -m sentiment.audit --shuffles 1000 --seed 7
    python -m sentiment.audit --live

Day 5's correlation between VADER ``compound`` and contemporaneous return
depends entirely on ``sentiment.market_hours.align_headline`` correctly
pairing each headline with the one trading session it could only have
reacted *after*. If that pairing were ever wrong - the exact bug class this
whole project exists to catch - a headline's apparent "predictive power"
would be coming from something other than genuine timing: ticker identity,
a scoring artifact, a leak.

The control: keep every headline's content and compound score exactly as
scored, but randomly reassign *which headline got which timestamp*. Re-run
the unmodified ``sentiment.correlate`` pairing and correlation on that
shuffled set. A shuffled timestamp carries no information about whether the
headline now attached to it was bullish or bearish, so across many
shuffles the resulting correlation should be noise centred on zero - *even
if the real, correctly-timed correlation is itself large*, because a real,
timing-dependent relationship is exactly what shuffling timestamps should
destroy.

If the shuffled-timestamp control's own 95% interval excludes zero instead
- a randomized timestamp still "predicting" returns as reliably as the real
one - the real result cannot be trusted: whatever produced it does not
depend on correct timing, so it is not evidence of the thing this pipeline
claims to measure.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_SHUFFLES = 500
DEFAULT_SEED = 0


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return headlines with the same content and the same compound score,
    but a random permutation of which headline carries which timestamp.

    Only ``published_at``/``published_raw`` move; title, link, source and
    scraped_at - everything the content and the scorer see - stay put. The
    *set* of timestamps (so the set of reachable trading sessions) is
    unchanged, only the pairing between a headline and a session is.
    """
    time_tags = [(h.published_at, h.published_raw) for h in headlines]
    rng.shuffle(time_tags)
    return [
        dataclasses.replace(h, published_at=pub_at, published_raw=pub_raw)
        for h, (pub_at, pub_raw) in zip(headlines, time_tags)
    ]


def contemporaneous_r(headlines: list[Headline], live: bool = False) -> tuple[float | None, int]:
    """Pearson r between VADER ``compound`` and contemporaneous return over
    whatever headlines resolve to a ticker with price data for their
    (possibly reassigned) session, or ``(None, n)`` if fewer than 2 rows
    resolve - too few to correlate."""
    rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(rows) < 2:
        return None, len(rows)
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, returns), len(rows)


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float | None
    real_n: int
    shuffled_rs: list[float]

    @property
    def n_shuffles(self) -> int:
        return len(self.shuffled_rs)

    @property
    def mean_shuffled(self) -> float:
        return sum(self.shuffled_rs) / len(self.shuffled_rs)

    @property
    def ci(self) -> tuple[float, float]:
        """95% percentile interval over the shuffled-timestamp r's - the
        null distribution a leak-free pipeline should produce."""
        ordered = sorted(self.shuffled_rs)
        n = len(ordered)
        lo_idx = int(0.025 * n)
        hi_idx = max(int(0.975 * n) - 1, lo_idx)
        return ordered[lo_idx], ordered[hi_idx]

    @property
    def leaking(self) -> bool:
        """True if the shuffled-timestamp control's own 95% interval
        excludes zero: a randomized timestamp still "predicts" returns,
        the exact failure NEXT_STEPS.md's Done when section names."""
        lo, hi = self.ci
        return lo > 0 or hi < 0

    @property
    def real_r_percentile(self) -> float | None:
        """Where the real (correctly-timed) r sits inside the shuffled
        null distribution, 0-1. Purely descriptive context for reading a
        null real result - the pass/fail gate is ``leaking``, not this."""
        if self.real_r is None or not self.shuffled_rs:
            return None
        below = sum(1 for s in self.shuffled_rs if s <= self.real_r)
        return below / len(self.shuffled_rs)


def run_shuffle_audit(
    headlines: list[Headline], live: bool, n_shuffles: int, seed: int
) -> ShuffleAuditResult:
    real_r, real_n = contemporaneous_r(headlines, live=live)
    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled = shuffle_timestamps(headlines, rng)
        r, _ = contemporaneous_r(shuffled, live=live)
        if r is not None:
            shuffled_rs.append(r)
    if not shuffled_rs:
        raise ValueError(
            "no shuffle produced 2+ resolvable headlines - cannot build a null distribution"
        )
    return ShuffleAuditResult(real_r=real_r, real_n=real_n, shuffled_rs=shuffled_rs)


def run(in_path: Path, live: bool, n_shuffles: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    try:
        result = run_shuffle_audit(headlines, live=live, n_shuffles=n_shuffles, seed=seed)
    except ValueError as exc:
        print(f"cannot run the audit: {exc}", file=sys.stderr)
        return 1

    if result.real_r is None:
        print(f"real: fewer than 2 headlines resolved (n={result.real_n}) - nothing to correlate")
    else:
        print(f"real (correctly-timed) contemporaneous r = {result.real_r:+.3f}  n={result.real_n}")

    lo, hi = result.ci
    print(
        f"{result.n_shuffles} shuffled-timestamp controls: mean r={result.mean_shuffled:+.3f}  "
        f"95% interval [{lo:+.3f}, {hi:+.3f}]"
    )
    if result.real_r_percentile is not None:
        print(f"real r sits at the {result.real_r_percentile:.0%} percentile of the shuffled null distribution")

    if result.leaking:
        print(
            "FAIL: the shuffled-timestamp control's own interval excludes zero - randomized "
            "timestamps still 'predict' returns. Something other than genuine timestamp "
            "alignment is driving this correlation; do not trust the real result until it is found."
        )
        return 1

    print(
        "PASS: shuffled-timestamp controls are consistent with zero - no evidence this "
        "pipeline's correlation depends on anything other than genuine timestamp alignment."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--shuffles", type=int, default=DEFAULT_SHUFFLES, help="number of timestamp-shuffle trials"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible CI run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.shuffles, args.seed))


if __name__ == "__main__":
    main()
