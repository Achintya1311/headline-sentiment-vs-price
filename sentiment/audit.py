"""Day 8: shuffled-timestamp leakage audit, plus a report on why the honest
Day 5/6/7 null result might still be spurious.

This is the "Done when" gate NEXT_STEPS.md set before Day 1 was written: take
every headline's ``published_at``, randomly reassign those timestamps among
the headlines (same values, different pairing), rebuild the correlation
pipeline exactly as ``sentiment.correlate`` does, and see what Pearson r that
produces. Do this many times to build a null distribution, then ask: does the
*real* (true-timestamp) correlation look different from what random timing
alone can produce?

Two honest readings this audit can give, and both are legitimate:

- The real result is not statistically significant in the first place (its
  95% CI contains zero) - which is exactly what Day 5 and Day 6 found on this
  fixture. There is no claimed signal for a leak to have produced, so the
  audit has nothing to fail on. This is the current, actual state.
- The real result *is* significant, and is also no more extreme than the
  shuffled null distribution produces by chance (a permutation-test p-value
  >= 0.05) - meaning true timestamp alignment does not matter to the strength
  of the correlation, which is the signature of a leak: something other than
  genuine, leak-free timing is driving the number. That is a hard CI failure.

A significant real result that *is* a clear outlier against the shuffled
null (p < 0.05) would be the one case this audit lets through as a genuine,
leak-free signal - this fixture has never produced that case, so the audit
below is demonstrated against synthetic data instead (see
``tests/test_audit.py``): a synthetic leaking pipeline where the "signal" is
really just sorted compound scores lined up against sorted returns regardless
of true timing (shuffling does *not* kill it - the audit must fail), and a
synthetic genuine-signal fixture where compound score really is wired to the
correctly-aligned session's return (shuffling *does* kill it - the audit must
pass). Running the same machinery against the real committed fixture is still
useful even though the real result is null: a null result could itself be an
artifact of a broken shuffle (e.g. a bug that makes shuffling a no-op), and
this audit's structural check (``shuffle_changes_alignment``) catches that
independently of what the correlation happens to be.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.market_hours import align_headline
from sentiment.stats import pearson_r, pearson_with_ci

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "outputs" / "shuffle_audit.json"
DEFAULT_N_SHUFFLES = 1000
DEFAULT_SEED = 0


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return headlines with the same set of ``published_at`` values, randomly
    reassigned to different headlines. Everything else (title, source, link,
    scraped_at) stays put - only the pairing between a headline's content and
    its publish time changes, which is exactly what the alignment and
    correlation pipeline downstream is sensitive to."""
    timestamps = [h.published_at for h in headlines]
    rng.shuffle(timestamps)
    return [
        Headline(
            source=h.source,
            title=h.title,
            link=h.link,
            published_at=ts,
            published_raw=f"shuffled:{ts.isoformat()}",
            scraped_at=h.scraped_at,
        )
        for h, ts in zip(headlines, timestamps)
    ]


def shuffle_changes_alignment(real: list[Headline], shuffled: list[Headline]) -> bool:
    """True if at least one headline's aligned session_date actually moved.

    A structural sanity check, not a statistical one (see
    ``stockstalker.audit``'s causality check for the same idea applied to a
    different pipeline): if shuffling the timestamps never changes a single
    headline's session_date, the shuffle - or the alignment step it is meant
    to stress - is not doing anything, and any "signal disappears" or
    "signal survives" reading from the correlation step downstream would be
    meaningless.
    """
    real_sessions = [align_headline(h.published_at).session_date for h in real]
    shuffled_sessions = [align_headline(h.published_at).session_date for h in shuffled]
    return real_sessions != shuffled_sessions


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float | None
    real_n: int
    real_ci_low: float | None
    real_ci_high: float | None
    real_significant: bool
    n_shuffles_requested: int
    n_shuffles_effective: int
    shuffled_rs: list[float]
    shuffled_ns: list[int]
    mean_shuffled_r: float | None
    std_shuffled_r: float | None
    p_value: float | None
    alignment_changed_every_shuffle: bool
    leak_suspected: bool

    def to_json(self) -> dict:
        return asdict(self)


def _std(xs: list[float]) -> float:
    n = len(xs)
    mean = sum(xs) / n
    return (sum((x - mean) ** 2 for x in xs) / n) ** 0.5


def run_shuffle_audit(
    in_path: Path,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
    live: bool = False,
) -> ShuffleAuditResult:
    headlines = read_csv(in_path)
    bars_cache: dict = {}

    real_rows, _ = build_rows_from_headlines(headlines, live=live, bars_cache=bars_cache, report_warnings=False)
    real_compound = [r["compound"] for r in real_rows]
    real_return = [r["contemporaneous_return"] for r in real_rows]

    real_r: float | None = None
    real_ci_low: float | None = None
    real_ci_high: float | None = None
    real_significant = False
    if len(real_rows) >= 2:
        real_r = pearson_r(real_compound, real_return)
    if len(real_rows) >= 4:
        stat = pearson_with_ci(real_compound, real_return)
        real_ci_low, real_ci_high = stat.ci_low, stat.ci_high
        real_significant = not (real_ci_low <= 0 <= real_ci_high)

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    shuffled_ns: list[int] = []
    alignment_changed_every_shuffle = True

    for _ in range(n_shuffles):
        shuffled_headlines = shuffle_timestamps(headlines, rng)
        if not shuffle_changes_alignment(headlines, shuffled_headlines):
            alignment_changed_every_shuffle = False
        rows, _ = build_rows_from_headlines(
            shuffled_headlines, live=False, bars_cache=bars_cache, report_warnings=False
        )
        if len(rows) < 2:
            continue
        r = pearson_r([row["compound"] for row in rows], [row["contemporaneous_return"] for row in rows])
        shuffled_rs.append(r)
        shuffled_ns.append(len(rows))

    mean_shuffled_r = sum(shuffled_rs) / len(shuffled_rs) if shuffled_rs else None
    std_shuffled_r = _std(shuffled_rs) if len(shuffled_rs) >= 2 else None

    p_value: float | None = None
    if real_r is not None and shuffled_rs:
        as_extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
        p_value = (as_extreme + 1) / (len(shuffled_rs) + 1)

    # The only failure mode this audit can detect: a real, statistically
    # significant correlation that random timestamps reproduce just as
    # easily (p >= 0.05 against the shuffled null). A real result that was
    # never significant has no claimed signal for a leak to have produced.
    leak_suspected = bool(real_significant and p_value is not None and p_value >= 0.05)

    return ShuffleAuditResult(
        real_r=real_r,
        real_n=len(real_rows),
        real_ci_low=real_ci_low,
        real_ci_high=real_ci_high,
        real_significant=real_significant,
        n_shuffles_requested=n_shuffles,
        n_shuffles_effective=len(shuffled_rs),
        shuffled_rs=shuffled_rs,
        shuffled_ns=shuffled_ns,
        mean_shuffled_r=mean_shuffled_r,
        std_shuffled_r=std_shuffled_r,
        p_value=p_value,
        alignment_changed_every_shuffle=alignment_changed_every_shuffle,
        leak_suspected=leak_suspected,
    )


def print_report(result: ShuffleAuditResult) -> None:
    print(f"shuffle audit: {result.n_shuffles_effective}/{result.n_shuffles_requested} shuffles produced a valid run")
    if not result.alignment_changed_every_shuffle:
        print(
            "  WARNING: at least one shuffle left every headline's session_date unchanged - "
            "the shuffle may not be exercising the alignment step"
        )
    if result.real_r is None:
        print(f"  real: not enough resolved headlines for a correlation (n={result.real_n})")
    else:
        sig = "significant" if result.real_significant else "not significant"
        ci = (
            f" 95% CI [{result.real_ci_low:+.3f}, {result.real_ci_high:+.3f}]"
            if result.real_ci_low is not None
            else ""
        )
        print(f"  real:     r={result.real_r:+.3f}{ci}  n={result.real_n}  ({sig})")
    if result.mean_shuffled_r is not None:
        print(
            f"  shuffled: mean r={result.mean_shuffled_r:+.3f}  std={result.std_shuffled_r:.3f}  "
            f"over {result.n_shuffles_effective} shuffles"
        )
    if result.p_value is not None:
        print(f"  permutation p-value (|shuffled r| >= |real r|): {result.p_value:.4f}")

    if result.real_r is None:
        print("  verdict: inconclusive - too few resolved headlines to correlate at all")
    elif not result.real_significant:
        print(
            "  verdict: PASS (vacuously) - the real result was never statistically significant, "
            "so there is no claimed signal for a shuffle to have leaked into existence. "
            "See README Limitations: this audit has not yet been exercised by a real, "
            "significant finding on this fixture."
        )
    elif result.leak_suspected:
        print(
            "  verdict: FAIL - the real result is significant, but random timestamps reproduce "
            "correlations this strong just as easily (p >= 0.05). True timing does not matter to "
            "the result, which is the signature of a leak."
        )
    else:
        print(
            "  verdict: PASS - the real result is significant and is a clear outlier against the "
            "shuffled null distribution: genuine timestamp alignment matters to it."
        )


def write_json(result: ShuffleAuditResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2) + "\n", encoding="utf-8")


def run(in_path: Path, out_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    result = run_shuffle_audit(in_path, n_shuffles=n_shuffles, seed=seed, live=live)
    write_json(result, out_path)
    print_report(result)

    if not result.alignment_changed_every_shuffle:
        return 1
    return 1 if result.leak_suspected else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT, help="JSON report to write")
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp shuffles")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible CI run")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.out_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
