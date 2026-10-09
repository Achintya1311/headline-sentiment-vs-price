"""Day 8 CLI: shuffled-timestamp leakage control (the "Correctness gate").

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --n-perm 500

NEXT_STEPS.md's "Done when" is specific: *"shuffle the headline timestamps
and the signal must disappear. If a shuffled-timestamp control still
predicts returns, the pipeline is leaking."* That is only a meaningful test
when there is a claimed signal to begin with - Day 5-7 already found none
(every reported 95% CI here includes zero), and a shuffle test that always
reports "no signal" regardless of whether the pipeline is sound would be
worthless as a gate. So each statistic below is audited in two steps:

1. **Is the real (correctly time-aligned) result even notable?** - its own
   95% CI (Day 5's Fisher-z CI for the correlation, Day 5's bootstrap CI for
   the event-study diff) must exclude zero. If it already includes zero,
   there is nothing for the shuffle to disprove and the check passes
   vacuously - this is exactly this repo's actual situation today.
2. **If it is notable, does it survive shuffling?** ``published_at`` is
   permuted across the same headlines (title, company resolution, and
   VADER score all travel with the headline - only which trading session
   Day 4's alignment pairs it with changes) ``--n-perm`` times, rebuilding
   the statistic each time. A real, alignment-dependent effect should
   mostly disappear under that control; a leak (e.g. a ticker-level
   confound that has nothing to do with the specific session) survives it.
   The check fails when a shuffled-timestamp rebuild reproduces a
   statistic at least as extreme as the real one too often to be chance
   (empirical two-sided p > 0.05).

``run()`` exits non-zero if either statistic is notable but fails to
disappear under shuffling, so this doubles as the CI gate NEXT_STEPS.md
requires - see also ``test_audit.py``'s synthetic leak/no-leak fixtures,
which exercise both branches directly, not just this repo's own (null)
result.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_EVENT_THRESHOLD, DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import bootstrap_mean_diff_ci, pearson_r, pearson_with_ci

DEFAULT_N_PERM = 200
DEFAULT_SEED = 0
P_VALUE_THRESHOLD = 0.05
MIN_RESOLVED_FOR_CORRELATION = 4


@dataclass(frozen=True)
class AuditOutcome:
    name: str
    real_stat: float | None
    notable: bool
    n_null: int
    p_value: float | None
    passed: bool
    reason: str


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list with ``published_at`` permuted across ``headlines``.

    Everything else about a headline (title, link, scraped_at) stays with
    it, so company/ticker resolution and the VADER score are unaffected -
    only the trading session Day 4's alignment pairs it with changes. That
    pairing is the one thing a real leak would have to route around.
    """
    shuffled_ts = [h.published_at for h in headlines]
    rng.shuffle(shuffled_ts)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_ts)]


def contemporaneous_r(rows: list[dict]) -> float | None:
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def event_diff(rows: list[dict], threshold: float) -> float | None:
    high = [r["contemporaneous_return"] for r in rows if abs(r["compound"]) > threshold]
    low = [r["contemporaneous_return"] for r in rows if abs(r["compound"]) <= threshold]
    if not high or not low:
        return None
    return sum(high) / len(high) - sum(low) / len(low)


def _notable(ci_low: float, ci_high: float) -> bool:
    """A 95% CI that excludes zero - i.e. an actual claim, not noise."""
    return not (ci_low <= 0 <= ci_high)


def _permutation_p_value(
    real_value: float,
    headlines: list[Headline],
    statistic,
    live: bool,
    n_perm: int,
    seed: int,
) -> tuple[float | None, int, int]:
    """Two-sided empirical p-value: the fraction of timestamp-shuffled
    rebuilds whose |statistic| is at least as extreme as the real one's,
    plus the real draw itself (add-one smoothing - p can't honestly be
    exactly 0 off a finite permutation count). Returns (p_value, n_null,
    n_extreme); p_value is None if no permutation produced a usable value.
    """
    rng = random.Random(seed)
    null_values: list[float] = []
    for _ in range(n_perm):
        shuffled = shuffle_timestamps(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=live)
        value = statistic(rows)
        if value is not None:
            null_values.append(value)

    if not null_values:
        return None, 0, 0

    n_extreme = sum(1 for v in null_values if abs(v) >= abs(real_value))
    p_value = (n_extreme + 1) / (len(null_values) + 1)
    return p_value, len(null_values), n_extreme


def audit_correlation(
    headlines: list[Headline],
    live: bool = False,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
) -> AuditOutcome:
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    xs = [r["compound"] for r in real_rows]
    ys = [r["contemporaneous_return"] for r in real_rows]
    name = "contemporaneous r"

    if len(xs) < MIN_RESOLVED_FOR_CORRELATION:
        return AuditOutcome(name, None, False, 0, None, True, f"only {len(xs)} resolved headline(s) - not enough to test")

    real = pearson_with_ci(xs, ys)
    if not _notable(real.ci_low, real.ci_high):
        return AuditOutcome(
            name, real.r, False, 0, None, True,
            f"95% CI [{real.ci_low:+.3f}, {real.ci_high:+.3f}] already includes zero - no claimed signal to disprove",
        )

    p_value, n_null, n_extreme = _permutation_p_value(real.r, headlines, contemporaneous_r, live, n_perm, seed)
    if p_value is None:
        return AuditOutcome(name, real.r, True, 0, None, False, "no permutation produced a usable statistic")

    passed = p_value <= P_VALUE_THRESHOLD
    reason = (
        f"shuffled-timestamp control reproduced |r|>={abs(real.r):.3f} in {n_extreme}/{n_null} permutations "
        f"(p={p_value:.3f})"
    )
    return AuditOutcome(name, real.r, True, n_null, p_value, passed, reason)


def audit_event_diff(
    headlines: list[Headline],
    threshold: float = DEFAULT_EVENT_THRESHOLD,
    live: bool = False,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = DEFAULT_SEED,
) -> AuditOutcome:
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    high = [r["contemporaneous_return"] for r in real_rows if abs(r["compound"]) > threshold]
    low = [r["contemporaneous_return"] for r in real_rows if abs(r["compound"]) <= threshold]
    name = "event-study diff"

    if not high or not low:
        return AuditOutcome(
            name, None, False, 0, None, True,
            f"threshold {threshold} leaves one group empty (high={len(high)}, low={len(low)})",
        )

    real = bootstrap_mean_diff_ci(high, low)
    if not _notable(real.ci_low, real.ci_high):
        return AuditOutcome(
            name, real.diff, False, 0, None, True,
            f"95% CI [{real.ci_low:+.4f}, {real.ci_high:+.4f}] already includes zero - no claimed signal to disprove",
        )

    p_value, n_null, n_extreme = _permutation_p_value(
        real.diff, headlines, lambda rows: event_diff(rows, threshold), live, n_perm, seed
    )
    if p_value is None:
        return AuditOutcome(name, real.diff, True, 0, None, False, "no permutation produced a usable statistic")

    passed = p_value <= P_VALUE_THRESHOLD
    reason = (
        f"shuffled-timestamp control reproduced |diff|>={abs(real.diff):.4f} in {n_extreme}/{n_null} permutations "
        f"(p={p_value:.3f})"
    )
    return AuditOutcome(name, real.diff, True, n_null, p_value, passed, reason)


def run(in_path: Path, live: bool, event_threshold: float, n_perm: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    outcomes = [
        audit_correlation(headlines, live=live, n_perm=n_perm, seed=seed),
        audit_event_diff(headlines, event_threshold, live=live, n_perm=n_perm, seed=seed + 1),
    ]

    for outcome in outcomes:
        stat_str = f"{outcome.real_stat:+.4f}" if outcome.real_stat is not None else "n/a"
        verdict = "PASS" if outcome.passed else "FAIL"
        print(f"{outcome.name}: real={stat_str}  notable={outcome.notable}  {verdict} - {outcome.reason}")

    return 0 if all(o.passed for o in outcomes) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=DEFAULT_EVENT_THRESHOLD,
        help="|compound| above this is 'high-magnitude' for the event-study statistic",
    )
    parser.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM, help="number of timestamp permutations")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed for the permutation RNG")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.event_threshold, args.n_perm, args.seed))


if __name__ == "__main__":
    main()
