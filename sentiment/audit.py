"""Day 8 CLI: shuffled-timestamp leakage audit.

    python -m sentiment.audit
    python -m sentiment.audit --n-shuffles 500 --seed 1

The "Done when" bar NEXT_STEPS.md sets for this repo: shuffle every
headline's ``published_at`` and the sentiment/return correlation must not
survive. If a shuffled-timestamp control still "predicts" returns, something
downstream of ``published_at`` is leaking real timing information back in
through a side channel, and every correlation Day 5-7 reported would be
suspect.

What a shuffle can and cannot prove here, honestly stated up front: Day 4's
``market_hours.align_headline`` already guarantees ``leak_free()`` by
construction for every alignment it produces, so this audit is not probing
*that* invariant directly - it is a permutation significance test. Re-pairing
each headline's (fixed) VADER score with a *different* headline's timing -
and therefore a different session/return - rebuilds the sampling
distribution of the correlation coefficient under the null hypothesis "the
compound score carries no timing-dependent information about returns at
all". If the correctly-timestamped correlation is unremarkable next to that
distribution, there is no evidence that the real pairing is doing anything a
random one couldn't - which, given Day 5's already-null finding (r=-0.185,
95% CI crossing zero), is exactly what should happen, and does not by
itself prove the *pipeline code* has no look-ahead bug. ``tests/test_audit.py``
closes that gap separately: it feeds this same permutation machinery a
deliberately broken alignment (same-calendar-day pairing, the exact mistake
Day 4 exists to prevent) wired to a hand-built signal, and confirms the
shuffle collapses *that* inflated correlation - proof the test has power to
catch a real leak, not just a test that always passes because there was
nothing to lose.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

BuildRowsFn = Callable[..., tuple[list[dict], list[tuple[str, str]]]]

DEFAULT_N_SHUFFLES = 500
DEFAULT_SEED = 0
SIGNIFICANCE_LEVEL = 0.05


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list with ``published_at`` (and its matching ``published_raw``
    string) randomly permuted across ``headlines``. Everything else - title,
    source, link - stays attached to its original headline, so only *when*
    each headline is believed to have been published changes, never what it
    says."""
    timestamps = [h.published_at for h in headlines]
    shuffled_timestamps = timestamps[:]
    rng.shuffle(shuffled_timestamps)
    return [
        replace(h, published_at=ts, published_raw=ts.isoformat())
        for h, ts in zip(headlines, shuffled_timestamps)
    ]


def _pearson_from_rows(rows: list[dict]) -> tuple[float | None, int]:
    """Pearson r between compound and contemporaneous_return, or (None, n) if
    there are too few rows to correlate at all."""
    n = len(rows)
    if n < 2:
        return None, n
    compounds = [r["compound"] for r in rows]
    returns = [r["contemporaneous_return"] for r in rows]
    return pearson_r(compounds, returns), n


@dataclass(frozen=True)
class ShuffleAuditResult:
    real_r: float | None
    real_n: int
    n_shuffles_requested: int
    shuffled_rs: tuple[float, ...]
    mean_abs_shuffled_r: float | None
    p_value: float | None
    passed: bool | None

    @property
    def inconclusive(self) -> bool:
        return self.passed is None


def run_shuffle_audit(
    headlines: list[Headline],
    build_rows_fn: BuildRowsFn = build_rows_from_headlines,
    live: bool = False,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    seed: int = DEFAULT_SEED,
) -> ShuffleAuditResult:
    """Run the shuffled-timestamp permutation audit against ``headlines``.

    ``build_rows_fn`` defaults to the real pipeline (``sentiment.correlate.
    build_rows_from_headlines``) but can be swapped for a different
    headline-to-row builder - ``tests/test_audit.py`` uses this to prove the
    audit actually has power to flag a deliberately broken one.

    Two-sided permutation p-value: the fraction of shuffles whose |r| is at
    least as extreme as the real, correctly-timestamped |r| (with the
    standard +1 smoothing so a p-value is never reported as exactly zero).
    ``passed`` is True when that p-value clears ``SIGNIFICANCE_LEVEL`` - the
    real pairing is not distinguishable from a random one, so there is
    nothing timing-dependent here for a leak to have produced. ``passed`` is
    None (inconclusive) when there are too few resolved headlines - real or
    shuffled - to compute a correlation at all.
    """
    real_rows, _ = build_rows_fn(headlines, live=live)
    real_r, real_n = _pearson_from_rows(real_rows)

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(n_shuffles):
        shuffled_headlines = shuffle_published_at(headlines, rng)
        shuffled_rows, _ = build_rows_fn(shuffled_headlines, live=live)
        r, _ = _pearson_from_rows(shuffled_rows)
        if r is not None:
            shuffled_rs.append(r)

    if real_r is None or not shuffled_rs:
        return ShuffleAuditResult(
            real_r=real_r,
            real_n=real_n,
            n_shuffles_requested=n_shuffles,
            shuffled_rs=tuple(shuffled_rs),
            mean_abs_shuffled_r=None,
            p_value=None,
            passed=None,
        )

    as_extreme = sum(1 for r in shuffled_rs if abs(r) >= abs(real_r))
    p_value = (as_extreme + 1) / (len(shuffled_rs) + 1)
    mean_abs_shuffled_r = sum(abs(r) for r in shuffled_rs) / len(shuffled_rs)

    return ShuffleAuditResult(
        real_r=real_r,
        real_n=real_n,
        n_shuffles_requested=n_shuffles,
        shuffled_rs=tuple(shuffled_rs),
        mean_abs_shuffled_r=mean_abs_shuffled_r,
        p_value=p_value,
        passed=p_value > SIGNIFICANCE_LEVEL,
    )


def run(in_path: Path, n_shuffles: int, seed: int, live: bool) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = run_shuffle_audit(headlines, live=live, n_shuffles=n_shuffles, seed=seed)

    if result.inconclusive:
        print(
            f"inconclusive: too few resolved headlines to correlate "
            f"(real n={result.real_n}, {len(result.shuffled_rs)}/{n_shuffles} shuffles usable)"
        )
        return 1

    print(f"real (correctly-timestamped):  r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled-timestamp null:       mean|r|={result.mean_abs_shuffled_r:.3f}  "
        f"over {len(result.shuffled_rs)}/{n_shuffles} usable shuffles"
    )
    print(f"two-sided permutation p-value: {result.p_value:.3f}")
    if result.passed:
        print(
            "PASS: the real pairing is not distinguishable from a randomly re-timed one "
            "(p > 0.05) - no evidence the correlation depends on genuine timing, consistent "
            "with Day 5-7's own null finding. See tests/test_audit.py for proof this audit "
            "can detect a real leak when one exists."
        )
    else:
        print(
            "FAIL: the real pairing is a significant outlier against the shuffled-timestamp "
            "null (p <= 0.05) - this correlation should not be trusted until the alignment "
            "and pairing code is reviewed for a look-ahead bug."
        )
    return 0 if result.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES, help="number of timestamp-shuffle iterations"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a reproducible report")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.n_shuffles, args.seed, args.live))


if __name__ == "__main__":
    main()
