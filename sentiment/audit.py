"""Day 8 CLI: the leakage audit. Shuffle headline timestamps, confirm the
sentiment/return correlation does not survive it.

    python -m sentiment.audit
    python -m sentiment.audit --trials 2000 --seed 1
    python -m sentiment.audit --live

This is the "Done when" gate NEXT_STEPS.md set before Day 1 was written: if a
shuffled-timestamp control reproduces the real pipeline's correlation about
as often as a real signal would, the apparent relationship does not actually
depend on Day 4's leak-free alignment and should not be trusted. It is a
permutation test, not a one-off manual check - ``tests/test_audit.py`` runs
it against the committed fixture on every ``pytest`` invocation, in CI, not
once by hand.

What gets shuffled: each headline keeps its title (so VADER's score and
``sentiment.tickers.resolve``'s ticker match are untouched) but the corpus's
``published_at`` values are redistributed across headlines at random. That
changes which trading session (Day 4's ``align_headline``) - and therefore
which return - each headline's fixed sentiment score ends up paired with,
without inventing timestamps outside the range the corpus actually covers.

Only the contemporaneous correlation is audited. The lagged correlation and
Day 6's regression are both structurally zero-variance on this fixture (see
README "Why this result might still be spurious") - shuffling a predictor
that is already identical across every row it is computed on tells us
nothing, so running the same test on them would be theatre, not evidence.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from sentiment.correlate import DEFAULT_IN, build_rows_from_headlines
from sentiment.headline import Headline, read_csv
from sentiment.stats import pearson_r

DEFAULT_TRIALS = 500
DEFAULT_SEED = 0
MIN_ROWS_FOR_CORRELATION = 4


def shuffle_timestamps(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a copy of ``headlines`` with ``published_at`` values permuted
    across the corpus - a shuffled-timestamp control.

    Every other field (title, source, link, scraped_at) stays attached to
    its original headline, so sentiment content and which ticker a headline
    resolves to are both untouched; only which trading session a headline's
    score gets paired with can change. The result is re-sorted by its new
    ``published_at``, matching the oldest-first order
    ``sentiment.headline.write_csv`` always produces and that
    ``sentiment.correlate``/``sentiment.regress`` assume.
    """
    times = [h.published_at for h in headlines]
    rng.shuffle(times)
    shuffled = [replace(h, published_at=t) for h, t in zip(headlines, times)]
    return sorted(shuffled, key=lambda h: h.published_at)


def contemporaneous_r(rows: list[dict]) -> float | None:
    """Pearson r between ``compound`` and ``contemporaneous_return`` across
    ``rows``, or ``None`` if there are too few rows to correlate at all."""
    if len(rows) < 2:
        return None
    return pearson_r([r["compound"] for r in rows], [r["contemporaneous_return"] for r in rows])


def permutation_p_value(real_stat: float, null_stats: list[float]) -> float:
    """Two-sided empirical p-value: the fraction of null-distribution draws
    at least as extreme as ``real_stat``.

    Uses the standard +1/+1 correction (Davison & Hinkley) so a finite
    sample never reports an impossible p-value of exactly 0 - with
    ``DEFAULT_TRIALS`` draws the smallest value this can return is
    1/(DEFAULT_TRIALS + 1), not 0.
    """
    if not null_stats:
        raise ValueError("need at least one null-distribution draw")
    extreme = sum(1 for s in null_stats if abs(s) >= abs(real_stat))
    return (extreme + 1) / (len(null_stats) + 1)


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    null_rs: list[float]

    @property
    def trials(self) -> int:
        return len(self.null_rs)

    @property
    def p_value(self) -> float:
        return permutation_p_value(self.real_r, self.null_rs)

    @property
    def null_mean(self) -> float:
        return sum(self.null_rs) / len(self.null_rs)

    def passes(self, alpha: float = 0.05) -> bool:
        """True if the real correlation is not a significant outlier against
        the shuffled-timestamp null - i.e. the signal does not survive
        scrutiny, which is what "the signal must disappear" means when
        (as here) there was barely a signal to begin with. False would mean
        the real statistic is unusually large next to what random timestamp
        pairings alone produce - the shape a genuine leak would take."""
        return self.p_value > alpha


def run_audit(
    in_path: Path,
    live: bool = False,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> AuditResult | None:
    """Run the permutation test against the headlines at ``in_path``.

    Returns ``None`` if the real pipeline does not even produce enough
    resolved headlines to compute a correlation - there is nothing to audit.
    """
    headlines = read_csv(in_path)
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    real_r = contemporaneous_r(real_rows)
    if real_r is None or len(real_rows) < MIN_ROWS_FOR_CORRELATION:
        return None

    rng = random.Random(seed)
    null_rs: list[float] = []
    for _ in range(trials):
        shuffled = shuffle_timestamps(headlines, rng)
        rows, _ = build_rows_from_headlines(shuffled, live=live)
        r = contemporaneous_r(rows)
        if r is not None:
            null_rs.append(r)

    if not null_rs:
        return None
    return AuditResult(real_r=real_r, real_n=len(real_rows), null_rs=null_rs)


def run(in_path: Path, live: bool, trials: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    result = run_audit(in_path, live=live, trials=trials, seed=seed)
    if result is None:
        print("not enough resolved headlines to audit a correlation", file=sys.stderr)
        return 1

    verdict = "PASS - signal does not survive the shuffle" if result.passes() else "FAIL - shuffled control still correlates"
    print(f"real contemporaneous r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled-timestamp null: {result.trials} trials, mean r={result.null_mean:+.3f}, "
        f"range [{min(result.null_rs):+.3f}, {max(result.null_rs):+.3f}]"
    )
    print(f"two-sided permutation p={result.p_value:.3f}  -> {verdict}")
    return 0 if result.passes() else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="number of shuffled-timestamp trials")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.trials, args.seed))


if __name__ == "__main__":
    main()
