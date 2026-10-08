"""Day 8 CLI: the ml-pipeline-audit leakage control.

    python -m sentiment.audit
    python -m sentiment.audit --live
    python -m sentiment.audit --trials 1000

NEXT_STEPS.md's "Done when" test: shuffle the headline timestamps and the
sentiment/return signal must disappear. If a shuffled-timestamp control
still predicts returns as well as the real, correctly-timed pipeline does,
the pipeline is leaking information a leak-free design should not have, and
the real result is an artifact of that leak rather than a finding.

What "shuffle the timestamps" means here, precisely: take the same set of
headlines (same titles, same VADER scores - ``sentiment.vader_score.
score_headline`` only ever reads ``title``) and redistribute their
``published_at`` values among each other. This breaks exactly one thing per
headline - whether the timestamp attached to it is genuinely its own - and
therefore which session Day 4's ``align_headline`` assigns it to, and so
which session's return Day 5's ``build_rows`` pairs its sentiment score
against. A headline's ticker (and hence which price series it draws from)
depends only on its title via ``sentiment.tickers.resolve``, so that part of
the pipeline is untouched by the shuffle - only the time axis moves.

The real pipeline's correlation (``sentiment.correlate``'s contemporaneous
r) is compared against a null distribution built from many independent
shuffles: a permutation p-value is the fraction of shuffled trials whose
|r| is at least as extreme as the real |r|. A small p-value here would mean
the real correlation sits outside what shuffled, deliberately-mistimed data
can produce by chance - evidence of a leak. A p-value that is not small
means the real correlation is unremarkable next to the shuffled noise floor,
consistent with (though it cannot on its own prove) no leak.

See the README's Day 8 Findings for why, on this fixture, a PASS here is a
necessary control that this pipeline already satisfies, not strong evidence
that a *real* signal - if this fixture ever produced one - would be trustworthy.
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


@dataclass(frozen=True)
class AuditResult:
    real_r: float
    real_n: int
    shuffled_rs: list[float]

    @property
    def trials(self) -> int:
        return len(self.shuffled_rs)

    @property
    def shuffled_mean(self) -> float:
        return sum(self.shuffled_rs) / len(self.shuffled_rs)

    @property
    def p_value(self) -> float:
        """Two-sided permutation p-value: the fraction of shuffled trials
        whose |r| is at least as large as the real pipeline's |r|. This, not
        the raw shuffled mean, is the actual leakage check - a real
        correlation that is a routine member of the shuffled null
        distribution is not evidence of a leak; one that is an outlier
        relative to it is."""
        extreme = sum(1 for r in self.shuffled_rs if abs(r) >= abs(self.real_r))
        return extreme / len(self.shuffled_rs)


def shuffle_published_at(headlines: list[Headline], rng: random.Random) -> list[Headline]:
    """Return a new list of the same headlines with ``published_at`` permuted
    across them - the same multiset of timestamps, reassigned to different
    titles. Title, link, and therefore sentiment score and resolved ticker
    are left exactly as they were; only which timestamp (and so which
    session) each headline is judged against changes."""
    shuffled_timestamps = [h.published_at for h in headlines]
    rng.shuffle(shuffled_timestamps)
    return [replace(h, published_at=ts) for h, ts in zip(headlines, shuffled_timestamps)]


def run_audit(headlines: list[Headline], live: bool, trials: int, seed: int) -> AuditResult | None:
    """Return the audit result, or ``None`` if there are not enough resolved
    headlines (real or shuffled) to compute a correlation at all."""
    real_rows, _ = build_rows_from_headlines(headlines, live=live)
    if len(real_rows) < MIN_ROWS_FOR_CORRELATION:
        return None

    real_r = pearson_r(
        [r["compound"] for r in real_rows],
        [r["contemporaneous_return"] for r in real_rows],
    )

    rng = random.Random(seed)
    shuffled_rs: list[float] = []
    for _ in range(trials):
        shuffled_headlines = shuffle_published_at(headlines, rng)
        shuffled_rows, _ = build_rows_from_headlines(shuffled_headlines, live=live)
        if len(shuffled_rows) < 2:
            continue
        shuffled_rs.append(
            pearson_r(
                [r["compound"] for r in shuffled_rows],
                [r["contemporaneous_return"] for r in shuffled_rows],
            )
        )

    if not shuffled_rs:
        return None

    return AuditResult(real_r=real_r, real_n=len(real_rows), shuffled_rs=shuffled_rs)


def run(in_path: Path, live: bool, trials: int, seed: int) -> int:
    if not in_path.exists():
        print(f"no headlines file at {in_path}; run sentiment.scrape first", file=sys.stderr)
        return 1

    headlines = read_csv(in_path)
    result = run_audit(headlines, live=live, trials=trials, seed=seed)
    if result is None:
        print("not enough resolved headlines (real or shuffled) to audit", file=sys.stderr)
        return 1

    print(f"real pipeline:    contemporaneous r={result.real_r:+.3f}  n={result.real_n}")
    print(
        f"shuffled control: mean r={result.shuffled_mean:+.3f}  "
        f"range [{min(result.shuffled_rs):+.3f}, {max(result.shuffled_rs):+.3f}]  "
        f"trials={result.trials}"
    )
    print(f"permutation p-value (|shuffled r| >= |real r|): {result.p_value:.3f}")

    if result.p_value < 0.05:
        print(
            "FAIL: the real correlation is an outlier against the shuffled-timestamp "
            "null distribution - on a leak-free pipeline this should not happen. "
            "Treat the real result as unverified until the leak is found."
        )
        return 1

    print(
        "PASS: the real correlation is not distinguishable from the shuffled-timestamp "
        "control. See the README's Day 8 'why this might be spurious' section for what "
        "this test does and does not rule out."
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN, help="headlines CSV to read")
    parser.add_argument("--live", action="store_true", help="re-fetch prices instead of reading fixtures")
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS, help="number of independent timestamp shuffles to run"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed, for a deterministic run")
    args = parser.parse_args()
    sys.exit(run(args.in_path, args.live, args.trials, args.seed))


if __name__ == "__main__":
    main()
